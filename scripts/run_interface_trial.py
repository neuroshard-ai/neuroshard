#!/usr/bin/env python3
"""Prospectively frozen joint training of independently owned expert interfaces."""
import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from neuroshard.evolution.fusion_data import correct
from neuroshard.evolution.fusion_score import score
from neuroshard.evolution.reference_data import identity, save, sha256
from neuroshard.evolution.sharded.cached_inference import generate_branch_cached
from neuroshard.evolution.sharded.expert_interface import ExpertInterface
from neuroshard.evolution.sharded.fused_graph import commitment, generate_fused
from neuroshard.evolution.sharded.fusion_features import produce
from neuroshard.evolution.sharded.fusion_trial import synchronize
from neuroshard.evolution.sharded.graph_execution import GraphNetwork
from neuroshard.evolution.sharded.interface_training import OwnedInterfaceTraining, initialize_weights
from neuroshard.evolution.sharded.mixture import ProbabilityMixture
from neuroshard.evolution.sharded.mixture_training import response_losses


def run(home, source):
    read = lambda name: json.loads((home/'inputs'/name).read_bytes())
    plan, graph, profile, freeze = [read(name+'.json') for name in ('plan', 'graph', 'profile', 'freeze')]
    contract = {'expert-interface': 'expert-interface-trial.json',
                'expert-interface-continuation': 'expert-interface-continuation.json'}.get(plan['method'])
    if contract is None:
        raise ValueError('Unknown interface experiment prescription')
    if (identity(plan) != freeze['plan'] or identity(graph) != freeze['graph']
            or sha256(source/'scripts/run_interface_trial.py') != freeze['driver']
            or sha256(source/'config/experiments'/contract) != sha256(home/'inputs/plan.json')
            or identity(graph['parent']) != plan['parent']
            or {name: identity(value) for name, value in graph['experts'].items()} != plan['experts']
            or identity(graph['interpreter_assets']) != plan['hub']
            or graph['tokenizer']['root'] != plan['tokenizer']):
        raise ValueError('Commit the complete interface prescription, model inventory and driver before training')
    rows = {}
    for role, spec in plan['data'].items():
        path = home/'inputs'/(role+'.jsonl')
        if sha256(path) != spec['sha256']:
            raise ValueError('Frozen interface data changed')
        rows[role] = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows[role]) != spec['rows'] or identity(rows[role]) != spec['root']:
            raise ValueError('Frozen interface selection changed')
    rank = int(os.environ['RANK'])
    output = home/'results'
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    dist.init_process_group('gloo', timeout=timedelta(seconds=1200))
    try:
        net = GraphNetwork(graph, profile, objects=home/'objects', interpreter=home/'interpreter',
                           seed=home/'seed', source_home=source, rank=rank)
        width, device = graph['parent']['config']['hidden_size'], net.shard.device_name
        torch.manual_seed(plan['initialization_seed'])
        initial_gate = ProbabilityMixture(width, {name: width for name in ['parent', *graph['experts']]},
                                          **plan['connection']).to(device).eval()
        adapter = None
        if rank >= 3:
            torch.manual_seed(plan['interface_seed']+rank)
            name = graph['descriptor']['rules'][rank-3]['id']
            adapter = ExpertInterface(net.shard, identity(graph['experts'][name]), plan['interface_rank']).eval()
        gates = {arm: copy.deepcopy(initial_gate) for arm in ('fusion', 'ablation')}
        if 'initial_weights' in plan:
            initial_weights = plan['initial_weights']
            if initial_weights['optimizer'] != 'new-per-owner-adam-at-step-zero':
                raise ValueError('A continuation must explicitly prescribe fresh optimizer state')
            if rank == 0:
                for arm, gate in gates.items():
                    initialize_weights(gate, home/'inputs', initial_weights['gates'][arm])
            if adapter is not None:
                initialize_weights(adapter, home/'inputs', initial_weights['interfaces'][str(rank)])
            for gate in gates.values():
                synchronize(net, gate)
        initial_adapter = copy.deepcopy(adapter)
        save(output/'ready.json', {'rank': rank, 'plan': identity(plan), 'graph': identity(graph),
            'runtime': net.runtime, 'gate_parameters': sum(p.numel() for p in initial_gate.parameters()),
            'interface_parameters': sum(p.numel() for p in adapter.parameters()) if adapter else 0,
            'owned_parameters': net.shard.resident_parameters,
            'preserved_parameters': net.preserved.shard.resident_parameters if net.preserved else 0})
        probe = net.tokenizer.apply_chat_template(rows['train'][0]['messages'][:-1],
                                                 tokenize=True, add_generation_prompt=True)
        baseline = generate_branch_cached(net.preserved, probe, 8, False) if rank < 3 else None
        baseline = net.all_owners.exchange(baseline)[0]
        initial = generate_fused(net, gates['fusion'], probe, 8, interface=adapter, adapt_interfaces=True)
        if 'initial_weights' not in plan and initial != baseline:
            raise ValueError('Zero interfaces and connection changed the actual GPU assistant')
        save(output/'initialization.json', {'hub_ids': baseline, 'candidate_ids': initial, 'equal': initial == baseline,
             'weight_sources': plan.get('initial_weights'), 'initial_gates': {arm: commitment(gate) for arm, gate in gates.items()}})
        bank, resources = produce(net, rows['train'], plan['batches']['train'], home/'features-train',
            max_length=plan['max_length'], max_seconds=plan['feature_max_seconds'], include_prefix=True)
        save(output/'features-train.json', {'bank': bank, 'resources': resources})
        models, roots = {}, {}
        for arm in ('fusion', 'ablation'):
            model = copy.deepcopy(gates[arm])
            enabled = arm == 'fusion'
            training = OwnedInterfaceTraining(net, model, adapter, rows['train'], bank, home/'features-train',
                                               plan['training'], plan['objective'], enabled=enabled)
            records = []
            for step in range(plan['training']['steps']):
                records.append(training.advance())
                if ((step+1) % plan['checkpoint_every'] == 0
                        or step+1 in (plan['training']['steps'], plan['training']['steps']-plan['replay_last_steps'])):
                    checkpoint = training.save(output/('checkpoints-'+arm))
                    save(output/('window-'+arm+'-'+str(step+1)+'.json'), checkpoint)
                    save(output/('training-'+arm+'.json'), records)
                save(output/'progress.json', {'phase': 'training', 'arm': arm, 'step': step+1,
                                              'seconds': time.monotonic()-started})
            terminal = training.save(output/('checkpoints-'+arm))
            replay = OwnedInterfaceTraining(net, copy.deepcopy(gates[arm]), copy.deepcopy(initial_adapter),
                rows['train'], bank, home/'features-train', plan['training'], plan['objective'], enabled=enabled)
            previous = json.loads((output/('window-'+arm+'-'+
                str(plan['training']['steps']-plan['replay_last_steps'])+'.json')).read_bytes())
            replay.restore(output/('checkpoints-'+arm), previous)
            for _ in range(plan['replay_last_steps']):
                replay.advance()
            reproduced = replay.save(output/('replayed-'+arm))
            if reproduced != terminal:
                raise ValueError('Owned interface optimizer restart changed terminal gate or adapter bytes')
            save(output/('restart-'+arm+'.json'), {'start': previous, 'terminal': terminal, 'equal': True,
                'scope': 'Same owners restore complete optimizer state, not an independent audit'})
            del replay, training
            roots[arm] = synchronize(net, model)
            models[arm] = model
        interface_roots = net.all_owners.exchange(commitment(adapter) if adapter is not None else None)
        reports = {}
        for role in ('dev', 'test'):
            if role == 'test' and not reports['dev']['passed']:
                break
            loss, answers = {}, {}
            for arm in ('fusion', 'ablation'):
                folder = home/('features-'+role+'-'+arm)
                enabled = arm == 'fusion'
                observed_bank, resources = produce(net, rows[role], plan['batches'][role], folder,
                    max_length=plan['max_length'], max_seconds=plan['feature_max_seconds'],
                    interface=adapter if enabled else None, adapt_interfaces=enabled)
                save(output/('features-'+role+'-'+arm+'.json'), {'bank': observed_bank, 'resources': resources})
                values = (response_losses({'fusion': models[arm], 'ablation': models[arm]}, net.preserved.shard,
                    rows[role], observed_bank, folder, plan['training'], source_head=net.shard) if rank == 0 else None)
                values = net.all_owners.exchange(values)[0]
                for key, value in values.items():
                    loss.setdefault(key, {})[arm] = value['fusion']
                    if 'hub' in loss[key] and loss[key]['hub'] != value['hub']:
                        raise ValueError('Interface evaluation changed the preserved baseline')
                    loss[key]['hub'] = value['hub']
            save(output/('losses-'+role+'.json'), loss)
            for number, row in enumerate(rows[role]):
                tokens = net.tokenizer.apply_chat_template(row['messages'][:-1], tokenize=True,
                                                          add_generation_prompt=True)
                answers[row['id']] = {}
                for arm in ('hub', 'fusion', 'ablation'):
                    observed = {}
                    if arm == 'hub':
                        ids = generate_branch_cached(net.preserved, tokens, plan['max_new_tokens'], False,
                                                     observed) if rank < 3 else None
                        ids = net.all_owners.exchange(ids)[0]
                    else:
                        ids = generate_fused(net, models[arm], tokens, plan['max_new_tokens'], observed,
                            interface=adapter if arm == 'fusion' else None, adapt_interfaces=arm == 'fusion')
                    text = net.tokenizer.decode(ids, skip_special_tokens=True).strip()
                    answers[row['id']][arm] = {'ids': ids, 'text': text, 'correct': correct(text, row)}
                    save(output/('generation-'+role+'-'+str(number)+'-'+arm+'.json'), observed)
                save(output/('answers-'+role+'.json'), answers)
                save(output/'progress.json', {'phase': 'generation', 'role': role, 'documents': number+1,
                                              'seconds': time.monotonic()-started})
            reports[role] = score(rows[role], answers, loss, plan['gates'][role], seed=plan['scoring_seed'])
            save(output/('score-'+role+'.json'), reports[role])
        result = {'plan': identity(plan), 'graph': identity(graph), 'gates': roots, 'interfaces': interface_roots,
                  'reports': reports, 'final_opened': 'test' in reports,
                  'passed': reports.get('test', {}).get('passed', False), 'limitations': plan['limitations']}
        if 'batched_audit' in plan:
            from neuroshard.evolution.sharded.batched_audit import verify
            specification = plan['batched_audit']
            development_answers = json.loads((output/'answers-dev.json').read_bytes())
            selected = {row['id']: (index, row) for index, row in enumerate(rows['dev'])}
            if (len(set(specification['cases'])) != len(specification['cases'])
                    or not set(specification['cases']) <= set(selected)):
                raise ValueError('Audit diagnostic must use only its frozen development cases')
            observations = []
            for case_number, key in enumerate(specification['cases']):
                index, row = selected[key]
                prompt = net.tokenizer.apply_chat_template(row['messages'][:-1], tokenize=True,
                                                           add_generation_prompt=True)
                tokens = development_answers[key]['fusion']['ids']
                checked = verify(net, models['fusion'], prompt, tokens, plan['max_new_tokens'],
                    specification['context'], home/('features-audit-'+str(case_number)),
                    interface=adapter, adapt_interfaces=True)
                generation = json.loads((output/('generation-dev-'+str(index)+'-fusion.json')).read_bytes())
                checked['generation_owners'] = net.all_owners.exchange(generation)
                checked['id'] = key
                if case_number < specification['tamper_cases']:
                    forged = list(tokens)
                    forged[0] = (forged[0]+1) % graph['parent']['config']['vocab_size']
                    if forged[0] == net.tokenizer.eos_token_id:
                        forged[0] = (forged[0]+1) % graph['parent']['config']['vocab_size']
                    if len(forged) == 1 and len(forged) < plan['max_new_tokens']:
                        forged.append(net.tokenizer.eos_token_id)
                    rejected = verify(net, models['fusion'], prompt, forged, plan['max_new_tokens'],
                        specification['context'], home/('features-forged-'+str(case_number)),
                        interface=adapter, adapt_interfaces=True)
                    if (rejected['result']['passed'] or rejected['result']['predicted'][0] != checked['result']['predicted'][0]):
                        raise ValueError('A changed future token altered the checked first-token prediction')
                    checked['forged'] = rejected
                observations.append(checked)
                save(output/'batched-audit.json', observations)
                save(output/'progress.json', {'phase': 'batched_audit', 'cases': case_number+1,
                                              'seconds': time.monotonic()-started})
            result['batched_audit'] = {'cases': len(observations),
                'cached_responses_matched': sum(value['result']['passed'] for value in observations),
                'tampered_responses_rejected': specification['tamper_cases'],
                'scope': 'Diagnostic only; does not change the quality gate or native execution acceptance'}
        if net.all_owners.exchange(identity(result)) != [identity(result)]*net.world_size:
            raise ValueError('Interface owners disagree on the complete quality result')
        save(output/'result.json', result)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    run(args.home, args.source)
