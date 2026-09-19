#!/usr/bin/env python3
"""Train one frozen connection prescription on the actual five owned paths."""
import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import time
from functools import partial

import torch
import torch.distributed as dist

from neuroshard.evolution.fusion_data import correct
from neuroshard.evolution.fusion_score import score
from neuroshard.evolution.reference_data import identity, save, sha256
from neuroshard.evolution.sharded.cached_inference import generate_branch_cached
from neuroshard.evolution.sharded.fused_graph import commitment, generate_fused
from neuroshard.evolution.sharded.fusion import CrossShardFusion
from neuroshard.evolution.sharded.fusion_features import produce
from neuroshard.evolution.sharded.fusion_training import Trainer
from neuroshard.evolution.sharded.fusion_trial import response_losses, synchronize
from neuroshard.evolution.sharded.graph_execution import GraphNetwork


def run(home, source):
    read = lambda name: json.loads((home/'inputs'/name).read_bytes())
    plan, graph, profile, freeze = [read(name+'.json') for name in ('plan', 'graph', 'profile', 'freeze')]
    method = plan.get('method', 'causal-fusion')
    contracts = {'causal-fusion': 'causal-fusion-trial.json', 'probability-mixture': 'probability-mixture-trial.json'}
    if method not in contracts:
        raise ValueError('Unknown committed connection method')
    if (identity(plan) != freeze['plan'] or identity(graph) != freeze['graph']
            or sha256(source/'scripts/run_fusion_trial.py') != freeze['driver']
            or sha256(source/'config/experiments'/contracts[method]) != sha256(home/'inputs/plan.json')):
        raise ValueError('Commit the complete fusion prescription and driver before training')
    rows = {}
    for role, spec in plan['data'].items():
        path = home/'inputs'/(role+'.jsonl')
        if sha256(path) != spec['sha256']:
            raise ValueError('Frozen fusion data changed')
        rows[role] = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows[role]) != spec['rows'] or identity(rows[role]) != spec['root']:
            raise ValueError('Frozen fusion selection changed')
    rank = int(os.environ['RANK'])
    output = home/'results'
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    dist.init_process_group('gloo', timeout=timedelta(seconds=1200))
    try:
        net = GraphNetwork(graph, profile, objects=home/'objects', interpreter=home/'interpreter',
                           seed=home/'seed', source_home=source, rank=rank)
        if (identity(graph['parent']) != plan['parent']
                or {name: identity(value) for name, value in graph['experts'].items()} != plan['experts']
                or identity(graph['interpreter_assets']) != plan['hub']
                or graph['tokenizer']['root'] != plan['tokenizer']):
            raise ValueError('Trial source model inventory changed')
        device = next(net.shard.parameters()).device
        model_class, trainer_class, loss_function = CrossShardFusion, Trainer, response_losses
        if method == 'probability-mixture':
            from neuroshard.evolution.sharded.mixture import ProbabilityMixture
            from neuroshard.evolution.sharded.mixture_training import MixtureTrainer, response_losses as mixture_losses
            model_class = ProbabilityMixture
            trainer_class = partial(MixtureTrainer, source_head=net.shard)
            loss_function = partial(mixture_losses, source_head=net.shard)
        torch.manual_seed(plan['initialization_seed'])
        width = graph['parent']['config']['hidden_size']
        initial = model_class(width, {name: width for name in ['parent', *graph['experts']]},
                                   **plan['connection']).to(device)
        save(output/'ready.json', {'rank': rank, 'plan': identity(plan), 'graph': identity(graph),
            'runtime': net.runtime, 'fusion_parameters': sum(p.numel() for p in initial.parameters()),
            'owned_parameters': net.shard.resident_parameters,
            'preserved_parameters': net.preserved.shard.resident_parameters if net.preserved else 0})
        probe = net.tokenizer.apply_chat_template(rows['train'][0]['messages'][:-1],
                                                 tokenize=True, add_generation_prompt=True)
        baseline = generate_branch_cached(net.preserved, probe, 8, False) if rank < 3 else None
        baseline = net.all_owners.exchange(baseline)[0]
        initial.eval()
        initialized = generate_fused(net, initial, probe, 8)
        if initialized != baseline:
            raise ValueError('Zero connection changed the actual GPU assistant before training')
        save(output/'initialization.json', {'hub_ids': baseline, 'fusion_ids': initialized, 'equal': True})
        banks = {}
        banks['train'], resources = produce(net, rows['train'], plan['batches']['train'], home/'features-train',
            max_length=plan['max_length'], max_seconds=plan['feature_max_seconds'])
        save(output/'features-train.json', {'bank': banks['train'], 'resources': resources})
        models, roots = {}, {}
        for arm in ('fusion', 'ablation'):
            model = copy.deepcopy(initial)
            trainer = (trainer_class(model, net.preserved.shard, rows['train'], banks['train'], home/'features-train',
                               plan['training'], source_ablation=arm == 'ablation') if rank == 0 else None)
            log = []
            for step in range(plan['training']['steps']):
                record = None
                if trainer:
                    record = trainer.advance()
                    log.append(record)
                    if (step+1) % plan['checkpoint_every'] == 0 or step+1 == plan['training']['steps']:
                        record = {**record, 'checkpoint': trainer.save(output/('checkpoints-'+arm))}
                        save(output/('training-'+arm+'.json'), log)
                packet = net.all_owners.exchange(record)
                if packet[0]['step'] != step+1 or any(value is not None for value in packet[1:]):
                    raise ValueError('Fusion owners disagree on prescribed progress')
                save(output/'progress.json', {'phase': 'training', 'arm': arm, 'step': step+1,
                                              'seconds': time.monotonic()-started})
            verification = None
            if trainer:
                terminal = trainer.save(output/('checkpoints-'+arm))
                previous = json.loads((output/('checkpoints-'+arm)/
                    ('step-'+str(plan['training']['steps']-plan['replay_last_steps'])+'.json')).read_bytes())
                replay = trainer_class(copy.deepcopy(initial), net.preserved.shard, rows['train'], banks['train'],
                                 home/'features-train', plan['training'], source_ablation=arm == 'ablation')
                replay.restore(output/('checkpoints-'+arm), previous)
            for _ in range(plan['replay_last_steps']):
                observed = replay.advance() if trainer else None
                net.all_owners.exchange(observed)
            if trainer:
                reconstructed = replay.save(output/('replayed-'+arm))
                if reconstructed != terminal:
                    raise ValueError('GPU optimizer restart changed the terminal fusion state')
                verification = {'start': previous, 'terminal': terminal, 'equal': True,
                                'scope': 'Same-owner numerical restart replay, not an independent audit'}
                del replay
            save(output/('restart-'+arm+'.json'), net.all_owners.exchange(verification)[0])
            roots[arm] = synchronize(net, model)
            models[arm] = model
        reports = {}
        for role in ('dev', 'test'):
            if role == 'test' and not reports['dev']['passed']:
                break
            banks[role], resources = produce(net, rows[role], plan['batches'][role], home/('features-'+role),
                max_length=plan['max_length'], max_seconds=plan['feature_max_seconds'])
            save(output/('features-'+role+'.json'), {'bank': banks[role], 'resources': resources})
            loss = (loss_function(models, net.preserved.shard, rows[role], banks[role], home/('features-'+role),
                                    plan['training']) if rank == 0 else None)
            loss = net.all_owners.exchange(loss)[0]
            save(output/('losses-'+role+'.json'), loss)
            answers = {}
            for number, row in enumerate(rows[role]):
                # Only original conversation tokens reach generation. References,
                # knowledge groups and task strata are used after generation.
                tokens = net.tokenizer.apply_chat_template(row['messages'][:-1], tokenize=True,
                                                          add_generation_prompt=True)
                responses = {}
                for arm in ('hub', 'fusion', 'ablation'):
                    at, before, observed = time.monotonic(), net.all_owners.sent_tensor_bytes, {}
                    if arm == 'hub':
                        ids = generate_branch_cached(net.preserved, tokens, plan['max_new_tokens'], False, observed) if rank < 3 else None
                        ids = net.all_owners.exchange(ids)[0]
                    else:
                        ids = generate_fused(net, models[arm], tokens, plan['max_new_tokens'], observed,
                                             source_ablation=arm == 'ablation')
                    text = net.tokenizer.decode(ids, skip_special_tokens=True).strip()
                    responses[arm] = {'text': text, 'ids': ids, 'correct': correct(text, row)}
                    save(output/('generation-'+role+'-'+str(number)+'-'+arm+'.json'),
                         {'id': row['id'], 'response': responses[arm], 'observation': observed,
                          'seconds': time.monotonic()-at,
                          'sent_tensor_bytes': observed.get('sent_tensor_bytes', 0) if arm == 'hub' else
                                               net.all_owners.sent_tensor_bytes-before})
                answers[row['id']] = responses
                save(output/('answers-'+role+'.json'), answers)
                save(output/'progress.json', {'phase': 'generation', 'role': role, 'documents': number+1,
                                              'seconds': time.monotonic()-started})
            reports[role] = score(rows[role], answers, loss, plan['gates'][role], plan['scoring_seed'])
            save(output/('score-'+role+'.json'), reports[role])
        net.verify_unchanged()
        result = {'plan': identity(plan), 'graph': identity(graph), 'fusion': roots,
                  'reports': reports, 'final_opened': 'test' in reports,
                  'passed': 'test' in reports and reports['test']['passed'],
                  'limitations': plan['limitations']}
        if net.all_owners.exchange(identity(result)) != [identity(result)]*net.world_size:
            raise ValueError('Owned trial reports disagree')
        save(output/'resources.json', {'seconds': time.monotonic()-started,
             'sent_tensor_bytes': net.all_owners.sent_tensor_bytes,
             'max_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0})
        save(output/'result.json', result)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    run(args.home, args.source)
