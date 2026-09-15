"""Real-model execution for a frozen-parent, independently trained shard tail."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, LlamaConfig

from .. import incremental_capacity as contract, incremental_facts as facts
from .. import reference, reference_data as data
from . import incremental, incremental_state, portable
from .model import Partition, batch_tensors
from .training import generate, score
from .wire import Wire


def emit(event, **values):
    print(json.dumps({'event': event, 'time': time.time(), **values}), flush=True)


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items() if key != 'seconds'}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


def tokenizer_for(plan, seed):
    for name, digest in plan['tokenizer_files'].items():
        if Path(name).name != name or data.sha256(seed / name) != digest:
            raise ValueError('Tokenizer asset differs from the frozen comparison')
    tokenizer = AutoTokenizer.from_pretrained(seed, local_files_only=True, trust_remote_code=False)
    if data.tokenizer_identity(tokenizer) != plan['tokenizer']:
        raise ValueError('Tokenizer identity differs')
    return tokenizer


def prepare(args):
    plan = contract.frozen_plan(args.plan, args.plan_commit)
    tokenizer = tokenizer_for(plan, args.seed)
    old_path = contract.ROOT / plan['replay']['prepared_path']
    old = json.loads(contract.committed(old_path))
    if (data.identity(old) != plan['replay']['prepared']
            or len(old['schedule']) != plan['replay']['completed_updates']
            or set(i for batch in old['schedule'] for i in batch['indices']) != set(range(old['roles']['train']['count']))):
        raise ValueError('Replay must come from all and only completed parent training windows')
    original = {}
    for role, spec in old['roles'].items():
        rows = data.read_records(args.replay_inputs / spec['file'], spec['sha256'])
        if [row['id'] for row in rows] != spec['ids']:
            raise ValueError('Original parent inputs differ from the completed manifest')
        original[role] = rows
    selected = contract.replay(plan, original['train'])
    roles = {}
    for target, source in (('train', 'train'), ('dev-knowledge', 'dev'), ('test-knowledge', 'test')):
        roles[target] = [{**row, **data.conversation(tokenizer, row['messages'], plan['max_length'])}
                         for row in facts.raw_examples(plan['seeds']['facts'], plan['cohort'], source)]
    for family, indices in selected.items():
        for index in indices:
            row = original['train'][index]
            roles['train'].append({**row, 'source_index': index,
                'stratum': 'replay-conversation' if family == 'conversation' else 'replay-skill', 'distill': True})
    for row in roles['train']:
        row['loss_weight'] = 1. / row['targets']
    rng = random.Random(plan['seeds']['skill_probes'])
    for target, source_roles in (('dev-skills', ('dev-prior', 'dev-new')),
                                  ('test-skills', ('test-prior', 'test-new'))):
        roles[target] = []
        for source in source_roles:
            rows = original[source]
            if target == 'dev-skills':
                chosen = []
                for family in ('sort', 'lookup', 'filter', 'total'):
                    indices = [i for i, row in enumerate(rows) if row['task']['family'] == family]
                    rng.shuffle(indices)
                    chosen.extend(indices[:32 if source == 'dev-prior' else 16])
                rows = [rows[i] for i in sorted(chosen)]
            roles[target].extend({**row, 'source_role': source,
                'task': {**row['task'], 'reasoning_allowed': source.endswith('new')}} for row in rows)
    roles['dev-conversation'] = original['dev-retention']
    roles['test-conversation'] = original['retention']
    if {role: len(rows) for role, rows in roles.items()} != plan['roles']:
        raise ValueError('Prepared role sizes differ from the frozen quotas')
    identifiers = [row['id'] for rows in roles.values() for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('Training and evaluation pools must have disjoint record identities')
    args.home.mkdir(parents=True, exist_ok=False)
    files = {}
    for role, rows in roles.items():
        path = args.home / (role + '.jsonl')
        with path.open('x') as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
        files[role] = {'file': path.name, 'count': len(rows), 'sha256': data.sha256(path),
                       'ids': [row['id'] for row in rows]}
    prepared = {'format': contract.FORMAT + '/prepared', 'plan_commit': args.plan_commit,
        'plan_sha256': data.sha256(args.plan), 'sources': contract.sources(), 'roles': files,
        'replay_indices': selected, 'schedule': contract.schedule(plan, roles['train'])}
    for role in roles:
        contract.read_role(prepared, args.home, role, tokenizer, plan['max_length'])
    data.save(args.home / 'prepared.json', prepared)
    emit('prepared', identity=data.identity(prepared), counts={role: len(rows) for role, rows in roles.items()})


def run(args):
    plan, prepared = contract.validate_prepared(args.plan, args.prepared)
    final = args.command == 'evaluate'
    baseline = args.arm == 'baseline'
    selection_path = contract.ROOT / plan['selection_path']
    selection = json.loads(contract.committed(selection_path)) if final else None
    if not final and selection_path.exists():
        raise ValueError('Development is closed once candidate selection is recorded')
    if selection is not None and (selection['plan'] != data.identity(plan)
                                   or selection['prepared'] != data.identity(prepared)):
        raise ValueError('Final selection belongs to different prepared inputs')
    if final and not baseline and selection['selected'].get(args.arm) is None:
        raise ValueError('A failed development arm cannot open its final evaluation')
    parent = json.loads(args.parent.read_bytes())
    portable.validate(parent)
    if data.identity(parent) != plan['parent']['checkpoint'] or parent['state_root'] != plan['parent']['state_root']:
        raise ValueError('Wrong established parent checkpoint')
    arm = 'tail-control' if baseline else args.arm
    if arm not in contract.ARMS or (args.command == 'train' and baseline):
        raise ValueError('Unknown comparison role')
    arm_plan = plan['arms'][arm]
    rate = selection['selected'][arm]['learning_rate'] if final and not baseline else args.rate
    recipe = contract.recipe(plan, rate) if not baseline else None
    job = contract.job(plan, prepared, arm, rate) if not baseline else data.identity(parent)
    target = json.loads(args.resume.read_bytes()) if args.resume else None
    if not baseline:
        if final and (target is None or data.identity(target) != selection['selected'][arm]['checkpoint']):
            raise ValueError('Finals must use the previously committed selected checkpoint')
        if target is not None:
            incremental_state.validate(target, parent)
            if target['job'] != job or target['step'] not in (64, 128) or (not final and target['step'] != 64):
                raise ValueError('Only the declared midpoint can resume training')
    elif target is not None:
        raise ValueError('The baseline is exactly the established parent')
    tokenizer = tokenizer_for(plan, args.seed)
    runtime = reference.configure('cuda', plan['threads'])
    runtime['allocator'] = os.environ['PYTORCH_CUDA_ALLOC_CONF']
    if {key: runtime[key] for key in plan['runtime']} != plan['runtime']:
        raise ValueError('Incremental numerical runtime differs from the freeze')
    roles = contract.FINALS if final else contract.DEVELOPMENT
    cache = {role: contract.read_role(prepared, args.inputs, role, tokenizer, plan['max_length']) for role in roles}
    training = args.command == 'train'
    if training:
        cache['train'] = contract.read_role(prepared, args.inputs, 'train', tokenizer, plan['max_length'])
        if contract.schedule(plan, cache['train']) != prepared['schedule']:
            raise ValueError('Training schedule differs from the completed preparation')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != len(arm_plan['boundaries']) - 1 or not 0 <= rank < world:
        raise ValueError('Every declared model owner must participate')
    torch.manual_seed(plan['seeds']['training'] + rank)
    random.seed(plan['seeds']['training'] + rank)
    np.random.seed(plan['seeds']['training'] + rank)
    config = LlamaConfig(**{**parent['config'], 'num_hidden_layers': arm_plan['layers']})
    config._attn_implementation = 'sdpa'
    shard = Partition(config, arm_plan['boundaries'], rank, 'cuda', plan['parameter_limit'])
    frozen = incremental_state.initialize(shard, parent, args.objects, arm, arm_plan['frozen_layers'])
    reference_layers = incremental.reference_tail(shard, arm_plan['frozen_layers']) if training and arm == 'tail-control' else None
    optimizer = incremental.configure(shard, arm_plan['frozen_layers'], recipe) if training else None
    if not training:
        shard.requires_grad_(False)
    if target is not None:
        frozen = incremental_state.load(args.resume.parent, shard, optimizer, target, parent,
            job, recipe, restore_optimizer=training)
    args.home.mkdir(parents=True, exist_ok=False)
    dist.init_process_group('gloo', timeout=timedelta(seconds=300))
    wire = Wire(rank, world)
    started = time.monotonic()
    network_before = network()
    try:
        binding = {'job': job, 'command': args.command, 'arm': args.arm, 'parent': data.identity(parent),
            'target': data.identity(target) if target else None, 'prepared': data.identity(prepared)}
        if any(value != binding for value in wire.exchange(binding)):
            raise ValueError('Incremental workers disagree on the complete computation')
        data.save(args.home / 'started.json', {**binding, 'rank': rank, 'runtime': runtime,
            'resident_parameters': shard.resident_parameters,
            'trainable_parameters': sum(p.numel() for p in shard.parameters() if p.requires_grad),
            'reference_parameters': sum(p.numel() for p in reference_layers.parameters()) if reference_layers else 0})

        def deadline():
            if time.monotonic() - started > plan['max_seconds']:
                raise TimeoutError('Incremental comparison operation deadline')

        def evaluate(role):
            rows = cache[role]
            result = {'answers': [], 'losses': []}
            if role.endswith('conversation'):
                for row in rows:
                    deadline()
                    result['losses'].extend(score(shard, wire, [row]))
            else:
                limit = plan['generation']['knowledge' if role.endswith('knowledge') else 'skills']
                for row in rows:
                    deadline()
                    prompt = tokenizer.apply_chat_template(row['messages'][:-1], tokenize=True, add_generation_prompt=True)
                    began = time.monotonic()
                    ids = generate(shard, wire, prompt, limit, tokenizer.eos_token_id)
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    check = facts.check_answer(row['task'], text) if role.endswith('knowledge') else contract.skill_check(row['task'], text)
                    result['answers'].append({'id': row['id'], 'text': text, 'output_ids': ids,
                                              'check': check, 'seconds': time.monotonic() - began})
                    if len(result['answers']) % 32 == 0:
                        emit('generated', rank=rank, role=role, count=len(result['answers']))
            if any(other != clean(result) for other in wire.exchange(clean(result))):
                raise ValueError('Workers disagree on generated answers or conversation losses')
            return result

        if training:
            first = target['step'] if target else 0
            if first == 0:
                ids, _, mask, _ = batch_tensors([cache['train'][0]], shard.device_name)
                with torch.no_grad():
                    _, _, before, after = incremental.forward(shard, wire, ids, mask,
                        arm_plan['frozen_layers'], reference_layers)
                local_equal = torch.equal(before, after) if rank == 0 else None
                equality = wire.exchange(local_equal)[0]
                if equality is not True:
                    raise ValueError('Initial candidate does not preserve its parent representation')
                data.save(args.home / 'identity-probe.json', {'passed': True, 'job': job,
                    'record': cache['train'][0]['id'], 'exact_hidden_state': True})
                del ids, mask, before, after
            target = incremental_state.commit(args.home, shard, optimizer, wire, parent,
                frozen, job, first, recipe, arm, arm_plan['frozen_layers'])
            for index in range(first, recipe['steps']):
                deadline()
                report = incremental.train_step(shard, optimizer, wire,
                    [cache['train'][i] for i in prepared['schedule'][index]], recipe,
                    index, plan['microbatch'], arm_plan['frozen_layers'],
                    reference_layers=reference_layers, **plan['reference'])
                with (args.home / 'metrics.jsonl').open('a') as output:
                    output.write(json.dumps(report, sort_keys=True) + '\n')
                emit('trained', rank=rank, **report)
                if index + 1 in (64, 128):
                    target = incremental_state.commit(args.home, shard, optimizer, wire, parent,
                        frozen, job, index + 1, recipe, arm, arm_plan['frozen_layers'])
            data.save(args.home / 'training-complete.json', {'checkpoint': data.identity(target),
                'job': job, 'steps': recipe['steps'], 'seconds': time.monotonic() - started})
        outcomes = {role: evaluate(role) for role in roles}
        result = {**binding, 'format': contract.FORMAT + '/evaluation', 'rank': rank,
            'learning_rate': rate, 'checkpoint': data.identity(target) if target else data.identity(parent),
            'checkpoint_step': target['step'] if target else parent['step'],
            'learned_state': target['state_root'] if target else parent['state_root'],
            'outcomes': outcomes, 'seconds': time.monotonic() - started,
            'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
            'network_before': network_before, 'network_after': network(), 'tokens_issued': 0}
        data.save(args.home / 'evaluation.json', result)
    finally:
        dist.destroy_process_group()


def network():
    return {path.name: {name: int((path / 'statistics' / file).read_text())
                       for name, file in (('tx', 'tx_bytes'), ('rx', 'rx_bytes'))}
            for path in Path('/sys/class/net').iterdir() if path.name != 'lo'}


def read_reports(directory, world):
    reports = [json.loads((directory / f'rank-{rank}' / 'evaluation.json').read_bytes())
               for rank in range(world)]
    common = reports[0]
    shared = ('format', 'job', 'command', 'arm', 'parent', 'prepared', 'learning_rate',
              'checkpoint', 'checkpoint_step', 'learned_state', 'tokens_issued')
    for rank, report in enumerate(reports):
        if (report['rank'] != rank or any(report[key] != common[key] for key in shared)
                or clean(report['outcomes']) != clean(common['outcomes'])):
            raise ValueError('Require agreeing reports from every declared shard owner')
    return common, [{'rank': rank, 'sha256': data.sha256(directory / f'rank-{rank}' / 'evaluation.json'),
                      'seconds': report['seconds'], 'peak_cuda_bytes': report['peak_cuda_bytes']}
                     for rank, report in enumerate(reports)]


def decide(args):
    plan, prepared = contract.validate_prepared(args.plan, args.prepared)
    tokenizer = tokenizer_for(plan, args.seed)
    development = args.command == 'select'
    roles = contract.DEVELOPMENT if development else contract.FINALS
    rows = {role: contract.read_role(prepared, args.inputs, role, tokenizer, plan['max_length']) for role in roles}
    world = len(plan['arms']['append']['boundaries']) - 1
    baseline, baseline_receipts = read_reports(args.baseline_report, world)
    if (baseline['arm'] != 'baseline' or baseline['checkpoint'] != plan['parent']['checkpoint']
            or baseline['prepared'] != data.identity(prepared)
            or baseline['command'] != ('baseline' if development else 'evaluate')
            or set(baseline['outcomes']) != set(roles)):
        raise ValueError('The comparison baseline must be the unchanged established parent')
    contract.validate_generation(plan, rows, baseline['outcomes'], tokenizer)
    selection_path = contract.ROOT / plan['selection_path']
    if development and selection_path.exists():
        raise ValueError('Do not overwrite a recorded candidate selection')
    selection = None if development else json.loads(contract.committed(selection_path))
    if selection is not None and (selection['plan'] != data.identity(plan)
                                   or selection['prepared'] != data.identity(prepared)):
        raise ValueError('Final selection differs from this experiment')
    candidates, outcomes, identities = [], {}, set()
    for directory in args.candidate_reports:
        report, receipts = read_reports(directory, world)
        arm, rate = report['arm'], report['learning_rate']
        if arm not in contract.ARMS or (arm, rate) in identities:
            raise ValueError('Each declared candidate must have exactly one complete report')
        identities.add((arm, rate))
        if (report['job'] != contract.job(plan, prepared, arm, rate)
                or report['parent'] != plan['parent']['checkpoint'] or report['checkpoint_step'] != 128
                or report['prepared'] != data.identity(prepared) or report['tokens_issued'] != 0
                or report['command'] != ('train' if development else 'evaluate')
                or set(report['outcomes']) != set(roles)):
            raise ValueError('Candidate does not match the frozen completed comparison')
        if not development:
            chosen = selection['selected'][arm]
            if chosen is None or report['checkpoint'] != chosen['checkpoint'] or rate != chosen['learning_rate']:
                raise ValueError('Finals cannot substitute an unselected checkpoint')
        contract.validate_generation(plan, rows, report['outcomes'], tokenizer)
        decision = contract.decision(plan, rows, baseline['outcomes'], report['outcomes'], development)
        candidates.append({'arm': arm, 'learning_rate': rate, 'checkpoint': report['checkpoint'],
            'learned_state': report['learned_state'], 'job': report['job'], 'decision': decision, 'reports': receipts})
        outcomes[arm] = report['outcomes']
    args.home.mkdir(parents=True, exist_ok=False)
    if development:
        if identities != {(arm, rate) for arm in contract.ARMS for rate in plan['learning_rates']}:
            raise ValueError('Selection requires every frozen candidate, including failures')
        selected = {}
        for arm in contract.ARMS:
            eligible = [row for row in candidates if row['arm'] == arm and row['decision']['passed']]
            eligible.sort(key=lambda row: (-row['decision']['knowledge']['accuracy'], row['learning_rate']))
            selected[arm] = eligible[0] if eligible else None
        result = {'format': contract.FORMAT + '/selection', 'plan': data.identity(plan),
            'prepared': data.identity(prepared), 'baseline': baseline_receipts,
            'candidates': candidates, 'selected': selected, 'finals_opened': False}
        data.save(args.home / 'selection.json', result)
    else:
        expected = {(arm, row['learning_rate']) for arm, row in selection['selected'].items() if row is not None}
        if identities != expected:
            raise ValueError('Score every selected arm; failed finals cannot be omitted')
        decisions = {row['arm']: row['decision'] for row in candidates}
        advantage = None
        if set(outcomes) == set(contract.ARMS):
            role = 'test-knowledge'
            advantage = contract.knowledge(plan, rows[role], outcomes['tail-control'][role]['answers'],
                                             outcomes['append'][role]['answers'])
            advantage['passed'] = (advantage['gains'] - advantage['losses'] >= plan['gate']['capacity_advantage_net_correct']
                and advantage['entity_cluster_delta']['lower'] > plan['gate']['capacity_advantage_lower'])
        learned = decisions.get('append', {}).get('passed', False)
        result = {'format': contract.FORMAT + '/results', 'plan': data.identity(plan),
            'prepared': data.identity(prepared), 'selection': data.identity(selection),
            'baseline': baseline_receipts, 'candidates': candidates,
            'learning_in_added_blocks_passed': learned, 'capacity_advantage': advantage,
            'passed': bool(learned and advantage and advantage['passed']),
            'tokens_issued': 0, 'native_activated': False}
        data.save(args.home / 'results.json', result)
    emit('decision', **{key: value for key, value in result.items()
                        if key in ('format', 'passed', 'learning_in_added_blocks_passed')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'train', 'baseline', 'evaluate', 'select', 'score'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--plan-commit')
    parser.add_argument('--replay-inputs', type=Path)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--seed', type=Path, required=True)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--objects', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--arm', choices=(*contract.ARMS, 'baseline'))
    parser.add_argument('--rate', type=float)
    parser.add_argument('--baseline-report', type=Path)
    parser.add_argument('--candidate-reports', type=Path, nargs='+')
    args = parser.parse_args()
    if args.command == 'prepare':
        if not args.plan_commit or args.replay_inputs is None:
            parser.error('Preparation requires --plan-commit and --replay-inputs')
        prepare(args)
    elif args.command in ('select', 'score'):
        if any(value is None for value in (args.prepared, args.inputs, args.baseline_report, args.candidate_reports)):
            parser.error('Decisions require prepared inputs and complete baseline/candidate report directories')
        decide(args)
    else:
        if any(value is None for value in (args.prepared, args.inputs, args.parent, args.objects, args.arm)):
            parser.error('Execution requires prepared inputs, parent objects and a comparison arm')
        if args.command == 'baseline' and args.arm != 'baseline':
            parser.error('Baseline execution must use --arm baseline')
        run(args)
