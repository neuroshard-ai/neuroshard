"""Repeated tail learning with stored frozen features and unchanged answer gates."""
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from transformers import LlamaConfig

from .. import incremental_capacity as base, incremental_facts as facts, rehearsal, reference
from .. import reference_data as data
from . import feature_bank, features, incremental, incremental_state
from .feature_probe import load_head
from .incremental_job import clean, emit, network, read_reports as original_reports, tokenizer_for
from .model import Partition
from .training import generate, score
from .wire import Wire


def read_reports(directory, world):
    common, receipts = original_reports(directory, world)
    if common['format'] == rehearsal.FORMAT + '/evaluation':
        for rank in range(world):
            report = rehearsal.read(directory / f'rank-{rank}/evaluation.json')
            if any(report[key] != common[key] for key in ('rehearsal_prepared', 'runtime', 'feature_root', 'target')):
                raise ValueError('Owners disagree on rehearsal identity or frozen features')
    return common, receipts


def run(args, device='cuda'):
    plan, original, prepared, bound = rehearsal.validate()
    training, final = args.command == 'train', args.command == 'evaluate'
    baseline = args.arm == 'baseline'
    if args.command == 'baseline' and not baseline:
        raise ValueError('Baseline execution must use the unchanged parent')
    arm = 'tail-control' if baseline else args.arm
    if arm not in plan['arms'] or (baseline and training):
        raise ValueError('Invalid rehearsal operation and model arm')
    selection_path = rehearsal.ROOT / plan['selection_path']
    if not final and selection_path.exists():
        raise ValueError('Development is closed after committing selection')
    selection = json.loads(base.committed(selection_path)) if final else None
    if selection is not None and (selection['plan'] != data.identity(plan)
                                  or selection['prepared'] != data.identity(bound)):
        raise ValueError('Final selection differs from this rehearsal')
    parent = rehearsal.read(args.parent)
    if data.identity(parent) != plan['parent']:
        raise ValueError('Wrong established parent')
    job = data.identity({'baseline': plan['parent'], 'prepared': data.identity(bound)}) if baseline else rehearsal.job(plan, bound, arm)
    target = rehearsal.read(args.resume) if args.resume else None
    if final and not baseline:
        chosen = selection['selected'][arm]
        if chosen is None or target is None or data.identity(target) != chosen['checkpoint']:
            raise ValueError('Only an eligible, committed terminal candidate may open finals')
    if target is not None:
        incremental_state.validate(target, parent)
        if (baseline or target['job'] != job
                or target['step'] != (1024 if final else 512)):
            raise ValueError('Invalid selected final or declared midpoint recovery')
    elif final and not baseline:
        raise ValueError('Candidate finals require their selected checkpoint')
    tokenizer = tokenizer_for(original, args.seed)
    runtime = reference.configure(device, original['threads'])
    runtime['allocator'] = os.environ['PYTORCH_CUDA_ALLOC_CONF']
    if {key: runtime[key] for key in original['runtime']} != original['runtime']:
        raise ValueError('Rehearsal changed the numerical runtime')
    profile = {key: runtime[key] for key in original['runtime']}
    roles = base.FINALS if final else base.DEVELOPMENT
    rows = {role: base.read_role(prepared, args.inputs, role, tokenizer, original['max_length']) for role in roles}
    if training:
        rows['train'] = base.read_role(prepared, args.inputs, 'train', tokenizer, original['max_length'])
        if base.schedule(original, rows['train']) != prepared['schedule']:
            raise ValueError('Rehearsal changed training examples or padding groups')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    layout = original['arms'][arm]
    if world != len(layout['boundaries']) - 1 or not 0 <= rank < world:
        raise ValueError('Rehearsal needs the complete declared model partition')
    torch.manual_seed(original['seeds']['training'] + rank)
    random.seed(original['seeds']['training'] + rank)
    np.random.seed(original['seeds']['training'] + rank)
    config = LlamaConfig(**{**parent['config'], 'num_hidden_layers': layout['layers']})
    config._attn_implementation = 'sdpa'
    shard = Partition(config, layout['boundaries'], rank, device, original['parameter_limit'])
    frozen = incremental_state.initialize(shard, parent, args.objects, arm, layout['frozen_layers'])
    recipe = plan['recipe']
    teacher_tail = incremental.reference_tail(shard, layout['frozen_layers']) if training and arm == 'tail-control' else None
    optimizer = incremental.configure(shard, layout['frozen_layers'], recipe) if training else None
    if not training:
        shard.requires_grad_(False)
    if target:
        frozen = incremental_state.load(args.resume.parent, shard, optimizer, target, parent, job, recipe,
                                        restore_optimizer=training)
    args.home.mkdir(parents=True, exist_ok=False)
    started, before_network = time.monotonic(), network()
    dist.init_process_group('gloo', timeout=timedelta(seconds=900))
    wire = Wire(rank, world)
    binding = {'format': rehearsal.FORMAT + '/evaluation', 'job': job, 'arm': args.arm,
        'parent': plan['parent'], 'prepared': data.identity(prepared), 'rehearsal_prepared': data.identity(bound),
        'command': args.command, 'learning_rate': None if baseline else recipe['learning_rate'],
        'target': data.identity(target) if target else None,
        'tokens_issued': 0, 'runtime': profile}
    try:
        if any(value != binding for value in wire.exchange(binding)):
            raise ValueError('Owners disagree on rehearsal execution')
        data.save(args.home / 'started.json', {**binding, 'rank': rank, 'local_runtime': runtime,
                                             'resident_parameters': shard.resident_parameters})
        bank_root, production = None, None
        if training:
            first = target['step'] if target else 0
            target = incremental_state.commit(args.home, shard, optimizer, wire, parent, frozen, job,
                first, recipe, arm, layout['frozen_layers'])
            bank_binding = {'job': job, 'parent': plan['parent'], 'base_prepared': data.identity(prepared),
                'rehearsal_prepared': data.identity(bound), 'cut_layer': layout['frozen_layers'],
                'runtime': profile, 'schedule': data.identity(prepared['schedule'])}
            began, sent = time.monotonic(), wire.sent_tensor_bytes
            bank_root = feature_bank.produce(shard, wire, rows['train'], prepared['schedule'],
                args.home / 'features', bank_binding, layout['frozen_layers'], teacher_tail, original['microbatch'])
            production = {'root': bank_root, 'seconds': time.monotonic() - began,
                          'sent_tensor_bytes': wire.sent_tensor_bytes - sent, 'unique_batches': 128}
            data.save(args.home / 'feature-production.json', production)
            del teacher_tail
            if rank == world - 1:
                bank = feature_bank.Reader(args.home / 'features', bank_root, bank_binding, config, original['microbatch'], 128)
                head = load_head(parent, args.objects, device)
            for index in range(first, recipe['steps']):
                if time.monotonic() - started > original['max_seconds']:
                    raise TimeoutError('Rehearsal exceeded the original numerical runtime bound')
                report = None
                if rank == world - 1:
                    batch_index = bound['schedule'][index]
                    batch = [rows['train'][i] for i in prepared['schedule'][batch_index]]
                    began = time.monotonic()
                    packets = bank.batch(batch_index, batch, device)
                    loaded = time.monotonic() - began
                    report = features.train_step(shard, head, optimizer, packets, batch, recipe, index,
                                                 original['microbatch'], **original['reference'])
                    report.update(feature_batch=batch_index, feature_load_seconds=loaded,
                                  total_seconds=time.monotonic() - began)
                    del packets
                report = wire.exchange(report)[world - 1]
                with (args.home / 'metrics.jsonl').open('a') as output:
                    output.write(json.dumps(report, sort_keys=True) + '\n')
                if (index + 1) % 32 == 0:
                    emit('trained', rank=rank, **report)
                if index + 1 in plan['checkpoints']:
                    target = incremental_state.commit(args.home, shard, optimizer, wire, parent, frozen,
                        job, index + 1, recipe, arm, layout['frozen_layers'])
            if rank == world - 1:
                del head, bank
            data.save(args.home / 'training-complete.json', {'checkpoint': data.identity(target),
                'job': job, 'steps': recipe['steps'], 'feature_root': bank_root,
                'seconds': time.monotonic() - started, 'tokens_issued': 0})
        outcomes = {}
        for role in roles:
            result = {'answers': [], 'losses': []}
            if role.endswith('conversation'):
                for row in rows[role]:
                    if time.monotonic() - started > original['max_seconds']:
                        raise TimeoutError('Rehearsal evaluation deadline')
                    result['losses'].extend(score(shard, wire, [row]))
            else:
                for row in rows[role]:
                    if time.monotonic() - started > original['max_seconds']:
                        raise TimeoutError('Rehearsal evaluation deadline')
                    began = time.monotonic()
                    prompt = tokenizer.apply_chat_template(row['messages'][:-1], tokenize=True, add_generation_prompt=True)
                    ids = generate(shard, wire, prompt,
                        original['generation']['knowledge' if role.endswith('knowledge') else 'skills'], tokenizer.eos_token_id)
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    check = (facts.check_answer if role.endswith('knowledge') else base.skill_check)(row['task'], text)
                    result['answers'].append({'id': row['id'], 'output_ids': ids, 'text': text,
                                             'check': check, 'seconds': time.monotonic() - began})
                    if len(result['answers']) % 32 == 0:
                        emit('generated', rank=rank, role=role, count=len(result['answers']))
            if any(value != clean(result) for value in wire.exchange(clean(result))):
                raise ValueError('Owners disagree on complete generated answers')
            outcomes[role] = result
        result = {**binding, 'rank': rank, 'checkpoint': data.identity(target) if target else data.identity(parent),
            'checkpoint_step': target['step'] if target else parent['step'],
            'learned_state': target['state_root'] if target else parent['state_root'], 'feature_root': bank_root,
            'feature_production': production, 'outcomes': outcomes, 'seconds': time.monotonic() - started,
            'peak_cuda_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
            'network_before': before_network, 'network_after': network()}
        data.save(args.home / 'evaluation.json', result)
    finally:
        dist.destroy_process_group()


def decide(args):
    plan, original, prepared, bound = rehearsal.validate()
    development = args.command == 'select'
    selection_path = rehearsal.ROOT / plan['selection_path']
    if development and selection_path.exists():
        raise ValueError('Do not overwrite the committed rehearsal selection')
    selection = None if development else json.loads(base.committed(selection_path))
    if selection and (selection['plan'] != data.identity(plan) or selection['prepared'] != data.identity(bound)):
        raise ValueError('Wrong rehearsal final selection')
    if selection and all(chosen is None for chosen in selection['selected'].values()):
        if args.candidate_reports:
            raise ValueError('Ineligible arms cannot supply final evidence')
        args.home.mkdir(parents=True, exist_ok=False)
        data.save(args.home / 'results.json', {'format': rehearsal.FORMAT + '/results',
            'prepared': data.identity(bound), 'selection': data.identity(selection), 'candidates': [],
            'finals_opened': False, 'learning_in_added_blocks_passed': False, 'capacity_advantage': None,
            'passed': False, 'tokens_issued': 0, 'native_activated': False})
        return
    roles = base.DEVELOPMENT if development else base.FINALS
    tokenizer = tokenizer_for(original, args.seed)
    rows = {role: base.read_role(prepared, args.inputs, role, tokenizer, original['max_length']) for role in roles}
    baseline, receipts = read_reports(args.baseline_report, 4)
    if (baseline['arm'] != 'baseline' or baseline['checkpoint'] != plan['parent']
            or baseline['prepared'] != data.identity(prepared) or set(baseline['outcomes']) != set(roles)
            or baseline['command'] != ('baseline' if development else 'evaluate')):
        raise ValueError('Use the complete unchanged-parent baseline')
    base.validate_generation(original, rows, baseline['outcomes'], tokenizer)
    candidates, outcomes = [], {}
    for directory in args.candidate_reports:
        report, receipts = read_reports(directory, 4)
        arm = report['arm']
        if (arm in outcomes or report['job'] != rehearsal.job(plan, bound, arm)
                or report['rehearsal_prepared'] != data.identity(bound) or report['prepared'] != data.identity(prepared)
                or report['checkpoint_step'] != 1024 or report['tokens_issued'] != 0
                or report['format'] != rehearsal.FORMAT + '/evaluation'
                or report['parent'] != plan['parent'] or report['learning_rate'] != plan['recipe']['learning_rate']
                or report['command'] != ('train' if development else 'evaluate') or set(report['outcomes']) != set(roles)):
            raise ValueError('Candidate differs from the completed bound rehearsal')
        if not development and (selection['selected'][arm] is None
                               or report['checkpoint'] != selection['selected'][arm]['checkpoint']):
            raise ValueError('Finals cannot substitute another checkpoint')
        base.validate_generation(original, rows, report['outcomes'], tokenizer)
        decision = base.decision(original, rows, baseline['outcomes'], report['outcomes'], development)
        candidates.append({'arm': arm, 'job': report['job'], 'step': 1024, 'checkpoint': report['checkpoint'],
            'learned_state': report['learned_state'], 'decision': decision, 'reports': receipts})
        outcomes[arm] = report['outcomes']
    args.home.mkdir(parents=True, exist_ok=False)
    if development:
        result = rehearsal.selection(plan, bound, candidates)
        data.save(args.home / 'selection.json', result)
    else:
        expected = {arm for arm, chosen in selection['selected'].items() if chosen is not None}
        if set(outcomes) != expected:
            raise ValueError('Score every selected final, including failures')
        learned = any(row['arm'] == 'append' and row['decision']['passed'] for row in candidates)
        advantage = None
        if expected == set(plan['arms']):
            advantage = base.knowledge(original, rows['test-knowledge'],
                outcomes['tail-control']['test-knowledge']['answers'], outcomes['append']['test-knowledge']['answers'])
            advantage['passed'] = (all(row['decision']['passed'] for row in candidates)
                and advantage['entity_cluster_delta']['lower'] > original['gate']['capacity_advantage_lower']
                and advantage['after'] - advantage['before'] >= original['gate']['capacity_advantage_net_correct'])
        result = {'format': rehearsal.FORMAT + '/results', 'prepared': data.identity(bound),
            'selection': data.identity(selection), 'candidates': candidates,
            'finals_opened': bool(expected), 'learning_in_added_blocks_passed': learned,
            'capacity_advantage': advantage, 'passed': bool(learned and advantage and advantage['passed']),
            'tokens_issued': 0, 'native_activated': False}
        data.save(args.home / 'results.json', result)
    emit('decision', format=result['format'], passed=result.get('passed'))
