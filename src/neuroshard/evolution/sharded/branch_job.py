"""Measure ordinary LLM generation through a retained parent and new expert."""
from datetime import timedelta
import copy
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from transformers import LlamaConfig

from .. import branch_experiment as contract, incremental_capacity as base
from .. import reference, reference_data as data
from . import incremental_state, portable
from .branch import Network, ParentWire, route
from .incremental_job import tokenizer_for
from .model import Partition, batch_tensors, weighted_loss
from .wire import Wire


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items() if key != 'seconds'}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


def run(args):
    plan, prepared = contract.validate()
    final = args.command == 'final'
    if final:
        selection = json.loads(base.committed(contract.SELECTION))
        if (not selection['eligible'] or selection['plan'] != data.identity(plan)
                or selection['prepared'] != data.identity(prepared)
                or selection['graph'] != data.identity(prepared['graph'])):
            raise ValueError('Only a committed eligible branch may open finals')
    elif contract.SELECTION.exists():
        raise ValueError('Development is closed after branch selection')
    parent = json.loads(args.parent.read_bytes())
    expert = json.loads(args.expert.read_bytes())
    descriptor = contract.graph(plan, parent, expert)
    incremental_state.validate(expert, parent)
    if (expert['mode'] != 'tail-control' or expert['frozen_layers'] != plan['split']
            or expert['step'] != 1024 or expert['job'] != plan['training_job']):
        raise ValueError('Use exactly the completed learned tail with its actual Adam ages')
    if descriptor != prepared['graph']:
        raise ValueError('The added branch differs from its committed tensor graph')
    # This cache is derived from the fixed parent, not a new quality dataset.
    # Its entire content is bound across owners and checked against actual
    # parent-path execution below, including every token and loss value.
    cache = json.loads((args.inputs / 'parent-baselines.json').read_bytes())
    if cache['parent'] != plan['parent'] or set(cache['roles']) != {
            'dev-skills', 'dev-conversation', 'test-skills', 'test-conversation'}:
        raise ValueError('Require exactly the original parent retention cache')
    for role, value in cache['roles'].items():
        if value['input_sha256'] != prepared['roles'][role]['sha256']:
            raise ValueError('Parent cache belongs to different retention inputs')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != 4 or not 0 <= rank < world:
        raise ValueError('Four separate declared owners are required')
    args.home.mkdir(parents=True, exist_ok=False)
    tokenizer = tokenizer_for(plan, args.seed)
    runtime = reference.configure('cuda', plan['threads'])
    runtime['allocator'] = os.environ['PYTORCH_CUDA_ALLOC_CONF']
    if {key: runtime[key] for key in plan['runtime']} != plan['runtime']:
        raise ValueError('Branch runtime differs from the frozen numerical profile')
    config = LlamaConfig(**parent['config'])
    config._attn_implementation = 'sdpa'
    shard = Partition(config, plan['parent_layout'] if rank < 3 else plan['expert_layout'],
                      rank, 'cuda', plan['parameter_limit'])
    records = incremental_state.records(parent) if rank < 3 else expert['tensors']
    objects = args.objects if rank < 3 else args.expert_objects
    with torch.no_grad():
        for name, parameter in shard.named_owned_parameters():
            value = incremental_state.tensor_values(portable.tensor_path(objects, records[name]['sha256']), records[name])
            parameter.copy_(value['weight'])
            parameter.requires_grad_(False)
            del value
    shard.eval()
    dist.init_process_group('gloo', timeout=timedelta(seconds=600))
    parent_group = dist.new_group([0, 1, 2], timeout=timedelta(seconds=600))
    wire = Wire(rank, 4)
    parent_wire = ParentWire(rank, parent_group) if rank < 3 else None
    binding = {'plan': data.identity(plan), 'prepared': data.identity(prepared), 'graph': data.identity(descriptor),
               'baseline_cache': data.identity(cache)}
    started = time.monotonic()
    try:
        declarations = wire.exchange({'binding': binding, 'rank': rank, 'runtime': plan['runtime']})
        if any(row != {'binding': binding, 'rank': index, 'runtime': plan['runtime']}
               for index, row in enumerate(declarations)):
            raise ValueError('Owners disagree on the fixed graph or numerical profile')
        data.save(args.home / 'started.json', {**binding, 'rank': rank, 'runtime': runtime,
            'owned_parameters': shard.resident_parameters})
        net = Network(shard, wire, parent_wire, tokenizer, plan['split'])

        def answer(question, cap, use_branch):
            value = net.answer(question, cap) if use_branch else net.generate(question, cap, False)
            values = wire.exchange(value)
            if values[0] is None or any(row is not None and row != values[0] for row in values):
                raise ValueError('Owners disagree on the complete generated answer')
            return values[0]

        @torch.no_grad()
        def loss(row):
            hidden = net.forward(row['input_ids'], False)
            value = None
            if rank == 0:
                _, labels, _, weights = batch_tensors([row], 'cuda')
                with reference.autocast('cuda'):
                    value = float(weighted_loss(shard.logits(hidden), labels, torch.ones_like(weights))) / row['targets']
            return {'id': row['id'], 'targets': row['targets'], 'loss': wire.exchange(value)[0]}

        if not final:
            examples = contract.rows(prepared, args.inputs, 'equivalence-knowledge')
            expected = contract.rows(prepared, args.inputs, 'equivalence-answers')
            results = []
            for index, (row, previous) in enumerate(zip(examples, expected)):
                value = answer(row['messages'][0]['content'], plan['generation']['knowledge'], True)
                if (row['id'] != previous['id'] or value['ids'] != previous['output_ids']
                        or value['text'] != previous['text'] or value['route'] != 'expert'):
                    raise ValueError('Branch composition changed the learned transformer answer')
                results.append({'id': row['id'], **value})
                if (index + 1) % 32 == 0:
                    print(json.dumps({'event': 'equivalence', 'rank': rank, 'count': index + 1}), flush=True)
            data.save(args.home / 'equivalence.json', {'passed': True, 'count': len(results), 'answers': results})
        before, after = {}, {}
        prefix = 'test' if final else 'dev'
        roles = [prefix + '-' + suffix for suffix in ('knowledge', 'skills', 'conversation')]
        all_rows = {role: contract.rows(prepared, args.inputs, role) for role in roles}
        for role, examples in all_rows.items():
            before[role], after[role] = ({'answers': [], 'losses': []} for _ in range(2))
            if not role.endswith('knowledge'):
                before[role] = copy.deepcopy(cache['roles'][role]['outcomes'])
            for index, row in enumerate(examples):
                if time.monotonic() - started > plan['max_seconds']:
                    raise TimeoutError('Frozen branch evaluation deadline')
                question = row['messages'][0]['content']
                if role.endswith('conversation'):
                    if route(question):
                        raise ValueError('Expert selector captured a general conversation probe')
                    after[role]['losses'].append(loss(row))
                else:
                    cap = plan['generation']['knowledge' if role.endswith('knowledge') else 'skills']
                    sides = ((before, False), (after, True)) if role.endswith('knowledge') else ((after, True),)
                    for destination, use_branch in sides:
                        began = time.monotonic()
                        value = answer(question, cap, use_branch)
                        destination[role]['answers'].append({'id': row['id'], **value,
                            'seconds': time.monotonic() - began})
                if (index + 1) % 32 == 0:
                    print(json.dumps({'event': 'evaluated', 'rank': rank, 'role': role, 'count': index + 1}), flush=True)
            data.save(args.home / (role + '.json'), {'before': before[role], 'after': after[role]})
        decision = base.decision(plan, all_rows, before, after, not final)
        decision['checks']['exact_parent_answers'] = all(clean(left) == clean(right)
            for left, right in zip(before[prefix + '-skills']['answers'], after[prefix + '-skills']['answers']))
        decision['checks']['exact_parent_losses'] = before[prefix + '-conversation'] == after[prefix + '-conversation']
        decision['checks']['question_only_selector'] = all(value['route'] == 'expert'
            for value in after[prefix + '-knowledge']['answers'])
        decision['passed'] = all(decision['checks'].values())
        digest = data.identity(clean({'before': before, 'after': after}))
        if any(value != digest for value in wire.exchange(digest)):
            raise ValueError('Owners disagree on the complete evaluation')
        data.save(args.home / 'result.json', {**binding, 'rank': rank, 'development': not final,
            'decision': decision, 'answer_identity': digest, 'seconds': time.monotonic() - started,
            'sent_tensor_bytes': wire.sent_tensor_bytes + (parent_wire.sent_tensor_bytes if parent_wire else 0),
            'owned_parameters': shard.resident_parameters, 'tokens_issued': 0, 'native_activated': False})
        wire.exchange('expert evaluation complete; parent continues independently')
        if rank == 3:
            return
        data.save(args.home / 'waiting-for-expert-exit.json', binding)
        receipt_path = args.home / 'expert-exited.json'
        began = time.monotonic()
        while not receipt_path.exists():
            if time.monotonic() - began > 300:
                raise TimeoutError('Controller did not confirm expert process exit')
            time.sleep(.1)
        receipt = json.loads(receipt_path.read_bytes())
        if receipt['binding'] != binding or receipt['expert_exit_code'] != 0:
            raise ValueError('Invalid observed expert exit receipt')
        index = next(i for i, (row, value) in enumerate(zip(all_rows[prefix + '-skills'], before[prefix + '-skills']['answers']))
                     if base.skill_check(row['task'], value['text'])['correct'])
        row = all_rows[prefix + '-skills'][index]
        value = net.answer(row['messages'][0]['content'], plan['generation']['skills'])
        expected = before[prefix + '-skills']['answers'][index]
        if value['ids'] != expected['ids'] or value['text'] != expected['text'] or value['route'] != 'parent':
            raise ValueError('Established response changed after the expert exited')
        if any(item != value for item in parent_wire.exchange(value)):
            raise ValueError('Remaining parent owners disagree')
        data.save(args.home / 'survival.json', {**binding, 'rank': rank, 'passed': True,
            'expert_exit': receipt, 'id': row['id'], 'answer': value})
    finally:
        dist.destroy_process_group()
