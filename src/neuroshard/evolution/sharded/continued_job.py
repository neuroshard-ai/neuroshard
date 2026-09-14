"""Operated continued-learning driver. Enforces the committed artifact freeze.

Starts from the passing phase-A portable checkpoint and its Adam state. Does
not grow the model, mint NEURO, or promote serving. Native settlement of these
updates requires a later job activation and reserved-window receipts.
"""
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

from .. import continued, grounded_tasks as tasks, reference, reference_data as data
from . import guarded, portable
from .model import Partition
from .training import score, generate
from .wire import Wire


def emit(event, **values):
    print(json.dumps({'event': event, 'time': time.time(), **values}), flush=True)


def main(argv, root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'evaluate'])
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--seed', type=Path, required=True)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--until', type=int)
    parser.add_argument('--roles', nargs='+')
    parser.add_argument('--selection', type=Path)
    args = parser.parse_args(argv)
    plan = continued.load(args.plan)
    prepared = json.loads(args.prepared.read_bytes())
    parent = json.loads(args.parent.read_bytes())
    common = json.loads(args.resume.read_bytes()) if args.resume else parent
    selection = json.loads(args.selection.read_bytes()) if args.selection else None
    continued.preflight(plan, prepared, parent, common, args.command, args.roles, selection)
    tokenizer = AutoTokenizer.from_pretrained(args.seed, local_files_only=True, trust_remote_code=False)
    runtime = reference.configure('cuda', plan['threads'])
    runtime['allocator'] = os.environ['PYTORCH_CUDA_ALLOC_CONF']
    profile = {k: v for k, v in runtime.items() if k != 'host'}
    if {key: profile[key] for key in continued.RUNTIME_KEYS} != plan['runtime']:
        raise ValueError('Runtime profile differs from the frozen artifact')
    continued.assert_artifact_freeze(plan, prepared, parent, data.tokenizer_identity(tokenizer), profile)
    if data.sha256(args.seed / 'config.json') != plan['config_sha256']:
        raise ValueError('Architecture config differs from the frozen artifact')
    if data.tokenizer_identity(tokenizer) != plan['tokenizer']:
        raise ValueError('Tokenizer identity differs from the frozen artifact')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    job = continued.job_identity(prepared, profile)
    config = portable.validate(parent)
    portable.validate(common)
    if world != len(parent['boundaries']) - 1 or parent['boundaries'] != plan['parent']['boundaries']:
        raise ValueError('Worker layout differs from the frozen parent')
    if portable.configuration(config) != portable.configuration(LlamaConfig.from_pretrained(args.seed, local_files_only=True)):
        raise ValueError('Architecture outside the prepared experiment')
    if config.num_hidden_layers not in plan['allowed_layers']:
        raise ValueError('Growth is forbidden in this experiment')
    config._attn_implementation = 'sdpa'
    torch.manual_seed(plan['training']['seed'] + rank)
    random.seed(plan['training']['seed'] + rank)
    np.random.seed(plan['training']['seed'] + rank)
    if args.home.exists():
        raise ValueError('Use a fresh result directory to preserve every attempt')
    args.home.mkdir(parents=True)
    shard = Partition(config, parent['boundaries'], rank, 'cuda', plan['parameter_limit'])
    optimizer = reference.optimizer_for(shard, plan['training']) if args.command == 'train' else None
    teacher = None
    if optimizer is not None:
        teacher = Partition(config, parent['boundaries'], rank, 'cuda', plan['parameter_limit'])
        portable.load(args.parent.parent, teacher, None, parent, parent['job'], restore_rng=False)
        teacher.eval().requires_grad_(False)
        required = shard.resident_parameters * 16 + teacher.resident_parameters * 4 + plan['memory_reserve_bytes']
        if required > torch.cuda.get_device_properties(0).total_memory:
            raise ValueError('Student, Adam and reference exceed the measured GPU budget')
    dist.init_process_group('gloo', timeout=timedelta(seconds=180))
    wire = Wire(rank, world)
    started = time.monotonic()
    try:
        agreement = {'job': job, 'parent': data.identity(parent), 'checkpoint': data.identity(common),
                     'boundaries': parent['boundaries'], 'reference': data.identity(parent)}
        if any(row != agreement for row in wire.exchange(agreement)):
            raise ValueError('Workers disagree on computation and checkpoint')
        inherit = data.identity(common) == plan['parent']['checkpoint']
        load_job = parent['job'] if inherit else job
        if not inherit and common['job'] != job:
            raise ValueError('Resume checkpoint belongs to another job')
        births = portable.load(args.resume.parent if args.resume else args.parent.parent,
                                shard, optimizer, common, load_job,
                                restore_rng=optimizer is not None)
        start = common['step']
        data.save(args.home / 'started.json', {'job': job, 'runtime': runtime, 'rank': rank,
            'boundaries': parent['boundaries'], 'parameters': plan['parameters'],
            'resident_parameters': shard.resident_parameters, 'restored_step': start,
            'parent': data.identity(parent), 'tokens_issued': 0})

        def records(role):
            return continued.read_role(args.prepared.parent, prepared, role)

        if args.command == 'train':
            until = args.until or plan['final_step']
            if not start < until <= plan['final_step'] or until not in plan['checkpoints']:
                raise ValueError('Stop only at a declared later checkpoint')
            rows = records('train')
            retention_rows = records('dev-retention')
            baseline_retention = score(teacher, wire, retention_rows)
            parent_root = data.identity(common)
            input_checkpoint = parent_root

            def check_development(checkpoint):
                report = continued.development_decision(plan, prepared, data.identity(checkpoint),
                    baseline_retention, score(shard, wire, retention_rows))
                if any(other != report for other in wire.exchange(report)):
                    raise ValueError('Workers disagree on development retention')
                data.save(args.home / f'development-{checkpoint["step"]:06d}.json', report)
                emit('development', rank=rank, step=checkpoint['step'],
                     mean_delta=report['mean_delta'], passed=report['passed'])
                if not report['passed']:
                    data.save(args.home / 'aborted.json', report)
                    raise ValueError('Development retention abort; preserve this rejected attempt')

            if start > plan['parent_step']:
                check_development(common)
            for index in range(start, until):
                if time.monotonic() - started > plan['max_seconds']:
                    raise TimeoutError('Training deadline reached')
                assignment = prepared['schedule'][index - plan['parent_step']]
                batch = [rows[i] for i in assignment['indices']]
                row = guarded.train_step(shard, teacher, optimizer, wire, batch, plan['training'],
                                          index - plan['parent_step'], plan['microbatch'],
                                          kl_strength=plan['reference']['kl_strength'])
                with (args.home / 'steps.jsonl').open('a') as log:
                    log.write(json.dumps(row) + '\n')
                    log.flush()
                emit('step', rank=rank, **row)
                if index + 1 in plan['checkpoints']:
                    common = portable.commit(args.home, shard, optimizer, wire, job, index + 1,
                                              parent_root, births)
                    parent_root = data.identity(common)
                    emit('checkpoint', rank=rank, step=index + 1, root=parent_root,
                         state_root=common['state_root'])
                    check_development(common)
            data.save(args.home / 'result.json', {'checkpoint': parent_root, 'state_root': common['state_root'],
                'rank': rank, 'start': start, 'end': until, 'seconds': time.monotonic() - started,
                'peak_cuda_bytes': torch.cuda.max_memory_allocated(), 'tokens_issued': 0,
                'input_checkpoint': input_checkpoint})
        else:
            roles = args.roles or plan['development_roles']
            outcomes = {}
            for role in roles:
                rows = records(role)
                outcomes[role] = {'losses': score(shard, wire, rows), 'answers': []}
                if role in ('test-new', 'test-prior', 'dev-new', 'dev-prior'):
                    for row in rows:
                        prompt = tokenizer.apply_chat_template(row['messages'][:-1], tokenize=True,
                                                               add_generation_prompt=True)
                        begun = time.monotonic()
                        ids = generate(shard, wire, prompt, plan['generation_tokens'], tokenizer.eos_token_id)
                        answer = tokenizer.decode(ids, skip_special_tokens=True)
                        outcomes[role]['answers'].append({
                            'id': row['id'], 'output_ids': ids, 'text': answer,
                            'check': tasks.check_answer(row['task'], answer),
                            'seconds': time.monotonic() - begun,
                        })
                emit('evaluated', rank=rank, role=role)
            data.save(args.home / 'evaluation.json', {
                'checkpoint': data.identity(common), 'prepared': data.identity(prepared),
                'rank': rank, 'outcomes': outcomes, 'seconds': time.monotonic() - started,
                'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
            })
    finally:
        dist.destroy_process_group()
