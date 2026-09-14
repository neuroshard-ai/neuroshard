"""Operated adaptive-shard experiment using the existing shard runner.

The prepared job binds all sources and data. Repartitioning changes ownership,
not the job identity or optimizer cursor. Nothing here issues native currency.
"""
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, LlamaConfig

from .. import reference, reference_data as data, grounded_tasks as tasks
from . import guarded, portable
from .model import Partition
from .training import score, generate
from .wire import Wire


def implementation(root):
    paths = ['scripts/run_sharded_training.py', 'scripts/prepare_adaptive_shards.py',
             'docs/learning-reference-requirements.txt', 'src/neuroshard/dataflow/store.py',
             'src/neuroshard/dataflow/collect.py']
    paths += ['src/neuroshard/evolution/'+n+'.py' for n in ['reference', 'reference_data', 'grounded_tasks', 'data']]
    paths += ['src/neuroshard/evolution/sharded/'+n+'.py' for n in
              ['__init__', 'model', 'training', 'wire', 'checkpoint', 'portable', 'guarded', 'expansion', 'adaptive_job']]
    return {p: data.sha256(Path(root)/p) for p in paths}


def emit(event, **values):
    print(json.dumps({'event': event, 'time': time.time(), **values}), flush=True)


def main(argv, root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'evaluate'])
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--seed', type=Path, required=True)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--until', type=int)
    parser.add_argument('--roles', nargs='+')
    parser.add_argument('--selection', type=Path)
    args = parser.parse_args(argv)
    prepared = json.loads(args.prepared.read_bytes())
    plan = prepared['plan']
    if implementation(root) != prepared['sources']:
        raise ValueError('Numerical source differs from the frozen job')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    seed_meta = json.loads((args.seed/'manifest.json').read_bytes())
    key = data.identity(seed_meta['boundaries'])
    if data.sha256(args.seed/'manifest.json') != prepared['seed_layouts'][key][rank]:
        raise ValueError('Uncommitted seed layout')
    if world != len(seed_meta['boundaries'])-1 or seed_meta['rank'] != rank:
        raise ValueError('Wrong reference rank declaration')
    if data.sha256(args.seed/'config.json') != prepared['config_sha256']:
        raise ValueError('Base architecture changed')
    # Tokenization is part of the computation, checked before GPU allocation.
    tokenizer = AutoTokenizer.from_pretrained(args.seed, local_files_only=True, trust_remote_code=False)
    if data.tokenizer_identity(tokenizer) != prepared['tokenizer']:
        raise ValueError('Tokenizer changed')
    runtime = reference.configure('cuda', plan['threads'])
    runtime['allocator'] = os.environ['PYTORCH_CUDA_ALLOC_CONF']
    profile = {k: v for k, v in runtime.items() if k != 'host'}
    job = data.identity({'prepared': data.identity(prepared), 'runtime': profile})
    common = json.loads(args.resume.read_bytes()) if args.resume else None
    anchor = json.loads(args.reference.read_bytes()) if args.reference else None
    config = portable.validate(common) if common else LlamaConfig.from_pretrained(args.seed, local_files_only=True)
    boundaries = common['boundaries'] if common else seed_meta['boundaries']
    if world != len(boundaries)-1:
        raise ValueError('Worker count differs from the committed layout')
    base = LlamaConfig.from_pretrained(args.seed, local_files_only=True)
    allowed = portable.configuration(base)
    allowed['num_hidden_layers'] = config.num_hidden_layers
    if portable.configuration(config) != allowed or config.num_hidden_layers not in plan['allowed_layers']:
        raise ValueError('Architecture outside the prepared experiment')
    config._attn_implementation = base._attn_implementation = 'sdpa'
    torch.manual_seed(plan['training']['seed']+rank)
    random.seed(plan['training']['seed']+rank)
    np.random.seed(plan['training']['seed']+rank)
    if args.home.exists():
        raise ValueError('Use a fresh result directory to preserve every attempt')
    args.home.mkdir(parents=True)
    shard = Partition(config, boundaries, rank, 'cuda', plan['parameter_limit'])
    optimizer = reference.optimizer_for(shard, plan['training']) if args.command == 'train' else None
    teacher = None
    if optimizer is not None:
        if common and common['step'] >= plan['cohort_steps']:
            if (not anchor or anchor['step'] != plan['cohort_steps'] or anchor['job'] != job
                    or anchor['config'] != portable.configuration(base)
                    or anchor['boundaries'] != seed_meta['boundaries']):
                raise ValueError('Cohort two requires its committed phase-one reference')
        elif anchor:
            raise ValueError('Cohort one uses the frozen seed reference')
        teacher = Partition(base, seed_meta['boundaries'], rank, 'cuda', plan['parameter_limit'])
        if anchor:
            portable.load(args.reference.parent, teacher, None, anchor, job, restore_rng=False)
        else:
            teacher.load_weights(args.seed, seed_meta)
        teacher.eval().requires_grad_(False)
        required = shard.resident_parameters*16+teacher.resident_parameters*4+plan['memory_reserve_bytes']
        if required > torch.cuda.get_device_properties(0).total_memory:
            raise ValueError('Student, Adam and reference exceed the measured GPU budget')
    dist.init_process_group('gloo', timeout=timedelta(seconds=180))
    wire = Wire(rank, world)
    started = time.monotonic()
    try:
        agreement = {'job': job, 'checkpoint': data.identity(common), 'boundaries': boundaries,
                     'reference': data.identity(anchor)}
        if any(r != agreement for r in wire.exchange(agreement)):
            raise ValueError('Workers disagree on computation and checkpoint')
        births, start = {}, 0
        if common:
            births = portable.load(args.resume.parent, shard, optimizer, common, job,
                                   restore_rng=optimizer is not None)
            start = common['step']
        else:
            shard.load_weights(args.seed, seed_meta)
            if optimizer is not None:
                common = portable.commit(args.home, shard, optimizer, wire, job, 0, None)
        data.save(args.home/'started.json', {'job': job, 'runtime': runtime, 'rank': rank,
            'boundaries': boundaries, 'parameters': sum(math.prod(s) for s in portable.shapes(config).values()),
            'resident_parameters': shard.resident_parameters, 'restored_step': start,
            'reference_resident_parameters': teacher.resident_parameters if teacher else 0,
            'started': time.time(), 'tokens_issued': 0})
        def records(role):
            spec = prepared['roles'][role]
            return data.read_records(args.prepared.parent/spec['file'], spec['sha256'])
        if args.command == 'train':
            until = args.until or plan['training']['steps']
            if not start < until <= plan['training']['steps'] or until not in plan['checkpoints']:
                raise ValueError('Stop only at a declared later checkpoint')
            if start < plan['cohort_steps'] < until:
                raise ValueError('Commit phase one before installing the next cohort reference')
            cohorts = {role: records(role) for role in ['train-a', 'train-b']}
            parent = data.identity(common)
            for index in range(start, until):
                if time.monotonic()-started > plan['max_seconds']:
                    raise TimeoutError('Training deadline reached')
                assignment = prepared['schedule'][index]
                batch = [cohorts[assignment['role']][i] for i in assignment['indices']]
                row = guarded.train_step(shard, teacher, optimizer, wire, batch,
                    plan['training'], index, plan['microbatch'], kl_strength=plan['kl_strength'])
                with (args.home/'steps.jsonl').open('a') as log:
                    log.write(json.dumps(row)+'\n')
                    log.flush()
                emit('step', rank=rank, **row)
                if index+1 in plan['checkpoints']:
                    common = portable.commit(args.home, shard, optimizer, wire, job, index+1, parent, births)
                    parent = data.identity(common)
                    emit('checkpoint', rank=rank, step=index+1, root=parent, state_root=common['state_root'])
            data.save(args.home/'result.json', {'checkpoint': parent, 'state_root': common['state_root'],
                'rank': rank, 'start': start, 'end': until, 'seconds': time.monotonic()-started,
                'peak_cuda_bytes': torch.cuda.max_memory_allocated(), 'tensor_wire_bytes': wire.sent_tensor_bytes,
                'tokens_issued': 0})
        else:
            roles = args.roles or plan['development_roles']
            if set(roles)-set(plan['development_roles']):
                if not args.selection:
                    raise ValueError('Commit development selection before opening final tests')
                selection = json.loads(args.selection.read_bytes())
                if (selection['prepared'] != data.identity(prepared)
                        or (common and data.identity(common) not in selection['checkpoints'])
                        or set(roles)-set(plan['final_roles'])):
                    raise ValueError('Evaluation not authorized by the frozen selection')
            outcomes = {}
            for role in roles:
                rows = records(role)
                outcomes[role] = {'losses': score(shard, wire, rows), 'answers': []}
                if role.startswith('test-'):
                    for row in rows[:plan['generation_cases_per_cohort']]:
                        prompt = tokenizer.apply_chat_template(row['messages'][:-1], tokenize=True, add_generation_prompt=True)
                        begun = time.monotonic()
                        ids = generate(shard, wire, prompt, plan['generation_tokens'], tokenizer.eos_token_id)
                        answer = tokenizer.decode(ids, skip_special_tokens=True)
                        outcomes[role]['answers'].append({'id': row['id'], 'output_ids': ids, 'text': answer,
                            'check': tasks.check_answer(row['task'], answer), 'seconds': time.monotonic()-begun})
                emit('evaluated', rank=rank, role=role)
            data.save(args.home/'evaluation.json', {'checkpoint': data.identity(common) if common else 'seed',
                'prepared': data.identity(prepared), 'rank': rank, 'outcomes': outcomes,
                'seconds': time.monotonic()-started, 'peak_cuda_bytes': torch.cuda.max_memory_allocated()})
    finally:
        dist.destroy_process_group()
