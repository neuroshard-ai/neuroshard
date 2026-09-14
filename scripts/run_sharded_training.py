#!/usr/bin/env python3
"""Train, recover and evaluate a pinned model using only rank-owned weights.

Research execution over an operated Gloo group. Membership denotes logical
shard slots, allowing another physical machine to restore the same slot.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import shutil
import time

os.environ.setdefault('ATEN_CPU_CAPABILITY', 'default')
os.environ.setdefault('MKL_ENABLE_INSTRUCTIONS', 'SSE4_2')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
import torch.distributed as dist
from transformers import LlamaConfig, AutoTokenizer

from neuroshard.evolution import reference as engine
from neuroshard.evolution import reference_data as data
from neuroshard.evolution import grounded_tasks as tasks
from neuroshard.evolution.sharded import checkpoint
from neuroshard.evolution.sharded.model import Partition
from neuroshard.evolution.sharded.training import train_step, score, generate
from neuroshard.evolution.sharded.wire import Wire

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['scripts/run_sharded_training.py', 'src/neuroshard/evolution/reference.py',
           'src/neuroshard/evolution/reference_data.py', 'src/neuroshard/evolution/grounded_tasks.py',
           'src/neuroshard/dataflow/store.py'] + [
               'src/neuroshard/evolution/sharded/'+name+'.py'
               for name in ['__init__', 'model', 'wire', 'training', 'checkpoint']]


def implementation():
    return {name: data.sha256(ROOT/name) for name in SOURCES}


def emit(event, **values):
    print(json.dumps({'event': event, 'time': time.time(), **values}), flush=True)


def network():
    return {p.name: {name: int((p/'statistics'/filename).read_text())
                    for name, filename in [('tx', 'tx_bytes'), ('rx', 'rx_bytes')]}
            for p in Path('/sys/class/net').iterdir() if p.name != 'lo'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'evaluate'])
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--seed', type=Path, required=True)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--until', type=int)
    parser.add_argument('--pause-at-step', type=int)
    args = parser.parse_args()
    prepared = json.loads(args.prepared.read_bytes())
    plan = prepared['plan']
    if implementation() != prepared['sources']:
        raise ValueError('Numerical source differs from the prepared job')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != len(plan['boundaries'])-1 or not 0 <= rank < world:
        raise ValueError('Rank declaration differs from shard layout')
    runtime = engine.configure('cuda', plan['threads'])
    runtime['allocator'] = os.environ['PYTORCH_CUDA_ALLOC_CONF']
    profile = {k: v for k, v in runtime.items() if k != 'host'}
    binding = data.identity({'prepared': data.identity(prepared), 'runtime': profile})
    seed_manifest = json.loads((args.seed/'manifest.json').read_bytes())
    if (seed_manifest['rank'] != rank or seed_manifest['boundaries'] != plan['boundaries']
            or data.sha256(args.seed/'manifest.json') != prepared['seed']['rank_manifests'][rank]):
        raise ValueError('Wrong seed partition manifest')
    if data.sha256(args.seed/'config.json') != prepared['config_sha256']:
        raise ValueError('Model configuration changed')
    config = LlamaConfig.from_pretrained(args.seed, local_files_only=True)
    config._attn_implementation = 'sdpa'
    torch.manual_seed(plan['training']['seed']+rank)
    random.seed(plan['training']['seed']+rank)
    np.random.seed(plan['training']['seed']+rank)
    shard = Partition(config, plan['boundaries'], rank, 'cuda', plan['parameter_limit'])
    optimizer = engine.optimizer_for(shard, plan['training']) if args.command == 'train' else None
    args.home.mkdir(parents=True, exist_ok=True)
    if (args.home/'result.json').exists() or (args.home/'evaluation.json').exists():
        raise ValueError('Preserve the prior completed run')
    dist.init_process_group('gloo', timeout=timedelta(seconds=90))
    wire = Wire(rank, world)
    started, before = time.monotonic(), network()
    try:
        declarations = wire.exchange({'binding': binding, 'rank': rank,
                                      'parameters': shard.resident_parameters})
        if any(r['binding'] != binding for r in declarations) or sum(r['parameters'] for r in declarations) != plan['parameters']:
            raise ValueError('Workers disagree on computation or total model coverage')
        common, start = None, 0
        if args.resume:
            common = json.loads(args.resume.read_bytes())
            if any(r != data.identity(common) for r in wire.exchange(data.identity(common))):
                raise ValueError('Workers must restore the same global checkpoint')
            start = checkpoint.load(args.resume.parent, shard, optimizer, common, binding,
                                    restore_rng=args.command == 'train')
        else:
            shard.load_weights(args.seed, seed_manifest)
            if optimizer is not None:
                common = checkpoint.commit(args.home, shard, optimizer, wire, binding, 0, None)
        data.save(args.home/'started.json', {'binding': binding, 'prepared': data.identity(prepared),
            'rank': rank, 'runtime': runtime, 'resident_parameters': shard.resident_parameters,
            'gpu_total_bytes': torch.cuda.get_device_properties(0).total_memory,
            'unsharded_fp32_training_state_bytes': plan['parameters']*16,
            'restored_step': start, 'network_start': before, 'started': time.time()})
        def records(role):
            spec = prepared['roles'][role]
            return data.read_records(args.prepared.parent/spec['file'], spec['sha256'])
        if args.command == 'train':
            until = args.until or plan['training']['steps']
            if not start < until <= plan['training']['steps'] or until not in plan['checkpoints']:
                raise ValueError('Stop only at a declared later checkpoint')
            cohorts = {role: records(role) for role in ['train-a', 'train-b']}
            parent = data.identity(common)
            journal = []
            def interruption(index, offset):
                if args.pause_at_step == index+1 and rank == world-1 and offset == 0:
                    data.save(args.home/'fault-ready.json', {'step': index+1, 'phase': 'after-forward',
                                                            'last_committed_step': common['step']})
                    emit('fault_ready', rank=rank, step=index+1)
                    deadline = time.monotonic()+180
                    while time.monotonic() < deadline:
                        time.sleep(2)
                    raise TimeoutError('Fault controller did not remove the paused worker')
            for index in range(start, until):
                if time.monotonic()-started > plan['max_seconds']:
                    raise TimeoutError('Training exceeded the job deadline')
                assignment = prepared['schedule'][index]
                batch = [cohorts[assignment['role']][i] for i in assignment['indices']]
                row = train_step(shard, optimizer, wire, batch, plan['training'], index,
                                 plan['microbatch'], after_forward=interruption)
                journal.append(row)
                with (args.home/'steps.jsonl').open('a') as log:
                    log.write(json.dumps(row)+'\n')
                emit('step', rank=rank, **row)
                if index+1 in plan['checkpoints']:
                    if shutil.disk_usage(args.home).free < 12*1024**3:
                        raise ValueError('Reserve checkpoint disk space before continuing')
                    common = checkpoint.commit(args.home, shard, optimizer, wire, binding, index+1, parent)
                    parent = data.identity(common)
                    emit('checkpoint', rank=rank, step=index+1, root=parent)
            data.save(args.home/'result.json', {'binding': binding, 'rank': rank, 'start': start, 'end': until,
                'root': parent, 'shards': common['shards'], 'steps': journal, 'seconds': time.monotonic()-started,
                'resident_parameters': shard.resident_parameters, 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
                'network_start': before, 'network_end': network(), 'tensor_wire_bytes': wire.sent_tensor_bytes,
                'tokens_issued': 0})
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.seed, local_files_only=True, trust_remote_code=False)
            if data.tokenizer_identity(tokenizer) != prepared['tokenizer']:
                raise ValueError('Tokenizer changed')
            outcomes = {}
            for role in ['test-a', 'test-b', 'retention']:
                rows = records(role)
                outcomes[role] = {'losses': score(shard, wire, rows), 'answers': []}
                if role != 'retention':
                    for row in rows[:plan['generation_cases_per_cohort']]:
                        prompt = tokenizer.apply_chat_template(row['messages'][:-1], tokenize=True, add_generation_prompt=True)
                        begun = time.monotonic()
                        ids = generate(shard, wire, prompt, plan['generation_tokens'], tokenizer.eos_token_id)
                        text = tokenizer.decode(ids, skip_special_tokens=True)
                        outcomes[role]['answers'].append({'id': row['id'], 'input_ids': prompt, 'output_ids': ids,
                            'text': text, 'check': tasks.check_answer(row['task'], text), 'seconds': time.monotonic()-begun})
                emit('evaluated', rank=rank, role=role)
            data.save(args.home/'evaluation.json', {'checkpoint': data.identity(common) if common else 'seed',
                'prepared': data.identity(prepared), 'rank': rank, 'outcomes': outcomes,
                'seconds': time.monotonic()-started, 'peak_cuda_bytes': torch.cuda.max_memory_allocated()})
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'adaptive':
        from neuroshard.evolution.sharded.adaptive_job import main as adaptive
        adaptive(sys.argv[2:], ROOT)
    elif len(sys.argv) > 1 and sys.argv[1] == 'continued':
        from neuroshard.evolution.sharded.continued_job import main as continued_job
        continued_job(sys.argv[2:], ROOT)
    elif len(sys.argv) > 1 and sys.argv[1] == 'consolidate':
        from neuroshard.evolution.sharded.consolidation_job import main as consolidation_job
        consolidation_job(sys.argv[2:], ROOT)
    else:
        main()
