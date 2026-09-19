#!/usr/bin/env python3
"""Fresh comparison of batched full-model learning with resident PowerSGD.

Research only. The frozen plan selects microbatches from training-only timing
probes. All final candidates must be committed before fresh test scoring.
"""
import argparse
from datetime import timedelta
import inspect
import json
import os
from pathlib import Path
import time

import study_learning_methods as base
from neuroshard.evolution import cooperative as group
from neuroshard.evolution import gradient_compression as hooks
from neuroshard.evolution import local_windows as windows
from neuroshard.evolution import reference as engine
from neuroshard.evolution import reference_data as data

_original_sources = base.sources
_original_runtime = base.runtime


def sources():
    values = _original_sources()
    for name in ['scripts/study_batched_learning.py', 'scripts/batch_scaling_probe.py', 'scripts/microbatch_probe.py']:
        values[name] = data.sha256(base.ROOT / name)
    return values


def runtime():
    value = _original_runtime()
    value['cuda_allocator'] = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default')
    if value['cuda_allocator'] != json.loads(base.PLAN.read_bytes())['cuda_allocator']:
        raise ValueError('Use the frozen CUDA allocator profile')
    return value


def configure():
    base.PLAN = base.ROOT / 'config/experiments/batched-learning-study.json'
    base.PREPARED = base.ROOT / 'config/experiments/batched-learning-study-inputs.json'
    base.SELECTED = base.ROOT / 'config/experiments/batched-learning-study-selection.json'
    base.sources = sources
    base.runtime = runtime


def train(args, plan):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as compression
    from batch_scaling_probe import batched_step

    prepared = base.inputs(args, plan)
    profile = base.runtime()
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if args.arm not in plan['arms'] or world != plan['arms'][args.arm] or not 0 <= rank < world:
        raise ValueError('Rank/world differs from the frozen arm')
    if base.model_snapshot(args.model_dir, plan['model']) != prepared['model_snapshot']:
        raise ValueError('Changed initial model')
    destination = args.home / args.arm / f'rank-{rank}'
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / 'started.json'
    if marker.exists():
        raise ValueError('Preserve the earlier attempt')
    binding = data.identity({'inputs': data.identity(prepared), 'arm': args.arm,
                             'runtime': group.runtime_profile(profile),
                             'upstream_hook': data.sha256(Path(inspect.getsourcefile(compression)))})
    data.save(marker, {'binding': binding, 'runtime': profile, 'started': time.time()})
    budget = engine.Budget(destination, time.time(), plan['budget'])
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(seconds=300))
    try:
        group.agree_digest(binding, world, 'cuda')
        recipe = plan['training']
        microbatch = plan['microbatches'][args.arm]
        if recipe['batch_documents'] % world or not 0 < microbatch <= recipe['batch_documents'] // world:
            raise ValueError('Global batch must divide between ranks and bound the local microbatch')
        torch.manual_seed(recipe['seed'])
        model = engine.load_model(args.model_dir, 'cuda', plan['model']['parameters'])
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()
        optimizer = engine.optimizer_for(model, recipe)
        wrapped = DDP(model, device_ids=[0], broadcast_buffers=False, gradient_as_bucket_view=True,
                      bucket_cap_mb=64) if world > 1 else model
        state = None
        if args.arm == 'compressed-pair':
            state = compression.PowerSGDState(process_group=dist.group.WORLD, **plan['compression'])
            wrapped.register_comm_hook(state, hooks.resident_power_sgd)
        rows = base.records(args, prepared, 'train')
        schedule = engine.schedule(len(rows), recipe['steps'], recipe['batch_documents'], recipe['seed'])
        before, started, journal = base.network_bytes(), time.monotonic(), []
        for index, indices in enumerate(schedule):
            budget.check()
            observation = batched_step(wrapped, optimizer, [rows[i] for i in indices], recipe,
                                       index, rank, world, microbatch)
            journal.append(observation)
            if index % 8 == 0 or index + 1 == recipe['steps']:
                base.emit('step', arm=args.arm, rank=rank, step=index + 1,
                          loss=observation['loss'], seconds=observation['seconds'])
            with (destination / 'steps.jsonl').open('a') as output:
                output.write(json.dumps(observation) + '\n')
        active, after = time.monotonic() - started, base.network_bytes()
        digest = group.parameter_digest(model)
        group.agree_digest(digest, world, 'cuda')
        compression_stats = None
        if state is not None:
            if state.iter != recipe['steps'] or any(value.device.type != 'cuda' for value in state.error_dict.values()):
                raise ValueError('Compression iterations or residency differ')
            ratio, original, transmitted = state.compression_stats()
            compression_stats = {'ratio': ratio, 'original_elements': original, 'transmitted_elements': transmitted,
                                 'error_digest': windows.state_digest((str(k), v) for k, v in sorted(state.error_dict.items()))}
        checkpoint_started, files = time.monotonic(), None
        if rank == 0:
            checkpoint = destination / 'model'
            model.save_pretrained(checkpoint, safe_serialization=True, max_shard_size='2GB')
            base.tokenizer_for(args.model_dir).save_pretrained(checkpoint)
            files = {path.name: data.sha256(path) for path in checkpoint.iterdir() if path.is_file()}
        if world > 1:
            dist.barrier()
        result = {'arm': args.arm, 'rank': rank, 'world': world, 'prepared': data.identity(prepared),
                  'binding': binding, 'runtime': profile, 'steps': journal, 'parameter_digest': digest,
                  'model_files': files, 'compression': compression_stats, 'active_seconds': active,
                  'checkpoint_seconds': time.monotonic() - checkpoint_started,
                  'network_start': before, 'network_end': after, 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
                  'tokens_issued': 0, 'scope': 'Batched research comparison. Model-only checkpoint; no native serving or recovery claim.'}
        data.save(destination / 'result.json', result)
        base.emit('completed', arm=args.arm, rank=rank, seconds=active, digest=digest)
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def main():
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'train', 'select', 'evaluate'])
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--previous-home', type=Path)
    parser.add_argument('--reference-home', type=Path)
    parser.add_argument('--candidate-dir', type=Path)
    parser.add_argument('--arm', choices=['seed', 'single', 'compressed-pair'])
    args = parser.parse_args()
    plan = json.loads(base.PLAN.read_bytes())
    with windows.exclusive_device('cuda' if args.command in ('train', 'evaluate') else 'cpu'):
        (train if args.command == 'train' else getattr(base, args.command))(args, plan)


if __name__ == '__main__':
    main()
