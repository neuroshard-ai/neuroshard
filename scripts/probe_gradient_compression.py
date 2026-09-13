#!/usr/bin/env python3
"""Check PowerSGD memory and rank agreement on two operated research GPUs.

This is a 12-update feasibility probe, not a quality experiment or a resumable
training profile. It reads only the already committed training partition.
"""
import argparse
from datetime import timedelta
import inspect
import json
import os
from pathlib import Path
import subprocess
import time

from neuroshard.evolution import cooperative as group
from neuroshard.evolution import local_windows as windows
from neuroshard.evolution import reference as engine
from neuroshard.evolution import reference_data as data
import run_local_training_windows as driver


CONTRACT = {
    'steps': 12,
    'world': 2,
    'matrix_approximation_rank': 32,
    'start_powerSGD_iter': 8,
    'min_compression_rate': 2,
    'use_error_feedback': True,
    'warm_start': True,
    'orthogonalization_epsilon': 1e-8,
    'random_seed': 20260913,
    'batch_tensors_with_same_shape': False,
}


def run(args):
    runtime = driver.numerical_runtime('cuda', 2)
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as compression

    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != CONTRACT['world'] or not 0 <= rank < world:
        raise ValueError('This feasibility probe requires exactly two GPU workers')
    plan = driver.validate(json.loads(driver.PLAN.read_bytes()))
    prepared = driver.inputs(args, plan)
    source = Path(__file__).resolve()
    relative = source.relative_to(driver.ROOT).as_posix()
    if subprocess.check_output(['git', 'show', 'HEAD:' + relative], cwd=driver.ROOT) != source.read_bytes():
        raise ValueError('Commit the probe before executing its fixed configuration')
    if driver.model_snapshot(args.model_dir, plan['model']) != prepared['model_snapshot']:
        raise ValueError('The pinned seed changed')
    if plan['training']['warmup_steps'] != CONTRACT['start_powerSGD_iter']:
        raise ValueError('Complete the declared learning-rate warmup before compression')
    binding = data.identity({
        'prepared': data.identity(prepared), 'source': data.sha256(source),
        'upstream_hook_source': data.sha256(Path(inspect.getsourcefile(compression))),
        'contract': CONTRACT, 'profile': group.runtime_profile(runtime),
    })
    output = args.output / f'rank-{rank}.json'
    marker = args.output / f'rank-{rank}.started.json'
    if output.exists() or marker.exists():
        raise ValueError('Preserve the previous probe attempt')
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    data.save(marker, {'binding': binding, 'rank': rank, 'started': started, 'runtime': runtime})
    budget = engine.Budget(args.output, started, {'seconds': 600, 'disk_gib': 1})
    dist.init_process_group('nccl', timeout=timedelta(seconds=180))
    try:
        group.agree_digest(binding, world, 'cuda')
        torch.manual_seed(plan['training']['seed'])
        model = engine.load_model(args.model_dir, 'cuda', plan['model']['parameters'])
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()
        optimizer = engine.optimizer_for(model, plan['training'])
        wrapped = DistributedDataParallel(model, device_ids=[0], broadcast_buffers=False,
                                          gradient_as_bucket_view=True, bucket_cap_mb=64)
        hook_options = {key: value for key, value in CONTRACT.items() if key not in {'steps', 'world'}}
        state = compression.PowerSGDState(process_group=dist.group.WORLD, **hook_options)
        wrapped.register_comm_hook(state, compression.powerSGD_hook)
        records = driver.partition(args.home, prepared, 'train')
        recipe = plan['training']
        schedule = engine.schedule(len(records), recipe['steps'], recipe['batch_documents'], recipe['seed'])
        before = driver.network_bytes()
        journal = []
        for index in range(CONTRACT['steps']):
            batch = [records[i] for i in schedule[index]]
            row = group.step(wrapped, optimizer, batch, 'cuda', recipe, index, rank, world, budget.check)
            journal.append(row)
            driver.emit('compression_probe_step', rank=rank, **row)
        after = driver.network_bytes()
        digest = group.parameter_digest(model)
        group.agree_digest(digest, world, 'cuda')
        if state.iter != CONTRACT['steps'] or not state.error_dict:
            raise ValueError('The hook did not reach the declared compressed iterations')
        for values in (state.error_dict, state.p_memory_dict, state.q_memory_dict):
            if any(not torch.isfinite(value).all() for value in values.values()):
                raise ValueError('Nonfinite compression state')
        ratio, original_elements, transmitted_elements = state.compression_stats()
        result = {
            'binding': binding, 'rank': rank, 'runtime': runtime, 'contract': CONTRACT,
            'prepared': data.identity(prepared), 'steps': journal, 'parameter_digest': digest,
            'network_start': before, 'network_end': after,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
            'compression_iterations': state.iter - CONTRACT['start_powerSGD_iter'],
            'upstream_compression_stats': {'ratio': ratio, 'original_elements': original_elements,
                                         'transmitted_elements': transmitted_elements},
            'error_elements': sum(value.numel() for value in state.error_dict.values()),
            'error_state_digest': windows.state_digest((str(key), value) for key, value in sorted(state.error_dict.items())),
            'seconds': time.time() - started, 'tokens_issued': 0, 'serving_approved': False,
            'scope': 'Two owned GPUs; eight dense warmup updates and four compressed updates. No held-out scoring, checkpoint recovery, quality or sustained-speed claim. Operator backups may run concurrently.',
        }
        data.save(output, result)
        driver.emit('compression_probe_completed', rank=rank, parameter_digest=digest)
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with windows.exclusive_device('cuda'):
        run(args)


if __name__ == '__main__':
    main()
