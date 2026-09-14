"""Measure shared-gradient scaling against efficient single-GPU microbatches.

Training partition only; these short runs make no model-quality claim.
"""
import argparse
import contextlib
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0, 'scripts')
import study_learning_methods as driver
from microbatch_probe import summed_loss
from neuroshard.evolution import cooperative as group
from neuroshard.evolution import gradient_compression as hooks
from neuroshard.evolution import local_windows as windows
from neuroshard.evolution import reference as engine
from neuroshard.evolution import reference_data as data


def batched_step(model, optimizer, records, recipe, index, rank, world, microbatch):
    import torch
    import torch.distributed as dist
    local = sorted(group.rank_records(records, rank, world), key=lambda row: len(row['input_ids']))
    denominator = sum(row['targets'] * row.get('loss_weight', 1) for row in records)
    rate = engine.learning_rate(recipe, index)
    for parameters in optimizer.param_groups:
        parameters['lr'] = rate
    optimizer.zero_grad(set_to_none=False)
    total, started = 0., time.monotonic()
    for offset in range(0, len(local), microbatch):
        subset = local[offset:offset + microbatch]
        context = model.no_sync() if world > 1 and offset + microbatch < len(local) else contextlib.nullcontext()
        with context:
            loss = summed_loss(model, subset, 'cuda')
            total += float(loss.detach())
            (loss * world / denominator).backward()
    if any(parameter.grad is None for parameter in model.parameters()):
        raise ValueError('Dense graph required to retain gradient buffers')
    if world > 1:
        value = torch.tensor(total, dtype=torch.float64, device='cuda')
        dist.all_reduce(value)
        total = float(value)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe['clip_norm'], error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize()
    return {'step': index + 1, 'seconds': time.monotonic() - started, 'loss': total / denominator,
            'learning_rate': rate, 'gradient_norm': float(norm), 'weighted_targets': denominator}


def run(args):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import PowerSGDState
    import microbatch_probe

    plan = json.loads(driver.PLAN.read_bytes())
    prepared = driver.inputs(args, plan)
    profile = driver.runtime()
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    assert world in (1, 2) and 0 <= rank < world
    assert driver.model_snapshot(args.model_dir, plan['model']) == prepared['model_snapshot']
    if world > 1:
        prior = json.loads((args.home / f'buffer-probe-rank-{rank}.json').read_bytes())
        assert prior['complete_numerical_equality'] is True
        dist.init_process_group('nccl', timeout=timedelta(seconds=180))
    contract = {'prepared': data.identity(prepared), 'source': data.sha256(Path(__file__)),
                'loss_source': data.sha256(Path(microbatch_probe.__file__)), 'steps': 16,
                'global_batch': args.global_batch, 'microbatch': args.microbatch,
                'world': world, 'warmup_updates_excluded': 8, 'retained_gradients': True,
                'compressed': world == 2, 'runtime': group.runtime_profile(profile)}
    output = args.home / f'batch-scaling-g{args.global_batch}-m{args.microbatch}-w{world}-rank-{rank}.json'
    assert not output.exists()
    journal = []
    result = {'contract': contract, 'rank': rank, 'scope': 'Short training-only throughput screen. No checkpoint is scored or promoted.'}
    try:
        group.agree_digest(data.identity(contract), world, 'cuda')
        recipe = {**plan['training'], 'batch_documents': args.global_batch}
        torch.manual_seed(recipe['seed'])
        model = engine.load_model(args.model_dir, 'cuda', plan['model']['parameters'])
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()
        optimizer = engine.optimizer_for(model, recipe)
        wrapped = DDP(model, device_ids=[0], broadcast_buffers=False, gradient_as_bucket_view=True,
                      bucket_cap_mb=64) if world > 1 else model
        if world > 1:
            state = PowerSGDState(process_group=dist.group.WORLD, **plan['compression'])
            wrapped.register_comm_hook(state, hooks.resident_power_sgd)
        rows = driver.records(args, prepared, 'train')
        schedule = engine.schedule(len(rows), recipe['steps'], recipe['batch_documents'], recipe['seed'])[:16]
        before, started = driver.network_bytes(), time.monotonic()
        for index, indices in enumerate(schedule):
            row = batched_step(wrapped, optimizer, [rows[i] for i in indices], recipe, index, rank, world, args.microbatch)
            journal.append(row)
            if index % 4 == 0 or index == len(schedule) - 1:
                print(json.dumps({'rank': rank, 'step': index + 1, 'seconds': row['seconds']}), flush=True)
        result.update(active_seconds=time.monotonic() - started, network_start=before, network_end=driver.network_bytes(),
                      peak_cuda_bytes=torch.cuda.max_memory_allocated(), median_seconds=statistics.median(r['seconds'] for r in journal[8:]),
                      parameter_digest=group.parameter_digest(model), succeeded=True)
        group.agree_digest(result['parameter_digest'], world, 'cuda')
    except Exception as error:
        result.update(succeeded=False, failure_type=type(error).__name__, failure=str(error))
        raise
    finally:
        result['steps'] = journal
        data.save(output, result)
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()
    print(json.dumps({key: value for key, value in result.items() if key != 'steps'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--global-batch', type=int, required=True)
    parser.add_argument('--microbatch', type=int, required=True)
    args = parser.parse_args()
    if args.global_batch % args.microbatch or args.global_batch > 4096:
        raise ValueError('Choose a divisible batch no larger than the committed training partition')
    with windows.exclusive_device('cuda'):
        run(args)
