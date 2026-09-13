import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from neuroshard.evolution.gradient_compression import offloaded_power_sgd, resident_power_sgd


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(item) for item in value)
    return value


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) == type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def compare_resident_and_offloaded(device, multiple_buckets=True):
    """Also invoked on the two real CUDA hosts to exercise CPU/GPU copies."""
    from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as upstream
    from torch.nn.parallel import DistributedDataParallel
    snapshots = []
    modes = ('resident', 'offloaded') if multiple_buckets else ('upstream', 'resident', 'offloaded')
    for mode in modes:
        torch.manual_seed(301)
        model = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.Tanh(),
                                    torch.nn.Linear(256, 256), torch.nn.Tanh(),
                                    torch.nn.Linear(256, 128)).to(device)
        wrapped = DistributedDataParallel(model, device_ids=[0] if device == 'cuda' else None,
                                          gradient_as_bucket_view=True,
                                          bucket_cap_mb=.05 if multiple_buckets else 25)
        state = upstream.PowerSGDState(dist.group.WORLD, matrix_approximation_rank=32,
                                       start_powerSGD_iter=8, random_seed=19,
                                       orthogonalization_epsilon=1e-8,
                                       use_error_feedback=True, warm_start=True)
        hooks = {'upstream': upstream.powerSGD_hook, 'resident': resident_power_sgd,
                 'offloaded': offloaded_power_sgd}
        wrapped.register_comm_hook(state, hooks[mode])
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, foreach=False)
        generator = torch.Generator().manual_seed(500 + dist.get_rank())
        losses = []
        for _ in range(12):
            inputs = torch.randn(16, 256, generator=generator).to(device)
            targets = torch.randn(16, 128, generator=generator).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (wrapped(inputs) - targets).square().mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if mode == 'offloaded':
                assert all(value.device.type == 'cpu' for value in state.error_dict.values())
        assert state.iter == 12 and state.error_dict and state.total_numel_after_compression > 0
        assert (len(state.error_dict) > 1) == multiple_buckets
        snapshots.append(cpu_copy({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                   'errors': state.error_dict, 'p': state.p_memory_dict,
                                   'q': state.q_memory_dict, 'losses': losses,
                                   'rng': state.rng.get_state()[1].tolist()}))
        del wrapped, model, optimizer, state
    for snapshot in snapshots[1:]:
        equal(snapshots[0], snapshot)
    return {'device': device, 'rank': dist.get_rank(), 'iterations': 12,
            'multiple_buckets': multiple_buckets, 'modes': modes,
            'exact_model_optimizer_errors_projections_losses_and_rng': True}


def worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        result = [compare_resident_and_offloaded('cpu', multiple) for multiple in (False, True)]
        (Path(output) / f'{rank}.json').write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


def test_offloaded_hook_preserves_upstream_state_through_multiple_bucket_futures(tmp_path):
    mp.spawn(worker, args=(str(tmp_path / 'rendezvous'), str(tmp_path)), nprocs=2, join=True)
    for rank in range(2):
        assert all(row['exact_model_optimizer_errors_projections_losses_and_rng']
                   for row in json.loads((tmp_path / f'{rank}.json').read_bytes()))
