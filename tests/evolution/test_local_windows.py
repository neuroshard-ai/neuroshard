from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from neuroshard.evolution import local_windows as windows


def tiny():
    model = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1., 2., 3.], [4., 5., 6.]]))
    return model


def process(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        model = tiny()
        outer = windows.OuterNesterov(model, .7, .9, chunk_bytes=8)
        for step in range(3):
            with torch.no_grad():
                model.weight.sub_((rank + 1) * (step + 1) / 16)
            result = outer.synchronize(model, 2)
            assert result['delta_payload_bytes_per_rank'] == 24
        torch.save({'model': model.state_dict(), 'velocity': outer.velocity}, Path(output) / f'{rank}.pt')
        values = {4: 'ab' * 32, 8: ('cd' if rank == 0 else 'ef') * 32}
        assert windows.common_checkpoint(values, [4, 8], 2, 'cpu') == 4
        assert windows.common_checkpoint({8: 'cd' * 32} if rank == 0 else {}, [8], 2, 'cpu') is None
    finally:
        dist.destroy_process_group()


def test_distributed_outer_matches_independent_torch_nesterov_and_common_checkpoint(tmp_path):
    torch.set_num_threads(1)
    reference = tiny()
    optimizer = torch.optim.SGD(reference.parameters(), lr=.7, momentum=.9, nesterov=True)
    for step in range(3):
        # The two synthetic local endpoints induce a known mean outer gradient.
        reference.weight.grad = torch.full_like(reference.weight, 1.5 * (step + 1) / 16)
        optimizer.step()
    mp.spawn(process, args=(str(tmp_path / 'rendezvous'), str(tmp_path)), nprocs=2, join=True)
    results = [torch.load(tmp_path / f'{rank}.pt', weights_only=True) for rank in range(2)]
    torch.testing.assert_close(results[0]['model']['weight'], reference.weight, rtol=0, atol=1e-6)
    assert torch.equal(results[0]['model']['weight'], results[1]['model']['weight'])
    assert torch.equal(results[0]['velocity'][0], results[1]['velocity'][0])


def test_restored_outer_state_reproduces_continuation_and_rejects_wrong_round(tmp_path):
    model = tiny()
    outer = windows.OuterNesterov(model, .7, .9, chunk_bytes=8)
    with torch.no_grad():
        model.weight.sub_(.25)
    outer.synchronize(model, 1)
    outer.save(tmp_path / 'outer.pt')
    restored_model = tiny()
    restored_model.load_state_dict(model.state_dict())
    restored = windows.OuterNesterov(restored_model, .7, .9, chunk_bytes=8)
    with pytest.raises(ValueError, match='identity'):
        restored.restore(tmp_path / 'outer.pt', 2)
    restored.restore(tmp_path / 'outer.pt', 1)
    for current, state in ((model, outer), (restored_model, restored)):
        with torch.no_grad():
            current.weight.add_(.5)
        state.synchronize(current, 1)
    assert torch.equal(model.weight, restored_model.weight)
    assert outer.digest() == restored.digest()


def test_rank_identity_prevents_reusing_another_workers_adam_checkpoint():
    one = windows.rank_binding('prepared', {'host': 'one', 'torch': 'pinned'}, 'diloco', 0, 4)
    moved = windows.rank_binding('prepared', {'host': 'two', 'torch': 'pinned'}, 'diloco', 0, 4)
    other = windows.rank_binding('prepared', {'host': 'one', 'torch': 'pinned'}, 'diloco', 1, 4)
    assert one == moved and one != other


def test_outer_rejects_nonfinite_update_and_invalid_bounds():
    model = tiny()
    outer = windows.OuterNesterov(model, .7, .9)
    with torch.no_grad():
        model.weight[0, 0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        outer.synchronize(model, 1)
    assert outer.round == 0
    with pytest.raises(ValueError):
        windows.OuterNesterov(tiny(), .7, 1.)
    with pytest.raises(ValueError):
        windows.OuterNesterov(tiny().double(), .7, .9)
    with pytest.raises(ValueError):
        windows.common_checkpoint({}, [8, 4], 1, 'cpu')


def test_exclusive_gpu_lock_rejects_overlap_and_releases_after_failure(tmp_path):
    with pytest.raises(RuntimeError, match='injected'):
        with windows.exclusive_device('cuda', tmp_path):
            with pytest.raises(ValueError, match='owns this host GPU'):
                with windows.exclusive_device('cuda', tmp_path):
                    pytest.fail('A concurrent GPU job acquired the device')
            raise RuntimeError('injected worker failure')
    with windows.exclusive_device('cuda', tmp_path):
        with windows.exclusive_device('cpu', tmp_path):
            pass
