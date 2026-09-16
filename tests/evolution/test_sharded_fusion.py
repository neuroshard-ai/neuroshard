"""Numerical and ownership checks, not evidence of improved LLM answers."""
import copy
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from neuroshard.evolution.sharded.fusion import CrossShardFusion, FusionCache
from neuroshard.evolution.sharded.wire import Wire


def fixture():
    torch.manual_seed(811)
    model = CrossShardFusion(12, {'directory': 8, 'protocol': 10}, rank=8, heads=2, max_context=32)
    hub = torch.randn(1, 7, 12)
    sources = {'directory': torch.randn(1, 7, 8), 'protocol': torch.randn(1, 7, 10)}
    return model, hub, sources


def test_initial_identity_still_learns_from_remote_states():
    model, hub, sources = fixture()
    hub[0, 0, 0] = -0.0
    with torch.no_grad():
        observed = model(hub, sources)
        assert torch.equal(observed.view(torch.int32), hub.view(torch.int32))
    target = hub + .3
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    initial = float((model(hub, sources)-target).square().mean().detach())
    for _ in range(40):
        optimizer.zero_grad()
        loss = (model(hub, sources)-target).square().mean()
        loss.backward()
        optimizer.step()
    assert float((model(hub, sources)-target).square().mean().detach()) < initial/20
    assert model.keys['directory'].weight.grad.abs().sum() > 0


def test_future_expert_answers_cannot_leak_into_current_prediction():
    model, hub, sources = fixture()
    torch.nn.init.normal_(model.output.weight, std=.1)
    sources = {key: value.requires_grad_() for key, value in sources.items()}
    expected = model(hub, sources)
    changed = {key: value.detach().clone() for key, value in sources.items()}
    for value in changed.values():
        value[:, 4:] += 100
    assert torch.equal(model(hub, changed)[:, :4], expected[:, :4])
    expected[:, :4].square().sum().backward()
    for value in sources.values():
        assert torch.equal(value.grad[:, 4:], torch.zeros_like(value.grad[:, 4:]))
        assert value.grad[:, :4].abs().sum() > 0


def test_projected_cache_matches_full_causal_execution_and_is_request_local():
    model, hub, sources = fixture()
    torch.nn.init.normal_(model.output.weight, std=.1)
    model.eval()
    with torch.no_grad():
        expected = model(hub, sources)
        cache = FusionCache(model)
        parts = []
        for start, stop in [(0, 3), (3, 4), (4, 5), (5, 6), (6, 7)]:
            projected = {key: model.project(key, value[:, start:stop]) for key, value in sources.items()}
            parts.append(cache.advance(hub[:, start:stop], projected, start))
        torch.testing.assert_close(torch.cat(parts, dim=1), expected, rtol=2e-6, atol=2e-6)
        assert cache.resident_bytes() == 2*2*7*8*4
        fresh = FusionCache(model)
        assert fresh.length == 0 and fresh.resident_bytes() == 0
        # Generation needs only the hub's last prefill position while keeping
        # every source position available in the receiving owner's cache.
        projected_all = {key: model.project(key, value) for key, value in sources.items()}
        last = fresh.advance(hub[:, -1:], projected_all, 0)
        torch.testing.assert_close(last, expected[:, -1:], rtol=2e-6, atol=2e-6)
        with pytest.raises(ValueError, match='continuing'):
            cache.advance(hub[:, :1], projected, 0)
        with pytest.raises(ValueError, match='failed'):
            cache.advance(hub[:, :1], projected, 7)


def test_cache_rejects_changed_weights_and_missing_sources():
    model, hub, sources = fixture()
    model.eval()
    cache = FusionCache(model)
    with torch.no_grad():
        model.output.weight.add_(.1)
    with pytest.raises(ValueError, match='changed'):
        cache.advance(hub, {}, 0)
    with pytest.raises(ValueError, match='every committed'):
        model(hub, {'directory': sources['directory']})


def test_padding_is_not_context_and_padding_queries_preserve_hub():
    model, hub, sources = fixture()
    torch.nn.init.normal_(model.output.weight, std=.1)
    valid = torch.tensor([[False, False, True, True, True, False, False]])
    expected = model(hub, sources, valid=valid)
    changed = {key: value.clone() for key, value in sources.items()}
    for value in changed.values():
        value[:, ~valid[0]] += 100
    assert torch.equal(model(hub, changed, valid=valid), expected)
    assert torch.equal(expected[:, ~valid[0]], hub[:, ~valid[0]])


def fusion_owner(rank, folder):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+str(Path(folder)/'meeting'),
        rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(911)
        model = CrossShardFusion(12, {'remote': 10}, rank=8, heads=2, max_context=32)
        torch.nn.init.normal_(model.output.weight, std=.1)
        hub, remote = torch.randn(1, 7, 12), torch.randn(1, 7, 10)
        target = torch.randn_like(hub)
        reference = copy.deepcopy(model)
        loss = (reference(hub, {'remote': remote})-target).square().mean()
        loss.backward()
        torch.optim.SGD(reference.parameters(), lr=.1).step()
        expected = copy.deepcopy(reference.state_dict())
        del reference
        wire = Wire(rank, 2)
        if rank == 0:
            # Source owner keeps only its projections; it has no hub parameters.
            key, value = model.keys['remote'], model.values['remote']
            del model
            projected = key(remote), value(remote)
            for tensor in projected:
                wire.send(tensor, 1)
            gradients = [wire.receive(1, (1, 7, 8), 'cpu') for _ in range(2)]
            torch.autograd.backward(projected, gradients)
            torch.optim.SGD([*key.parameters(), *value.parameters()], lr=.1).step()
            assert torch.equal(key.weight, expected['keys.remote.weight'])
            assert torch.equal(value.weight, expected['values.remote.weight'])
        else:
            # Receiver needs projections, never the sending owner's weights.
            del model.keys, model.values
            projected = tuple(wire.receive(0, (1, 7, 8), 'cpu').requires_grad_() for _ in range(2))
            result = model.receive(hub, {'remote': projected})
            (result-target).square().mean().backward()
            for tensor in projected:
                wire.send(tensor.grad, 0)
            torch.optim.SGD(model.parameters(), lr=.1).step()
            assert torch.equal(model.query.weight, expected['query.weight'])
            assert torch.equal(model.output.weight, expected['output.weight'])
        assert wire.sent_tensor_bytes == 2*(1*7*8*4+24)
        (Path(folder)/str(rank)).write_text('matched')
    finally:
        dist.destroy_process_group()


def test_two_owners_match_joint_gradient_update_without_exchanging_weights(tmp_path):
    mp.spawn(fusion_owner, args=(str(tmp_path),), nprocs=2, join=True)
    assert [(tmp_path/str(rank)).read_text() for rank in range(2)] == ['matched', 'matched']
