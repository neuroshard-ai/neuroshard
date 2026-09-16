import torch

from neuroshard.evolution.sharded.mixture import ProbabilityMixture


def test_inactive_connection_is_exact_hub_and_can_learn_to_use_a_source():
    torch.manual_seed(91)
    model = ProbabilityMixture(12, {'expert': 12}, rank=8, max_context=8)
    hub = torch.randn(2, 4, 12)
    logits = {'hub': torch.zeros(2, 4, 7), 'expert': torch.zeros(2, 4, 7)}
    logits['hub'][:, :, 0] = 4
    logits['expert'][:, :, 3] = 8
    with torch.no_grad():
        assert torch.equal(model.log_probs(hub, logits), logits['hub'].log_softmax(dim=-1))
    loss = -model.log_probs(hub, logits)[:, :, 3].mean()
    loss.backward()
    assert torch.isfinite(model.output.bias.grad).all()
    assert float(model.output.bias.grad[0]) < 0
    with torch.no_grad():
        model.output.bias.fill_(8)
        values = model.log_probs(hub, logits)
    assert bool((values.argmax(dim=-1) == 3).all())
    torch.testing.assert_close(values.exp().sum(dim=-1), torch.ones(2, 4), rtol=1e-6, atol=1e-6)


def test_future_states_cannot_change_earlier_mixture_outputs():
    torch.manual_seed(5)
    model = ProbabilityMixture(12, {'expert': 12}, rank=8, max_context=8)
    with torch.no_grad():
        model.output.weight.normal_(std=.1)
        model.output.bias.fill_(1)
    hub = torch.randn(1, 6, 12, requires_grad=True)
    logits = {name: torch.randn(1, 6, 7) for name in ('hub', 'expert')}
    before = model.log_probs(hub, logits)
    other = hub.detach().clone()
    other[:, 3:] += 100
    changed = {name: value.clone() for name, value in logits.items()}
    for value in changed.values():
        value[:, 3:] *= -50
    assert torch.equal(before[:, :3], model.log_probs(other, changed)[:, :3])
    before[:, :3, 0].sum().backward()
    assert not bool(hub.grad[:, 3:].any())
