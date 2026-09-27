"""Episode-level selection between the parent and one trained arm.

Labels come from estimated success rates on integration episodes (one greedy and
several sampled outcomes per model), never from evaluation goals. The gate learns
only where the arm is better or worse, in proportion to the difference; ties
carry no weight. The degenerate rules are fixed before any outcome is seen.
"""

import torch

from neuroshard.evolution.modular_reference_execution import identity


def success_rate(outcomes):
    if not outcomes or any(type(value) is not bool for value in outcomes):
        raise ValueError('success rate needs boolean episode outcomes')
    return sum(outcomes) / len(outcomes)


def targets(parent, arm):
    """Per-episode target (1 selects the arm) and weight |difference|."""
    if set(parent) != set(arm):
        raise ValueError('parent and arm outcomes cover different episodes')
    rows = {}
    for key in sorted(parent):
        difference = success_rate(arm[key]) - success_rate(parent[key])
        rows[key] = (1.0 if difference > 0 else 0.0, abs(difference))
    return rows


def fit(features, rows, recipe):
    """Declared logistic gate; constant rules when no weighted example favours one side."""
    keys = sorted(rows)
    if set(features) != set(keys):
        raise ValueError('features and targets cover different episodes')
    favour_arm = sum(weight for key in keys for target, weight in [rows[key]] if target == 1.0)
    favour_parent = sum(weight for key in keys for target, weight in [rows[key]] if target == 0.0)
    counts = {'arm_better': sum(rows[k][0] == 1.0 for k in keys),
              'parent_better': sum(rows[k][0] == 0.0 and rows[k][1] > 0 for k in keys),
              'ties': sum(rows[k][1] == 0 for k in keys)}
    if favour_arm == 0:
        return {'rule': 'constant-parent', 'counts': counts}
    if favour_parent == 0:
        return {'rule': 'constant-arm', 'counts': counts}
    generator = torch.Generator().manual_seed(recipe['seed'])
    x = torch.nn.functional.normalize(torch.tensor([features[k] for k in keys], dtype=torch.float32),
                                      dim=1, eps=recipe['epsilon'])
    y = torch.tensor([rows[k][0] for k in keys])
    w = torch.tensor([rows[k][1] for k in keys])
    weight = torch.zeros(x.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.AdamW([weight, bias], lr=recipe['learning_rate'], weight_decay=recipe['weight_decay'])
    for _ in range(recipe['updates']):
        batch = torch.randperm(len(keys), generator=generator)[:recipe['batch']]
        logits = x[batch] @ weight + bias
        losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[batch], reduction='none')
        loss = (losses * w[batch]).sum() / w[batch].sum().clamp_min(1e-12)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    gate = {'rule': 'logistic', 'weight': weight.detach().tolist(), 'bias': float(bias.detach()),
            'epsilon': recipe['epsilon'], 'threshold': recipe['threshold'], 'counts': counts}
    gate['sha256'] = identity(gate)
    return gate


def choose(gate, feature):
    """True selects the trained arm for the whole episode; an exact tie selects the parent."""
    if gate['rule'] == 'constant-parent':
        return False
    if gate['rule'] == 'constant-arm':
        return True
    x = torch.nn.functional.normalize(torch.tensor([feature], dtype=torch.float32), dim=1, eps=gate['epsilon'])[0]
    probability = torch.sigmoid(x @ torch.tensor(gate['weight']) + gate['bias'])
    return bool(probability > gate['threshold'])
