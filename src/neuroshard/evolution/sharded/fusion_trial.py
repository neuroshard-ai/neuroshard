"""Shared numerical operations for the prospectively frozen fusion trial."""
import copy

import torch
from torch.nn import functional as F

from ..reference import autocast
from .fused_graph import commitment
from .fusion_training import Trainer


@torch.no_grad()
def synchronize(net, model):
    """Publish only the small trained connection, then agree on its bytes."""
    wire = net.all_owners
    expected = wire.exchange(commitment(model) if net.rank == 0 else None)[0]
    device = next(model.parameters()).device
    for parameter in model.parameters():
        shape = (1, 1, parameter.numel())
        if net.rank == 0:
            for owner in range(1, net.world_size):
                wire.send(parameter.reshape(shape), owner)
        else:
            parameter.copy_(wire.receive(0, shape, device).reshape(parameter.shape))
    if wire.exchange(commitment(model)) != [expected]*net.world_size:
        raise ValueError('Owners disagree on the trained fusion parameters')
    model.eval()
    return expected


@torch.no_grad()
def response_losses(models, head, rows, bank, home, recipe):
    # Reuse the byte, token, padding and shape checks from the training reader.
    reading_recipe = {**copy.deepcopy(recipe), 'steps': 1, 'warmup_steps': 0, 'schedule': [0]}
    reader = Trainer(models['fusion'], head, rows, bank, home, reading_recipe)
    for model in models.values():
        model.eval()
    result = {}
    device = next(head.parameters()).device
    for batch in range(len(bank['batches'])):
        data, selected = reader.batch(batch)
        for start in range(0, len(selected), recipe['microbatch']):
            stop = start+recipe['microbatch']
            values = {key: value[start:stop].to(device) for key, value in data.items()}
            labels = values['labels'][:, 1:]
            observations = {}
            for arm in ('hub', 'fusion', 'ablation'):
                hidden = values['hub']
                if arm != 'hub':
                    model = models[arm]
                    sources = {name: values['parent' if arm == 'ablation' else name]
                               for name in model.source_widths}
                    hidden = model(hidden, sources, valid=values['valid'])
                with autocast(head.device_name):
                    logits = head.logits(hidden).float()[:, :-1]
                losses = F.cross_entropy(logits.transpose(1, 2), labels, reduction='none')
                observations[arm] = (losses.sum(dim=1)/(labels != -100).sum(dim=1)).cpu().tolist()
            for index, row in enumerate(selected[start:stop]):
                result[row['id']] = {arm: values[index] for arm, values in observations.items()}
    return result
