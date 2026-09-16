"""A causal learned mixture in the models' common token vocabulary.

Preserve native output distributions instead of compressing expert knowledge
through a low-rank hidden-to-hidden map. The gate sees only the preserved hub's
causal hidden state. Source identities name installed paths, never answer labels.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ProbabilityMixture(nn.Module):
    def __init__(self, hub_width, source_widths, *, rank=128, heads=1, max_context=1024):
        super().__init__()
        if (type(hub_width) is not int or not 1 <= hub_width <= 16384
                or not isinstance(source_widths, dict) or not 1 <= len(source_widths) <= 64
                or any(not isinstance(name, str) or not name or len(name) > 128 or '.' in name
                       or type(width) is not int or width != hub_width
                       for name, width in source_widths.items())
                or type(rank) is not int or not 1 <= rank <= 1024 or type(heads) is not int or heads != 1
                or type(max_context) is not int or not 1 <= max_context <= 16384):
            raise ValueError('Invalid bounded probability mixture')
        self.hub_width, self.source_widths = hub_width, dict(sorted(source_widths.items()))
        self.rank, self.max_context = rank, max_context
        self.query = nn.Linear(hub_width, rank)
        self.output = nn.Linear(rank, len(source_widths))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def descriptor(self):
        return {'format': 'neuroshard-probability-mixture-v1', 'hub_width': self.hub_width,
                'sources': self.source_widths, 'rank': self.rank, 'max_context': self.max_context,
                'gate': 'expm1(clamp(linear(silu(linear(layer_norm(hub)))),0,10))',
                'layer_norm_epsilon': 1e-5, 'source_probability_floor': 1e-12,
                'inactive_inference': 'exact-hub-log-softmax', 'initialization': 'zero-output'}

    def log_probs(self, hub, logits):
        if (hub.ndim != 3 or hub.shape[-1] != self.hub_width or not 1 <= hub.shape[0] <= 64
                or not 1 <= hub.shape[1] <= self.max_context or hub.numel() > 2**24
                or hub.dtype != torch.float32 or hub.device != self.query.weight.device
                or not bool(torch.isfinite(hub).all()) or set(logits) != {'hub', *self.source_widths}):
            raise ValueError('Invalid causal mixture activations or source inventory')
        shape = logits['hub'].shape
        if (len(shape) != 3 or shape[:2] != hub.shape[:2] or not 1 <= shape[-1] <= 262144
                or math.prod(shape)*len(logits) > 2**28
                or any(value.shape != shape or value.dtype != torch.float32 or value.device != hub.device
                       or not bool(torch.isfinite(value).all()) for value in logits.values())):
            raise ValueError('Native heads must use the identical bounded token vocabulary')
        hidden = F.silu(self.query(F.layer_norm(hub, (self.hub_width,), eps=1e-5)))
        # clamp has derivative one at zero in this pinned numerical profile.
        # Thus a newly added, initially inactive connection can start learning.
        weights = self.output(hidden).clamp(0, 10).expm1()
        if not bool(torch.isfinite(weights).all()):
            raise ValueError('Nonfinite probability gate')
        probability = lambda value: value.softmax(dim=-1).clamp_min(1e-12)
        base = probability(logits['hub'])
        combined = base/base.sum(dim=-1, keepdim=True)
        for index, source in enumerate(self.source_widths):
            value = probability(logits[source])
            combined = combined+weights[:, :, index:index+1]*(value/value.sum(dim=-1, keepdim=True))
        result = (combined/(1+weights.sum(dim=-1, keepdim=True))).log()
        if not torch.is_grad_enabled():
            # Exact fallback avoids even smoothing an inactive source's hub.
            result = torch.where((weights.sum(dim=-1) == 0)[:, :, None],
                                 logits['hub'].log_softmax(dim=-1), result)
        if not bool(torch.isfinite(result).all()):
            raise ValueError('Nonfinite mixed token distribution')
        return result
