"""Trainable causal exchange between separately owned frozen model paths.

The receiver learns from expert hidden states, instead of requiring an LLM to
rewrite a request into perfect subquestions. Each sender projects only its own
activations. The receiver never needs sender model weights. This is an
experimental numerical layer, not an admitted native execution profile.

Inspired by model composition with cross-attention (CALM, arXiv:2401.02412)
and frozen-expert stitching (BTS, arXiv:2502.00075). The small projection,
identity initialization and transport layout here are our own design choices.
"""
import math
from collections.abc import Mapping

import torch
from torch import nn


class CrossShardFusion(nn.Module):
    """Causal low-rank cross-attention with an exactly preserved initial hub.

    Source order, dimensions, context bound and trained projection weights form
    part of the eventual execution commitment. Source IDs identify installed
    model paths; they are not per-request task labels. Every enabled source is
    evaluated. Sparse source selection needs a separate measured policy.
    """

    def __init__(self, hub_width, source_widths, *, rank=64, heads=4, max_context=4096):
        super().__init__()
        if (type(hub_width) is not int or not 1 <= hub_width <= 16384
                or not isinstance(source_widths, dict) or not 1 <= len(source_widths) <= 64
                or any(not isinstance(key, str) or not key or '.' in key or len(key) > 128
                       or type(width) is not int or not 1 <= width <= 16384
                       for key, width in source_widths.items())
                or type(rank) is not int or not 1 <= rank <= 1024
                or type(heads) is not int or not 1 <= heads <= 32 or rank % heads
                or type(max_context) is not int or not 1 <= max_context <= 16384):
            raise ValueError('Invalid bounded cross-shard fusion layout')
        if rank*(2*sum(source_widths.values())+2*hub_width) > 2**26:
            raise ValueError('Fusion projections exceed the parameter bound')
        self.hub_width = hub_width
        self.source_widths = dict(sorted(source_widths.items()))
        self.rank, self.heads, self.max_context = rank, heads, max_context
        self.query = nn.Linear(hub_width, rank, bias=False)
        self.keys = nn.ModuleDict({key: nn.Linear(width, rank, bias=False)
                                   for key, width in self.source_widths.items()})
        self.values = nn.ModuleDict({key: nn.Linear(width, rank, bias=False)
                                     for key, width in self.source_widths.items()})
        self.output = nn.Linear(rank, hub_width, bias=False)
        # Initial inference is bit-for-bit the input hub, including signed zero.
        # Training still has a nonzero derivative through output.weight.
        nn.init.zeros_(self.output.weight)

    def descriptor(self):
        return {'format': 'neuroshard-causal-fusion-v1', 'hub_width': self.hub_width,
                'sources': self.source_widths.copy(), 'rank': self.rank, 'heads': self.heads,
                'max_context': self.max_context, 'attention': 'causal-all-sources-float32-softmax',
                'initialization': 'zero-output', 'dropout': 0}

    def _hidden(self, value, width):
        if (not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[2] != width
                or not 1 <= value.shape[0] <= 64 or not 1 <= value.shape[1] <= self.max_context
                or value.numel() > 2**24
                or value.dtype != self.query.weight.dtype or value.device != self.query.weight.device
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid finite owned activation or numerical profile')

    def project(self, source, hidden):
        """Run at a source owner; only these bounded K/V tensors cross the wire."""
        if source not in self.source_widths:
            raise ValueError('Unknown committed fusion source')
        self._hidden(hidden, self.source_widths[source])
        return self.keys[source](hidden), self.values[source](hidden)

    def receive(self, hub, projected, *, positions=None, valid=None):
        """Run at the hub owner with checked projections from every source.

        `positions` are absolute token positions for hub queries. Projection
        entries cover positions zero through T-1 under the SAME tokenizer and
        request; transport must bind those identities. `valid` masks right or
        left padding. Every query must itself be at a valid position.
        """
        self._hidden(hub, self.hub_width)
        if not isinstance(projected, Mapping) or set(projected) != set(self.source_widths):
            raise ValueError('Require every committed fusion source exactly once')
        batch, queries, _ = hub.shape
        length = None
        keys, values = [], []
        for source in self.source_widths:
            pair = projected[source]
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError('Require source key and value projections')
            for tensor in pair:
                self._hidden(tensor, self.rank)
                if tensor.shape[0] != batch or (length is not None and tensor.shape[1] != length):
                    raise ValueError('Fusion sources differ in request batch or context length')
                length = tensor.shape[1]
            keys.append(pair[0])
            values.append(pair[1])
        if positions is None:
            if queries != length:
                raise ValueError('Incremental queries require explicit absolute positions')
            positions = torch.arange(length, device=hub.device)
        if (not isinstance(positions, torch.Tensor) or positions.dtype != torch.int64
                or positions.device != hub.device or positions.shape != (queries,)
                or bool((positions < 0).any()) or bool((positions >= length).any())
                or (queries > 1 and not bool((positions[1:] > positions[:-1]).all()))):
            raise ValueError('Invalid increasing absolute fusion query positions')
        if valid is None:
            valid = torch.ones((batch, length), dtype=torch.bool, device=hub.device)
        if (not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool
                or valid.shape != (batch, length) or valid.device != hub.device
                or not bool(valid.any(dim=1).all())):
            raise ValueError('Every fusion sequence must contain valid source context')
        if batch*self.heads*queries*length*len(keys) > 2**26:
            raise ValueError('Fusion attention exceeds its declared memory bound')
        query_valid = valid[:, positions]
        width = self.rank // self.heads
        query = self.query(hub).reshape(batch, queries, self.heads, width).transpose(1, 2)
        key = torch.cat(keys, dim=1).reshape(batch, length*len(keys), self.heads, width).transpose(1, 2)
        value = torch.cat(values, dim=1).reshape(batch, length*len(values), self.heads, width).transpose(1, 2)
        causal = torch.arange(length, device=hub.device)[None, :] <= positions[:, None]
        allowed = (valid[:, None, :] & causal[None, :, :]).repeat(1, 1, len(keys))
        # Left-padding queries may have no causal real token. Give only those
        # ignored queries a dummy key, then preserve their hub value below.
        allowed[:, :, 0] |= ~query_valid
        weights = (query.float() @ key.float().transpose(-1, -2)) / math.sqrt(width)
        weights = weights.masked_fill(~allowed[:, None], -torch.inf).softmax(dim=-1)
        mixed = (weights @ value.float()).transpose(1, 2).reshape(batch, queries, self.rank)
        if not bool(torch.isfinite(mixed).all()):
            raise ValueError('Nonfinite projected attention')
        residual = self.output(mixed.to(hub.dtype))
        # Avoid altering signed zeros at the exact initial (zero-output) state,
        # while keeping the normal differentiable graph during training.
        if not torch.is_grad_enabled() and not bool(self.output.weight.any()):
            return hub.clone()
        result = torch.where(query_valid[:, :, None], hub + residual, hub)
        if not bool(torch.isfinite(result).all()):
            raise ValueError('Nonfinite cross-shard fusion output')
        return result

    def forward(self, hub, experts, *, positions=None, valid=None):
        if not isinstance(experts, Mapping) or set(experts) != set(self.source_widths):
            raise ValueError('Require owned hidden states for every committed source')
        return self.receive(hub, {source: self.project(source, experts[source]) for source in self.source_widths},
                            positions=positions, valid=valid)


class FusionCache:
    """One request's projected source history at the receiving owner."""

    def __init__(self, fusion):
        if any(module.training for module in fusion.modules()):
            raise ValueError('Fusion caching requires evaluation mode')
        self.fusion = fusion
        self.versions = tuple(parameter._version for parameter in fusion.parameters())
        self.projected, self.length, self.failed = {}, 0, False

    @torch.no_grad()
    def advance(self, hub, projected, position):
        if self.failed:
            raise ValueError('Discard a failed fusion cache')
        self.failed = True
        model = self.fusion
        if (any(module.training for module in model.modules())
                or self.versions != tuple(parameter._version for parameter in model.parameters())):
            raise ValueError('Fusion weights or mode changed within a request')
        model._hidden(hub, model.hub_width)
        if not isinstance(projected, Mapping) or set(projected) != set(model.source_widths):
            raise ValueError('Invalid continuing fusion request')
        first = next(iter(projected.values()))
        if not isinstance(first, tuple) or len(first) != 2:
            raise ValueError('Require a source projection pair')
        model._hidden(first[0], model.rank)
        count = first[0].shape[1]
        if (type(position) is not int or position != self.length or hub.shape[0] != 1
                or not 1 <= hub.shape[1] <= count or (position and count != 1)
                or position+count > model.max_context):
            raise ValueError('Invalid continuing fusion request')
        updated = {}
        for source in model.source_widths:
            pair = projected[source]
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError('Require a source projection pair')
            for tensor in pair:
                model._hidden(tensor, model.rank)
                if tuple(tensor.shape) != (1, count, model.rank):
                    raise ValueError('Source projection does not continue the current request')
            updated[source] = (tuple(torch.cat((old, new), dim=1)
                for old, new in zip(self.projected[source], pair)) if position else
                tuple(tensor.clone() for tensor in pair))
        positions = torch.arange(position+count-hub.shape[1], position+count, device=hub.device)
        result = model.receive(hub, updated, positions=positions)
        self.projected, self.length, self.failed = updated, position+count, False
        return result

    def resident_bytes(self):
        return sum(t.numel()*t.element_size() for pair in self.projected.values() for t in pair)
