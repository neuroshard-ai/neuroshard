"""Train appended transformer blocks over a frozen, distributed parent model.

The parent forward pass supplies both the input to the added blocks and the
reference logits. Only the added blocks participate in backward propagation
and hold optimizer state. Freezing parent tensors preserves their bytes, not
the grown model's answers: held-out learning and retention checks remain needed.

This is a numerical research kernel, not a native checkpoint/settlement profile.
"""
import copy
import math
import time

import torch
from torch.utils.checkpoint import checkpoint
from transformers.masking_utils import create_causal_mask

from ..reference import autocast, learning_rate
from .guarded import correct_margin, objective, response_logits
from .model import batch_tensors, owner
from .portable import shapes


def trainable_names(config, frozen_layers):
    if type(frozen_layers) is not int or not 0 < frozen_layers < config.num_hidden_layers:
        raise ValueError('Train a nonempty tail after a nonempty frozen prefix')
    return {name for name in shapes(config)
            if name.startswith('model.layers.') and int(name.split('.')[2]) >= frozen_layers}


def configure(shard, frozen_layers, recipe):
    names = trainable_names(shard.config, frozen_layers)
    if owner(f'model.layers.{frozen_layers}.input_layernorm.weight', shard.boundaries) == 0:
        raise ValueError('The frozen output-head owner must precede added-block owners')
    groups = [[], []]
    for name, parameter in shard.named_owned_parameters():
        parameter.requires_grad_(name in names)
        parameter.grad = None
        if parameter.requires_grad:
            groups[parameter.ndim < 2].append(parameter)
    if not any(groups):
        return None
    return torch.optim.AdamW([
        {'params': groups[0], 'weight_decay': recipe['weight_decay']},
        {'params': groups[1], 'weight_decay': 0.},
    ], lr=recipe['learning_rate'], betas=(.9, .95), eps=1e-8, foreach=False)


def reference_tail(shard, first_trainable_layer):
    """Frozen original tail for a control that fine-tunes existing final blocks.

    The control and grown candidate can each train the same number of blocks.
    Both compute the parent once and a trainable tail once: the control copies
    only its small original tail, while growth reuses the complete parent output.
    This control currently requires the trainable tail to have one owner.
    """
    first = owner(f'model.layers.{first_trainable_layer}.input_layernorm.weight', shard.boundaries)
    if first != len(shard.boundaries) - 2:
        raise ValueError('The comparison reference tail must fit its last owner')
    if shard.rank != first:
        return None
    return torch.nn.ModuleDict({name: copy.deepcopy(layer).eval().requires_grad_(False)
                                for name, layer in shard.layers.items()
                                if int(name) >= first_trainable_layer})


def forward(shard, wire, ids, mask, frozen_layers, reference_layers=None):
    """Send the shared parent representation once, then the adapted output."""
    first = owner(f'model.layers.{frozen_layers}.input_layernorm.weight', shard.boundaries)
    shape = (*ids.shape, shard.config.hidden_size)
    incoming = None if wire.rank == 0 else wire.receive(wire.rank - 1, shape, shard.device_name)
    if incoming is not None and wire.rank > first:
        incoming.requires_grad_(True)
    with autocast(shard.device_name):
        hidden = shard.embedding(ids) if wire.rank == 0 else incoming
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        position_ids = positions.unsqueeze(0)
        causal = create_causal_mask(shard.config, hidden, mask, positions, None, position_ids)
        rotary = shard.rotary(hidden, position_ids)
        for number, layer in shard.layers.items():
            if int(number) == frozen_layers:
                if hidden.requires_grad:
                    raise ValueError('Frozen parent unexpectedly retained a backward graph')
                reference = hidden
                if reference_layers is not None:
                    with torch.no_grad():
                        for reference_layer in reference_layers.values():
                            reference = reference_layer(reference, attention_mask=causal,
                                position_ids=position_ids, cache_position=positions,
                                position_embeddings=rotary, use_cache=False)
                wire.send(reference, 0)
                del reference
            def apply(value, layer=layer):
                return layer(value, attention_mask=causal, position_ids=position_ids,
                             cache_position=positions, position_embeddings=rotary, use_cache=False)
            if int(number) < frozen_layers:
                with torch.no_grad():
                    hidden = apply(hidden)
            else:
                hidden = checkpoint(apply, hidden, use_reentrant=False) if shard.recompute else apply(hidden)
    wire.send(hidden, wire.rank + 1 if wire.rank + 1 < wire.world else 0)
    reference = wire.receive(first, shape, shard.device_name) if wire.rank == 0 else None
    final = wire.receive(wire.world - 1, shape, shard.device_name) if wire.rank == 0 else None
    return incoming, hidden, reference, final


def train_step(shard, optimizer, wire, records, recipe, index, microbatch, frozen_layers,
               kl_strength=2., margin_strength=0., margin_min=.5, margin_max=2., reference_layers=None):
    expected = trainable_names(shard.config, frozen_layers)
    pairs = shard.named_owned_parameters()
    if any(parameter.requires_grad != (name in expected) for name, parameter in pairs):
        raise ValueError('Only the declared tail blocks may be trained')
    local_trainable = [(name, p) for name, p in pairs if name in expected]
    if (optimizer is None) != (not local_trainable):
        raise ValueError('Only added-block owners hold an optimizer')
    if optimizer is not None:
        actual = [id(p) for group in optimizer.param_groups for p in group['params']]
        if sorted(actual) != sorted(id(p) for _, p in local_trainable):
            raise ValueError('Optimizer must cover exactly the added local blocks')
    if (not isinstance(microbatch, int) or isinstance(microbatch, bool) or microbatch <= 0
            or not 0 <= index < recipe['steps']
            or not all(math.isfinite(v) and v >= 0 for v in (kl_strength, margin_strength))
            or not 0 < margin_min <= margin_max):
        raise ValueError('Invalid incremental training recipe')
    denominator = sum(r['targets'] * r.get('loss_weight', 1) for r in records)
    anchors = sum(r['targets'] for r in records if r.get('distill', False))
    if denominator <= 0 or ((kl_strength or margin_strength) and not anchors):
        raise ValueError('Training needs targets and declared retention anchors')
    started = time.monotonic()
    rate = learning_rate(recipe, index)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group['lr'] = rate
    shard.train()
    total_ce = total_kl = total_margin = 0.
    first = owner(f'model.layers.{frozen_layers}.input_layernorm.weight', shard.boundaries)
    if reference_layers is not None:
        expected_layers = {str(n) for n in range(frozen_layers, shard.config.num_hidden_layers)}
        if (wire.rank != first or first != wire.world - 1 or set(reference_layers) != expected_layers
                or any(p.requires_grad for p in reference_layers.parameters())):
            raise ValueError('Invalid frozen reference tail')
    for offset in range(0, len(records), microbatch):
        batch = records[offset:offset + microbatch]
        ids, labels, mask, weights = batch_tensors(batch, shard.device_name)
        incoming, outgoing, reference, final = forward(shard, wire, ids, mask, frozen_layers, reference_layers)
        if wire.rank == 0:
            final.requires_grad_(True)
            with autocast(shard.device_name):
                with torch.no_grad():
                    reference_logits, _, _ = response_logits(shard, reference, labels)
                logits, targets, active = response_logits(shard, final, labels)
                weights = weights[:, None].expand_as(active)[active]
                anchor_rows = torch.tensor([r.get('distill', False) for r in batch],
                                           dtype=torch.bool, device=shard.device_name)
                anchor_mask = anchor_rows[:, None].expand_as(active)[active]
                loss, ce, kl = objective(logits, targets, weights, reference_logits, anchor_mask,
                                         denominator, max(1, anchors), kl_strength)
                margin = correct_margin(logits, targets, reference_logits, anchor_mask,
                                        max(1, anchors), margin_min, margin_max) if margin_strength else logits.new_zeros(())
                loss = loss + margin_strength * margin
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite incremental objective')
            total_ce += float(ce)
            total_kl += float(kl)
            total_margin += float(margin.detach())
            loss.backward()
            wire.send(final.grad, wire.world - 1)
            del loss, ce, kl, margin, logits, reference_logits, final, reference
        if wire.rank >= first:
            shape = (*ids.shape, shard.config.hidden_size)
            gradient = wire.receive(wire.rank + 1 if wire.rank + 1 < wire.world else 0,
                                    shape, shard.device_name)
            outgoing.backward(gradient)
            if wire.rank > first:
                wire.send(incoming.grad, wire.rank - 1)
            del gradient
        del incoming, outgoing
    if any(p.grad is not None for name, p in pairs if name not in expected):
        raise ValueError('Frozen parent acquired gradients')
    if any(p.grad is None for _, p in local_trainable):
        raise ValueError('An added parameter is missing its gradient')
    local = {name: float(torch.linalg.vector_norm(p.grad, dtype=torch.float64).square())
             for name, p in local_trainable}
    norms = {}
    for part in wire.exchange(local):
        if set(norms) & set(part):
            raise ValueError('Duplicate added-parameter ownership')
        norms.update(part)
    if set(norms) != expected or any(not math.isfinite(v) or v < 0 for v in norms.values()):
        raise ValueError('Incomplete finite added-parameter gradient inventory')
    norm = math.sqrt(math.fsum(norms[name] for name in sorted(norms)))
    scale = min(1., recipe['clip_norm'] / (norm + 1e-6))
    for _, parameter in local_trainable:
        parameter.grad.mul_(scale)
    if optimizer is not None:
        optimizer.step()
    ce, kl, margin = wire.sum(total_ce), wire.sum(total_kl), wire.sum(total_margin)
    if shard.device_name == 'cuda':
        torch.cuda.synchronize()
    return {'step': index + 1, 'loss': ce + kl_strength * kl + margin_strength * margin,
            'response_loss': ce, 'reference_kl': kl, 'reference_margin': margin,
            'gradient_norm': norm, 'learning_rate': rate, 'weighted_targets': denominator,
            'anchor_targets': anchors, 'seconds': time.monotonic() - started}
