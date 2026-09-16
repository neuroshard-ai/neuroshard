"""Response learning with a frozen, equally sharded reference distribution.

Distillation is applied only to declared replay examples. It is a regularizer,
not a guarantee of retention; independent quality gates remain mandatory.
"""
import math
import time

import torch
import torch.nn.functional as F

from ..reference import autocast, learning_rate
from .model import batch_tensors
from .training import forward
from .portable import shapes


def response_logits(shard, hidden, labels):
    active = labels[:, 1:] != -100
    # Prompt/padding positions never contribute, and need no vocabulary logits.
    selected = hidden[:, :-1][active]
    return shard.logits(selected), labels[:, 1:][active], active


def objective(logits, targets, weights, teacher_logits=None, anchor_mask=None,
              ce_denominator=1., kl_denominator=1., strength=0.):
    ce = (F.cross_entropy(logits.float(), targets, reduction='none')*weights).sum()/ce_denominator
    kl = logits.new_zeros((), dtype=torch.float32)
    if teacher_logits is not None and bool(anchor_mask.any()):
        reference = F.softmax(teacher_logits[anchor_mask].float(), dim=-1)
        prediction = F.log_softmax(logits[anchor_mask].float(), dim=-1)
        kl = F.kl_div(prediction, reference, reduction='sum')/kl_denominator
    return ce+strength*kl, ce.detach(), kl.detach()


def correct_margin(logits, targets, teacher_logits, anchor_mask, denominator, minimum, maximum):
    """Protect known-correct reference token choices, with a bounded target gap.

    A positive reference margin identifies agreement with the supervised label;
    an incorrect teacher winner receives no margin protection. This is a
    differentiable penalty, not a guarantee about unseen greedy answers.
    """
    reference = teacher_logits.detach().float()
    index = targets[:, None]
    teacher_target = reference.gather(1, index).squeeze(1)
    teacher_other = reference.scatter(1, index, float('-inf')).max(dim=1).values
    gap = teacher_target - teacher_other
    protected = anchor_mask & (gap > 0)
    target = gap.clamp(min=minimum, max=maximum)
    prediction = logits.float()
    student_target = prediction.gather(1, index).squeeze(1)
    student_other = prediction.scatter(1, index, float('-inf')).max(dim=1).values
    return (F.relu(target - (student_target - student_other)) * protected).sum() / denominator


def train_step(shard, teacher, optimizer, wire, records, recipe, index, microbatch,
               kl_strength=1., margin_strength=0., margin_min=0.5, margin_max=2.0):
    if not math.isfinite(kl_strength) or kl_strength < 0:
        raise ValueError('Invalid retention regularizer')
    if (not all(math.isfinite(x) for x in (margin_strength, margin_min, margin_max))
            or margin_strength < 0 or not 0 < margin_min <= margin_max):
        raise ValueError('Invalid correct-token margin regularizer')
    started = time.monotonic()
    denominator = sum(r['targets']*r.get('loss_weight', 1) for r in records)
    anchors = sum(r['targets'] for r in records if r.get('distill', False))
    if denominator <= 0 or ((kl_strength or margin_strength) and anchors == 0):
        raise ValueError('Each guarded batch requires targets and declared replay anchors')
    shard.train()
    teacher.eval()
    if any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('The reference partition must be frozen')
    optimizer.zero_grad(set_to_none=True)
    rate = learning_rate(recipe, index)
    for group in optimizer.param_groups:
        group['lr'] = rate
    total_ce, total_kl, total_margin = 0., 0., 0.
    for offset in range(0, len(records), microbatch):
        batch = records[offset:offset+microbatch]
        ids, labels, mask, row_weights = batch_tensors(batch, shard.device_name)
        teacher_logits = None
        if any(r.get('distill', False) for r in batch) and (kl_strength or margin_strength):
            with torch.no_grad(), autocast(shard.device_name):
                _, _, final = forward(teacher, wire, ids, mask)
                if wire.rank == 0:
                    teacher_logits, _, _ = response_logits(teacher, final, labels)
                del final
        incoming, outgoing, final = forward(shard, wire, ids, mask)
        shape = (*ids.shape, shard.config.hidden_size)
        if wire.rank == 0:
            final.requires_grad_(True)
            with autocast(shard.device_name):
                logits, targets, active = response_logits(shard, final, labels)
                weights = row_weights[:, None].expand_as(labels[:, 1:])[active]
                rows = torch.tensor([r.get('distill', False) for r in batch], dtype=torch.bool,
                                    device=shard.device_name)
                anchor_mask = rows[:, None].expand_as(active)[active]
                loss, ce, kl = objective(logits, targets, weights, teacher_logits, anchor_mask,
                                          denominator, max(1, anchors), kl_strength)
                if margin_strength and teacher_logits is not None:
                    margin = correct_margin(logits, targets, teacher_logits, anchor_mask,
                                            max(1, anchors), margin_min, margin_max)
                    loss = loss + margin_strength * margin
                    total_margin += float(margin.detach())
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite guarded objective')
            total_ce += float(ce)
            total_kl += float(kl)
            loss.backward()
            wire.send(final.grad, wire.world-1)
            del loss, ce, kl, logits, final, teacher_logits
        gradient = wire.receive(wire.rank+1 if wire.rank+1 < wire.world else 0, shape, shard.device_name)
        outgoing.backward(gradient)
        if wire.rank > 0:
            wire.send(incoming.grad, wire.rank-1)
        del incoming, outgoing, gradient
    pairs = shard.named_owned_parameters()
    if any(p.grad is None for _, p in pairs):
        raise ValueError('Every trainable tensor needs a gradient')
    # Canonical per-parameter summation keeps clipping independent of shard layout.
    local = {name: float(torch.linalg.vector_norm(p.grad, dtype=torch.float64).square())
             for name, p in pairs}
    norms = {}
    for part in wire.exchange(local):
        if set(norms) & set(part):
            raise ValueError('Duplicate gradient ownership')
        norms.update(part)
    if set(norms) != set(shapes(shard.config)) or any(not math.isfinite(v) or v < 0 for v in norms.values()):
        raise ValueError('Invalid complete gradient inventory')
    norm = math.sqrt(math.fsum(norms[name] for name in sorted(norms)))
    if not math.isfinite(norm):
        raise ValueError('Nonfinite gradient norm')
    scale = min(1., recipe['clip_norm']/(norm+1e-6))
    for _, p in pairs:
        p.grad.mul_(scale)
    optimizer.step()
    ce, kl = wire.sum(total_ce), wire.sum(total_kl)
    margin = wire.sum(total_margin) if margin_strength else 0.
    if shard.device_name == 'cuda':
        torch.cuda.synchronize()
    result = {'step': index+1, 'loss': ce+kl_strength*kl, 'response_loss': ce, 'reference_kl': kl,
            'gradient_norm': norm, 'learning_rate': rate, 'weighted_targets': denominator,
            'anchor_targets': anchors, 'seconds': time.monotonic()-started}
    if margin_strength:
        result.update(loss=result['loss'] + margin_strength * margin, reference_margin=margin)
    return result
