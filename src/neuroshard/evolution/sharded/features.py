"""Optimize an owned tail from immutable, previously verified prefix features.

The frozen prefix and reference must be computed and audited for the exact
padded microbatch before reuse. Tensor hashes alone do not prove that work.
This numerical kernel does not activate a native feature-certification profile.
"""

import math
import time

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from ..reference import autocast, learning_rate
from .guarded import correct_margin, objective, response_logits
from .model import batch_tensors


class FrozenHead(nn.Module):
    """Read-only replica of the parent's tied output matrix and final norm."""

    def __init__(self, config, embedding, norm, device='cpu'):
        super().__init__()
        if (not config.tie_word_embeddings or embedding.shape != (config.vocab_size, config.hidden_size)
                or norm.shape != (config.hidden_size,) or embedding.dtype != torch.float32
                or norm.dtype != torch.float32 or not bool(torch.isfinite(embedding).all())
                or not bool(torch.isfinite(norm).all())):
            raise ValueError('Invalid read-only tied output head')
        self.weight = nn.Parameter(torch.empty(embedding.shape, dtype=torch.float32, device=device), requires_grad=False)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(device).requires_grad_(False)
        with torch.no_grad():
            self.weight.copy_(embedding)
            self.norm.weight.copy_(norm)
        self.versions = tuple(parameter._version for parameter in self.parameters())

    def unchanged(self):
        return (all(not parameter.requires_grad and parameter.grad is None for parameter in self.parameters())
                and self.versions == tuple(parameter._version for parameter in self.parameters()))

    def logits(self, hidden):
        if not self.unchanged():
            raise ValueError('Read-only output head changed')
        return F.linear(self.norm(hidden), self.weight)


def validate_packet(packet, rows, config, device):
    ids, labels, mask, weights = batch_tensors(rows, device)
    if set(packet) != {'prefix', 'reference', 'ids', 'labels', 'mask', 'weights'}:
        raise ValueError('Incomplete frozen feature packet')
    for key, expected in (('ids', ids), ('labels', labels), ('mask', mask), ('weights', weights)):
        if (packet[key].dtype != expected.dtype or packet[key].device != expected.device
                or not torch.equal(packet[key], expected)):
            raise ValueError('Feature packet tokenization, labels, padding or weights changed')
    shape = (*ids.shape, config.hidden_size)
    for name in ('prefix', 'reference'):
        value = packet[name]
        if (value.shape != shape or value.dtype != torch.float32 or value.requires_grad
                or value.device != ids.device or value.numel() > 4 * 1024 * 4096
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid finite frozen representation')


def train_step(shard, head, optimizer, packets, records, recipe, index, microbatch,
               kl_strength=2., margin_strength=1., margin_min=.5, margin_max=2.):
    """Same objective and Adam update as the distributed incremental kernel.

    This version requires the complete trainable tail to have one owner. Its
    head replica is outside the unique trainable-parameter/optimizer inventory.
    """
    if shard.rank != len(shard.boundaries) - 2 or shard.rank == 0:
        raise ValueError('Cached learner must own the complete final tail')
    pairs = shard.named_owned_parameters()
    if (any(not parameter.requires_grad for _, parameter in pairs) or not head.unchanged()
            or sorted(id(p) for group in optimizer.param_groups for p in group['params'])
            != sorted(id(p) for _, p in pairs)):
        raise ValueError('Train exactly the owned tail and keep the head frozen')
    if (type(microbatch) is not int or microbatch <= 0 or type(index) is not int or not 0 <= index < recipe['steps']
            or len(packets) != math.ceil(len(records) / microbatch)
            or not all(math.isfinite(v) and v >= 0 for v in (kl_strength, margin_strength))
            or not 0 < margin_min <= margin_max):
        raise ValueError('Invalid feature training recipe or packet coverage')
    denominator = sum(row['targets'] * row.get('loss_weight', 1) for row in records)
    anchors = sum(row['targets'] for row in records if row.get('distill', False))
    if denominator <= 0 or ((kl_strength or margin_strength) and not anchors):
        raise ValueError('Feature training needs targets and declared retention anchors')
    began = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    rate = learning_rate(recipe, index)
    for group in optimizer.param_groups:
        group['lr'] = rate
    shard.train()
    ce_total = kl_total = margin_total = 0.
    for packet, offset in zip(packets, range(0, len(records), microbatch)):
        subset = records[offset:offset + microbatch]
        validate_packet(packet, subset, shard.config, shard.device_name)
        with autocast(shard.device_name):
            hidden = shard(packet['prefix'], packet['mask'])
            logits, targets, active = response_logits(head, hidden, packet['labels'])
            with torch.no_grad():
                teacher, _, _ = response_logits(head, packet['reference'], packet['labels'])
            weights = packet['weights'][:, None].expand_as(active)[active]
            anchor_rows = torch.tensor([row.get('distill', False) for row in subset],
                                       dtype=torch.bool, device=shard.device_name)
            anchor_mask = anchor_rows[:, None].expand_as(active)[active]
            loss, ce, kl = objective(logits, targets, weights, teacher, anchor_mask,
                                    denominator, max(1, anchors), kl_strength)
            margin = (correct_margin(logits, targets, teacher, anchor_mask, max(1, anchors), margin_min, margin_max)
                      if margin_strength else logits.new_zeros(()))
            loss = loss + margin_strength * margin
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite feature objective')
        ce_total += float(ce)
        kl_total += float(kl)
        margin_total += float(margin.detach())
        loss.backward()
        del hidden, logits, teacher, loss, ce, kl, margin
    if any(parameter.grad is None for _, parameter in pairs):
        raise ValueError('Missing owned-tail gradient')
    norm = math.sqrt(math.fsum(float(torch.linalg.vector_norm(parameter.grad, dtype=torch.float64).square())
                              for _, parameter in pairs))
    if not math.isfinite(norm):
        raise ValueError('Nonfinite owned-tail gradient norm')
    scale = min(1., recipe['clip_norm'] / (norm + 1e-6))
    for _, parameter in pairs:
        parameter.grad.mul_(scale)
    optimizer.step()
    if not head.unchanged():
        raise ValueError('Output head acquired a gradient or changed')
    if shard.device_name == 'cuda':
        torch.cuda.synchronize()
    return {'step': index + 1, 'loss': ce_total + kl_strength * kl_total + margin_strength * margin_total,
        'response_loss': ce_total, 'reference_kl': kl_total, 'reference_margin': margin_total,
        'gradient_norm': norm, 'learning_rate': rate, 'weighted_targets': denominator,
        'anchor_targets': anchors, 'seconds': time.monotonic() - began}
