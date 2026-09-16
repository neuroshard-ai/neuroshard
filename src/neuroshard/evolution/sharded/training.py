"""Synchronous microbatched backpropagation with one optimizer partition per rank."""
import math
import time
import torch

from ..reference import autocast, learning_rate
from .model import batch_tensors, weighted_loss


def forward(shard, wire, ids, mask):
    rank, world, device = wire.rank, wire.world, shard.device_name
    shape = (*ids.shape, shard.config.hidden_size)
    incoming = None if rank == 0 else wire.receive(rank-1, shape, device)
    if incoming is not None and torch.is_grad_enabled():
        incoming.requires_grad_(True)
    with autocast(device):
        outgoing = shard(ids if rank == 0 else incoming, mask)
    wire.send(outgoing, rank+1 if rank+1 < world else 0)
    final = wire.receive(world-1, shape, device) if rank == 0 else None
    return incoming, outgoing, final


def train_step(shard, optimizer, wire, records, recipe, index, microbatch, after_forward=None):
    started = time.monotonic()
    rank, world, device = wire.rank, wire.world, shard.device_name
    denominator = sum(r['targets'] * r.get('loss_weight', 1) for r in records)
    if denominator <= 0:
        raise ValueError('Training batch requires response targets')
    optimizer.zero_grad(set_to_none=True)
    shard.train()
    rate = learning_rate(recipe, index)
    for group in optimizer.param_groups:
        group['lr'] = rate
    total = 0.
    for offset in range(0, len(records), microbatch):
        ids, labels, mask, weights = batch_tensors(records[offset:offset+microbatch], device)
        incoming, outgoing, final = forward(shard, wire, ids, mask)
        if after_forward is not None:
            after_forward(index, offset)
        shape = (*ids.shape, shard.config.hidden_size)
        if rank == 0:
            final.requires_grad_(True)
            with autocast(device):
                loss = weighted_loss(shard.logits(final), labels, weights)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite response loss')
            total += float(loss.detach())
            (loss / denominator).backward()
            wire.send(final.grad, world-1)
            del final, loss
        gradient = wire.receive(rank+1 if rank+1 < world else 0, shape, device)
        outgoing.backward(gradient)
        if rank > 0:
            wire.send(incoming.grad, rank-1)
        del incoming, outgoing, gradient
    pairs = shard.named_owned_parameters()
    if any(p.grad is None for _, p in pairs):
        raise ValueError('Every owned trainable tensor must receive a gradient')
    squared = math.fsum(float(torch.linalg.vector_norm(p.grad, dtype=torch.float64).square()) for _, p in pairs)
    norm = math.sqrt(wire.sum(squared))
    if not math.isfinite(norm):
        raise ValueError('Nonfinite global gradient norm')
    scale = min(1., recipe['clip_norm'] / (norm + 1e-6))
    for _, p in pairs:
        p.grad.mul_(scale)
    optimizer.step()
    loss = wire.sum(total) / denominator
    if device == 'cuda':
        torch.cuda.synchronize()
    return {'step': index+1, 'loss': loss, 'gradient_norm': norm, 'learning_rate': rate,
            'weighted_targets': denominator, 'seconds': time.monotonic()-started}


@torch.no_grad()
def score(shard, wire, records, microbatch=1):
    shard.eval()
    result = []
    for row in records:
        ids, labels, mask, weights = batch_tensors([row], shard.device_name)
        _, _, final = forward(shard, wire, ids, mask)
        loss = 0.
        if wire.rank == 0:
            with autocast(shard.device_name):
                loss = float(weighted_loss(shard.logits(final), labels, torch.ones_like(weights)))
        result.append({'id': row['id'], 'targets': row['targets'], 'loss': wire.sum(loss)/row['targets']})
    return result


@torch.no_grad()
def generate(shard, wire, token_ids, max_tokens, eos_id):
    shard.eval()
    ids, output = list(token_ids), []
    if not ids or len(ids)+max_tokens > shard.config.max_position_embeddings:
        raise ValueError('Generation exceeds context; no silent truncation')
    for _ in range(max_tokens):
        tokens = torch.tensor([ids], dtype=torch.long, device=shard.device_name)
        mask = torch.ones_like(tokens)
        _, _, final = forward(shard, wire, tokens, mask)
        token = None
        if wire.rank == 0:
            with autocast(shard.device_name):
                token = int(shard.logits(final[:, -1:]).float().argmax(-1)[0, 0])
        token = wire.exchange({'token': token})[0]['token']
        if type(token) is not int or not 0 <= token < shard.config.vocab_size:
            raise ValueError('Invalid generated token')
        output.append(token)
        ids.append(token)
        if token == eos_id:
            break
    return output
