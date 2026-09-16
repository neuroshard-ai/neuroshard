"""One autoregressive stream using causal activations from owned model paths.

No generated subquestions, answer lookup, task labels or specialist text parsing
occur in this executor. Every token runs the two frozen backbones and all expert
tails. The common trained backbone prefix is evaluated once and shared by tails.
Only projected expert activations reach the fusion receiver. This research path
requires measured learning and a separate native admission before public use.
"""
import hashlib
import time

import torch

from ..reference import autocast
from ..reference_data import identity
from .cached_inference import CachedPartition
from .fusion import FusionCache


def commitment(fusion):
    weights = {}
    for name, value in fusion.named_parameters():
        if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError('Fusion requires finite FP32 parameters')
        weights[name] = {'shape': list(value.shape),
            'sha256': hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
    return identity({'layout': fusion.descriptor(), 'weights': weights})


@torch.no_grad()
def generate_fused(net, fusion, token_ids, max_tokens, observation=None, *, source_ablation=False):
    """Execute an identical committed request on every graph owner."""
    graph, wire, rank = net.graph, net.all_owners, net.rank
    width = graph['parent']['config']['hidden_size']
    sources = {'parent': 2, **{row['id']: row['owner'] for row in graph['descriptor']['rules']}}
    if (type(source_ablation) is not bool or set(fusion.source_widths) != set(sources) or fusion.hub_width != width
            or any(value != width for value in fusion.source_widths.values())
            or any(module.training for module in fusion.modules())
            or next(fusion.parameters()).device != next(net.shard.parameters()).device):
        raise ValueError('Fusion must bind every installed source and local numerical device')
    tokens = list(token_ids)
    if (not tokens or any(type(value) is not int or not 0 <= value < graph['parent']['config']['vocab_size']
                         for value in tokens) or type(max_tokens) is not int or not 1 <= max_tokens <= 256
            or len(tokens)+max_tokens > fusion.max_context):
        raise ValueError('Invalid bounded fused request')
    net.check_context(tokens, max_tokens)
    root = commitment(fusion)
    request = {'graph': identity(graph), 'fusion': root, 'tokens': tokens, 'max_tokens': max_tokens,
               'source_ablation': source_ablation}
    if wire.exchange(identity(request)) != [identity(request)]*net.world_size:
        raise ValueError('Owners disagree on fused weights or the complete request')
    versions = tuple(parameter._version for parameter in fusion.parameters())
    trained = CachedPartition(net.shard)
    preserved = CachedPartition(net.preserved.shard) if rank < 3 else None
    cache = FusionCache(fusion) if rank == 0 else None
    captured, hook = {}, None
    if rank == 2:
        layer = net.shard.layers[str(graph['descriptor']['split']-1)]
        hook = layer.register_forward_hook(lambda module, args, output: captured.update(prefix=output.clone()))
    device = net.shard.device_name
    current, output = tokens, []
    started, first, before = time.monotonic(), None, wire.sent_tensor_bytes
    try:
        for _ in range(max_tokens):
            if versions != tuple(parameter._version for parameter in fusion.parameters()):
                raise ValueError('Fusion changed within an autoregressive request')
            count = len(current)
            shape = (1, count, width)
            projected = {}
            if rank < 3:
                incoming = (torch.tensor([current], dtype=torch.long, device=device) if rank == 0
                            else wire.receive(rank-1, shape, device))
                with autocast(device):
                    hidden = trained.advance(incoming, trained.length)
                if rank < 2:
                    wire.send(hidden, rank+1)
                else:
                    for value in fusion.project('parent', hidden):
                        wire.send(value, 0)
                    for name, owner in sources.items():
                        if name != 'parent':
                            wire.send(captured['prefix'], owner)
                            if source_ablation:
                                wire.send(hidden, owner)
                    captured.clear()
            else:
                name = next(name for name, owner in sources.items() if owner == rank)
                incoming = wire.receive(2, shape, device)
                with autocast(device):
                    hidden = trained.advance(incoming, trained.length)
                if source_ablation:
                    hidden = wire.receive(2, shape, device)
                for value in fusion.project(name, hidden):
                    wire.send(value, 0)
            if rank == 0:
                for name, owner in sources.items():
                    projected[name] = tuple(wire.receive(owner, (1, count, fusion.rank), device) for _ in range(2))
            # The preserved general backbone receives the same global tokens.
            if rank < 3:
                incoming = (torch.tensor([current], dtype=torch.long, device=device) if rank == 0
                            else wire.receive(rank-1, shape, device))
                with autocast(device):
                    hidden = preserved.advance(incoming, preserved.length)
                wire.send(hidden if rank < 2 else hidden[:, -1:], rank+1 if rank < 2 else 0)
            token = None
            if rank == 0:
                hub = wire.receive(2, (1, 1, width), device)
                merged = cache.advance(hub, projected, cache.length)
                with autocast(device):
                    logits = net.preserved.shard.logits(merged).float()
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError('Nonfinite fused model logits')
                token = int(logits.argmax(-1)[0, 0])
            packets = wire.exchange(token)
            token = packets[0]
            if (any(value is not None for value in packets[1:]) or type(token) is not int
                    or not 0 <= token < graph['parent']['config']['vocab_size']):
                raise ValueError('Only the tied output owner may produce the next token')
            output.append(token)
            if first is None:
                first = time.monotonic()-started
            if token == net.tokenizer.eos_token_id:
                break
            current = [token]
        net.verify_unchanged()
        if observation is not None:
            observation.update({'graph': identity(graph), 'fusion': root, 'sources': sources,
                'source_ablation': source_ablation,
                'seconds': time.monotonic()-started, 'first_token_seconds': first,
                'sent_tensor_bytes': wire.sent_tensor_bytes-before,
                'fusion_cache_bytes': cache.resident_bytes() if cache else 0,
                'owned_cache_bytes': trained.cache.resident_bytes() +
                    (preserved.cache.resident_bytes() if preserved else 0),
                'executed_positions': trained.length, 'tokens': output})
        return output
    finally:
        if hook is not None:
            hook.remove()
