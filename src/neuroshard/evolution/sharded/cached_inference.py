"""Research-only greedy inference with request-local, owner-local KV state.

The existing native and frozen learning profiles keep their original generator.
Caching changes matrix shapes and may change floating-point rounding, so this
executor requires its own numerical identity before any consensus activation.
"""

import time

import torch
from transformers.cache_utils import Cache, DynamicLayer
from transformers.masking_utils import create_causal_mask

from ..reference import autocast


class OwnedCache(Cache):
    """Translate global decoder indices to only the layers this worker owns."""

    def __init__(self, begin, end):
        self.begin, self.end = begin, end
        super().__init__(layers=[DynamicLayer() for _ in range(begin, end)])

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if not self.begin <= layer_idx < self.end:
            raise ValueError('Cannot cache an unowned layer')
        return super().update(key_states, value_states, layer_idx - self.begin, cache_kwargs)

    def resident_bytes(self):
        return sum(value.numel() * value.element_size()
                   for layer in self.layers if layer.is_initialized
                   for value in (layer.keys, layer.values))


class CachedPartition:
    """One immutable shard and one unpadded request; discard on any failure.

    Cache tensors are derived execution state. They are never accepted from a
    peer or carried across requests. Replaying the prompt reconstructs them.
    """

    def __init__(self, shard):
        if any(module.training for module in shard.modules()):
            raise ValueError('Cached execution requires evaluation mode')
        self.shard = shard
        self.versions = tuple(parameter._version for parameter in shard.parameters())
        self.cache = OwnedCache(*shard.boundaries[shard.rank:shard.rank + 2])
        self.length = 0
        self.failed = False

    @torch.no_grad()
    def advance(self, value, position):
        shard = self.shard
        if self.failed:
            raise ValueError('Discard a failed cache session')
        # A failed validation also invalidates the session: callers must not
        # continue with a partially updated subset of local layer caches.
        self.failed = True
        if (any(module.training for module in shard.modules())
                or self.versions != tuple(parameter._version for parameter in shard.parameters())):
            raise ValueError('Weights or execution mode changed within a request')
        if type(position) is not int or position != self.length:
            raise ValueError('Cache position must continue the current request')
        expected_dims = 2 if shard.rank == 0 else 3
        if value.ndim != expected_dims or value.shape[0] != 1 or value.shape[1] < 1:
            raise ValueError('Cache profile requires one nonempty unpadded request')
        count = value.shape[1]
        if self.length and count != 1:
            raise ValueError('Decode exactly one new token after prefill')
        if position + count > shard.config.max_position_embeddings:
            raise ValueError('Cached generation exceeds context')
        if shard.rank == 0:
            if value.dtype != torch.long or bool(((value < 0) | (value >= shard.config.vocab_size)).any()):
                raise ValueError('Invalid input token')
        elif value.shape[2] != shard.config.hidden_size or value.dtype != torch.float32:
            raise ValueError('Invalid cached boundary tensor')
        if value.device != next(shard.parameters()).device:
            raise ValueError('Cached input must be on the shard device')
        if any(layer.get_seq_length() != position for layer in self.cache.layers):
            raise ValueError('Incomplete local cache')

        hidden = shard.embedding(value) if shard.rank == 0 else value
        positions = torch.arange(position, position + count, device=hidden.device)
        position_ids = positions.unsqueeze(0)
        attention_mask = torch.ones((1, position + count), dtype=torch.long, device=hidden.device)
        causal = create_causal_mask(shard.config, hidden, attention_mask,
                                    positions, self.cache, position_ids)
        rotary = shard.rotary(hidden, position_ids)
        for layer in shard.layers.values():
            hidden = layer(hidden, attention_mask=causal, position_ids=position_ids,
                           past_key_values=self.cache, cache_position=positions,
                           position_embeddings=rotary, use_cache=True)
        self.length += count
        if any(layer.get_seq_length() != self.length for layer in self.cache.layers):
            raise ValueError('Incomplete cache update')
        self.failed = False
        return hidden


@torch.no_grad()
def generate_cached(shard, wire, token_ids, max_tokens, eos_id, observation=None):
    """Prefill once, then send one hidden vector per boundary per output token.

    Every owner gets the same request. ``observation`` is an optional mutable
    dict for local timing, cache memory and actual Wire tensor-byte counters.
    """
    tokens = list(token_ids)
    if (not tokens or any(type(token) is not int or not 0 <= token < shard.config.vocab_size for token in tokens)
            or type(max_tokens) is not int or max_tokens < 0
            or len(tokens) + max_tokens > shard.config.max_position_embeddings
            or type(eos_id) is not int or not -1 <= eos_id < shard.config.vocab_size):
        raise ValueError('Invalid cached generation request or context limit')
    if wire.rank != shard.rank or wire.world != len(shard.boundaries) - 1 or wire.world < 2:
        raise ValueError('Wire must match the shard partition')
    shard.eval()
    session = CachedPartition(shard)
    device, rank, world = shard.device_name, wire.rank, wire.world
    output, current = [], tokens
    transport = getattr(wire, 'wire', wire)
    sent_before = getattr(transport, 'sent_tensor_bytes', None)
    started, first_token_seconds = time.monotonic(), None
    for _ in range(max_tokens):
        shape = (1, len(current), shard.config.hidden_size)
        incoming = (torch.tensor([current], dtype=torch.long, device=device) if rank == 0
                    else wire.receive(rank - 1, shape, device))
        with autocast(device):
            outgoing = session.advance(incoming, session.length)
        # Only the final position is consumed by the tied output head.
        wire.send(outgoing if rank + 1 < world else outgoing[:, -1:],
                  rank + 1 if rank + 1 < world else 0)
        token = None
        if rank == 0:
            final = wire.receive(world - 1, (1, 1, shard.config.hidden_size), device)
            with autocast(device):
                logits = shard.logits(final).float()
            if not bool(torch.isfinite(logits).all()):
                raise ValueError('Nonfinite cached output logits')
            token = int(logits.argmax(-1)[0, 0])
        token = wire.exchange({'token': token})[0]['token']
        if type(token) is not int or not 0 <= token < shard.config.vocab_size:
            raise ValueError('Invalid generated token')
        output.append(token)
        if first_token_seconds is None:
            first_token_seconds = time.monotonic() - started
        if token == eos_id:
            break
        current = [token]
    if device == 'cuda':
        torch.cuda.synchronize()
    if observation is not None:
        observation.update({'seconds': time.monotonic() - started,
                            'first_token_seconds': first_token_seconds,
                            'generated_tokens': len(output), 'prefill_tokens': len(tokens),
                            'processed_positions_per_layer': session.length,
                            'resident_cache_bytes': session.cache.resident_bytes(),
                            'sent_tensor_bytes': (transport.sent_tensor_bytes - sent_before
                                                  if sent_before is not None else None)})
    return output
