"""Grow by adding a learned transformer tail while retaining the original path.

The explicit domain selector is an experimental boundary. The neural decoder
and tokenizer remain unchanged: both paths perform ordinary greedy generation.
"""
import json

import torch
import torch.distributed as dist
from transformers.masking_utils import create_causal_mask

from ..reference import autocast
from .wire import Wire


def route(question):
    return isinstance(question, str) and 'fictional luma directory' in question.casefold()


class ParentWire(Wire):
    """The established path has no collective dependency on the new owner."""
    def __init__(self, rank, group):
        super().__init__(rank, 3)
        self.group = group

    def send(self, value, destination):
        if value.dtype != torch.float32 or value.ndim != 3 or value.numel() > self.max_elements:
            raise ValueError('Unsupported parent boundary tensor')
        payload = value.detach().to('cpu').contiguous()
        dist.send(torch.tensor(list(payload.shape), dtype=torch.int64), dst=destination, group=self.group)
        dist.send(payload, dst=destination, group=self.group)
        self.sent_tensor_bytes += payload.numel() * 4 + 24

    def receive(self, source, shape, device):
        header = torch.empty(3, dtype=torch.int64)
        dist.recv(header, src=source, group=self.group)
        actual = tuple(header.tolist())
        if actual != tuple(shape) or any(n <= 0 for n in actual) or int(header.prod()) > self.max_elements:
            raise ValueError('Unexpected parent boundary shape')
        value = torch.empty(actual, dtype=torch.float32)
        dist.recv(value, src=source, group=self.group)
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite parent boundary')
        return value.to(device)

    def exchange(self, value):
        raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        if not 0 < len(raw) <= 2 * 1024**2:
            raise ValueError('Bounded parent control message required')
        sizes = [torch.zeros(1, dtype=torch.int64) for _ in range(3)]
        dist.all_gather(sizes, torch.tensor([len(raw)], dtype=torch.int64), group=self.group)
        length = max(int(size) for size in sizes)
        if not 0 < length <= 2 * 1024**2:
            raise ValueError('Parent control message exceeds bound')
        payload = torch.zeros(length, dtype=torch.uint8)
        payload[:len(raw)] = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        received = [torch.empty_like(payload) for _ in sizes]
        dist.all_gather(received, payload, group=self.group)
        return [json.loads(bytes(value[:int(size)].tolist())) for value, size in zip(received, sizes)]


@torch.no_grad()
def prefix(shard, hidden, mask, stop):
    positions = torch.arange(hidden.shape[1], device=hidden.device)
    position_ids = positions.unsqueeze(0)
    causal = create_causal_mask(shard.config, hidden, mask, positions, None, position_ids)
    rotary = shard.rotary(hidden, position_ids)
    for number, layer in shard.layers.items():
        if int(number) >= stop:
            break
        hidden = layer(hidden, attention_mask=causal, position_ids=position_ids,
            cache_position=positions, position_embeddings=rotary, use_cache=False)
    return hidden


class Network:
    def __init__(self, shard, wire, parent_wire, tokenizer, split):
        if wire.world != 4 or not 0 < split < shard.config.num_hidden_layers:
            raise ValueError('Three parent owners and one added tail owner are required')
        self.shard, self.wire, self.parent_wire = shard, wire, parent_wire
        self.tokenizer, self.split = tokenizer, split
        self.device, self.config = shard.device_name, shard.config
        if wire.rank == 2 and not shard.boundaries[2] < split < shard.boundaries[3]:
            raise ValueError('The original final owner must retain both prefix and parent tail')
        if wire.rank == 3 and shard.boundaries[-2] != split:
            raise ValueError('The new owner must hold exactly the learned tail')

    @torch.no_grad()
    def forward(self, token_ids, expert):
        if (not token_ids or len(token_ids) > self.config.max_position_embeddings
                or any(type(token) is not int or not 0 <= token < self.config.vocab_size for token in token_ids)):
            raise ValueError('Invalid causal input')
        rank, final_rank = self.wire.rank, 3 if expert else 2
        if rank == 3 and not expert:
            return None
        wire = self.wire if expert else self.parent_wire
        shape = (1, len(token_ids), self.config.hidden_size)
        ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        mask = torch.ones_like(ids)
        incoming = ids if rank == 0 else wire.receive(rank - 1, shape, self.device)
        with autocast(self.device):
            outgoing = (prefix(self.shard, incoming, mask, self.split) if rank == 2 and expert
                        else self.shard(incoming, mask))
        wire.send(outgoing, rank + 1 if rank < final_rank else 0)
        return wire.receive(final_rank, shape, self.device) if rank == 0 else None

    @torch.no_grad()
    def generate(self, question, max_tokens, expert):
        """Only user text chooses the path; no evaluation/task identifier enters."""
        if self.wire.rank == 3 and not expert:
            return None
        if not isinstance(question, str) or not question or len(question.encode()) > 32768:
            raise ValueError('Require a bounded user question')
        ids = self.tokenizer.apply_chat_template([{'role': 'user', 'content': question}],
            tokenize=True, add_generation_prompt=True)
        if not 0 < max_tokens <= 256 or len(ids) + max_tokens > self.config.max_position_embeddings:
            raise ValueError('No silent generation truncation')
        output = []
        control = self.wire if expert else self.parent_wire
        for _ in range(max_tokens):
            final = self.forward(ids, expert)
            token = None
            if self.wire.rank == 0:
                with autocast(self.device):
                    token = int(self.shard.logits(final[:, -1:]).float().argmax(-1)[0, 0])
            token = control.exchange(token)[0]
            if type(token) is not int or not 0 <= token < self.config.vocab_size:
                raise ValueError('Invalid branch token')
            output.append(token)
            ids.append(token)
            if token == self.tokenizer.eos_token_id:
                break
        return {'ids': output, 'text': self.tokenizer.decode(output, skip_special_tokens=True),
                'route': 'expert' if expert else 'parent'}

    def answer(self, question, max_tokens):
        return self.generate(question, max_tokens, route(question))
