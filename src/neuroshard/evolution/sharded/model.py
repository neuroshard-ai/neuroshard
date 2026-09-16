"""A Llama layer partition that never constructs an unowned parameter."""
import contextlib
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import LlamaConfig
from transformers.masking_utils import create_causal_mask
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding,
)

from ..reference import autocast


def owner(name, boundaries):
    if name in ('model.embed_tokens.weight', 'model.norm.weight'):
        return 0
    match = re.fullmatch(r'model\.layers\.(\d+)\..+', name)
    if match:
        layer = int(match[1])
        for rank, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
            if begin <= layer < end:
                return rank
    raise ValueError('Unsupported model tensor: ' + name)


class Partition(nn.Module):
    def __init__(self, config, boundaries, rank, device='cpu', parameter_limit=None):
        super().__init__()
        if (config.model_type != 'llama' or not config.tie_word_embeddings
                or config.hidden_act != 'silu' or config.attention_dropout != 0
                or config.rope_scaling is not None):
            raise ValueError('Use the pinned, tied-weight, dropout-free Llama profile')
        if (boundaries[0] != 0 or boundaries[-1] != config.num_hidden_layers
                or any(a >= b for a, b in zip(boundaries, boundaries[1:]))
                or not 0 <= rank < len(boundaries)-1):
            raise ValueError('Layer ranges must partition the entire model')
        self.config, self.boundaries, self.rank = config, tuple(boundaries), rank
        self.device_name, self.recompute = device, device == 'cuda'
        begin, end = boundaries[rank:rank+2]
        # Meta allocation describes ONLY this shard, then allocates its tensors.
        with torch.device('meta'):
            self.layers = nn.ModuleDict({str(i): LlamaDecoderLayer(config, i) for i in range(begin, end)})
            if rank == 0:
                self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
                self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.resident_parameters = sum(p.numel() for p in self.parameters())
        if parameter_limit is not None and self.resident_parameters > parameter_limit:
            raise ValueError('Partition exceeds the worker parameter limit')
        if device == 'cuda' and self.resident_parameters * 16 + 2 * 1024**3 > torch.cuda.mem_get_info()[0]:
            raise ValueError('Insufficient memory for local weights, gradients, Adam and activations')
        self.to_empty(device=device)
        # Real rotary buffers are tiny; to_empty must not leave them uninitialized.
        self.rotary = LlamaRotaryEmbedding(config, device=device)
        for parameter in self.parameters():
            if parameter.dtype != torch.float32:
                raise ValueError('This profile retains FP32 parameters')

    def named_owned_parameters(self):
        pairs = []
        if self.rank == 0:
            pairs += [('model.embed_tokens.weight', self.embedding.weight), ('model.norm.weight', self.norm.weight)]
        for index, layer in self.layers.items():
            pairs += [('model.layers.' + index + '.' + name, value) for name, value in layer.named_parameters()]
        return sorted(pairs)

    def load_weights(self, directory, manifest):
        from safetensors.torch import load_file
        from ..reference_data import sha256
        directory = Path(directory)
        expected = dict(self.named_owned_parameters())
        if set(expected) != set(manifest['tensors']):
            raise ValueError('Seed must contain exactly this partition, without unowned tensors')
        with torch.no_grad():
            for name, spec in manifest['tensors'].items():
                if Path(spec['file']).name != spec['file']:
                    raise ValueError('Unsafe tensor path')
                path = directory / spec['file']
                if sha256(path) != spec['sha256']:
                    raise ValueError('Seed tensor digest differs')
                values = load_file(path, device='cpu')
                if set(values) != {'weight'}:
                    raise ValueError('Unexpected seed tensor keys')
                value, parameter = values['weight'], expected[name]
                if value.shape != parameter.shape or value.dtype not in (torch.float32, torch.bfloat16):
                    raise ValueError('Seed tensor shape or dtype differs')
                if not bool(torch.isfinite(value).all()):
                    raise ValueError('Nonfinite seed tensor')
                parameter.copy_(value)

    def forward(self, value, attention_mask):
        hidden = self.embedding(value) if self.rank == 0 else value
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        position_ids = positions.unsqueeze(0)
        causal = create_causal_mask(self.config, hidden, attention_mask, positions, None, position_ids)
        rotary = self.rotary(hidden, position_ids)
        for layer in self.layers.values():
            def apply(x, layer=layer):
                return layer(x, attention_mask=causal, position_ids=position_ids,
                             cache_position=positions, position_embeddings=rotary, use_cache=False)
            hidden = checkpoint(apply, hidden, use_reentrant=False) if self.recompute and self.training else apply(hidden)
        return hidden

    def logits(self, hidden):
        if self.rank != 0:
            raise ValueError('Only the tied embedding owner computes the output head')
        return F.linear(self.norm(hidden), self.embedding.weight)


def batch_tensors(rows, device):
    length = max(len(row['input_ids']) for row in rows)
    ids = torch.zeros((len(rows), length), dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    weights = torch.tensor([r.get('loss_weight', 1) for r in rows], device=device, dtype=torch.float32)
    for i, row in enumerate(rows):
        n = len(row['input_ids'])
        if n < 2 or len(row['labels']) != n or row['targets'] != sum(x != -100 for x in row['labels'][1:]):
            raise ValueError('Invalid response labels or target count')
        if any(y != -100 and y != x for x, y in zip(row['input_ids'], row['labels'])):
            raise ValueError('Labels cannot introduce tokens absent from the input')
        if row['targets'] <= 0 or not 0 < row.get('loss_weight', 1) <= 16:
            raise ValueError('Invalid target count or objective weight')
        ids[i, :n] = torch.tensor(row['input_ids'], device=device)
        labels[i, :n] = torch.tensor(row['labels'], device=device)
        mask[i, :n] = 1
    return ids, labels, mask, weights


def weighted_loss(logits, labels, weights):
    values = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                             labels[:, 1:].reshape(-1), ignore_index=-100, reduction='none')
    return (values.reshape(labels.shape[0], -1) * weights[:, None]).sum()
