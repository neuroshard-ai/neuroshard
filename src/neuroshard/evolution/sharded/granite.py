"""Granite layer partitions: each owner holds one contiguous layer range and no unowned tensor.

Owner 0 holds the tied embedding and the final norm, so it embeds the prompt and
scores the last hidden state; every owner applies its layers with its own
key/value cache, indexed by local layer. The numerics are those of the canonical
Granite runtime: the same decoder modules, masks, rotary tables, multipliers and
last-position head, so a pipeline can be compared token for token against it.
"""
import hashlib
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

FORMAT = 'neuroshard-granite-partition/1'
EDGE = ('model.embed_tokens.weight', 'model.norm.weight')
LAYER = re.compile(r'model\.layers\.(\d+)\.(.+)')


def check_profile(config):
    rope = getattr(config, 'rope_parameters', None) or getattr(config, 'rope_scaling', None) or {}
    if (config.model_type != 'granite' or not config.tie_word_embeddings or config.hidden_act != 'silu'
            or config.attention_dropout != 0 or rope.get('rope_type', 'default') != 'default'
            or config._attn_implementation != 'eager'):
        raise ValueError('use the pinned tied-weight, dropout-free, eager Granite profile')


def check_boundaries(boundaries, layers):
    if (len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != layers
            or any(a >= b for a, b in zip(boundaries, boundaries[1:]))):
        raise ValueError('layer ranges must partition the entire model')


def owner(name, boundaries):
    if name in EDGE:
        return 0
    match = LAYER.fullmatch(name)
    if match:
        layer = int(match[1])
        for rank, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
            if begin <= layer < end:
                return rank
    raise ValueError('unsupported Granite tensor: ' + name)


def load_config(directory):
    from transformers import GraniteConfig

    config = GraniteConfig.from_pretrained(directory, local_files_only=True)
    config._attn_implementation = 'eager'
    return config


class Partition(nn.Module):
    def __init__(self, config, boundaries, rank, *, dtype=torch.bfloat16, device='cpu'):
        from transformers.models.granite.modeling_granite import (
            GraniteDecoderLayer, GraniteRMSNorm, GraniteRotaryEmbedding,
        )

        super().__init__()
        check_profile(config)
        check_boundaries(boundaries, config.num_hidden_layers)
        if not 0 <= rank < len(boundaries) - 1:
            raise ValueError('unknown partition owner')
        self.config, self.boundaries, self.rank = config, tuple(boundaries), rank
        self.begin, self.end = boundaries[rank], boundaries[rank + 1]
        # Meta allocation describes only this owner's tensors before any storage exists.
        with torch.device('meta'):
            self.layers = nn.ModuleList(GraniteDecoderLayer(config, local) for local in range(self.end - self.begin))
            if rank == 0:
                self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
                self.norm = GraniteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.to_empty(device=device)
        self.to(dtype)
        self.rotary = GraniteRotaryEmbedding(config, device=device)
        self.requires_grad_(False)
        self.eval()

    def owned_names(self):
        names = [f'model.layers.{self.begin + i}.{name}' for i, layer in enumerate(self.layers)
                 for name, _ in layer.named_parameters()]
        return sorted(names + (list(EDGE) if self.rank == 0 else []))

    def parameter(self, name):
        if owner(name, self.boundaries) != self.rank:
            raise ValueError('tensor belongs to another owner: ' + name)
        if name == EDGE[0]:
            return self.embedding.weight
        if name == EDGE[1]:
            return self.norm.weight
        index, rest = LAYER.fullmatch(name).groups()
        return self.layers[int(index) - self.begin].get_parameter(rest)

    def load(self, read):
        """Copy every owned tensor from ``read(name)``; no other tensor is requested."""
        with torch.no_grad():
            for name in self.owned_names():
                value, target = read(name), self.parameter(name)
                if value.shape != target.shape or value.dtype != target.dtype:
                    raise ValueError('tensor shape or dtype differs: ' + name)
                if not bool(torch.isfinite(value).all()):
                    raise ValueError('nonfinite tensor: ' + name)
                target.copy_(value)
        return self

    def resident_bytes(self):
        return sum(p.numel() * p.element_size() for p in self.parameters())

    def warm_up(self, lengths=(1024, 1)):
        """Discarded passes through every owned module before any declared work.

        A fresh process's first pass can round differently from every later one;
        the same pass later in the process is reproducible across processes.
        """
        from transformers import DynamicCache

        dtype = self.layers[0].input_layernorm.weight.dtype
        with torch.inference_mode():
            for length in lengths:
                if self.rank == 0:
                    hidden = self(self.embed(torch.zeros((1, length), dtype=torch.long)), None, DynamicCache())
                    self.logits(hidden[:, -1:])
                else:
                    self(torch.zeros((1, length, self.config.hidden_size), dtype=dtype), None, DynamicCache())

    def embed(self, ids):
        if self.rank != 0:
            raise ValueError('only owner 0 holds the embedding')
        return self.embedding(ids) * self.config.embedding_multiplier

    def forward(self, hidden, attention_mask=None, cache=None):
        from transformers.masking_utils import create_causal_mask

        past = cache.get_seq_length() if cache is not None else 0
        positions = torch.arange(past, past + hidden.shape[1], device=hidden.device)
        position_ids = positions.unsqueeze(0)
        mask = create_causal_mask(self.config, hidden, attention_mask, positions,
                                  past_key_values=cache, position_ids=position_ids)
        rotary = self.rotary(hidden, position_ids)
        for layer in self.layers:
            out = layer(hidden, attention_mask=mask, position_ids=position_ids, past_key_values=cache,
                        use_cache=cache is not None, cache_position=positions, position_embeddings=rotary)
            hidden = out[0] if isinstance(out, tuple) else out
        return hidden

    def logits(self, hidden):
        if self.rank != 0:
            raise ValueError('only owner 0 holds the tied output head')
        return F.linear(self.norm(hidden), self.embedding.weight) / self.config.logits_scaling


def checkpoint_reader(directory):
    """Lazy reads of named tensors from a safetensors checkpoint, plus its complete tensor list."""
    from safetensors import safe_open

    directory = Path(directory)
    index = directory / 'model.safetensors.index.json'
    if index.exists():
        weight_map = json.loads(index.read_text())['weight_map']
    else:
        with safe_open(directory / 'model.safetensors', 'pt') as handle:
            weight_map = {name: 'model.safetensors' for name in handle.keys()}
    handles = {}

    def read(name):
        file = weight_map[name]
        if file not in handles:
            handles[file] = safe_open(directory / file, 'pt')
        return handles[file].get_tensor(name)

    return read, sorted(weight_map)


def safetensors_header(read_range):
    """Header of one safetensors file and the absolute offset where its data begins."""
    size = int.from_bytes(read_range(0, 8), 'little')
    if not 0 < size <= 100 * 1024 ** 2:
        raise ValueError('implausible safetensors header')
    return json.loads(read_range(8, 8 + size)), 8 + size


def tensor_inventory(directory, files):
    """Byte range and digest of every tensor in a checkpoint whose files match ``files`` digests."""
    directory = Path(directory)
    _, names = checkpoint_reader(directory)
    weight_map = json.loads((directory / 'model.safetensors.index.json').read_text())['weight_map']
    tensors = {}
    for file in sorted(set(weight_map.values())):
        path = directory / file
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1 << 24), b''):
                digest.update(block)
        if digest.hexdigest() != files[file]:
            raise ValueError('checkpoint file differs from its pinned digest: ' + file)
        with path.open('rb') as handle:
            def read_range(begin, end):
                handle.seek(begin)
                return handle.read(end - begin)

            header, base = safetensors_header(read_range)
            for name, spec in header.items():
                if name == '__metadata__':
                    continue
                begin, end = (base + offset for offset in spec['data_offsets'])
                tensors[name] = {'file': file, 'dtype': spec['dtype'], 'shape': spec['shape'], 'begin': begin,
                                 'end': end, 'sha256': hashlib.sha256(read_range(begin, end)).hexdigest()}
    if sorted(tensors) != names or any(weight_map[name] != spec['file'] for name, spec in tensors.items()):
        raise ValueError('tensor inventory differs from the checkpoint index')
    return {'format': 'neuroshard-granite-tensors/1', 'files': dict(sorted(files.items())), 'tensors': tensors}


DTYPES = {'BF16': torch.bfloat16, 'F32': torch.float32, 'F16': torch.float16}


def fetch(inventory, boundaries, rank, out, read_range):
    """Build one owner's shard from verified byte ranges; ``read_range(file, begin, end)`` sees nothing else.

    Adjacent owned ranges in one file are fetched together; every tensor is checked
    against its inventory digest before it is written.
    """
    from safetensors.torch import save_file

    owned = sorted((spec['file'], spec['begin'], spec['end'], name)
                   for name, spec in inventory['tensors'].items() if owner(name, boundaries) == rank)
    spans = []
    for file, begin, end, name in owned:
        if spans and spans[-1][0] == file and spans[-1][2] == begin:
            spans[-1][2] = end
            spans[-1][3].append((name, begin, end))
        else:
            spans.append([file, begin, end, [(name, begin, end)]])
    values, fetched = {}, 0
    for file, begin, end, members in spans:
        raw = read_range(file, begin, end)
        if len(raw) != end - begin:
            raise ValueError('short range read')
        fetched += len(raw)
        for name, first, last in members:
            spec, chunk = inventory['tensors'][name], raw[first - begin:last - begin]
            if hashlib.sha256(chunk).hexdigest() != spec['sha256']:
                raise ValueError('fetched tensor differs from its inventory digest: ' + name)
            values[name] = torch.frombuffer(bytearray(chunk), dtype=DTYPES[spec['dtype']]).reshape(spec['shape'])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'partition-{rank}.safetensors'
    save_file(values, str(path))
    manifest = {'format': FORMAT, 'boundaries': list(boundaries), 'rank': rank, 'tensors': sorted(values),
                'file': path.name, 'bytes': path.stat().st_size, 'fetched_bytes': fetched, 'requests': len(spans),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / f'partition-{rank}.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def http_ranges(base_url, timeout=120):
    """``read_range`` over HTTP Range requests against a pinned repository revision."""
    import urllib.request

    def read_range(file, begin, end):
        request = urllib.request.Request(f'{base_url}/{file}', headers={'Range': f'bytes={begin}-{end - 1}'})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 206:
                raise ValueError('server ignored the byte range')
            return response.read()

    return read_range


def export(directory, boundaries, rank, out):
    """Write one owner's tensors and a digest manifest; every checkpoint tensor must have an owner."""
    from safetensors.torch import save_file

    read, names = checkpoint_reader(directory)
    owned = [name for name in names if owner(name, boundaries) == rank]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'partition-{rank}.safetensors'
    save_file({name: read(name).contiguous() for name in owned}, str(path))
    manifest = {'format': FORMAT, 'boundaries': list(boundaries), 'rank': rank, 'tensors': owned,
                'file': path.name, 'bytes': path.stat().st_size,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / f'partition-{rank}.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def load_partition(config, directory, rank, *, dtype=torch.bfloat16, device='cpu'):
    """An owner from its exported shard, after checking the manifest digest and tensor set."""
    from safetensors import safe_open

    directory = Path(directory)
    manifest = json.loads((directory / f'partition-{rank}.json').read_text())
    path = directory / manifest['file']
    if (manifest['format'] != FORMAT or manifest['rank'] != rank or Path(manifest['file']).name != manifest['file']
            or hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']):
        raise ValueError('partition shard differs from its manifest')
    partition = Partition(config, manifest['boundaries'], rank, dtype=dtype, device=device)
    with safe_open(path, 'pt') as handle:
        if sorted(handle.keys()) != partition.owned_names() or manifest['tensors'] != partition.owned_names():
            raise ValueError('partition shard must hold exactly its owned tensors')
        partition.load(handle.get_tensor)
    return partition, manifest


def forward(partitions, ids, attention_mask=None, caches=None, last=False):
    """One in-process pass through every owner in order, then owner 0's head."""
    hidden = partitions[0].embed(ids)
    for partition, cache in zip(partitions, caches or [None] * len(partitions)):
        hidden = partition(hidden, attention_mask, cache)
    return partitions[0].logits(hidden[:, -1:] if last else hidden)


def generate(step, ids, max_new_tokens, eos_ids):
    """Greedy decoding: ``step(tokens, mask)`` returns last-position logits and keeps its own caches."""
    tokens, current, length = [], ids, ids.shape[1]
    for _ in range(max_new_tokens):
        logits = step(current, torch.ones((1, length), dtype=torch.long))
        token = int(logits[0, -1].float().argmax())
        tokens.append(token)
        if token in eos_ids:
            break
        current, length = torch.tensor([[token]]), length + 1
    return tokens


def local_step(partitions):
    from transformers import DynamicCache

    caches = [DynamicCache() for _ in partitions]
    return lambda tokens, mask: forward(partitions, tokens, mask, caches, last=True)
