"""Named-tensor checkpoints whose learned-state root survives repartitioning.

The numerical profile is dropout-free and consumes no RNG during updates.
Process RNG files are retained for recovery; the layout-independent state root
commits weights, Adam state, parameter birth cursors and the optimizer recipe.
Changing the layout changes the checkpoint root, but not that learned-state root.
"""
import copy
import functools
import json
import math
import os
import random
import shutil
import uuid
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from ..reference_data import identity, save, sha256
from . import checkpoint as files
from .model import owner

FORMAT = 'neuroshard-portable-shards-v1'


def shapes(config):
    """Supported tied, bias-free Llama tensor inventory, without model allocation."""
    if (config.model_type != 'llama' or not config.tie_word_embeddings
            or config.attention_dropout != 0 or config.attention_bias or config.mlp_bias
            or config.hidden_act != 'silu' or config.rope_scaling is not None):
        raise ValueError('Require the deterministic tied Llama profile')
    for field, maximum in [('num_hidden_layers', 512), ('hidden_size', 16384),
                           ('intermediate_size', 65536), ('vocab_size', 262144),
                           ('num_attention_heads', 2048), ('num_key_value_heads', 2048)]:
        value = getattr(config, field)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError('Architecture exceeds the bounded numerical profile')
    if config.num_attention_heads % config.num_key_value_heads:
        raise ValueError('Attention heads must form complete key/value groups')
    h, m = config.hidden_size, config.intermediate_size
    d = config.head_dim or h // config.num_attention_heads
    q, kv = config.num_attention_heads*d, config.num_key_value_heads*d
    result = {'model.embed_tokens.weight': [config.vocab_size, h], 'model.norm.weight': [h]}
    block = {'input_layernorm.weight': [h], 'post_attention_layernorm.weight': [h],
             'self_attn.q_proj.weight': [q, h], 'self_attn.k_proj.weight': [kv, h],
             'self_attn.v_proj.weight': [kv, h], 'self_attn.o_proj.weight': [h, q],
             'mlp.gate_proj.weight': [m, h], 'mlp.up_proj.weight': [m, h],
             'mlp.down_proj.weight': [h, m]}
    for layer in range(config.num_hidden_layers):
        result.update({f'model.layers.{layer}.{name}': size for name, size in block.items()})
    return dict(sorted(result.items()))


def configuration(config):
    # Model identity is independent of local paths, host names and transformers metadata.
    return {name: getattr(config, name) for name in ['model_type', 'vocab_size', 'hidden_size',
        'intermediate_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads',
        'head_dim', 'max_position_embeddings', 'rms_norm_eps', 'rope_theta', 'rope_scaling',
        'tie_word_embeddings', 'hidden_act', 'attention_dropout', 'attention_bias', 'mlp_bias',
        'pad_token_id', 'bos_token_id', 'eos_token_id']}


def recipe(optimizer):
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError('Portable profile requires AdamW')
    return [{k: v for k, v in group.items() if k != 'params'}
            for group in optimizer.param_groups]


def groups(optimizer):
    result = {}
    for index, group in enumerate(optimizer.param_groups):
        for parameter in group['params']:
            if id(parameter) in result:
                raise ValueError('Repeated optimizer parameter')
            result[id(parameter)] = index
    return result


def learned_root(common):
    return identity({name: common[name] for name in ['format', 'job', 'step', 'config', 'optimizer', 'tensors']})


def validate(common):
    from transformers import LlamaConfig
    if common['format'] != FORMAT or common['state_root'] != learned_root(common):
        raise ValueError('Invalid portable state commitment')
    config = LlamaConfig(**common['config'])
    expected = shapes(config)
    if not isinstance(common['optimizer'], list) or not 1 <= len(common['optimizer']) <= 16:
        raise ValueError('Invalid optimizer groups')
    if set(common['tensors']) != set(expected):
        raise ValueError('Incomplete or additional model tensors')
    boundaries = common['boundaries']
    if (not isinstance(boundaries, list) or not 3 <= len(boundaries) <= 513
            or any(type(a) is not int for a in boundaries)
            or boundaries[0] != 0 or boundaries[-1] != config.num_hidden_layers
            or any(a >= b for a, b in zip(boundaries, boundaries[1:]))):
        raise ValueError('Invalid contiguous shard layout')
    if type(common['step']) is not int or not 0 <= common['step'] < 2**24:
        raise ValueError('Adam FP32 cursor exceeds the exact integer profile')
    for name, spec in common['tensors'].items():
        if (set(spec) != {'sha256', 'bytes', 'shape', 'born', 'group'} or spec['shape'] != expected[name]
                or type(spec['born']) is not int or not 0 <= spec['born'] <= common['step']
                or type(spec['group']) is not int or not 0 <= spec['group'] < len(common['optimizer'])
                or type(spec['bytes']) is not int or spec['bytes'] <= 0
                or len(spec['sha256']) != 64 or any(c not in '0123456789abcdef' for c in spec['sha256'])):
            raise ValueError('Invalid tensor identity, shape or birth cursor')
    return config


def directory(home, step):
    return Path(home)/f'shard-{step:06d}'


def tensor_path(folder, digest):
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Unsafe tensor digest')
    return Path(folder)/(digest+'.safetensors')


def write(home, shard, optimizer, job, step, births=None):
    births = births or {}
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    pending = home/('.writing-'+uuid.uuid4().hex)
    pending.mkdir()
    tensors = {}
    membership = groups(optimizer)
    if set(membership) != {id(p) for _, p in shard.named_owned_parameters()}:
        raise ValueError('Optimizer must cover exactly the owned parameters')
    for name, parameter in shard.named_owned_parameters():
        born = births.get(name, 0)
        if type(born) is not int or not 0 <= born <= step:
            raise ValueError('Invalid parameter birth')
        age = step-born
        state = optimizer.state.get(parameter, {})
        if (age == 0 and state) or (age > 0 and set(state) != {'step', 'exp_avg', 'exp_avg_sq'}):
            raise ValueError('Incomplete age-aware Adam state')
        if state and (state['step'].numel() != 1 or float(state['step']) != age):
            raise ValueError('Adam age differs from parameter birth')
        values = {'weight': parameter.detach().cpu().contiguous(),
                  **{k: v.detach().cpu().contiguous() for k, v in state.items()}}
        temporary = pending/'tensor.pending'
        spec = files.tensor_file(temporary, values)
        os.replace(temporary, tensor_path(pending, spec['sha256']))
        tensors[name] = {**spec, 'shape': list(parameter.shape), 'born': born,
                         'group': membership[id(parameter)]}
        del values
    numpy = np.random.get_state()
    rng = {'torch_cpu': torch.get_rng_state(), 'numpy': torch.from_numpy(numpy[1].astype(np.int64))}
    if shard.device_name == 'cuda':
        rng['torch_cuda'] = torch.cuda.get_rng_state()
    rng_file = files.tensor_file(pending/'rng.safetensors', rng)
    meta = {'format': FORMAT, 'job': job, 'step': step, 'rank': shard.rank,
            'boundaries': list(shard.boundaries), 'config': configuration(shard.config),
            'optimizer': recipe(optimizer), 'tensors': tensors, 'rng': rng_file,
            'python_rng': random.getstate(), 'numpy_rng': [numpy[0], *numpy[2:]]}
    save(pending/'manifest.json', meta)
    files.sync_directory(pending)
    destination = directory(home, step)
    if destination.exists():
        raise ValueError('Preserve the prior checkpoint; choose another output directory')
    os.replace(pending, destination)
    files.sync_directory(home)
    return meta


def assemble(metas, parent, transition=None):
    first = metas[0]
    common = {k: first[k] for k in ['format', 'job', 'step', 'config', 'optimizer', 'boundaries']}
    tensors, refs = {}, []
    if len(metas) != len(first['boundaries'])-1:
        raise ValueError('Incomplete shard coverage')
    for rank, meta in enumerate(metas):
        if meta['rank'] != rank or any(identity(meta[k]) != identity(v) for k, v in common.items()):
            raise ValueError('Shard job, version, profile or layout differs')
        if set(tensors) & set(meta['tensors']):
            raise ValueError('Duplicate parameter ownership')
        if any(owner(name, common['boundaries']) != rank for name in meta['tensors']):
            raise ValueError('Parameter stored by wrong shard')
        tensors.update(meta['tensors'])
        refs.append(identity(meta))
    common.update(tensors=dict(sorted(tensors.items())), shards=refs, parent=parent, transition=transition)
    common['state_root'] = learned_root(common)
    validate(common)
    return common


def commit(home, shard, optimizer, wire, job, step, parent, births=None, transition=None):
    meta = write(home, shard, optimizer, job, step, births)
    common = assemble(wire.exchange(meta), parent, transition)
    root = identity(common)
    save(Path(home)/f'commit-{step:06d}.json', common)
    if any(r != root for r in wire.exchange(root)):
        raise ValueError('Workers disagree on the complete checkpoint')
    save(Path(home)/'HEAD.json', {'root': root, 'state_root': common['state_root'], 'step': step})
    return common


def load(home, shard, optimizer, common, job, restore_rng=True):
    validate(common)
    if common['job'] != job or common['config'] != configuration(shard.config) or common['boundaries'] != list(shard.boundaries):
        raise ValueError('Wrong job, model or layout')
    folder = directory(home, common['step'])
    meta = json.loads((folder/'manifest.json').read_bytes())
    if identity(meta) != common['shards'][shard.rank] or meta['rank'] != shard.rank:
        raise ValueError('Wrong shard manifest')
    expected = dict(shard.named_owned_parameters())
    if set(expected) != set(meta['tensors']):
        raise ValueError('Incomplete owned checkpoint')
    if optimizer is not None:
        actual = recipe(optimizer)
        semantics = lambda rows: [{k: v for k, v in row.items() if k != 'lr'} for row in rows]
        if identity(semantics(actual)) != identity(semantics(common['optimizer'])):
            raise ValueError('Optimizer semantics changed')
        membership = groups(optimizer)
        if set(membership) != {id(p) for p in expected.values()} or any(
                membership[id(p)] != common['tensors'][name]['group'] for name, p in expected.items()):
            raise ValueError('Optimizer group membership changed')
        for group, saved in zip(optimizer.param_groups, common['optimizer']):
            group['lr'] = saved['lr']
    with torch.no_grad():
        for name, parameter in expected.items():
            spec = meta['tensors'][name]
            if spec != common['tensors'][name]:
                raise ValueError('Tensor identity differs from global commitment')
            path = tensor_path(folder, spec['sha256'])
            if path.stat().st_size != spec['bytes'] or sha256(path) != spec['sha256']:
                raise ValueError('Missing or corrupted tensor state')
            values = load_file(path, device='cpu')
            age = common['step']-spec['born']
            if set(values) != ({'weight'} if age == 0 else {'weight', 'step', 'exp_avg', 'exp_avg_sq'}):
                raise ValueError('Incomplete Adam state for parameter age')
            if age and (values['step'].numel() != 1 or float(values['step']) != age):
                raise ValueError('Wrong Adam age')
            for key, value in values.items():
                if key != 'step' and (value.shape != parameter.shape or value.dtype != parameter.dtype or not bool(torch.isfinite(value).all())):
                    raise ValueError('Invalid checkpoint weight or moment')
            parameter.copy_(values['weight'])
            if optimizer is not None:
                optimizer.state.pop(parameter, None)
                if age:
                    optimizer.state[parameter] = {'step': values['step'].cpu(),
                        'exp_avg': values['exp_avg'].to(shard.device_name),
                        'exp_avg_sq': values['exp_avg_sq'].to(shard.device_name)}
            del values
    path = folder/'rng.safetensors'
    if sha256(path) != meta['rng']['sha256']:
        raise ValueError('Corrupted RNG state')
    if restore_rng:
        rng = load_file(path, device='cpu')
        torch.set_rng_state(rng['torch_cpu'])
        if shard.device_name == 'cuda':
            torch.cuda.set_rng_state(rng['torch_cuda'])
        random.setstate(files.tuples(meta['python_rng']))
        np.random.set_state((meta['numpy_rng'][0], rng['numpy'].numpy().astype(np.uint32), *meta['numpy_rng'][1:]))
    return {name: spec['born'] for name, spec in common['tensors'].items()}


def layout(config, capacities, bytes_per_parameter=20, reserve_bytes=3*1024**3):
    """Minimize maximum memory utilization over contiguous layer assignments."""
    sizes = shapes(config)
    if not 2 <= len(capacities) <= config.num_hidden_layers:
        raise ValueError('Need two or more workers with nonempty layer ranges')
    budgets = [int(n)-reserve_bytes for n in capacities]
    if any(n <= 0 for n in budgets):
        raise ValueError('Insufficient memory reserve')
    fixed = sum(math.prod(s) for n, s in sizes.items() if not n.startswith('model.layers.'))*bytes_per_parameter
    layers = [sum(math.prod(s) for n, s in sizes.items() if n.startswith(f'model.layers.{i}.'))*bytes_per_parameter
              for i in range(config.num_hidden_layers)]
    prefix = [0]
    for value in layers:
        prefix.append(prefix[-1]+value)
    @functools.lru_cache(None)
    def solve(rank, begin):
        if rank == len(budgets):
            return (0., ()) if begin == len(layers) else (math.inf, ())
        best = (math.inf, ())
        for end in range(begin+1, len(layers)-(len(budgets)-rank-1)+1):
            load = prefix[end]-prefix[begin]+(fixed if rank == 0 else 0)
            if load > budgets[rank]:
                continue
            score, suffix = solve(rank+1, end)
            candidate = (max(load/budgets[rank], score), (end,)+suffix)
            if candidate < best:
                best = candidate
        return best
    score, boundaries = solve(0, 0)
    if not math.isfinite(score):
        raise ValueError('Advertised capacity cannot hold this model and optimizer')
    return [0, *boundaries]


def repartition(common, boundaries):
    """Plan a metadata-only ownership change; each recipient fetches its own bytes."""
    validate(common)
    proposal = copy.deepcopy(common)
    proposal.update(boundaries=list(boundaries), parent=identity(common), shards=[],
                    transition={'kind': 'repartition', 'source': identity(common),
                                'preserved_state_root': common['state_root']})
    validate(proposal)
    assert proposal['state_root'] == common['state_root']
    return proposal


def install_partition(home, rank, proposal, sources, rng_meta, rng_path):
    """Stage only owned immutable files. Global commit/HEAD requires all owners."""
    validate(proposal)
    if not 0 <= rank < len(proposal['boundaries'])-1:
        raise ValueError('Invalid destination rank')
    folder = directory(home, proposal['step'])
    folder.mkdir(parents=True, exist_ok=False)
    tensors = {n: v for n, v in proposal['tensors'].items() if owner(n, proposal['boundaries']) == rank}
    if set(sources) != set(tensors):
        raise ValueError('Only the exact owned tensor set may be installed')
    for name, spec in tensors.items():
        path = Path(sources[name])
        if path.stat().st_size != spec['bytes'] or sha256(path) != spec['sha256']:
            raise ValueError('Missing or corrupt migration source')
        shutil.copyfile(path, tensor_path(folder, spec['sha256']))
        with tensor_path(folder, spec['sha256']).open('rb') as stream:
            os.fsync(stream.fileno())
    if sha256(rng_path) != rng_meta['rng']['sha256']:
        raise ValueError('Corrupt source RNG file')
    shutil.copyfile(rng_path, folder/'rng.safetensors')
    with (folder/'rng.safetensors').open('rb') as stream:
        os.fsync(stream.fileno())
    meta = {k: proposal[k] for k in ['format', 'job', 'step', 'config', 'optimizer', 'boundaries']}
    meta.update(rank=rank, tensors=tensors, **{k: rng_meta[k] for k in ['rng', 'python_rng', 'numpy_rng']})
    save(folder/'manifest.json', meta)
    files.sync_directory(folder)
    files.sync_directory(folder.parent)
    return meta
