"""Complete shard checkpoints and a common, content-bound commit manifest.

Only completed optimizer steps can commit. Prepared local checkpoints alone do
not advance HEAD; all shard manifests and acknowledgements must agree first.
The operated controller replicates a committed checkpoint before host removal.
"""
import hashlib
import json
import os
import random
import uuid
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ..reference_data import identity, save, sha256


def tensor_digest(value):
    value = value.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update((str(value.dtype) + ':' + str(list(value.shape)) + ':').encode())
    h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def tensor_file(path, values):
    save_file(values, str(path))
    with path.open('rb') as source:
        os.fsync(source.fileno())
    return {'sha256': sha256(path), 'bytes': path.stat().st_size}


def optimizer_groups(shard, optimizer):
    names = {id(p): name for name, p in shard.named_owned_parameters()}
    return [{**{k: v for k, v in group.items() if k != 'params'},
             'params': [names[id(p)] for p in group['params']]} for group in optimizer.param_groups]


def write_shard(home, shard, optimizer, binding, step):
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    destination = home / f'shard-{step:06d}'
    pending = home / ('.writing-' + uuid.uuid4().hex)
    pending.mkdir()
    files, tensors, logical = {}, {}, {}
    for index, (name, parameter) in enumerate(shard.named_owned_parameters()):
        values = {'weight': parameter.detach().cpu().contiguous()}
        state = optimizer.state.get(parameter, {})
        if (step == 0 and state) or (step > 0 and not state):
            raise ValueError('Optimizer state must cover every completed update')
        if state and (state['step'].numel() != 1 or float(state['step']) != step):
            raise ValueError('Optimizer step differs from the global cursor')
        if state and set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('Unsupported optimizer state')
        values.update({key: value.detach().cpu().contiguous() for key, value in state.items()})
        filename = f'tensor-{index:04d}.safetensors'
        files[filename] = tensor_file(pending / filename, values)
        tensors[name] = filename
        logical[name] = {key: tensor_digest(value) for key, value in values.items()}
        del values
    numpy_state = np.random.get_state()
    rng = {'torch_cpu': torch.get_rng_state(), 'numpy': torch.from_numpy(numpy_state[1].astype(np.int64))}
    if shard.device_name == 'cuda':
        rng['torch_cuda'] = torch.cuda.get_rng_state()
    files['rng.safetensors'] = tensor_file(pending/'rng.safetensors', rng)
    meta = {'format': 'neuroshard-shard-checkpoint-v1', 'binding': binding,
            'rank': shard.rank, 'boundaries': list(shard.boundaries), 'step': step,
            'tensors': tensors, 'files': files, 'optimizer_groups': optimizer_groups(shard, optimizer),
            'python_rng': random.getstate(), 'numpy_rng': [numpy_state[0], *numpy_state[2:]],
            'parameter_digest': identity({name: v['weight'] for name, v in logical.items()}),
            'state_digest': identity(logical)}
    save(pending/'manifest.json', meta)
    sync_directory(pending)
    if destination.exists():
        previous = json.loads((destination/'manifest.json').read_bytes())
        if identity(previous) != identity(meta):
            raise ValueError('Refuse to overwrite a different prepared checkpoint')
        # A retry can prepare the same immutable state. Keep its first copy.
        import shutil
        shutil.rmtree(pending)
    else:
        os.replace(pending, destination)
        sync_directory(home)
    return {'rank': shard.rank, 'manifest': identity(meta), 'parameter_digest': meta['parameter_digest'],
            'state_digest': meta['state_digest']}


def commit(home, shard, optimizer, wire, binding, step, parent):
    local = write_shard(home, shard, optimizer, binding, step)
    refs = wire.exchange(local)
    if [r['rank'] for r in refs] != list(range(wire.world)):
        raise ValueError('Incomplete checkpoint coverage')
    common = {'format': 'neuroshard-sharded-commit-v1', 'binding': binding,
              'step': step, 'parent': parent, 'shards': refs}
    root = identity(common)
    path = Path(home) / f'commit-{step:06d}.json'
    if path.exists() and json.loads(path.read_bytes()) != common:
        raise ValueError('Refuse to replace a different committed model version')
    save(path, common)
    acknowledgements = wire.exchange({'commit': root, 'step': step})
    if any(a != {'commit': root, 'step': step} for a in acknowledgements):
        raise ValueError('Peers disagree on the model version')
    save(Path(home)/'HEAD.json', {'root': root, 'step': step})
    return common


def tuples(value):
    return tuple(tuples(x) for x in value) if isinstance(value, list) else value


def load(home, shard, optimizer, common, binding, restore_rng=True):
    if common['binding'] != binding or common['format'] != 'neuroshard-sharded-commit-v1':
        raise ValueError('Checkpoint belongs to a different computation')
    if len(common['shards']) != len(shard.boundaries)-1:
        raise ValueError('Checkpoint membership differs')
    step = common['step']
    if type(step) is not int or not 0 <= step <= 2**53-1:
        raise ValueError('Invalid checkpoint cursor')
    folder = Path(home) / f'shard-{step:06d}'
    meta = json.loads((folder/'manifest.json').read_bytes())
    ref = common['shards'][shard.rank]
    if (identity(meta) != ref['manifest'] or meta['binding'] != binding or meta['step'] != step
            or meta['rank'] != shard.rank or meta['boundaries'] != list(shard.boundaries)):
        raise ValueError('Wrong shard, version or manifest')
    owned = dict(shard.named_owned_parameters())
    if set(owned) != set(meta['tensors']):
        raise ValueError('Checkpoint tensor ownership differs')
    if optimizer is not None:
        actual = optimizer_groups(shard, optimizer)
        for left, right in zip(actual, meta['optimizer_groups']):
            # LR is scheduled state; all other optimizer semantics are fixed.
            if identity({k: v for k, v in left.items() if k != 'lr'}) != identity({k: v for k, v in right.items() if k != 'lr'}):
                raise ValueError('Optimizer configuration differs')
        if len(actual) != len(meta['optimizer_groups']):
            raise ValueError('Optimizer group count differs')
        for group, saved in zip(optimizer.param_groups, meta['optimizer_groups']):
            group['lr'] = saved['lr']
    with torch.no_grad():
        for name, filename in meta['tensors'].items():
            if Path(filename).name != filename or filename not in meta['files']:
                raise ValueError('Unsafe checkpoint tensor path')
            path, parameter = folder/filename, owned[name]
            if sha256(path) != meta['files'][filename]['sha256']:
                raise ValueError('Checkpoint tensor is missing or corrupted')
            values = load_file(path, device='cpu')
            if set(values) != ({'weight'} if step == 0 else {'weight', 'step', 'exp_avg', 'exp_avg_sq'}):
                raise ValueError('Incomplete optimizer checkpoint')
            if step and (values['step'].numel() != 1 or float(values['step']) != step):
                raise ValueError('Optimizer step differs from the global cursor')
            if values['weight'].shape != parameter.shape or values['weight'].dtype != parameter.dtype:
                raise ValueError('Checkpoint weight shape or dtype differs')
            parameter.copy_(values['weight'])
            if optimizer is not None:
                optimizer.state.pop(parameter, None)
                if 'step' in values:
                    for key in ('exp_avg', 'exp_avg_sq'):
                        if values[key].shape != parameter.shape or values[key].dtype != parameter.dtype:
                            raise ValueError('Checkpoint moment shape or dtype differs')
                    optimizer.state[parameter] = {'step': values['step'].cpu(),
                        'exp_avg': values['exp_avg'].to(shard.device_name),
                        'exp_avg_sq': values['exp_avg_sq'].to(shard.device_name)}
            del values
    rng_file = folder/'rng.safetensors'
    if sha256(rng_file) != meta['files']['rng.safetensors']['sha256']:
        raise ValueError('Checkpoint RNG state is corrupted')
    if restore_rng:
        rng = load_file(rng_file, device='cpu')
        torch.set_rng_state(rng['torch_cpu'])
        if shard.device_name == 'cuda':
            torch.cuda.set_rng_state(rng['torch_cuda'])
        random.setstate(tuples(meta['python_rng']))
        np.random.set_state((meta['numpy_rng'][0], rng['numpy'].numpy().astype(np.uint32), *meta['numpy_rng'][1:]))
    return step
