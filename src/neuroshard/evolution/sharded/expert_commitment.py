"""Hash the actual tail and Adam state without persisting every replay snapshot.

The returned roots match ordinary Safetensors checkpoints. A commitment does
not imply that every intermediate object is currently available to another
peer; a serving/availability protocol must supply or reproduce those bytes.
"""
import hashlib

import torch
from safetensors.torch import save as tensor_bytes

from .. import expert_checkpoint as codec
from ..reference_data import identity


def snapshot(shard, optimizer, parent, job, step, recipe):
    if shard.rank != len(shard.boundaries) - 2 or shard.rank == 0:
        raise ValueError('Commit one complete separately owned terminal expert')
    names = dict(shard.named_owned_parameters())
    if any(not parameter.requires_grad for parameter in names.values()):
        raise ValueError('The actual trainable tail must retain its optimizer')
    if sorted(id(p) for group in optimizer.param_groups for p in group['params']) != sorted(id(p) for p in names.values()):
        raise ValueError('Optimizer must cover exactly the active owner')
    versions = tuple(p._version for p in names.values())
    tensors = {}
    for name, parameter in names.items():
        state = optimizer.state.get(parameter, {})
        if ((step == 0 and state) or (step > 0 and (set(state) != {'step', 'exp_avg', 'exp_avg_sq'}
                or state['step'].dtype != torch.float32 or state['step'].numel() != 1
                or float(state['step']) != step))):
            raise ValueError('Snapshot must preserve the actual optimizer update count')
        values = {'weight': parameter.detach().cpu().contiguous(),
                  **{key: value.detach().cpu().contiguous() for key, value in state.items()}}
        for key, value in values.items():
            if (value.dtype != torch.float32 or not bool(torch.isfinite(value).all())
                    or (key != 'step' and value.shape != parameter.shape)):
                raise ValueError('Invalid finite weight or Adam moment')
        raw = tensor_bytes(values)
        tensors[name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
                         'shape': list(parameter.shape), 'optimizer_step': step}
        del raw, values
    if tuple(p._version for p in names.values()) != versions:
        raise ValueError('Snapshotting changed a model parameter')
    value = {'format': codec.FORMAT, 'parent': identity(parent), 'job': job, 'step': step,
             'split': shard.boundaries[-2], 'recipe': dict(recipe), 'boundaries': list(shard.boundaries),
             'tensors': tensors, 'checkpoint': '0' * 64, 'state_root': '0' * 64}
    common = codec.reconstruct(parent, value)
    value.update(checkpoint=identity(common), state_root=common['state_root'])
    if codec.unpack(parent, value) != common:
        raise ValueError('In-memory commitment failed reconstruction')
    return value
