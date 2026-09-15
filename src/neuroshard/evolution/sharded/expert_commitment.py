"""Hash the actual tail and Adam state without persisting every replay snapshot.

The returned roots match ordinary Safetensors checkpoints. A commitment does
not imply that every intermediate object is currently available to another
peer; a serving/availability protocol must supply or reproduce those bytes.
"""
import hashlib
import json
import struct
import sys

import numpy as np
import torch

from .. import expert_checkpoint as codec
from ..reference_data import identity


def tensor_commitment(values):
    """Hash this profile's exact F32 Safetensors encoding without copying it.

    Field names, dtype and metadata are deliberately restricted. Compatibility
    tests compare this byte stream with the pinned Safetensors writer, including
    complete weight/Adam checkpoints. This is a storage optimization only.
    """
    if (sys.byteorder != 'little' or set(values) not in (
            {'weight'}, {'weight', 'step', 'exp_avg', 'exp_avg_sq'})):
        raise ValueError('Require the fixed little-endian weight/Adam encoding')
    header, arrays, offset = {}, [], 0
    for key, value in sorted(values.items()):
        if value.dtype != torch.float32 or value.device.type != 'cpu' or not value.is_contiguous():
            raise ValueError('Commit contiguous CPU float32 arrays')
        array = value.numpy()
        if not bool(np.isfinite(array).all()):
            raise ValueError('Invalid finite weight or Adam moment')
        header[key] = {'dtype': 'F32', 'shape': list(value.shape),
                       'data_offsets': [offset, offset + array.nbytes]}
        offset += array.nbytes
        arrays.append(array)
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    digest = hashlib.sha256(struct.pack('<Q', len(encoded)) + encoded)
    for array in arrays:
        digest.update(memoryview(array).cast('B'))
    return {'sha256': digest.hexdigest(), 'bytes': 8 + len(encoded) + offset}


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
            if (value.dtype != torch.float32
                    or (key != 'step' and value.shape != parameter.shape)):
                raise ValueError('Invalid finite weight or Adam moment')
        tensors[name] = {**tensor_commitment(values),
                         'shape': list(parameter.shape), 'optimizer_step': step}
        del values
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
