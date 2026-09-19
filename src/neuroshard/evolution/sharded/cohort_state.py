"""Checkpoint one new learner while referencing the immutable parent prefix.

Only the new tail is written. Prefix manifests are derived from the already
validated parent inventory; they do not claim that those owners trained again
or that any public consensus accepted this research checkpoint.
"""
from pathlib import Path
import math

import torch

from .. import expert_checkpoint
from ..reference_data import identity, save
from . import incremental_state, portable
from .model import Partition, owner


def commit_tail(home, shard, optimizer, parent, objects, job, step, recipe, split, *, initial_expert=None):
    if (shard.rank != 3 or len(shard.boundaries) != 5 or shard.boundaries[-2] != split
            or shard.config.num_hidden_layers != parent['config']['num_hidden_layers']):
        raise ValueError('Commit exactly one separate inherited tail')
    inherited = incremental_state.records(parent)
    names = {name for name, _ in shard.named_owned_parameters()}
    sources = {name: portable.tensor_path(objects, inherited[name]['sha256']) for name in names}
    meta = incremental_state.write(home, shard, optimizer, parent, sources, job, step,
                                   recipe, mode='tail-control', frozen_layers=split,
                                   initial_expert=initial_expert, initial_objects=objects)
    frozen = []
    for rank in range(3):
        item = {key: value for key, value in meta.items() if key not in ('rank', 'tensors')}
        item.update(rank=rank, tensors={name: spec for name, spec in inherited.items()
                                       if owner(name, shard.boundaries) == rank})
        frozen.append(item)
    common = incremental_state.assemble([*frozen, meta], parent)
    active = {name for name in common['tensors'] if owner(name, shard.boundaries) == 3}
    if active != names or any(common['tensors'][name] != spec for name, spec in inherited.items()
                              if name not in active):
        raise ValueError('New expert changed an inherited prefix or its Adam age')
    save(Path(home) / f'commit-{step:06d}.json', common)
    save(portable.directory(home, step) / 'prefix-references.json', {
        'parent': identity(parent), 'derived_manifests': frozen,
        'scope': 'References to unchanged parent objects, not new prefix computation or consensus signatures.'})
    return common


def initialize_tail(shard, parent, objects, split, initial_expert=None):
    """Load a parent tail or the exact accepted expert weights, without Adam."""
    sources = incremental_state.initialize(shard, parent, objects, 'tail-control', split)
    if initial_expert is None:
        return sources
    expert_checkpoint.unpack(parent, initial_expert)
    if (initial_expert['step'] == 0 or initial_expert['split'] != split
            or initial_expert['boundaries'] != list(shard.boundaries)
            or set(initial_expert['tensors']) != {name for name, _ in shard.named_owned_parameters()}):
        raise ValueError('Warm start must cover the exact trained tail and frozen parent')
    with torch.no_grad():
        for name, parameter in shard.named_owned_parameters():
            spec = initial_expert['tensors'][name]
            path = portable.tensor_path(objects, spec['sha256'])
            if path.is_symlink():
                raise ValueError('An accepted expert requires regular owned tensor objects')
            values = incremental_state.tensor_values(path, spec)
            parameter.copy_(values['weight'])
            del values
    # Checkpoint writes after an update still reference the unchanged parent;
    # the warm start is separately committed in the job's initial state.
    return sources


def frozen_reference(shard, parent, objects, split, checkpoint, parameter_limit):
    """Load an explicitly counted, read-only accepted tail on the last parent owner."""
    expert_checkpoint.unpack(parent, checkpoint)
    if (checkpoint['step'] == 0 or checkpoint['split'] != split
            or checkpoint['boundaries'] != [*parent['boundaries'][:-1], split, parent['boundaries'][-1]]):
        raise ValueError('Replay reference must be the complete compatible accepted expert')
    if shard.rank != 2:
        return None
    extra = sum(math.prod(spec['shape']) for spec in checkpoint['tensors'].values())
    if type(parameter_limit) is not int or shard.resident_parameters + extra > parameter_limit:
        raise ValueError('Parent partition and frozen expert reference exceed the owner limit')
    teacher = Partition(shard.config, checkpoint['boundaries'], 3, shard.device_name, parameter_limit)
    with torch.no_grad():
        for name, parameter in teacher.named_owned_parameters():
            spec = checkpoint['tensors'][name]
            path = portable.tensor_path(objects, spec['sha256'])
            if path.is_symlink():
                raise ValueError('Accepted reference requires regular owned tensor objects')
            values = incremental_state.tensor_values(path, spec)
            parameter.copy_(values['weight'])
            del values
    return teacher.eval().requires_grad_(False)
