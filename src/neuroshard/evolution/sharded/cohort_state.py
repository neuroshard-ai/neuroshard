"""Checkpoint one new learner while referencing the immutable parent prefix.

Only the new tail is written. Prefix manifests are derived from the already
validated parent inventory; they do not claim that those owners trained again
or that any public consensus accepted this research checkpoint.
"""
from pathlib import Path

from ..reference_data import identity, save
from . import incremental_state, portable
from .model import owner


def commit_tail(home, shard, optimizer, parent, objects, job, step, recipe, split):
    if (shard.rank != 3 or len(shard.boundaries) != 5 or shard.boundaries[-2] != split
            or shard.config.num_hidden_layers != parent['config']['num_hidden_layers']):
        raise ValueError('Commit exactly one separate inherited tail')
    inherited = incremental_state.records(parent)
    names = {name for name, _ in shard.named_owned_parameters()}
    sources = {name: portable.tensor_path(objects, inherited[name]['sha256']) for name in names}
    meta = incremental_state.write(home, shard, optimizer, parent, sources, job, step,
                                   recipe, mode='tail-control', frozen_layers=split)
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
