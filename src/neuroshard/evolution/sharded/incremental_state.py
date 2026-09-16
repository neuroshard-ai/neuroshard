"""Research checkpoints for a frozen parent and newly trained model blocks.

Each parameter retains its actual Adam update count. Freezing a parameter does
not advance that count or rewrite its immutable tensor object. The explicit
tail-control comparison starts fresh Adam on its trainable existing tail while
retaining the original parent objects. This format is not accepted by the
existing native portable-work profile.
"""
import errno
import math
import os
from pathlib import Path
import shutil
import uuid

import torch
from safetensors.torch import load_file
from transformers import LlamaConfig

from ..reference_data import identity, save, sha256
from . import checkpoint, incremental, portable
from .model import owner

FORMAT = 'neuroshard-incremental-shards-v1'
FIELDS = ('format', 'job', 'step', 'parent', 'parent_layers', 'mode', 'frozen_layers', 'config', 'recipe', 'tensors')


def state_root(common):
    return identity({key: common[key] for key in FIELDS})


def records(parent):
    if parent['format'] == portable.FORMAT:
        portable.validate(parent)
        return {name: {key: spec[key] for key in ('sha256', 'bytes', 'shape')} |
                {'optimizer_step': parent['step'] - spec['born']}
                for name, spec in parent['tensors'].items()}
    if parent['format'] != FORMAT or parent['state_root'] != state_root(parent):
        raise ValueError('Invalid frozen parent commitment')
    if set(parent['tensors']) != set(portable.shapes(LlamaConfig(**parent['config']))):
        raise ValueError('Incomplete frozen parent')
    return parent['tensors']


def validate(common, parent):
    inherited = records(parent)
    if (common['format'] != FORMAT or common['state_root'] != state_root(common)
            or common['parent'] != identity(parent)):
        raise ValueError('Wrong incremental state or frozen parent')
    config = LlamaConfig(**common['config'])
    previous = parent['config']['num_hidden_layers']
    if (type(common['parent_layers']) is not int or common['parent_layers'] != previous
            or {**common['config'], 'num_hidden_layers': previous} != parent['config']):
        raise ValueError('Only model depth may change')
    if common['mode'] == 'append':
        if not 1 <= config.num_hidden_layers - previous <= 16 or common['frozen_layers'] != previous:
            raise ValueError('Growth may only append one to sixteen blocks')
    elif common['mode'] == 'tail-control':
        if config.num_hidden_layers != previous or not 0 < common['frozen_layers'] < previous:
            raise ValueError('The control fine-tunes an existing tail without growth')
    else:
        raise ValueError('Unsupported incremental experiment mode')
    active = incremental.trainable_names(config, common['frozen_layers'])
    if type(common['step']) is not int or not 0 <= common['step'] < 2**24:
        raise ValueError('Invalid cohort update cursor')
    expected = portable.shapes(config)
    boundaries = common['boundaries']
    if (not isinstance(boundaries, list) or not 3 <= len(boundaries) <= 513
            or any(type(n) is not int for n in boundaries)
            or boundaries[0] != 0 or boundaries[-1] != config.num_hidden_layers
            or any(a >= b for a, b in zip(boundaries, boundaries[1:]))):
        raise ValueError('Invalid complete model partition')
    if owner(f'model.layers.{common["frozen_layers"]}.input_layernorm.weight', boundaries) == 0:
        raise ValueError('Added blocks need a separate owner from the frozen head')
    if (not isinstance(common['shards'], list) or len(common['shards']) != len(boundaries) - 1
            or any(not isinstance(root, str) or len(root) != 64
                   or any(c not in '0123456789abcdef' for c in root) for root in common['shards'])):
        raise ValueError('Invalid complete shard commitments')
    if set(common['tensors']) != set(expected):
        raise ValueError('Incomplete incremental tensor inventory')
    for name, spec in common['tensors'].items():
        if (set(spec) != {'sha256', 'bytes', 'shape', 'optimizer_step'}
                or spec['shape'] != expected[name]
                or type(spec['bytes']) is not int or spec['bytes'] <= 0
                or type(spec['optimizer_step']) is not int or not 0 <= spec['optimizer_step'] < 2**24
                or not isinstance(spec['sha256'], str) or len(spec['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in spec['sha256'])):
            raise ValueError('Invalid incremental tensor record')
        if name in inherited and name not in active:
            if spec != inherited[name]:
                raise ValueError('Frozen parent state changed')
        elif spec['optimizer_step'] != common['step']:
            raise ValueError('Added parameter Adam age differs from the cohort cursor')
    recipe = common['recipe']
    if (type(recipe['steps']) is not int or not 0 < recipe['steps'] < 2**24 or common['step'] > recipe['steps']
            or type(recipe['warmup_steps']) is not int or not 0 <= recipe['warmup_steps'] <= recipe['steps']
            or not all(type(recipe[key]) in (int, float) and math.isfinite(recipe[key])
                       for key in ('learning_rate', 'weight_decay', 'clip_norm'))
            or recipe['learning_rate'] <= 0 or recipe['clip_norm'] <= 0 or recipe['weight_decay'] < 0):
        raise ValueError('Invalid incremental optimizer recipe')
    return config


def tensor_values(path, spec):
    if path.stat().st_size != spec['bytes'] or sha256(path) != spec['sha256']:
        raise ValueError('Missing or corrupted incremental tensor')
    values = load_file(path, device='cpu')
    age = spec['optimizer_step']
    if set(values) != ({'weight'} if age == 0 else {'weight', 'step', 'exp_avg', 'exp_avg_sq'}):
        raise ValueError('Incomplete incremental Adam state')
    if age and (values['step'].dtype != torch.float32 or values['step'].numel() != 1
                or float(values['step']) != age):
        raise ValueError('Wrong incremental Adam age')
    for key, value in values.items():
        if key != 'step' and (list(value.shape) != spec['shape'] or value.dtype != torch.float32
                              or not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid incremental weight or moment')
    return values


def write(home, shard, optimizer, parent, frozen_sources, job, step, recipe,
          mode='append', frozen_layers=None, initial_expert=None, initial_objects=None):
    inherited = records(parent)
    frozen_layers = parent['config']['num_hidden_layers'] if frozen_layers is None else frozen_layers
    active = incremental.trainable_names(shard.config, frozen_layers)
    if initial_expert is not None:
        from .. import expert_checkpoint
        expert_checkpoint.unpack(parent, initial_expert)
        if (initial_objects is None or step != 0 or mode != 'tail-control' or initial_expert['step'] == 0
                or initial_expert['boundaries'] != list(shard.boundaries)
                or initial_expert['split'] != frozen_layers):
            raise ValueError('Initialize only an identical trained expert with fresh Adam')
    pairs = shard.named_owned_parameters()
    owned_frozen = {name for name, _ in pairs if name in inherited}
    if set(frozen_sources) != owned_frozen:
        raise ValueError('Provide exactly the owned frozen tensor objects')
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    pending = home / ('.writing-' + uuid.uuid4().hex)
    pending.mkdir()

    def stage(source, spec):
        target = portable.tensor_path(pending, spec['sha256'])
        if not target.exists():
            try:
                os.link(source, target)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.copyfile(source, target)
            if sha256(target) != spec['sha256']:
                raise ValueError('Frozen object changed while staging checkpoint')

    tensors = {}
    for name, parameter in pairs:
        if parameter.requires_grad != (name in active):
            raise ValueError('Wrong trainable mask for incremental checkpoint')
        if name in inherited:
            spec = inherited[name]
            source = Path(frozen_sources[name])
            values = tensor_values(source, spec)
            if name not in active and (parameter.grad is not None or not torch.equal(parameter.detach().cpu(), values['weight'])):
                raise ValueError('Frozen parameter changed during incremental learning')
            if name in active and step == 0:
                if initial_expert is not None:
                    seed_spec = initial_expert['tensors'][name]
                    seed_source = portable.tensor_path(initial_objects, seed_spec['sha256'])
                    values = tensor_values(seed_source, seed_spec)
                    stage(seed_source, seed_spec)
                if not torch.equal(parameter.detach().cpu(), values['weight']):
                    raise ValueError('Control must start from exactly the inherited weights')
            stage(source, spec)
            del values
            if name not in active:
                tensors[name] = spec
                continue
        if optimizer is None or not parameter.requires_grad:
            raise ValueError('Added blocks need their declared optimizer')
        state = optimizer.state.get(parameter, {})
        if (step == 0 and state) or (step > 0 and (set(state) != {'step', 'exp_avg', 'exp_avg_sq'}
                or state['step'].numel() != 1 or float(state['step']) != step)):
            raise ValueError('Added parameter Adam age differs from the cohort cursor')
        values = {'weight': parameter.detach().cpu().contiguous(),
                  **{key: value.detach().cpu().contiguous() for key, value in state.items()}}
        temporary = pending / 'tensor.pending'
        spec = checkpoint.tensor_file(temporary, values)
        target = portable.tensor_path(pending, spec['sha256'])
        os.replace(temporary, target)
        tensors[name] = {**spec, 'shape': list(parameter.shape), 'optimizer_step': step}
        tensor_values(target, tensors[name])
        del values
    meta = {'format': FORMAT, 'job': job, 'step': step, 'parent': identity(parent),
            'parent_layers': parent['config']['num_hidden_layers'],
            'mode': mode, 'frozen_layers': frozen_layers,
            'config': portable.configuration(shard.config), 'recipe': dict(recipe),
            'boundaries': list(shard.boundaries), 'rank': shard.rank, 'tensors': tensors}
    save(pending / 'manifest.json', meta)
    checkpoint.sync_directory(pending)
    destination = portable.directory(home, step)
    if destination.exists():
        raise ValueError('Preserve the existing incremental checkpoint')
    os.replace(pending, destination)
    checkpoint.sync_directory(home)
    return meta


def assemble(metas, parent):
    if not metas:
        raise ValueError('Missing incremental partitions')
    first = metas[0]
    common = {key: first[key] for key in (*FIELDS[:-1], 'boundaries')}
    if len(metas) != len(common['boundaries']) - 1:
        raise ValueError('Missing incremental partitions')
    tensors = {}
    for rank, meta in enumerate(metas):
        if meta['rank'] != rank or any(identity(meta[key]) != identity(value) for key, value in common.items()):
            raise ValueError('Incremental partition disagreement')
        if set(tensors) & set(meta['tensors']) or any(owner(name, common['boundaries']) != rank for name in meta['tensors']):
            raise ValueError('Duplicate or incorrect incremental ownership')
        tensors.update(meta['tensors'])
    common.update(tensors=dict(sorted(tensors.items())), shards=[identity(meta) for meta in metas])
    common['state_root'] = state_root(common)
    validate(common, parent)
    return common


def commit(home, shard, optimizer, wire, parent, frozen_sources, job, step, recipe,
           mode='append', frozen_layers=None):
    meta = write(home, shard, optimizer, parent, frozen_sources, job, step, recipe, mode, frozen_layers)
    common = assemble(wire.exchange(meta), parent)
    root = identity(common)
    if any(value != root for value in wire.exchange(root)):
        raise ValueError('Incremental checkpoint disagreement')
    save(Path(home) / f'commit-{step:06d}.json', common)
    return common


def initialize(shard, parent, objects, mode, frozen_layers):
    """Load owned parent objects and initialize appended identity blocks.

    ``objects`` contains content-addressed parent tensors, including the final
    parent block for owners of newly appended blocks. No full model is built.
    Original optimizer objects are retained on disk, never allocated on frozen
    workers. The training optimizer is configured separately and starts empty.
    """
    inherited = records(parent)
    previous = parent['config']['num_hidden_layers']
    config = portable.configuration(shard.config)
    if {**config, 'num_hidden_layers': previous} != parent['config']:
        raise ValueError('Incremental initialization only changes model depth')
    if mode == 'append':
        if frozen_layers != previous or not 1 <= config['num_hidden_layers'] - previous <= 16:
            raise ValueError('Invalid appended-block initialization')
    elif mode == 'tail-control':
        if config['num_hidden_layers'] != previous or not 0 < frozen_layers < previous:
            raise ValueError('Invalid existing-tail initialization')
    else:
        raise ValueError('Unsupported incremental initialization')
    frozen_sources = {}
    with torch.no_grad():
        for name, parameter in shard.named_owned_parameters():
            source_name = name
            if name not in inherited:
                parts = name.split('.')
                if parts[:2] != ['model', 'layers'] or int(parts[2]) < previous:
                    raise ValueError('Only appended layer tensors may be initialized')
                parts[2] = str(previous - 1)
                source_name = '.'.join(parts)
            spec = inherited[source_name]
            path = portable.tensor_path(objects, spec['sha256'])
            values = tensor_values(path, spec)
            if list(parameter.shape) != spec['shape']:
                raise ValueError('Owned parameter shape differs from the parent template')
            parameter.copy_(values['weight'])
            if name in inherited:
                frozen_sources[name] = path
            elif name.endswith(('self_attn.o_proj.weight', 'mlp.down_proj.weight')):
                parameter.zero_()
            del values
    return frozen_sources


def load(home, shard, optimizer, common, parent, job, recipe, restore_optimizer=True, *, initial_expert=None):
    import json
    validate(common, parent)
    if initial_expert is not None:
        from .. import expert_checkpoint
        expert_checkpoint.unpack(parent, initial_expert)
        if (common['step'] != 0 or initial_expert['step'] == 0
                or initial_expert['boundaries'] != common['boundaries']
                or initial_expert['split'] != common['frozen_layers']):
            raise ValueError('Restore only the declared initial accepted expert weights')
    if (common['job'] != job or common['recipe'] != recipe
            or common['config'] != portable.configuration(shard.config)
            or common['boundaries'] != list(shard.boundaries)):
        raise ValueError('Wrong incremental job, optimizer recipe or layout')
    if not restore_optimizer and (optimizer is not None or any(p.requires_grad for p in shard.parameters())):
        raise ValueError('Evaluation restores require frozen weights and no optimizer')
    folder = portable.directory(home, common['step'])
    meta = json.loads((folder / 'manifest.json').read_bytes())
    if identity(meta) != common['shards'][shard.rank] or meta['rank'] != shard.rank:
        raise ValueError('Wrong incremental partition commitment')
    pairs = dict(shard.named_owned_parameters())
    if set(meta['tensors']) != set(pairs):
        raise ValueError('Incomplete owned incremental checkpoint')
    inherited = records(parent)
    active = incremental.trainable_names(shard.config, common['frozen_layers'])
    frozen_sources = {}
    with torch.no_grad():
        for name, parameter in pairs.items():
            spec = meta['tensors'][name]
            if spec != common['tensors'][name]:
                raise ValueError('Incremental tensor differs from global commitment')
            path = portable.tensor_path(folder, spec['sha256'])
            values = tensor_values(path, spec)
            parameter.copy_(values['weight'])
            if name not in active:
                if parameter.requires_grad:
                    raise ValueError('Frozen parent marked trainable')
            else:
                if restore_optimizer and (optimizer is None or not parameter.requires_grad):
                    raise ValueError('Added blocks need their declared optimizer')
                if restore_optimizer:
                    optimizer.state.pop(parameter, None)
                if restore_optimizer and spec['optimizer_step']:
                    optimizer.state[parameter] = {'step': values['step'].cpu(),
                        'exp_avg': values['exp_avg'].to(shard.device_name),
                        'exp_avg_sq': values['exp_avg_sq'].to(shard.device_name)}
            if name in inherited:
                source = portable.tensor_path(folder, inherited[name]['sha256'])
                if name in active:
                    original = tensor_values(source, inherited[name])
                    if initial_expert is not None:
                        seed_spec = initial_expert['tensors'][name]
                        original = tensor_values(portable.tensor_path(folder, seed_spec['sha256']), seed_spec)
                    if common['step'] == 0 and not torch.equal(values['weight'], original['weight']):
                        raise ValueError('Control must start from exactly the inherited weights')
                    del original
                frozen_sources[name] = source
            del values
    return frozen_sources
