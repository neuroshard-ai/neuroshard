"""Compact metadata for an independently trained tail of an immutable parent.

This is a consensus-side codec and transition check. It imports no numerical
runtime and does not verify tensor bytes, execute an update, issue currency or
promote a model. Funded execution verification remains a separate obligation.
"""
import copy
import math
import re

from . import checkpoint_schema
from .reference_data import identity
from .schema import integer, root

FORMAT = 'neuroshard-expert-checkpoint-v1'
INCREMENTAL = 'neuroshard-incremental-shards-v1'
STATE_FIELDS = ('format', 'job', 'step', 'parent', 'parent_layers', 'mode',
                'frozen_layers', 'config', 'recipe', 'tensors')
FIELDS = {'format', 'parent', 'job', 'step', 'split', 'recipe', 'boundaries',
          'tensors', 'checkpoint', 'state_root'}


def shapes(config):
    """The supported tied Llama profile, using metadata and integer arithmetic."""
    if (config['model_type'] != 'llama' or config['tie_word_embeddings'] is not True
            or config['attention_dropout'] != 0 or config['attention_bias'] is not False
            or config['mlp_bias'] is not False or config['hidden_act'] != 'silu'
            or config['rope_scaling'] is not None):
        raise ValueError('Require the supported deterministic tied Llama profile')
    for key, maximum in [('num_hidden_layers', 512), ('hidden_size', 16384),
                         ('intermediate_size', 65536), ('vocab_size', 262144),
                         ('num_attention_heads', 2048), ('num_key_value_heads', 2048)]:
        integer(config[key], 1, maximum)
    if config['num_attention_heads'] % config['num_key_value_heads']:
        raise ValueError('Incomplete grouped attention heads')
    h, m = config['hidden_size'], config['intermediate_size']
    d = config['head_dim'] or h // config['num_attention_heads']
    integer(d, 1, 16384)
    q, kv = config['num_attention_heads'] * d, config['num_key_value_heads'] * d
    expected = {'model.embed_tokens.weight': [config['vocab_size'], h], 'model.norm.weight': [h]}
    block = {'input_layernorm.weight': [h], 'post_attention_layernorm.weight': [h],
             'self_attn.q_proj.weight': [q, h], 'self_attn.k_proj.weight': [kv, h],
             'self_attn.v_proj.weight': [kv, h], 'self_attn.o_proj.weight': [h, q],
             'mlp.gate_proj.weight': [m, h], 'mlp.up_proj.weight': [m, h],
             'mlp.down_proj.weight': [h, m]}
    for layer in range(config['num_hidden_layers']):
        expected.update({f'model.layers.{layer}.{name}': shape for name, shape in block.items()})
    return dict(sorted(expected.items()))


def parent_records(parent):
    checkpoint_schema.validate(parent)
    expected = shapes(parent['config'])
    if set(parent['tensors']) != set(expected) or any(
            spec['shape'] != expected[name] for name, spec in parent['tensors'].items()):
        raise ValueError('Incomplete or misshaped frozen parent')
    return {name: {key: spec[key] for key in ('sha256', 'bytes', 'shape')} |
            {'optimizer_step': parent['step'] - spec['born']}
            for name, spec in parent['tensors'].items()}


def owner(name, boundaries):
    if name in ('model.embed_tokens.weight', 'model.norm.weight'):
        return 0
    layer = int(name.split('.')[2])
    return next(i for i, (a, b) in enumerate(zip(boundaries, boundaries[1:])) if a <= layer < b)


def reconstruct(parent, value):
    """Derive frozen references and complete shard commitments without tensors."""
    if not isinstance(value, dict) or set(value) != FIELDS or value['format'] != FORMAT:
        raise ValueError('Invalid compact expert schema')
    inherited = parent_records(parent)
    if value['parent'] != identity(parent):
        raise ValueError('Expert belongs to a different immutable parent')
    root(value['job'])
    integer(value['step'], 0, 2**24 - 1)
    layers = parent['config']['num_hidden_layers']
    integer(value['split'], 1, layers - 1)
    boundaries = value['boundaries']
    if (not isinstance(boundaries, list) or not 3 <= len(boundaries) <= 513
            or any(type(n) is not int for n in boundaries) or boundaries[0] != 0
            or boundaries[-1] != layers or boundaries[-2] != value['split']
            or any(a >= b for a, b in zip(boundaries, boundaries[1:]))):
        raise ValueError('Require a separate terminal expert owner')
    expected = {name for name in inherited if owner(name, boundaries) == len(boundaries) - 2}
    if not isinstance(value['tensors'], dict) or set(value['tensors']) != expected:
        raise ValueError('Transmit only complete active expert tensors; frozen references are derived')
    for name, spec in value['tensors'].items():
        if (set(spec) != {'sha256', 'bytes', 'shape', 'optimizer_step'}
                or spec['shape'] != inherited[name]['shape']):
            raise ValueError('Invalid active tensor schema or shape')
        root(spec['sha256'])
        integer(spec['bytes'], 1, 16 * 1024**3)
        integer(spec['optimizer_step'], 0, 2**24 - 1)
        if spec['optimizer_step'] != value['step']:
            raise ValueError('Active Adam age must equal the actual expert update cursor')
    recipe = value['recipe']
    required = {'steps', 'warmup_steps', 'learning_rate', 'weight_decay', 'clip_norm'}
    if (not isinstance(recipe, dict) or not required <= set(recipe)
            or set(recipe) - required - {'batch_documents', 'seed'}):
        raise ValueError('Require the complete fixed expert optimizer recipe')
    if 'batch_documents' in recipe:
        integer(recipe['batch_documents'], 1, 65536)
    if 'seed' in recipe:
        integer(recipe['seed'], 0, 2**64 - 1)
    integer(recipe['steps'], 1, 2**24 - 1)
    integer(recipe['warmup_steps'], 0, recipe['steps'])
    if (value['step'] > recipe['steps'] or any(type(recipe[k]) not in (int, float)
            or not math.isfinite(recipe[k]) for k in ('learning_rate', 'weight_decay', 'clip_norm'))
            or recipe['learning_rate'] <= 0 or recipe['weight_decay'] < 0 or recipe['clip_norm'] <= 0):
        raise ValueError('Invalid expert optimizer recipe')
    common = {'format': INCREMENTAL, 'job': value['job'], 'step': value['step'],
              'parent': value['parent'], 'parent_layers': layers, 'mode': 'tail-control',
              'frozen_layers': value['split'], 'config': copy.deepcopy(parent['config']),
              'recipe': copy.deepcopy(recipe), 'boundaries': list(boundaries),
              'tensors': dict(sorted((inherited | copy.deepcopy(value['tensors'])).items()))}
    common['shards'] = [identity({**common, 'rank': rank, 'tensors': {
        name: spec for name, spec in common['tensors'].items() if owner(name, boundaries) == rank}})
        for rank in range(len(boundaries) - 1)]
    common['state_root'] = identity({key: common[key] for key in STATE_FIELDS})
    return common


def unpack(parent, value):
    common = reconstruct(parent, value)
    if root(value['checkpoint']) != identity(common) or root(value['state_root']) != common['state_root']:
        raise ValueError('Compact metadata must reconstruct the exact numerical checkpoint')
    return common


def pack(parent, common):
    if common.get('format') != INCREMENTAL or common.get('mode') != 'tail-control':
        raise ValueError('This profile represents separately trained existing tails')
    split = common['frozen_layers']
    value = {'format': FORMAT, 'parent': identity(parent), 'job': common['job'], 'step': common['step'],
             'split': split, 'recipe': copy.deepcopy(common['recipe']), 'boundaries': list(common['boundaries']),
             'tensors': {name: copy.deepcopy(spec) for name, spec in common['tensors'].items()
                         if re.fullmatch(r'model\.layers\.(\d+)\..+', name) and int(name.split('.')[2]) >= split},
             'checkpoint': identity(common), 'state_root': common['state_root']}
    if unpack(parent, value) != common:
        raise ValueError('Common checkpoint changed a derived frozen reference or shard commitment')
    return value


def transition(parent, before, after, max_steps=4):
    unpack(parent, before)
    unpack(parent, after)
    integer(max_steps, 1, 4)
    integer(after['step'] - before['step'], 1, max_steps)
    if any(before[key] != after[key] for key in ('parent', 'job', 'split', 'recipe', 'boundaries')):
        raise ValueError('A training window cannot replace its job, recipe or ownership')
    return after['step'] - before['step']


def work_identity(parent, before, batch_root, numerical_profile):
    """Identify one prescribed update independently of graph/job/owner labels.

    The numerical auditor must derive batch_root from canonical tensors actually
    consumed by the update, excluding document IDs and descriptive metadata.
    Object hashes identify this serialization profile, not arbitrary numerical
    equivalence across different file formats or floating-point profiles.
    """
    common = unpack(parent, before)
    root(batch_root)
    root(numerical_profile)
    return identity({'domain': FORMAT + '/update', 'step': before['step'],
        'parameters': {name: spec['sha256'] for name, spec in common['tensors'].items()},
        'batch': batch_root, 'recipe': before['recipe'], 'numerical_profile': numerical_profile})
