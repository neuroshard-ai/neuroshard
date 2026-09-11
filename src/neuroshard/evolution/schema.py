"""Tensor-free validation of model and worker commitments."""
import math
import re

MAX_PARAMETERS = 48_000_000


def block_order(name):
    return int(name.removeprefix('block_'))


def root(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Invalid artifact root')
    return value


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError('Integer outside execution bounds')
    return value


def shapes(config, component):
    h, v, m = config['hidden_size'], config['vocab_size'], config['intermediate_size']
    kv = h // config['num_attention_heads'] * config['num_key_value_heads']
    if component == 'embed':
        return {'weight': [v, h]}
    if component == 'norm':
        return {'weight': [h]}
    return {
        'input_layernorm.weight': [h], 'post_attention_layernorm.weight': [h],
        'self_attn.q_proj.weight': [h, h], 'self_attn.o_proj.weight': [h, h],
        'self_attn.k_proj.weight': [kv, h], 'self_attn.v_proj.weight': [kv, h],
        'mlp.gate_proj.weight': [m, h], 'mlp.up_proj.weight': [m, h],
        'mlp.down_proj.weight': [h, m],
    }


def model(value):
    if value['format'] != 'neuroshard-model-v1':
        raise ValueError('Unsupported model format')
    c = value['config']
    for field, low, high in (
        ('hidden_size', 8, 4096), ('vocab_size', 16, 131072),
        ('intermediate_size', 8, 16384), ('num_hidden_layers', 1, 1024),
        ('num_attention_heads', 1, 128), ('num_key_value_heads', 1, 128),
        ('max_position_embeddings', 256, 131072),
    ):
        integer(c[field], low, high)
    if (c['hidden_size'] % c['num_attention_heads'] or
            (c['hidden_size'] // c['num_attention_heads']) % 2 or
            c['num_attention_heads'] % c['num_key_value_heads']):
        raise ValueError('Invalid attention dimensions')
    for name, low, high in [('rms_norm_eps', 1e-8, .01), ('rope_theta', 100, 1e9)]:
        if type(c[name]) not in (float, int) or not math.isfinite(c[name]) or not low <= c[name] <= high:
            raise ValueError('Invalid normalization or rotary configuration')
    expected = {'embed', 'norm'} | {f'block_{i:03}' for i in range(c['num_hidden_layers'])}
    if set(value['components']) != expected:
        raise ValueError('Model component coverage differs from architecture')
    total = 0
    for name, component in value['components'].items():
        root(component['root'])
        count = sum(math.prod(s) for s in shapes(c, name).values())
        if type(component['parameters']) is not int or component['parameters'] != count or count > MAX_PARAMETERS:
            raise ValueError('Component size differs from architecture or exceeds worker limit')
        total += count
    if type(value['parameters']) is not int or value['parameters'] != total:
        raise ValueError('Incorrect model parameter count')
    if value['parent'] is not None:
        root(value['parent'])
    if 'tokenizer_root' in value:
        root(value['tokenizer_root'])
    return value


def partition(value, assignment):
    model(value)
    names = assignment['components']
    if (not isinstance(names, list) or not names or len(names) != len(set(names)) or
            any(n not in value['components'] for n in names)):
        raise ValueError('Invalid partition components')
    integer(assignment['capacity'], 1, MAX_PARAMETERS)
    integer(assignment['parameters'], 1, MAX_PARAMETERS)
    count = sum(value['components'][n]['parameters'] for n in names)
    if assignment['parameters'] != count or count > assignment['capacity']:
        raise ValueError('Invalid partition capacity')
    blocks = [n for n in names if n.startswith('block_')]
    if blocks != sorted(blocks,key=block_order):
        raise ValueError('Partition blocks are not in execution order')
    if ('embed' in names) != ('norm' in names):
        raise ValueError('Tied embedding and final norm must share a worker')
    return assignment
