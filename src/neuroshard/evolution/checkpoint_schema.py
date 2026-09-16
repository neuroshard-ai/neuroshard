"""Existing portable checkpoint metadata validation without neural imports."""
import re

from .reference_data import identity
from .schema import root, integer

COMMIT_FIELDS = {'format', 'job', 'step', 'config', 'optimizer', 'boundaries',
                 'tensors', 'shards', 'parent', 'transition', 'state_root'}


def validate(value):
    if set(value) != COMMIT_FIELDS or value['format'] != 'neuroshard-portable-shards-v1':
        raise ValueError('Invalid portable checkpoint schema')
    root(value['job'])
    integer(value['step'], 0, 2**24-1)
    state = {k: value[k] for k in ['format', 'job', 'step', 'config', 'optimizer', 'tensors']}
    if root(value['state_root']) != identity(state):
        raise ValueError('Portable learned-state commitment differs')
    boundaries = value['boundaries']
    if not isinstance(boundaries, list) or not 3 <= len(boundaries) <= 513:
        raise ValueError('Invalid shard layout')
    if any(type(n) is not int for n in boundaries) or boundaries[0] != 0 or any(a >= b for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError('Invalid layer boundaries')
    if boundaries[-1] != value['config']['num_hidden_layers'] or len(value['shards']) != len(boundaries)-1:
        raise ValueError('Incomplete checkpoint owners')
    for digest in value['shards']:
        root(digest)
    if not isinstance(value['optimizer'], list) or not 1 <= len(value['optimizer']) <= 16:
        raise ValueError('Invalid Adam groups')
    if any(not isinstance(group, dict) for group in value['optimizer']):
        raise ValueError('Invalid Adam group metadata')
    if not isinstance(value['tensors'], dict) or not 1 <= len(value['tensors']) <= 8192:
        raise ValueError('Invalid tensor inventory')
    for name, spec in value['tensors'].items():
        match = re.fullmatch(r'model\.layers\.(\d+)\..+', name)
        if match:
            integer(int(match[1]), 0, boundaries[-1]-1)
        elif name not in ('model.embed_tokens.weight', 'model.norm.weight'):
            raise ValueError('Unsupported parameter name')
        if set(spec) != {'sha256', 'bytes', 'shape', 'born', 'group'}:
            raise ValueError('Invalid tensor metadata')
        root(spec['sha256'])
        integer(spec['bytes'], 1, 16*1024**3)
        integer(spec['born'], 0, value['step'])
        integer(spec['group'], 0, len(value['optimizer'])-1)
        if not isinstance(spec['shape'], list) or not 1 <= len(spec['shape']) <= 2:
            raise ValueError('Invalid parameter shape')
        for dimension in spec['shape']:
            integer(dimension, 1, 262144)
    return value
