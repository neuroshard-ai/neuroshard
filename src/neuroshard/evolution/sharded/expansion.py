"""Append residual blocks at a committed cursor without changing old tensors.

Zero output projections make each added block an identity before its first
update. New parameters have their own Adam age; existing moments are retained.
This changes capacity, and is not evidence that the larger model is better.
"""
import copy
import os
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import LlamaConfig

from ..reference_data import identity, sha256
from . import checkpoint, portable


def materialize(common, additional, sources, destination):
    config = portable.validate(common)
    if type(additional) is not int or not 1 <= additional <= 16:
        raise ValueError('Append between one and sixteen blocks per transition')
    prefix = f'model.layers.{config.num_hidden_layers-1}.'
    template = {n: s for n, s in common['tensors'].items() if n.startswith(prefix)}
    if set(sources) != set(template):
        raise ValueError('Growth requires exactly the last block, without a whole model')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    result = {}
    for name, spec in template.items():
        path = Path(sources[name])
        if sha256(path) != spec['sha256'] or path.stat().st_size != spec['bytes']:
            raise ValueError('Corrupt growth source')
        weight = load_file(path, device='cpu')['weight']
        if list(weight.shape) != spec['shape'] or weight.dtype != torch.float32 or not bool(torch.isfinite(weight).all()):
            raise ValueError('Invalid growth source weight')
        suffix = name[len(prefix):]
        if suffix in ('self_attn.o_proj.weight', 'mlp.down_proj.weight'):
            weight.zero_()
        temporary = destination/'tensor.pending'
        content = checkpoint.tensor_file(temporary, {'weight': weight})
        os.replace(temporary, portable.tensor_path(destination, content['sha256']))
        for index in range(config.num_hidden_layers, config.num_hidden_layers+additional):
            result[f'model.layers.{index}.{suffix}'] = {
                **content, 'shape': spec['shape'], 'born': common['step'], 'group': spec['group']}
        del weight
    checkpoint.sync_directory(destination)
    return result


def propose(common, additional, new_tensors, capacities):
    config = portable.validate(common)
    if type(additional) is not int or not 1 <= additional <= 16:
        raise ValueError('Invalid growth increment')
    config.num_hidden_layers += additional
    expected = set(portable.shapes(config))-set(common['tensors'])
    if set(new_tensors) != expected or any(s['born'] != common['step'] for s in new_tensors.values()):
        raise ValueError('Growth must introduce exactly the new block parameters at this cursor')
    proposal = copy.deepcopy(common)
    proposal.update(config=portable.configuration(config),
                    boundaries=portable.layout(config, capacities), shards=[], parent=identity(common),
                    transition={'kind': 'append-residual-blocks', 'source': identity(common),
                                'blocks': additional, 'template': config.num_hidden_layers-additional-1})
    proposal['tensors'].update(new_tensors)
    proposal['tensors'] = dict(sorted(proposal['tensors'].items()))
    proposal['state_root'] = portable.learned_root(proposal)
    portable.validate(proposal)
    return proposal
