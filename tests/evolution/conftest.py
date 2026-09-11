import math

import pytest

from neuroshard.evolution import schema
from neuroshard.evolution.model import torch
from neuroshard.evolution.objects import Objects


@pytest.fixture
def seed(tmp_path):
    store = Objects(tmp_path / 'objects')
    config = dict(hidden_size=16, intermediate_size=32, num_attention_heads=4,
                  num_key_value_heads=2, vocab_size=64, rms_norm_eps=1e-5,
                  rope_theta=100000.0, max_position_embeddings=8192, num_hidden_layers=4)
    torch.manual_seed(42)
    components = {}
    for name in ['embed', 'norm', *[f'block_{i:03}' for i in range(4)]]:
        values = {k: torch.ones(shape) if len(shape) == 1 else torch.randn(shape)*.05
                  for k, shape in schema.shapes(config, name).items()}
        components[name] = {'root':store.put_tensors(values), 'parameters':sum(t.numel() for t in values.values())}
    model = dict(format='neuroshard-model-v1', config=config, components=components,
                 parameters=sum(c['parameters'] for c in components.values()), parent=None,
                 origin={'test':True})
    return store, store.put_json(model), model
