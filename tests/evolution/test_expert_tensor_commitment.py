"""The optimized recorder must preserve existing canonical checkpoint bytes."""
import hashlib

import pytest
import torch
from safetensors.torch import save

from neuroshard.evolution.sharded.expert_commitment import tensor_commitment


@pytest.mark.parametrize('shape', [(), (1,), (0,), (7,), (3, 11), (17, 31), (1024, 1024)])
def test_stream_matches_the_actual_safetensors_writer(shape):
    torch.manual_seed(61)
    weight = torch.randn(shape)
    for values in ({'weight': weight}, {'weight': weight, 'step': torch.tensor(25.),
            'exp_avg_sq': weight.square(), 'exp_avg': weight * .01}):
        raw = save(values)
        assert tensor_commitment(values) == {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        # Mapping order must not change the prescribed file's bytes.
        assert tensor_commitment(dict(reversed(list(values.items())))) == tensor_commitment(values)


def test_finite_edge_bits_remain_exact_and_nonfinite_arrays_are_rejected():
    values = {'weight': torch.tensor([0., -0., 1e-44, -1e-44, torch.finfo(torch.float32).max])}
    raw = save(values)
    assert tensor_commitment(values)['sha256'] == hashlib.sha256(raw).hexdigest()
    for bad in (float('nan'), float('inf'), -float('inf')):
        with pytest.raises(ValueError, match='finite'):
            tensor_commitment({'weight': torch.tensor([bad])})
    with pytest.raises(ValueError, match='float32'):
        tensor_commitment({'weight': torch.ones(1, dtype=torch.float64)})
