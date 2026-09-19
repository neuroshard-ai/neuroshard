import pytest
import torch
from safetensors.torch import save_file

from neuroshard.evolution import expert_router
from neuroshard.evolution.reference_data import sha256
from neuroshard.evolution.sharded.router_features import EmbeddingFeatures


class Tokens:
    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return [int(word) for word in text.split()]


def make(tmp_path, weights=None):
    path = tmp_path / 'embedding.safetensors'
    save_file({'weight': weights if weights is not None else
               torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.bfloat16)}, path)
    return path, sha256(path)


def test_features_use_only_prompt_tokens_and_preserve_repetition(tmp_path):
    path, digest = make(tmp_path)
    features = EmbeddingFeatures(path, digest, Tokens(), '1' * 64, max_tokens=4)
    assert features('0 1') == expert_router.normalize([1, 1, 0])
    assert features('0 0 1') == expert_router.normalize([2, 1, 0])
    assert features('2') == [0, 0, expert_router.SCALE]
    assert features.profile['embedding_sha256'] == digest
    assert EmbeddingFeatures(path, digest, Tokens(), '2' * 64).root != features.root
    for text in ('', ' ', '3', '-1', '0 1 0 1 0'):
        with pytest.raises(ValueError):
            features(text)


def test_corrupted_nonfinite_or_out_of_profile_embeddings_never_route(tmp_path):
    path, digest = make(tmp_path)
    with pytest.raises(ValueError, match='commitment'):
        EmbeddingFeatures(path, '0' * 64, Tokens(), '1' * 64)
    for invalid in (float('nan'), float('inf'), 100):
        path, digest = make(tmp_path, torch.tensor([[invalid, 0, 0]], dtype=torch.float32))
        with pytest.raises(ValueError, match='quantization'):
            EmbeddingFeatures(path, digest, Tokens(), '1' * 64)
