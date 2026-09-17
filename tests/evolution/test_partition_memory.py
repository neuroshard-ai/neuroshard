import pytest
import torch
from transformers import LlamaConfig

from neuroshard.evolution.sharded.model import Partition


def configuration():
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        tie_word_embeddings=True)
    config._attn_implementation = 'sdpa'
    return config


def test_inference_budget_does_not_change_owned_tensors_or_numerical_output():
    training = Partition(configuration(), [0, 1, 2], 0)
    serving = Partition(configuration(), [0, 1, 2], 0, inference_only=True)
    torch.manual_seed(7)
    with torch.no_grad():
        for parameter in training.parameters():
            parameter.normal_(0, .05)
        serving.load_state_dict(training.state_dict())
        training.eval(); serving.eval()
        ids=torch.tensor([[3, 4, 5]]); mask=torch.ones_like(ids)
        assert torch.equal(training.logits(training(ids,mask)), serving.logits(serving(ids,mask)))
    assert any(parameter.requires_grad for parameter in training.parameters())
    assert not any(parameter.requires_grad for parameter in serving.parameters())
    assert [name for name,_ in training.named_owned_parameters()] == [name for name,_ in serving.named_owned_parameters()]


def test_low_free_memory_accepts_serving_without_relaxing_training_or_parameter_limits(monkeypatch):
    size = Partition(configuration(), [0,1,2], 0).resident_parameters
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda: (size*8+2*1024**3, 24*1024**3))
    class AllocationReached(Exception):
        pass
    def allocation(*args, **kwargs):
        raise AllocationReached('The reservation passed before any CUDA allocation')
    monkeypatch.setattr(Partition, 'to_empty', allocation)
    with pytest.raises(ValueError, match='Insufficient memory'):
        Partition(configuration(),[0,1,2],0,'cuda')
    with pytest.raises(AllocationReached):
        Partition(configuration(),[0,1,2],0,'cuda',inference_only=True)
    with pytest.raises(ValueError, match='parameter limit'):
        Partition(configuration(),[0,1,2],0,'cuda',parameter_limit=size-1,inference_only=True)
