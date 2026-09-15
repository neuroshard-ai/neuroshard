"""Exercise the four-owner wire, causal vectors and actual parent fallback."""
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import LlamaConfig

from neuroshard.evolution.sharded.model import Partition
from neuroshard.evolution.sharded.readout_job import Network
from neuroshard.evolution.sharded.wire import Wire


class Tokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        assert len(messages) == 1 and messages[0]['role'] == 'user'
        return [7, 11, 19]

    def decode(self, values, **kwargs):
        return ','.join(map(str, values))


def initialize(shard):
    import hashlib
    with torch.no_grad():
        for name, parameter in shard.named_owned_parameters():
            seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
            generator = torch.Generator().manual_seed(seed)
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * .02)
            parameter.requires_grad_(False)
    shard.eval()


def owner_process(rank, rendezvous, output):
    torch.set_num_threads(1)
    config = LlamaConfig(vocab_size=40000, hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
        tie_word_embeddings=True, attention_dropout=0, max_position_embeddings=64)
    config._attn_implementation = 'sdpa'
    shard = Partition(config, [0, 1, 2, 3], rank) if rank < 3 else None
    if shard is not None:
        initialize(shard)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=60))
    try:
        net = Network(shard, Wire(rank, 4), Tokenizer(), config)
        vector = net.feature('A question with no answer')
        if rank == 3:
            full = Partition(config, [0, 3], 0)
            initialize(full)
            with torch.no_grad():
                ids = torch.tensor([[7, 11, 19, 39428, 11247, 25535]])
                expected = full(ids, torch.ones_like(ids))[0, -1].numpy()
            np.testing.assert_array_equal(vector, expected)
        before = net.generate('Add 17 and 19.', 3)
        after, route = net.answer('Add 17 and 19.', None, ['Ada Alden'], 3)
        assert route == 'parent' and before == after
        class Decoder:
            def predict(self, feature):
                assert feature.shape == (16,) and feature.dtype == np.float32
                return 'Oslo'
        result, route = net.answer('fictional Luma directory: Ada Alden', Decoder(), ['Ada Alden'], 3)
        assert result == '{"answer":"Oslo"}' and route == 'readout'
        Path(output, str(rank)).write_text('passed')
    finally:
        dist.destroy_process_group()


def test_real_four_owner_causal_path_and_parent_fallback(tmp_path):
    mp.spawn(owner_process, args=(str(tmp_path / 'group'), str(tmp_path)), nprocs=4, join=True)
    assert all((tmp_path / str(rank)).read_text() == 'passed' for rank in range(4))
