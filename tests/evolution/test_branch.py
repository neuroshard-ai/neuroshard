"""Exact branch equivalence and parent generation after the expert exits."""
from datetime import timedelta
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import LlamaConfig

from neuroshard.evolution.sharded.branch import Network, ParentWire, route
from neuroshard.evolution.sharded.model import Partition
from neuroshard.evolution.sharded.wire import Wire


class Tokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        return [7, 11, 19]

    def decode(self, values, **kwargs):
        return ','.join(map(str, values))


def initialize(shard, expert=False):
    with torch.no_grad():
        for name, parameter in shard.named_owned_parameters():
            key = name + ('/expert' if expert and name.startswith('model.layers.5.') else '')
            seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
            generator = torch.Generator().manual_seed(seed)
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * .02)
            parameter.requires_grad_(False)
    shard.eval()


def process(rank, rendezvous, output):
    torch.set_num_threads(1)
    config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=6,
        num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=True, attention_dropout=0,
        max_position_embeddings=64)
    config._attn_implementation = 'sdpa'
    shard = Partition(config, [0, 2, 4, 6] if rank < 3 else [0, 2, 4, 5, 6], rank)
    initialize(shard, rank == 3)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=60))
    group = dist.new_group([0, 1, 2], timeout=timedelta(seconds=60))
    wire = Wire(rank, 4)
    net = Network(shard, wire, ParentWire(rank, group) if rank < 3 else None, Tokenizer(), 5)
    try:
        actual = {}
        for expert in (False, True):
            hidden = net.forward([7, 11, 19], expert)
            if rank == 0:
                reference = Partition(config, [0, 6], 0)
                initialize(reference, expert)
                with torch.no_grad():
                    ids = torch.tensor([[7, 11, 19]])
                    expected = reference(ids, torch.ones_like(ids))
                np.testing.assert_array_equal(hidden.numpy(), expected.numpy())
                actual[expert] = hidden.numpy()
        if rank == 0:
            assert not np.array_equal(actual[False], actual[True])
        before = net.answer('Add 17 and 19.', 4)
        fresh = net.answer('fictional Luma directory: any natural question', 4)
        assert fresh['route'] == 'expert'
        wire.exchange('expert may exit now')
        if rank == 3:
            return
        # The fourth process destroys its group and exits. Established generation
        # uses only the three-owner subgroup and continues to match its output.
        import time
        end = time.monotonic() + 10
        while not Path(output, 'expert-exited').exists():
            assert time.monotonic() < end
            time.sleep(.01)
        after = net.answer('Add 17 and 19.', 4)
        assert before == after and after['route'] == 'parent'
        Path(output, str(rank)).write_text('passed')
    finally:
        dist.destroy_process_group()


def test_real_branch_and_parent_subgroup_survives_expert_exit(tmp_path):
    context = mp.spawn(process, args=(str(tmp_path / 'group'), str(tmp_path)), nprocs=4, join=False)
    try:
        context.processes[3].join(timeout=60)
        assert context.processes[3].exitcode == 0
        (tmp_path / 'expert-exited').write_text('Controller observed process exit code 0')
        while not context.join(timeout=60):
            pass
    finally:
        for worker in context.processes:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=10)
    assert all((tmp_path / str(rank)).read_text() == 'passed' for rank in range(3))


def test_route_has_no_name_attribute_or_benchmark_id_input():
    assert route('In the FICTIONAL LUMA DIRECTORY, describe this person.')
    assert not route('Explain how a solar cell works.')
    assert not route({'id': 'test-knowledge', 'expected': 'Oslo'})
