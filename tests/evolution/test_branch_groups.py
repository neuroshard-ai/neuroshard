"""Five real processes, two independent paths, and observed departures."""
from datetime import timedelta
import hashlib
from pathlib import Path
import time

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import LlamaConfig

from neuroshard.evolution.sharded.branch_groups import OrderedRoutes, RoutedNetwork
from neuroshard.evolution.sharded.model import Partition
from neuroshard.evolution.sharded.wire import Wire


RULES = [{'id': 'directory', 'needle': 'fictional luma directory', 'owner': 3},
         {'id': 'protocol', 'needle': 'neuroshard 0.4.0', 'owner': 4}]
QUESTIONS = ['Add 17 and 19.', 'fictional Luma directory question', 'NeuroShard 0.4.0 question']


class Tokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        return [7, 11, 19]

    def decode(self, values, **kwargs):
        return ','.join(map(str, values))


def initialize(shard, variant):
    with torch.no_grad():
        for name, parameter in shard.named_owned_parameters():
            suffix = '/' + variant if name.startswith('model.layers.5.') else ''
            seed = int(hashlib.sha256((name + suffix).encode()).hexdigest()[:8], 16)
            generator = torch.Generator().manual_seed(seed)
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * .02)
            parameter.requires_grad_(False)
    shard.eval()


def wait_for(path):
    deadline = time.monotonic() + 30
    while not path.exists():
        assert time.monotonic() < deadline
        time.sleep(.01)


def process(rank, rendezvous, output):
    torch.set_num_threads(1)
    config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=6,
        num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=True, attention_dropout=0,
        max_position_embeddings=64)
    config._attn_implementation = 'sdpa'
    shard = Partition(config, [0, 2, 4, 6] if rank < 3 else [0, 2, 4, 5, 6], min(rank, 3))
    initialize(shard, 'parent' if rank < 3 else RULES[rank - 3]['id'])
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=5,
                            timeout=timedelta(seconds=60))
    parent = dist.new_group([0, 1, 2], timeout=timedelta(seconds=60))
    groups = {rule['id']: dist.new_group([0, 1, 2, rule['owner']], timeout=timedelta(seconds=60))
              for rule in RULES}
    graph = RoutedNetwork(rank, shard, Tokenizer(), 5, OrderedRoutes(RULES), parent, groups)
    all_owners = Wire(rank, 5)
    outputs = {}
    try:
        for variant in ('parent', 'directory', 'protocol'):
            expert = variant != 'parent'
            active = rank < 3 or rank == (3 if variant == 'directory' else 4) and expert
            hidden = None
            if active:
                net = graph.networks[variant if expert else 'directory']
                hidden = net.forward([7, 11, 19], expert)
            if rank == 0:
                reference = Partition(config, [0, 6], 0)
                initialize(reference, variant)
                with torch.no_grad():
                    ids = torch.tensor([[7, 11, 19]])
                    expected = reference(ids, torch.ones_like(ids))
                np.testing.assert_array_equal(hidden.numpy(), expected.numpy())
        for question in QUESTIONS:
            value = graph.answer(question, 4)
            values = all_owners.exchange(value)
            assert all(item is None or item == values[0] for item in values)
            outputs[question] = values[0]
        all_owners.exchange('second expert may exit')
        if rank == 4:
            return
        wait_for(Path(output, 'second-expert-exited'))
        assert graph.answer(QUESTIONS[1], 4) == outputs[QUESTIONS[1]]
        if rank < 3:
            assert graph.answer(QUESTIONS[0], 4) == outputs[QUESTIONS[0]]
        graph.networks['directory'].wire.exchange('first expert may exit')
        if rank == 3:
            return
        wait_for(Path(output, 'first-expert-exited'))
        assert graph.answer(QUESTIONS[0], 4) == outputs[QUESTIONS[0]]
        Path(output, str(rank)).write_text('passed')
    finally:
        dist.destroy_process_group()


def test_two_experts_exact_and_old_paths_survive_observed_process_exits(tmp_path):
    context = mp.spawn(process, args=(str(tmp_path / 'group'), str(tmp_path)), nprocs=5, join=False)
    try:
        context.processes[4].join(timeout=60)
        assert context.processes[4].exitcode == 0
        (tmp_path / 'second-expert-exited').write_text('Observed exit code 0')
        context.processes[3].join(timeout=60)
        assert context.processes[3].exitcode == 0
        (tmp_path / 'first-expert-exited').write_text('Observed exit code 0')
        while not context.join(timeout=60):
            pass
    finally:
        for worker in context.processes:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=10)
    assert all((tmp_path / str(rank)).read_text() == 'passed' for rank in range(3))


def test_extension_keeps_existing_precedence_even_for_overlapping_questions():
    old = OrderedRoutes(RULES[:1])
    extended = OrderedRoutes(RULES)
    extended.require_extension_of(old)
    for question in [QUESTIONS[1], QUESTIONS[1] + QUESTIONS[2], QUESTIONS[2] + QUESTIONS[1]]:
        assert extended.select(question) == old.select(question) == 'directory'
    assert old.select(QUESTIONS[2]) is None and extended.select(QUESTIONS[2]) == 'protocol'
    changed = OrderedRoutes(list(reversed(RULES)))
    with pytest.raises(ValueError, match='preserve every existing choice'):
        changed.require_extension_of(old)
    copy = extended.rules
    copy[0]['needle'] = 'a different question'
    assert extended.select(QUESTIONS[1]) == 'directory'
    with pytest.raises(ValueError, match='user text'):
        extended.select({'task': 'protocol', 'expected': 'neuroshard doctor'})
