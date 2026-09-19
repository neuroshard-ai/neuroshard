"""Two actual partitioned models, neural argument generation, and expert exit."""
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import LlamaConfig

from neuroshard.evolution import incremental_facts as facts
from neuroshard.evolution.sharded.branch import Network, ParentWire
from neuroshard.evolution.sharded.interpretation import InterpretedNetwork, interpretation
from neuroshard.evolution.sharded.model import Partition
from neuroshard.evolution.sharded.wire import Wire


class ToyTokenizer:
    """A finite test vocabulary with distinct atomic JSON strings.

    The real neural argmax chooses the field; no factual answers enter this
    tokenizer. The real-model test separately checks normal subword decoding.
    """
    eos_token_id = 2
    def apply_chat_template(self, messages, **kwargs):
        digest = hashlib.sha256(json.dumps(messages).encode()).digest()
        return [3 + value % 61 for value in digest[:3]]
    def decode(self, values, **kwargs):
        if len(values) == 1:
            field = ('city', 'profession', 'instrument', 'hobby')[values[0] % 4]
            return '{"name":' + ' ' * (values[0] // 4) + '"Robin Finch","field":"' + field + '"}'
        return ','.join(map(str, values))


def initialize(shard, kind):
    with torch.no_grad():
        for name, parameter in shard.named_owned_parameters():
            suffix = '/seed' if kind == 'seed' else '/expert' if kind == 'expert' and name.startswith('model.layers.5.') else ''
            seed = int(hashlib.sha256((name + suffix).encode()).hexdigest()[:8], 16)
            parameter.copy_(torch.randn(parameter.shape, generator=torch.Generator().manual_seed(seed)) * .02)
    shard.eval().requires_grad_(False)


@torch.no_grad()
def generate(model, tokenizer, messages, cap):
    ids = tokenizer.apply_chat_template(messages)
    output = []
    for _ in range(cap):
        values = torch.tensor([ids])
        hidden = model(values, torch.ones_like(values))
        token = int(model.logits(hidden[:, -1:]).float().argmax(-1)[0, 0])
        ids.append(token); output.append(token)
        if token == tokenizer.eos_token_id:
            break
    return {'ids': output, 'text': tokenizer.decode(output)}


def process(rank, rendezvous, home):
    torch.set_num_threads(1)
    config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=6,
        num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=True, attention_dropout=0,
        max_position_embeddings=64)
    config._attn_implementation = 'sdpa'
    shard = Partition(config, [0, 2, 4, 6] if rank < 3 else [0, 2, 4, 5, 6], rank)
    initialize(shard, 'expert' if rank == 3 else 'parent')
    seed = Partition(config, [0, 2, 4, 6], rank) if rank < 3 else None
    if seed is not None:
        initialize(seed, 'seed')
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=60))
    group = dist.new_group([0, 1, 2], timeout=timedelta(seconds=60))
    wire, tokenizer = Wire(rank, 4), ToyTokenizer()
    parent_wire = ParentWire(rank, group) if rank < 3 else None
    trained = Network(shard, wire, parent_wire, tokenizer, 5)
    preserved = Network(seed, wire, parent_wire, tokenizer, 5) if seed else None
    records = []
    net = InterpretedNetwork(trained, preserved, 'Interpret the quoted question.', [], 1, records.append)
    question = 'fictional Luma directory: what is the city for Robin Finch?'
    prior = 'Add 17 and 19.'
    try:
        before = net.answer(prior, 4)
        assert not records, 'Ordinary parent serving must bypass the interpreter'
        actual = net.answer(question, 4)
        assert len(records) == 1
        assert records[0]['interpretation'] is not None
        if rank == 0:
            reference_seed = Partition(config, [0, 6], 0)
            reference_expert = Partition(config, [0, 6], 0)
            initialize(reference_seed, 'seed'); initialize(reference_expert, 'expert')
            parsed = generate(reference_seed, tokenizer, [*net.prefix,
                {'role': 'user', 'content': json.dumps(question) + '\n\n' + net.instruction}], 1)
            assert records[0]['parser'] == parsed
            value = interpretation(parsed['text'], question)
            canonical = facts.question({'name': value['name']}, value['field'], 'train', 0)
            expected = generate(reference_expert, tokenizer, [{'role': 'user', 'content': canonical}], 4)
            assert records[0]['expert_question'] == canonical
            assert actual == {**expected, 'route': 'expert'}
        net.verify_unchanged()
        wire.exchange('expert may exit')
        if rank == 3:
            return
        until = time.monotonic() + 15
        while not Path(home, 'expert-exited').exists():
            assert time.monotonic() < until
            time.sleep(.01)
        assert net.answer(prior, 4) == before
        assert len(records) == 1
        net.verify_unchanged()
        Path(home, str(rank)).write_text('passed')
    finally:
        dist.destroy_process_group()


def test_actual_neural_interpretation_composition_and_parent_survival(tmp_path):
    context = mp.spawn(process, args=(str(tmp_path / 'group'), str(tmp_path)), nprocs=4, join=False)
    try:
        context.processes[3].join(timeout=60)
        assert context.processes[3].exitcode == 0
        (tmp_path / 'expert-exited').write_text('Controller observed actual process exit')
        while not context.join(timeout=60):
            pass
    finally:
        for worker in context.processes:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=10)
    assert all((tmp_path / str(rank)).read_text() == 'passed' for rank in range(3))


def test_interpretation_uses_only_model_output_and_raw_question():
    question = 'Where does Robin Finch live?'
    assert interpretation('{"name":"Robin Finch","field":"city"}', question) == {'name': 'Robin Finch', 'field': 'city'}
    for text in ('{"answer":"Paris"}', '{"name":"Other Person","field":"city"}',
                 '{"name":"Robin Finch","field":"occupation"}',
                 '{"name":"Robin Finch","name":"Robin Finch","field":"city"}',
                 '{"name":"Robin Finch","field":"city","answer":"Paris"}',
                 '{"name":"Robin Finch","field":["city"]}', 'not json', '[]'):
        assert interpretation(text, question) is None
