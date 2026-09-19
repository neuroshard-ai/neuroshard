"""Cached owner-local execution against the independent full HF model."""

from datetime import timedelta
import copy
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache
from safetensors.torch import load_file

from neuroshard.evolution.sharded.cached_inference import CachedPartition, generate_cached
from neuroshard.evolution.sharded import checkpoint, portable
from neuroshard.evolution.sharded.model import Partition
from neuroshard.evolution.sharded.training import generate
from neuroshard.evolution.sharded.transcript import Recorder, Replay, validate
from neuroshard.evolution.sharded.wire import Wire


def configuration(attention='eager'):
    config = LlamaConfig(vocab_size=47, hidden_size=24, intermediate_size=48,
                         num_hidden_layers=5, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=96,
                         tie_word_embeddings=True, attention_dropout=0.)
    config._attn_implementation = attention
    return config


def loaded_shard(model, boundaries, rank):
    shard = Partition(model.config, boundaries, rank)
    reference = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in shard.named_owned_parameters():
            value.copy_(reference[name])
    return shard.eval()


@torch.no_grad()
def full_greedy(model, prompt, limit):
    generated, cache = [], DynamicCache(config=model.config)
    current = list(prompt)
    for _ in range(limit):
        value = model(torch.tensor([current]), past_key_values=cache, use_cache=True)
        generated.append(int(value.logits[0, -1].argmax()))
        current = generated[-1:]
    return generated


def distributed_worker(rank, rendezvous, folder, boundaries, attention):
    torch.set_num_threads(1)
    torch.manual_seed(812)
    model = LlamaForCausalLM(configuration(attention)).float().eval()
    # Full construction is allowed only for the independent test oracle.
    def forbidden(*args, **kwargs):
        raise AssertionError('A serving owner constructed the whole model')
    LlamaForCausalLM.__init__ = forbidden
    shard = loaded_shard(model, boundaries, rank)
    world = len(boundaries) - 1
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
                            rank=rank, world_size=world, timeout=timedelta(seconds=60))
    wire = Wire(rank, world)
    try:
        records = []
        for prompt in ([1, 5, 7, 9, 11, 13, 15, 17, 19], [3, 4]):
            observed = {}
            streamed = []
            result = generate_cached(shard, wire, prompt, 12, -1, observed, on_tokens=streamed.append)
            assert result == full_greedy(model, prompt, 12)
            assert streamed == ([tuple(result[:i]) for i in range(1, 13)] if rank == 0 else [])
            before = wire.sent_tensor_bytes
            uncached = generate(shard, wire, prompt, 12, -1)
            assert uncached == result
            observed['uncached_sent_tensor_bytes'] = wire.sent_tensor_bytes - before
            count = len(prompt) + 11
            sent_positions = count if rank + 1 < world else 12
            assert observed['sent_tensor_bytes'] == sent_positions * 24 * 4 + 12 * 24
            assert observed['sent_tensor_bytes'] < observed['uncached_sent_tensor_bytes']
            assert observed['processed_positions_per_layer'] == count
            local_layers = boundaries[rank + 1] - boundaries[rank]
            assert observed['resident_cache_bytes'] == local_layers * 2 * count * 2 * 6 * 4
            # New requests rebuild their own caches. A second run is exact,
            # and EOS stops at the first emitted token without another pass.
            def disconnected(_tokens):
                raise ConnectionError('The customer disconnected during live generation')
            assert generate_cached(shard, wire, prompt, 12, -1, on_tokens=disconnected) == result
            assert generate_cached(shard, wire, prompt, 12, result[0]) == result[:1]
            records.append(observed)
        Path(folder, f'rank-{rank}.json').write_text(json.dumps(records))
        recorder = Recorder(wire, Path(folder, f'witness-{rank}'))
        prompt = [1, 4, 7, 17]
        tokens = generate_cached(shard, recorder, prompt, 10, -1)
        transcript = recorder.finish({'method': 'owner-local-kv', 'prompt': prompt, 'tokens': tokens})
        transcripts = wire.exchange(transcript)
        validate(transcripts)
        replay = Replay(recorder.home, transcript)
        assert generate_cached(shard, replay, prompt, 10, -1) == tokens
        replay.finish()
        if rank == 0:
            # Matching hashes at both endpoints do not prove an activation.
            # Forge a closed graph; numerical cached replay must still reject it.
            forged = copy.deepcopy(transcripts)
            send = next(event for event in forged[0]['events'] if event['kind'] == 'send')
            receive = next(event for event in forged[1]['events'] if event['kind'] == 'receive')
            value = load_file(portable.tensor_path(recorder.home, send['tensor']['sha256']))['value']
            pending = Path(folder, 'forged.pending')
            spec = checkpoint.tensor_file(pending, {'value': value + 1})
            raw = pending.read_bytes()
            for owner in (0, 1):
                portable.tensor_path(Path(folder, f'witness-{owner}'), spec['sha256']).write_bytes(raw)
            send['tensor'] = receive['tensor'] = {**spec, 'shape': list(value.shape)}
            validate(forged)
            replay = Replay(recorder.home, forged[0])
            with pytest.raises(ValueError, match='forward value'):
                generate_cached(shard, replay, prompt, 10, -1)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('boundaries', [[0, 1, 3, 5], [0, 1, 2, 3, 5]])
@pytest.mark.parametrize('attention', ['eager', 'sdpa'])
def test_cached_generation_matches_full_model_with_only_owned_state(tmp_path, boundaries, attention):
    mp.spawn(distributed_worker, args=(str(tmp_path / 'rendezvous'), str(tmp_path), boundaries, attention),
             nprocs=len(boundaries) - 1, join=True)


@torch.no_grad()
def test_each_cached_logit_matches_full_model_and_partial_layer_failure_is_terminal():
    torch.set_num_threads(1)
    torch.manual_seed(111)
    model = LlamaForCausalLM(configuration()).float().eval()
    boundaries = [0, 2, 3, 5]
    shards = [loaded_shard(model, boundaries, rank) for rank in range(3)]
    sessions = [CachedPartition(shard) for shard in shards]
    full_cache = DynamicCache(config=model.config)
    prompt = [1, 6, 2, 17, 25]
    position = 0
    for current in (prompt, [7], [13], [2]):
        value = torch.tensor([current])
        reference = model(value, past_key_values=full_cache, use_cache=True).logits
        hidden = value
        for session in sessions:
            hidden = session.advance(hidden, position)
        logits = shards[0].logits(hidden)
        torch.testing.assert_close(logits, reference, atol=2e-7, rtol=2e-5)
        position += len(current)
    assert [len(session.cache.layers) for session in sessions] == [2, 1, 2]
    with pytest.raises(ValueError, match='unowned'):
        sessions[1].cache.update(torch.zeros(1), torch.zeros(1), 0)
    # A mid-layer exception could leave only some layers advanced. Such a
    # session cannot silently continue or mix states from two executions.
    layer = shards[0].layers['1']
    original = layer.forward
    def fail(*args, **kwargs):
        raise RuntimeError('interrupted layer')
    layer.forward = fail
    with pytest.raises(RuntimeError, match='interrupted'):
        sessions[0].advance(torch.tensor([[4]]), position)
    layer.forward = original
    with pytest.raises(ValueError, match='failed cache'):
        sessions[0].advance(torch.tensor([[4]]), position)


@torch.no_grad()
def test_cached_session_rejects_stale_weights_cursor_and_excess_context():
    torch.set_num_threads(1)
    model = LlamaForCausalLM(configuration()).float().eval()
    shard = loaded_shard(model, [0, 2, 5], 0)
    session = CachedPartition(shard)
    session.advance(torch.tensor([[1, 2, 3]]), 0)
    with pytest.raises(ValueError, match='position'):
        session.advance(torch.tensor([[4]]), 2)
    session = CachedPartition(shard)
    session.advance(torch.tensor([[1, 2]]), 0)
    next(shard.parameters()).add_(.001)
    with pytest.raises(ValueError, match='Weights'):
        session.advance(torch.tensor([[4]]), 2)
    session = CachedPartition(shard)
    with pytest.raises(ValueError, match='context'):
        session.advance(torch.ones(1, 97, dtype=torch.long), 0)
    shard.train()
    with pytest.raises(ValueError, match='evaluation mode'):
        CachedPartition(shard)
