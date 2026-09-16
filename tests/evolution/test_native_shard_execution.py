"""Real autograd, complete partition replay and greedy serving on tiny shards."""
import copy
from datetime import timedelta
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from neuroshard.evolution import reference, portable_lifecycle as lifecycle
from neuroshard.evolution.reference_data import identity, save
from neuroshard.evolution.sharded import native_execution as native, portable, transcript
from neuroshard.evolution.sharded.model import Partition
from neuroshard.evolution.sharded.wire import Wire
from test_adaptive_shards import config, models, records, RECIPE

JOB = identity({'native-window': 'balanced-objective'})
PLAN = {'parent': {'step': 0, 'checkpoint': 'a'*64}, 'training': RECIPE,
        'microbatch': 2, 'reference': {'kl_strength': 2., 'margin_strength': 1.,
                                    'margin_min': .1, 'margin_max': 2.}}
PREPARED = {'schedule': [{'indices': list(range(5))} for _ in range(4)]}


def pair(rank):
    student, teacher = models()
    shards = [Partition(config(), [0, 3, 6], rank) for _ in range(2)]
    with torch.no_grad():
        for shard, model in zip(shards, (student, teacher)):
            parameters = dict(model.named_parameters())
            for name, parameter in shard.named_owned_parameters():
                parameter.copy_(parameters[name])
    shards[1].eval().requires_grad_(False)
    return shards


def worker(rank, home, rendezvous):
    torch.set_num_threads(1)
    random.seed(19 + rank)
    np.random.seed(19 + rank)
    shard, teacher = pair(rank)
    optimizer = reference.optimizer_for(shard, RECIPE)
    home = Path(home) / f'rank-{rank}'
    home.mkdir()
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=90))
    wire = Wire(rank, 2)
    try:
        initial = portable.commit(home / 'initial', shard, optimizer, wire, 'b'*64, 0, None)
        before = initial
        for start, end in ((0, 2), (2, 4)):
            folder = home / f'window-{end}'
            before = native.train_window(folder, shard, teacher, optimizer, wire, records(),
                PREPARED, PLAN, before, end, {}, JOB)
            manifests = wire.exchange(json.loads((folder / 'transcript/transcript.json').read_bytes()))
            transcript.validate(manifests)
            save(folder / 'transcripts.json', manifests)
        final = before
        # A continuous four-update run must produce exactly the same learned
        # weights and Adam moments despite native boundaries every two updates.
        portable.load(home / 'initial', shard, optimizer, initial, initial['job'])
        native.update_window(shard, teacher, optimizer, wire, records(), PREPARED['schedule'], PLAN, 0, 4)
        control = portable.commit(home / 'control', shard, optimizer, wire, JOB, 4, identity(initial))
        assert control['state_root'] == final['state_root']
        assert control['shards'] == final['shards']
        request = {'prompt_ids': [1, 3], 'max_tokens': 3, 'eos_id': 2, 'tokenizer_root': 'e'*64}
        captured = transcript.Recorder(wire, home / 'inference')
        tokens = native.generate_response(shard, captured, request)
        claim = {'kind': 'portable_inference', 'input_checkpoint': final, 'model_root': final['state_root'],
            'record_root': None, 'stages': len(tokens), 'executor_root': 'f'*64, 'job_id': 'd'*64,
            'request': request, 'token_ids': tokens}
        local = captured.finish(lifecycle.transcript_binding(claim))
        manifests = wire.exchange(local)
        claim['record_root'] = transcript.validate(manifests)
        save(home / 'inference-claim.json', claim)
        save(home / 'inference-transcripts.json', manifests)
        segmented = native.SegmentedRecorder(wire, home / 'segmented')
        answers = []
        for prompt in ([1, 3], [3, 5]):
            answers.append(native.generate_response(shard, segmented, {**request, 'prompt_ids': prompt}))
            segmented.segment()
        local = segmented.finish({'quality': 'bounded-test'})
        manifests = wire.exchange(local)
        native.validate_service_transcripts(manifests)
        save(home / 'segmented.json', {'rows': manifests, 'answers': answers})
        # A real quality manifest exceeds the original 2 MiB control-message
        # bound. Exchange several chunks over actual Gloo without relaxing it.
        large = {'rank': rank, 'payload': 'x' * (3 * 1024**2 + rank)}
        with pytest.raises(ValueError, match='Control message exceeds bound'):
            wire.exchange(large)
        gathered = native.exchange_manifests(wire, large)
        assert gathered == [{'rank': i, 'payload': 'x' * (3 * 1024**2 + i)} for i in range(2)]
        save(home / 'large-manifest.json', {'passed': True, 'ranks': len(gathered)})
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope='module')
def executed(tmp_path_factory):
    home = tmp_path_factory.mktemp('native-window')
    mp.spawn(worker, args=(str(home), str(home / 'rendezvous')), nprocs=2, join=True)
    return home


def test_native_windows_match_continuous_learning_and_every_shard_replays(executed):
    torch.set_num_threads(1)
    for rank in range(2):
        home = executed / f'rank-{rank}'
        shard, teacher = pair(rank)
        optimizer = reference.optimizer_for(shard, RECIPE)
        for start, end in ((0, 2), (2, 4)):
            source = home / ('initial' if start == 0 else f'window-{start}')
            target = home / f'window-{end}'
            before = json.loads((source / f'commit-{start:06d}.json').read_bytes())
            after = json.loads((target / f'commit-{end:06d}.json').read_bytes())
            births = portable.load(source, shard, optimizer, before, before['job'])
            rows = json.loads((target / 'transcripts.json').read_bytes())
            result = native.replay_window(home / f'audit-{end}', shard, teacher, optimizer, records(),
                PREPARED, PLAN, before, after, births, JOB, rows, target / 'transcript')
            assert result['valid'] and result['binding']['output'] == identity(after)


def test_paid_response_replays_with_only_one_partition_in_memory(executed):
    torch.set_num_threads(1)
    for rank in range(2):
        home = executed / f'rank-{rank}'
        claim = json.loads((home / 'inference-claim.json').read_bytes())
        rows = json.loads((home / 'inference-transcripts.json').read_bytes())
        shard, _ = pair(rank)
        checkpoint = claim['input_checkpoint']
        portable.load(home / 'window-4', shard, None, checkpoint, checkpoint['job'], restore_rng=False)
        assert native.service_report(claim, rank, rows)['valid']
        replay = transcript.Replay(home / 'inference', rows[rank])
        assert native.generate_response(shard, replay, claim['request']) == claim['token_ids']
        replay.finish()
        # Neither a different paid prompt nor an invented answer can reuse an
        # honest witness. A self-consistent forged witness must also replay.
        for field in ('request', 'token_ids'):
            broken = copy.deepcopy(claim)
            if field == 'request':
                broken[field]['prompt_ids'][0] = 7
            else:
                broken[field][0] = (broken[field][0] + 1) % 32
            with pytest.raises(ValueError, match='native statement'):
                native.service_report(broken, rank, rows)


def test_segmented_quality_witness_preserves_complete_ordered_coverage(executed):
    for rank in range(2):
        home = executed / f'rank-{rank}'
        stored = json.loads((home / 'segmented.json').read_bytes())
        shard, _ = pair(rank)
        checkpoint = json.loads((home / 'window-4/commit-000004.json').read_bytes())
        portable.load(home / 'window-4', shard, None, checkpoint, checkpoint['job'], restore_rng=False)
        replay = native.SegmentedReplay(home / 'segmented', stored['rows'][rank])
        for index, prompt in enumerate(([1, 3], [3, 5])):
            assert native.generate_response(shard, replay, {'prompt_ids': prompt, 'max_tokens': 3,
                                                           'eos_id': 2}) == stored['answers'][index]
            replay.segment()
        replay.finish()
        broken = copy.deepcopy(stored['rows'])
        for row in broken:
            row['segments'].reverse()
        with pytest.raises(ValueError, match='reordered'):
            native.validate_service_transcripts(broken)


def test_large_quality_manifests_use_bounded_transport(executed):
    for rank in range(2):
        assert json.loads((executed / f'rank-{rank}/large-manifest.json').read_bytes()) == {'passed': True, 'ranks': 2}


@pytest.mark.parametrize('corruption', ['allocation', 'length', 'digest'])
def test_manifest_transport_rejects_untrusted_declarations_and_chunks(corruption):
    class Peer:
        world = 2

        def exchange(self, value):
            peer = copy.deepcopy(value)
            if isinstance(value, dict) and corruption == 'allocation':
                peer['bytes'] = 128 * 1024**2 + 1
            elif isinstance(value, str):
                if corruption == 'length':
                    peer += 'AAAA'
                elif corruption == 'digest':
                    peer = ('A' if peer[0] != 'A' else 'B') + peer[1:]
            return [value, peer]

    with pytest.raises(ValueError, match='manifest (declaration|chunk exceeds|differs)'):
        native.exchange_manifests(Peer(), {'actual': 'committed data'})
