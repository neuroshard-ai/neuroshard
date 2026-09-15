"""Replay actual feature-backed training, including every intermediate root."""
import json

import torch
from transformers import LlamaConfig

from neuroshard.evolution import cohort_experiment as contract, reference_data as data
from neuroshard.evolution.sharded import expert_replay, feature_bank, incremental, incremental_state
from neuroshard.evolution.sharded.feature_probe import load_head
from neuroshard.evolution.sharded.model import Partition
from test_expert_cohort_job import test_actual_training_job_serves_and_survives_learner_exit as train_fixture


def test_replay_reconstructs_the_actual_five_owner_training_checkpoint(tmp_path):
    train_fixture(tmp_path, False)
    read = lambda path: json.loads(path.read_bytes())
    plan, prepared = read(tmp_path / 'plan.json'), read(tmp_path / 'prepared.json')
    parent = read(tmp_path / 'parent.json')
    expected = {step: data.identity(read(tmp_path / f'rank-4/commit-{step:06d}.json')) for step in (0, 2, 4)}
    config = LlamaConfig(**parent['config'])
    config._attn_implementation = 'sdpa'
    torch.set_num_threads(1)
    shard = Partition(config, plan['expert_layout'], 3)
    incremental_state.initialize(shard, parent, tmp_path / 'objects', 'tail-control', plan['split'])
    optimizer = incremental.configure(shard, plan['split'], plan['training'])
    head = load_head(parent, tmp_path / 'objects', 'cpu')
    index = read(tmp_path / 'rank-4/features/index.json')
    bank = feature_bank.Reader(tmp_path / 'rank-4/features', data.identity(index), index['binding'],
                               config, plan['microbatch'], len(prepared['batches']))
    rows = contract.rows(prepared, tmp_path / 'inputs', 'train')
    metrics = [json.loads(line) for line in (tmp_path / 'rank-4/updates.jsonl').read_text().splitlines()]
    result = expert_replay.replay(shard, head, optimizer, parent, contract.job(plan, prepared),
        plan['training'], plan['objective'], bank, rows, prepared['batches'], prepared['schedule'],
        metrics, expected, 'b' * 64, tmp_path / 'replay', 60)
    assert result['passed'] and result['checkpoint'] == expected[4] and len(result['windows']) == 1
    window = read(tmp_path / 'replay/window-000000.json')
    assert len({step['work_identity'] for step in window['steps']}) == 4
    assert window['input']['checkpoint'] == expected[0] and window['output']['checkpoint'] == expected[4]
    assert len(list((tmp_path / 'replay').glob('checkpoint-*.json'))) == 5
    assert result['prefix_recomputed'] is False and result['tokens_issued'] == 0


def test_numerical_batch_identity_ignores_descriptive_metadata():
    a = {'records': 'a' * 64, 'files': [{'file': 'first', 'records': 'b' * 64,
                                      'sha256': 'c' * 64, 'reference_aliases_prefix': False}]}
    b = {'records': 'd' * 64, 'files': [{**a['files'][0], 'file': 'renamed', 'records': 'e' * 64}]}
    assert expert_replay.batch_identity(a) == expert_replay.batch_identity(b)
    b['files'][0]['sha256'] = 'f' * 64
    assert expert_replay.batch_identity(a) != expert_replay.batch_identity(b)
