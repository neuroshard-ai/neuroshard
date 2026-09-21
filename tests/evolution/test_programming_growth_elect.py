import json
from pathlib import Path

import pytest
import torch

from neuroshard.evolution import programming_growth as growth
from neuroshard.evolution import programming_growth_elect as elect
from neuroshard.evolution import programming_growth_ties as ties
from neuroshard.evolution.reference_data import identity, sha256


ROOT = Path(__file__).resolve().parents[2]


def test_elect_keeps_unique_directions_and_zeros_tied_conflicts():
    parent = torch.zeros(4)
    incumbent = torch.tensor([4.0, 0.0, -1.0, 2.0])
    added = torch.tensor([2.0, 3.0, 1.0, -2.0])
    merged = elect.elect_merge(parent, incumbent, added)
    # idx0 both + → mean 3; idx1 unique added 3; idx2 opposite equal mag → 0; idx3 opposite equal → 0
    assert torch.equal(merged, torch.tensor([3.0, 3.0, 0.0, 0.0]))


def test_elect_larger_magnitude_wins_conflicts():
    parent = torch.zeros(1)
    incumbent = torch.tensor([-2.0])
    added = torch.tensor([1.0])
    merged = elect.elect_merge(parent, incumbent, added)
    assert torch.equal(merged, torch.tensor([-2.0]))


def test_elect_is_not_ties_trim_and_not_unit_sum():
    parent = torch.zeros(5)
    incumbent = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
    added = torch.tensor([0.5, 0.0, 0.0, 0.0, 10.0])
    unit = growth.task_vector_merge(parent, incumbent, added, 1, 1)
    composed = elect.elect_merge(parent, incumbent, added)
    trimmed = ties.ties_merge(parent, incumbent, added)
    assert not torch.equal(unit, composed)
    assert not torch.equal(trimmed, composed)
    # untrimmed elect keeps the small unique added direction; TIES keep 0.2 drops it
    assert float(composed[0]) == pytest.approx(0.5)
    assert float(trimmed[0]) == pytest.approx(0.0)


def test_elect_rejects_a_changed_scale():
    with pytest.raises(ValueError, match='frozen default'):
        elect.elect_merge(torch.zeros(1), torch.ones(1), torch.ones(1), scale=0.5)


def test_elect_hyperparameters_are_frozen():
    assert elect.LAMBDA == 1.0
    assert elect.GROWTH_FREEZE_COMMIT == 'dcc66936a746f7fdc6b6dfa25d1c47ad6f366758'
    assert elect.ADDED_EXPERT == '3abb54eecf62610a1116cc1b5eb67110bea17ee9608f946002ed9b4ad95703b6'
    plan = json.loads((ROOT / 'config/experiments/programming-growth.json').read_text())
    assert identity(plan) == elect.GROWTH_PLAN
    spec = json.loads((ROOT / 'config/experiments/programming-growth-elect.json').read_text())
    expert = json.loads((ROOT / 'config/experiments/programming-growth-elect-expert.json').read_text())
    assert identity(spec) == elect.CONTRACT_IDENTITY
    assert identity(expert) == elect.EXPERT_IDENTITY
    assert spec['trim'] is False
    assert spec.get('keep') in (None, False)
    assert spec['train'] is False
    assert spec['admission_evidence'] is False


def test_frozen_elect_execution_binds_the_cpu_merge():
    freeze = json.loads((ROOT / 'config/experiments/programming-growth-elect-freeze.json').read_text())
    assert freeze['contract'] == elect.CONTRACT_IDENTITY
    assert freeze['expert'] == elect.EXPERT_IDENTITY
    assert freeze['gpu_launch_authorized'] is True
    assert freeze['decode_performed'] is False
    assert freeze['trim'] is False
    assert freeze['parent_stopped_ties_commit'] == '9e855852f529514ba7d9eff603e8fff67b15b519'
    for name, digest in freeze['files'].items():
        assert sha256(ROOT / name) == digest


def test_recorded_elect_screen_keeps_the_failed_gates():
    score_path = ROOT / 'config/experiments/programming-growth-elect-screen-score.json'
    outputs_path = ROOT / 'config/experiments/programming-growth-elect-outputs.json'
    record = json.loads((ROOT / 'config/experiments/programming-growth-elect-screen-record.json').read_text())
    score = json.loads(score_path.read_text())
    assert sha256(score_path) == '7491a09b3814d1c4d4ecb5354d422b8efc27fd17494836e02e76f7c151aa9ba1'
    assert sha256(outputs_path) == '3d1e80f207149f4b906829553766a9cde4817ff02828d43189e658841a83ddc8'
    assert record['screen_score_sha256'] == sha256(score_path)
    assert record['outputs_sha256'] == sha256(outputs_path)
    assert record['status'] == 'stop-this-composition'
    assert record['later_method'] == 'do-not-iterate-sign-election-on-these-64'
    assert record['closed_family'] == 'task-vector-sign-consensus'
    assert record['matched_ties_task_outcomes'] is True
    assert score['passed'] is False
    assert score['elect_correct'] == 31
    assert score['extractable']['elect'] == 38
    assert score['unique_added_recovered'] == 1
    assert score['incumbent_successes_preserved'] == 29
    assert score['old_successes_preserved'] == 12
    assert score['next'] == 'stop-this-composition'
    assert score['admission_evidence'] is False
    assert score['gpu_authorized_by_screen'] is False
    assert record['unique_added_recovered_task_ids'] == [503]
    assert record['unique_added_missed_task_ids'] == [276, 265]
    assert record['elect_only_task_ids'] == [54]
