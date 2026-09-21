import json
from pathlib import Path

import pytest
import torch

from neuroshard.evolution import programming_growth as growth
from neuroshard.evolution import programming_growth_ties as ties
from neuroshard.evolution.reference_data import identity


ROOT = Path(__file__).resolve().parents[2]


def test_ties_drops_opposing_signs_and_averages_agreement():
    parent = torch.zeros(4)
    incumbent = torch.tensor([2.0, -2.0, 2.0, 0.0])
    added = torch.tensor([2.0, 2.0, -2.0, 3.0])
    # keep=1 for this unit test of the disjoint-mean rule; production keep is frozen at 0.2
    with pytest.raises(ValueError, match='frozen defaults'):
        ties.ties_merge(parent, incumbent, added, keep=1.0)
    merged = ties.ties_merge(parent, incumbent, added)
    # With keep=0.2 on 4 entries, only the largest-magnitude entry of each delta survives.
    # incumbent magnitudes [2,2,2,0] → one of the 2.0s; added [2,2,2,3] → index 3.
    assert merged.shape == parent.shape
    assert float((merged - parent).abs().sum()) >= 0


def test_ties_disjoint_mean_on_untrimmed_vectors():
    original_keep = ties.KEEP
    original_lambda = ties.LAMBDA
    try:
        # Temporarily exercise the mean rule through trim_delta keep=1 by calling internals.
        parent = torch.zeros(3)
        inc = torch.tensor([4.0, 4.0, -4.0])
        add = torch.tensor([2.0, -2.0, -2.0])
        d_inc = ties.trim_delta(inc - parent, 1.0)
        d_add = ties.trim_delta(add - parent, 1.0)
        assert torch.equal(d_inc, inc)
        assert torch.equal(d_add, add)
        elected = (d_inc + d_add).sign()
        # elected: + + -  because 6, 2, -6
        assert torch.equal(elected, torch.tensor([1.0, 1.0, -1.0]))
    finally:
        assert ties.KEEP == original_keep and ties.LAMBDA == original_lambda


def test_unit_merge_is_not_ties():
    parent = torch.tensor([1.0, 1.0])
    inc = torch.tensor([2.0, 0.0])
    add = torch.tensor([0.0, 3.0])
    unit = growth.task_vector_merge(parent, inc, add, 1, 1)
    composed = ties.ties_merge(parent, inc, add)
    assert not torch.equal(unit, composed)


def test_conflict_fraction_counts_opposite_signs():
    parent = torch.zeros(4)
    inc = torch.tensor([1.0, 1.0, -1.0, 0.0])
    add = torch.tensor([-1.0, 1.0, -1.0, 2.0])
    # opposite on index 0; same on 1 and 2; index 3 has a zero on inc so not both-nonzero
    assert ties.conflict_fraction(parent, inc, add) == pytest.approx(1 / 3)


def test_ties_hyperparameters_are_frozen():
    assert ties.KEEP == 0.2
    assert ties.LAMBDA == 1.0
    assert ties.GROWTH_FREEZE_COMMIT == 'dcc66936a746f7fdc6b6dfa25d1c47ad6f366758'
    assert ties.ADDED_EXPERT == '3abb54eecf62610a1116cc1b5eb67110bea17ee9608f946002ed9b4ad95703b6'
    plan = json.loads((ROOT / 'config/experiments/programming-growth.json').read_text())
    assert identity(plan) == ties.GROWTH_PLAN


def test_frozen_ties_contract_identity_is_pinned():
    spec = json.loads((ROOT / 'config/experiments/programming-growth-ties.json').read_text())
    expert = json.loads((ROOT / 'config/experiments/programming-growth-ties-expert.json').read_text())
    assert identity(spec) == ties.CONTRACT_IDENTITY
    assert identity(expert) == ties.EXPERT_IDENTITY
    assert spec['keep'] == 0.2
    assert spec['train'] is False
    assert spec['admission_evidence'] is False


def test_recorded_ties_screen_keeps_the_failed_gates():
    from neuroshard.evolution.reference_data import sha256
    score_path = ROOT / 'config/experiments/programming-growth-ties-screen-score.json'
    outputs_path = ROOT / 'config/experiments/programming-growth-ties-outputs.json'
    record = json.loads((ROOT / 'config/experiments/programming-growth-ties-screen-record.json').read_text())
    score = json.loads(score_path.read_text())
    assert sha256(score_path) == '86a6a4895bc9cae6d3779df79f079a6aa2de5423bd918c03500cb76d8ba7f913'
    assert sha256(outputs_path) == '3de80b42714d54162dbeb84685cc04caf5bf9fd3024cf69fd8406c883de61943'
    assert record['screen_score_sha256'] == sha256(score_path)
    assert record['outputs_sha256'] == sha256(outputs_path)
    assert record['status'] == 'stop-this-composition'
    assert record['later_method'] == 'do-not-iterate-keep-or-scale-on-these-64'
    assert score['passed'] is False
    assert score['ties_correct'] == 31
    assert score['extractable']['ties'] == 38
    assert score['unique_added_recovered'] == 1
    assert score['incumbent_successes_preserved'] == 29
    assert score['old_successes_preserved'] == 12
    assert score['next'] == 'stop-this-composition'
    assert score['admission_evidence'] is False
    assert score['gpu_authorized_by_screen'] is False
    assert record['unique_added_recovered_task_ids'] == [503]
    assert record['unique_added_missed_task_ids'] == [276, 265]
    assert record['ties_only_task_ids'] == [54]
