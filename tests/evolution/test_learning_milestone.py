import copy
import json

import pytest

from neuroshard.evolution.milestone import (
    PLAN_PATH, decide_learning, load, selection_path, training_allowed, validate,
)

PLAN = json.loads(PLAN_PATH.read_text())


def test_frozen_plan_loads_and_forbids_training():
    plan = load()
    assert plan['status'] == 'plan-frozen'
    assert plan['seed']['growth_layers'] == 0
    assert plan['optimizer']['recipes'] == 1
    assert training_allowed(plan) is False
    assert not selection_path(plan).is_file()


def test_quality_margins_match_existing_gate():
    from neuroshard.evolution.evaluation import decide
    evaluation = PLAN['evaluation']
    baseline = [1.2] * 64
    better = [1.19] * 64
    worse = [1.3] * 64
    reference = decide(baseline, better, baseline, better,
                       evaluation['retention_margin'], evaluation['fresh_min_gain'])
    assert reference['retention']['margin'] == evaluation['retention_margin']
    assert reference['fresh']['margin'] == -evaluation['fresh_min_gain']
    rejected = decide(baseline, worse, baseline, worse,
                      evaluation['retention_margin'], evaluation['fresh_min_gain'])
    assert rejected['promote'] is False


def test_fresh_cannot_rescue_or_veto_the_sealed_set():
    good = [1.2] * 64
    improved = [1.1] * 64
    worse = [1.3] * 64
    rescued = decide_learning({
        'baseline': {'test': good, 'retention': good, 'fresh': good},
        'candidate': {'test': [1.1999] * 64, 'retention': improved, 'fresh': improved},
    }, PLAN)
    assert rescued['fresh']['passes'] is True
    assert rescued['pass'] is False
    vetoed = decide_learning({
        'baseline': {'test': good, 'retention': good, 'fresh': good},
        'candidate': {'test': improved, 'retention': improved, 'fresh': worse},
    }, PLAN)
    assert vetoed['fresh']['passes'] is False
    assert vetoed['pass'] is True


def test_stop_rules_reject_growth_and_margin_changes():
    plan = copy.deepcopy(PLAN)
    plan['seed']['growth_layers'] = 4
    with pytest.raises(ValueError, match='no growth'):
        validate(plan)
    plan = copy.deepcopy(PLAN)
    plan['evaluation']['fresh_min_gain'] = 0.0001
    with pytest.raises(ValueError, match='cannot be relaxed'):
        validate(plan)
    plan = copy.deepcopy(PLAN)
    plan['continual']['blocked_on'] = 'optional'
    with pytest.raises(ValueError, match='only after a learning pass'):
        validate(plan)
    plan = copy.deepcopy(PLAN)
    plan['open_source']['secret_evaluation'] = True
    with pytest.raises(ValueError, match='secret evaluation'):
        validate(plan)


def test_training_status_without_selection_is_invalid():
    plan = copy.deepcopy(PLAN)
    plan['status'] = 'learning-running'
    with pytest.raises(ValueError, match='committed sealed-set'):
        validate(plan)
