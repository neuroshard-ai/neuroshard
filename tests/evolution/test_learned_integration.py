import json
from pathlib import Path

import pytest

from neuroshard.evolution.learned_integration import (
    CONTRACT_IDENTITY, FORMAT, GROWTH_PLAN, UNIQUE_ADDED_HISTORY,
    UNIQUE_INCUMBENT_HISTORY, bind_spec, load_spec,
)
from neuroshard.evolution.reference_data import identity


ROOT = Path(__file__).resolve().parents[2]


def spec():
    return json.loads((ROOT / 'config/experiments/learned-integration.json').read_text())


def test_frozen_contract_identity_is_pinned():
    current = spec()
    assert identity(current) == CONTRACT_IDENTITY
    assert current['format'] == FORMAT
    assert bind_spec(current) == {
        'learned_integration': CONTRACT_IDENTITY,
        'growth_plan': GROWTH_PLAN,
        'gpu_launch_authorized': False,
        'admission_evidence': False,
    }
    assert identity(load_spec()) == CONTRACT_IDENTITY


def test_specification_does_not_authorize_training_or_gpus():
    current = spec()
    assert current['train'] is False
    assert current['gpu_launch_authorized'] is False
    assert current['admission_evidence'] is False
    assert current['confirmation_opened'] is False
    assert current['stage2']['authorized'] is False
    assert current['fresh_dataset_alone_is_not_the_experiment'] is True


def test_opened_programming_cases_are_ineligible():
    current = spec()
    forbidden = set(current['forbidden_evaluation_task_ids'])
    growth = json.loads((ROOT / 'config/experiments/programming-growth.json').read_text())
    assert identity(growth) == GROWTH_PLAN
    assert set(growth['parent_final_task_ids']) <= forbidden
    assert len(growth['parent_final_task_ids']) == 128
    assert set(UNIQUE_ADDED_HISTORY + UNIQUE_INCUMBENT_HISTORY) <= forbidden
    for role in ('train_new', 'train_replay', 'retention', 'development', 'confirmation'):
        assert not set(current['splits'][role]) & forbidden


def test_control_and_gate_are_frozen_before_outcomes():
    current = spec()
    assert current['control']['matched_data_and_steps'] is True
    assert current['success']['loss_alone_is_success'] is False
    assert current['architecture']['active_experts_per_token'] == 1
    assert current['architecture']['experts_after_expansion'] == 2
    assert current['architecture']['accepted_modules_frozen'] is True
    assert current['architecture']['existing_moe_layer_is_this_runtime'] is False
    assert current['gate']['minimum_confirmation_gain_tasks_vs_control'] == 1
    assert current['gate']['development_must_pass_before_confirmation'] is True
    assert current['general_retention']['opened'] is False
    assert current['general_retention']['documents'] == 8


@pytest.mark.parametrize('field,value,match', [
    ('train', True, 'does not authorize training'),
    ('gpu_launch_authorized', True, 'does not authorize training'),
    ('admission_evidence', True, 'may not count as admission'),
    ('original_final_opened', True, 'does not open the original'),
    ('opened_64_reusable_as_evaluation', True, 'remain development history'),
    ('confirmation_opened', True, 'Confirmation remains closed'),
    ('fresh_dataset_alone_is_not_the_experiment', False, 'fresh dataset alone'),
])
def test_bind_rejects_campaign_reopen_flags(field, value, match, monkeypatch):
    current = spec()
    current[field] = value
    from neuroshard.evolution import learned_integration as module
    monkeypatch.setattr(module, 'CONTRACT_IDENTITY', identity(current))
    with pytest.raises(ValueError, match=match):
        bind_spec(current)


def test_bind_rejects_identity_drift():
    current = spec()
    current['seed'] = 0
    with pytest.raises(ValueError, match='does not bind this measurement'):
        bind_spec(current)


def test_bind_rejects_runtime_and_success_drift_when_identity_is_forced(monkeypatch):
    from neuroshard.evolution import learned_integration as module

    def reject(mutate, match):
        broken = spec()
        mutate(broken)
        monkeypatch.setattr(module, 'CONTRACT_IDENTITY', identity(broken))
        with pytest.raises(ValueError, match=match):
            bind_spec(broken)

    reject(lambda current: current['architecture'].__setitem__('active_experts_per_token', 2),
           'Active expert count')
    reject(lambda current: current['architecture'].__setitem__('existing_moe_layer_is_this_runtime', True),
           'not this experiment runtime')
    reject(lambda current: current['success'].__setitem__('loss_alone_is_success', True),
           'Lower loss is not success')
    reject(lambda current: current['control'].__setitem__('matched_data_and_steps', False),
           'reuse the expansion data')
    reject(lambda current: current['stage2'].__setitem__('authorized', True),
           'Stage 2 is not authorized')

    def overlap(current):
        current['splits']['confirmation'] = list(current['splits']['confirmation'])
        current['splits']['confirmation'][0] = UNIQUE_ADDED_HISTORY[0]

    reject(overlap, 'overlaps burned')
