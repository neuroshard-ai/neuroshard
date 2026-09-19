import copy
import json

import pytest

from neuroshard.evolution import continued, grounded_tasks as tasks

PLAN = json.loads(continued.PLAN_PATH.read_text())


def answers(records, correct):
    return [{'id': row['id'], 'correct': bool(correct[index])} for index, row in enumerate(records)]


def records(role, count, seed):
    return [{'id': f'{index:064x}', 'task': tasks.make_case(seed, role, index)} for index in range(count)]


def measurements(new_records, prior_records, new_correct, prior_correct, losses):
    return {
        'test-new': {'answers': answers(new_records, new_correct), 'losses': list(losses['new'])},
        'test-prior': {'answers': answers(prior_records, prior_correct), 'losses': list(losses['prior'])},
        'retention': {'losses': list(losses['retention']), 'answers': []},
    }


def gain_on_totals(rows, count):
    candidate = [False] * len(rows)
    wins = 0
    for index, row in enumerate(rows):
        if row['task']['family'] == 'total' and wins < count:
            candidate[index] = True
            wins += 1
    return candidate


@pytest.fixture
def fixtures():
    new_rows = records('test', PLAN['test_new_cases'], PLAN['task_seed'])
    prior_rows = records('test', PLAN['test_prior_cases'], PLAN['prior_probe_seed'])
    assert [row['task']['family'] for row in new_rows].count('total') == 64
    assert [row['task']['family'] for row in prior_rows].count('total') == 32
    return new_rows, prior_rows


def test_frozen_plan_loads_and_forbids_training():
    plan = continued.load()
    plan['status'] = 'plan-frozen'
    assert plan['method']['growth_layers'] == 0
    assert plan['quality_gate']['primary'] == 'generated-answer-improvement'
    assert plan['settlement']['existing_checkpoint_insufficient'] is True
    assert continued.training_allowed(plan) is False


def test_stop_rules_reject_growth_margin_and_checkpoint_payment():
    plan = copy.deepcopy(PLAN)
    plan['method']['growth_layers'] = 2
    with pytest.raises(ValueError, match='Method constants'):
        continued.validate(plan)
    plan = copy.deepcopy(PLAN)
    plan['quality_gate']['min_net_gain'] = 1
    with pytest.raises(ValueError, match='Quality margins'):
        continued.validate(plan)
    plan = copy.deepcopy(PLAN)
    plan['settlement']['existing_checkpoint_insufficient'] = False
    with pytest.raises(ValueError, match='activated replay'):
        continued.validate(plan)
    plan = copy.deepcopy(PLAN)
    plan['allowed_layers'] = [24, 26]
    with pytest.raises(ValueError, match='24-layer'):
        continued.validate(plan)


def test_fresh_identities_cannot_overlap_training_or_prior_finals():
    with pytest.raises(ValueError, match='overlaps'):
        continued.overlap_ids(['aa'], ['aa'])
    with pytest.raises(ValueError, match='Repeated'):
        continued.overlap_ids(['aa', 'aa'])
    prior = continued.prior_role_ids(PLAN)
    assert len(prior) == 13120
    continued.overlap_ids(['0' * 64], ['1' * 64])
    assert '0' * 64 not in prior


def test_family_override_does_not_change_default_generation():
    default = tasks.make_case(1, 'train', 0)
    assert default['family'] == 'lookup'
    total = tasks.make_case(1, 'train', 0, family='total')
    assert total['family'] == 'total'
    assert total['rows'] == default['rows']
    with pytest.raises(ValueError, match='Unknown grounded task family'):
        tasks.make_case(1, 'train', 0, family='chat')


def test_generated_answer_gain_is_required_and_loss_cannot_pass(fixtures):
    new_rows, prior_rows = fixtures
    baseline_new = [False] * 256
    candidate_new = gain_on_totals(new_rows, 8)
    prior_correct = [False] * 128
    losses = {'new': [1.0] * 256, 'prior': [1.0] * 128, 'retention': [1.0] * 128}
    improved_loss = {'new': [0.5] * 256, 'prior': [0.5] * 128, 'retention': [1.0] * 128}
    payload = {'test-new': new_rows, 'test-prior': prior_rows, 'retention': []}
    passed = continued.decide(
        PLAN, payload,
        measurements(new_rows, prior_rows, baseline_new, prior_correct, losses),
        measurements(new_rows, prior_rows, candidate_new, prior_correct, losses))
    assert passed['passed']
    assert passed['outcomes']['test-new']['generation']['wins'] == 8
    assert passed['serving'] == 'not authorized by this decision'
    loss_only = continued.decide(
        PLAN, payload,
        measurements(new_rows, prior_rows, baseline_new, prior_correct, losses),
        measurements(new_rows, prior_rows, baseline_new, prior_correct, improved_loss))
    assert loss_only['passed'] is False
    assert loss_only['outcomes']['test-new']['loss']['mean_delta'] < 0


def test_sorting_gains_cannot_hide_lost_arithmetic(fixtures):
    new_rows, prior_rows = fixtures
    baseline_new = [row['task']['family'] == 'total' for row in new_rows]
    candidate_new = [row['task']['family'] == 'sort' for row in new_rows]
    prior_correct = [False] * 128
    losses = {'new': [1.0] * 256, 'prior': [1.0] * 128, 'retention': [1.0] * 128}
    result = continued.decide(
        PLAN, {'test-new': new_rows, 'test-prior': prior_rows, 'retention': []},
        measurements(new_rows, prior_rows, baseline_new, prior_correct, losses),
        measurements(new_rows, prior_rows, candidate_new, prior_correct, losses))
    assert result['passed'] is False
    assert result['outcomes']['test-new']['generation']['families']['total']['passed'] is False
    assert result['outcomes']['test-new']['generation']['families']['sort']['candidate_correct'] == 64


def test_zero_net_gain_fails_even_when_every_family_holds(fixtures):
    new_rows, prior_rows = fixtures
    baseline_new = [row['task']['family'] == 'lookup' for row in new_rows]
    losses = {'new': [1.0] * 256, 'prior': [1.0] * 128, 'retention': [1.0] * 128}
    result = continued.decide(
        PLAN, {'test-new': new_rows, 'test-prior': prior_rows, 'retention': []},
        measurements(new_rows, prior_rows, baseline_new, [False] * 128, losses),
        measurements(new_rows, prior_rows, baseline_new, [False] * 128, losses))
    generation = result['outcomes']['test-new']['generation']
    assert generation['wins'] == generation['losses'] == 0
    assert all(row['passed'] for row in generation['families'].values())
    assert result['passed'] is False


def test_conversation_retention_can_veto_an_answer_gain(fixtures):
    new_rows, prior_rows = fixtures
    baseline_new = [False] * 256
    candidate_new = gain_on_totals(new_rows, 8)
    losses = {'new': [1.0] * 256, 'prior': [1.0] * 128, 'retention': [1.0] * 128}
    worse = dict(losses)
    worse['retention'] = [2.0] * 128
    result = continued.decide(
        PLAN, {'test-new': new_rows, 'test-prior': prior_rows, 'retention': []},
        measurements(new_rows, prior_rows, baseline_new, [False] * 128, losses),
        measurements(new_rows, prior_rows, candidate_new, [False] * 128, worse))
    assert result['outcomes']['test-new']['generation']['passed']
    assert result['outcomes']['retention']['passed'] is False
    assert result['passed'] is False


def test_implementation_digest_covers_the_freeze_sources():
    for name in continued.SOURCE_PATHS:
        assert (continued.repo_root() / name).is_file()
    assert len(continued.implementation_digest()) == 64
