import copy
import importlib.util
from pathlib import Path

import pytest

from neuroshard.evolution import grounded_tasks as tasks, reference_data as data

spec = importlib.util.spec_from_file_location(
    'adaptive_quality', Path(__file__).resolve().parents[2]/'scripts/score_adaptive_shards.py')
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


def fixture():
    plan = {'generation_cases_per_cohort': 2, 'quality_gate': {
        'bootstrap_seed': 17, 'bootstrap_samples': 10000, 'confidence': .95,
        'phase_one_checkpoint': 128, 'phase_two_checkpoint': 256,
        'gain_upper_below_nats': -.001, 'retention_upper_at_most_nats': .02}}
    prepared = {'plan': plan}
    rows = [{'id': str(i), 'task': tasks.make_case(13, 'test', i)} for i in range(4)]
    records = {'test-a': rows, 'retention': rows}
    checkpoint = {'step': 128}
    selection = {'prepared': data.identity(prepared), 'checkpoints': [data.identity(checkpoint)]}
    outcomes = {}
    for role in records:
        outcomes[role] = {'losses': [{'id': r['id'], 'loss': 1.} for r in rows], 'answers': []}
        if role == 'test-a':
            import json
            for row in rows[:2]:
                text = json.dumps(tasks.expected(row['task']))
                outcomes[role]['answers'].append({'id': row['id'], 'text': text,
                                                 'check': tasks.check_answer(row['task'], text)})
    before = {'prepared': data.identity(prepared), 'checkpoint': 'seed', 'outcomes': outcomes}
    after = copy.deepcopy(before)
    after['checkpoint'] = data.identity(checkpoint)
    for row in after['outcomes']['test-a']['losses']:
        row['loss'] -= .05
    return prepared, selection, before, after, checkpoint, records


def test_learning_must_preserve_retention_and_generated_answers():
    args = fixture()
    assert quality.score(*args, phase='a')['passed']
    args[3]['outcomes']['retention']['losses'][0]['loss'] = 2.
    assert not quality.score(*args, phase='a')['passed']
    args = fixture()
    answer = args[3]['outcomes']['test-a']['answers'][0]
    answer['text'] = '{}'
    answer['check'] = tasks.check_answer(args[5]['test-a'][0]['task'], '{}')
    assert not quality.score(*args, phase='a')['passed']


def test_partial_or_falsely_scored_outputs_are_not_quality_evidence():
    args = fixture()
    args[3]['outcomes']['test-a']['losses'].pop()
    with pytest.raises(ValueError, match='Incomplete'):
        quality.score(*args, phase='a')
    args = fixture()
    args[3]['outcomes']['test-a']['answers'][0]['text'] = '{}'
    with pytest.raises(ValueError, match='Reported correctness'):
        quality.score(*args, phase='a')


def test_endpoint_and_job_bindings_are_enforced():
    args = fixture()
    args[4]['step'] = 64
    with pytest.raises(ValueError, match='Unselected'):
        quality.score(*args, phase='a')
    args = fixture()
    args[3]['prepared'] = 'another job'
    with pytest.raises(ValueError, match='another job'):
        quality.score(*args, phase='a')


def test_positive_mean_is_not_rescued_by_bootstrap_or_nonfinite_values():
    gate = fixture()[0]['plan']['quality_gate']
    result = quality.paired([1., 1., 1.], [1.01, 1.02, 1.03], gate)
    assert result['upper'] > result['mean_delta'] > 0
    with pytest.raises(ValueError, match='Nonfinite'):
        quality.paired([1.], [float('nan')], gate)
