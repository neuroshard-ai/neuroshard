import json
from pathlib import Path

import pytest

from neuroshard.evolution import programming_growth as growth
from neuroshard.evolution import programming_growth_diagnosis as diagnosis
from neuroshard.evolution.programming_expert import MBPP_SHA
from neuroshard.evolution.reference_data import identity, sha256


ROOT = Path(__file__).resolve().parents[2]


def plan():
    return json.loads((ROOT / 'config/experiments/programming-growth.json').read_text())


def spec():
    return json.loads((ROOT / 'config/experiments/programming-growth-diagnosis.json').read_text())


def messages(text, example='assert example'):
    return [{'role': 'user', 'content': text + '\n\nUse this callable interface and behavior:\n'
             + example + '\n\nReturn only the complete Python code, including needed imports.'}]


def row(task_id, text, tests):
    return {'id': identity({'dataset': MBPP_SHA, 'task': task_id}), 'task_id': task_id,
            'kind': 'code', 'messages': messages(text, tests[0]), 'setup': '', 'tests': tests}


def output(item, arm, text, *, generated=True, path='expert', prompt_ids=None, ids=None):
    prompt_ids = [10] if prompt_ids is None else prompt_ids
    ids = ids or [2]
    packed = {
        'id': item['id'], 'arm': arm, 'text': '```python\n' + text + '\n```',
        'ids': ids, 'seconds': 1.0, 'prompt_kind': 'original' if generated else 'unused',
        'generated': generated, 'path': path, 'prompt_ids': prompt_ids,
        'input_tokens': len(prompt_ids), 'output_tokens': len(ids),
        'check_seconds': 0.05 if arm == 'base' else 0.0,
    }
    if not generated:
        packed['seconds'] = 0.0
        packed['path'] = 'parent'
    return packed


def unused(item, arm, text):
    return output(item, arm, text, generated=False, path='parent', ids=[1], prompt_ids=[10])


def fixture(unique_added=False):
    current = plan()
    fail = set(diagnosis.VISIBLE_FAIL_TASK_IDS)
    preservation = [row(n, 'preserve %s' % n, ['assert example', 'assert hidden'])
                    for n in current['splits']['preservation_task_ids']]
    new_rows = [row(n, 'new %s' % n, ['assert example', 'assert hidden'])
                for n in current['splits']['new_task_ids']]
    outputs = []
    added = []
    for item in preservation + new_rows:
        if item['task_id'] not in fail:
            outputs += [
                output(item, 'base', 'GOOD', path='parent', ids=[1], prompt_ids=[10]),
                unused(item, growth.INCUMBENT, 'GOOD'),
                unused(item, growth.MERGED, 'GOOD'),
            ]
            continue
        incumbent_good = item['task_id'] in {169, 258, 115, 249, 224}
        added_good = unique_added and (incumbent_good or item['task_id'] == 183)
        outputs += [
            output(item, 'base', 'BAD', path='parent', ids=[1], prompt_ids=[10]),
            output(item, growth.INCUMBENT, 'GOOD' if incumbent_good else 'BAD',
                   path='expert', ids=[2], prompt_ids=[10]),
            output(item, growth.MERGED, 'BAD', path='expert', ids=[3], prompt_ids=[10]),
        ]
        added.append(output(item, diagnosis.ADDED, 'GOOD' if added_good else 'BAD',
                            path='expert', ids=[4], prompt_ids=[10]))
    return current, spec(), preservation, new_rows, outputs, added


def test_committed_diagnosis_pins_the_failed_merge_artifacts():
    current = spec()
    assert current['train'] is False
    assert current['oracle_is_not_a_policy'] is True
    assert current['admission_evidence'] is False
    assert current['original_final_opened'] is False
    assert current['growth_plan'] == identity(plan())
    assert current['growth_outputs'] == sha256(ROOT / 'config/experiments/programming-growth-outputs.json')
    assert current['visible_fail_task_ids'] == diagnosis.VISIBLE_FAIL_TASK_IDS
    assert len(current['visible_fail_task_ids']) == 38


def test_oracle_is_an_upper_bound_not_a_policy():
    current, bound, preservation, new_rows, outputs, added = fixture(unique_added=True)
    check = lambda code, setup, tests: {'passed': code.strip() == 'GOOD'}
    result = diagnosis.score_complementarity(
        preservation, new_rows, outputs, added, current, bound, check)
    assert result['growth_merge_passed'] is False
    assert result['oracle_is_not_a_policy'] is True
    assert result['admission_evidence'] is False
    assert result['unique_incumbent'] == 0
    assert result['unique_added'] == 1
    assert result['both'] == 5
    assert result['oracle_gain_vs_incumbent'] == 1
    assert result['useful_additional_coverage'] is True
    assert result['next'] == 'selection'
    assert result['visible_fail'] == 38


def test_no_unique_added_coverage_stops_the_second_tail():
    current, bound, preservation, new_rows, outputs, added = fixture(unique_added=False)
    check = lambda code, setup, tests: {'passed': code.strip() == 'GOOD'}
    result = diagnosis.score_complementarity(
        preservation, new_rows, outputs, added, current, bound, check)
    assert result['unique_added'] == 0
    assert result['unique_incumbent'] == 5
    assert result['both'] == 0
    assert result['useful_additional_coverage'] is False
    assert result['next'] == 'stop'
    assert result['oracle_policy'] == result['incumbent_policy']


def test_bind_diagnosis_keeps_the_failed_growth_freeze():
    current = spec()
    added_manifest = {'plan': 'other'}
    with pytest.raises(ValueError, match='Added checkpoint'):
        diagnosis.bind_diagnosis(plan(), current, current['growth_outputs'], added_manifest)
    trained = dict(current)
    trained['train'] = True
    with pytest.raises(ValueError, match='may not train'):
        diagnosis.bind_diagnosis(plan(), trained, current['growth_outputs'],
                                 {'unused': True})
