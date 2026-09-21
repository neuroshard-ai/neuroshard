import json
import math
from pathlib import Path

import pytest

from neuroshard.evolution.programming_selector import (
    CONTRACT_IDENTITY, FORMAT, NearestTrainPicker, bind_contract, bind_picker_freeze,
    build_view, decide_picker_calls, jaccard, nearest_rank_p95, public_example_from_question,
    serve, validate_view,
)
from neuroshard.evolution.reference_data import identity


ROOT = Path(__file__).resolve().parents[2]


def contract():
    return json.loads((ROOT / 'config/experiments/programming-selector-contract.json').read_text())


def question(text='sort a list', example='assert f([1])==[1]'):
    return (text + '\n\nUse this callable interface and behavior:\n' + example
            + '\n\nReturn only the complete Python code, including needed imports.')


def view(text='sort a list', example='assert f([1])==[1]', parent='```python\npass\n```',
         status='execution-error'):
    prompt = question(text, example)
    return {
        'question': prompt,
        'failed_parent_program': parent,
        'public_example': example,
        'public_feedback': {'passed': False, 'status': status},
    }


def assets(incumbent=None, added=None):
    incumbent = incumbent or [question('incumbent train on matrices', 'assert m()==1')]
    added = added or [question('added train on string palindromes', 'assert p("aba")==True')]
    return {
        'format': FORMAT + '/assets',
        'incumbent_prompts': incumbent,
        'added_prompts': added,
        'provenance': {'synthetic': True},
    }


def spec_for(payload):
    return {
        'format': FORMAT + '/picker',
        'rule': 'nearest-train-jaccard',
        'assets': identity(payload),
        'margin': 0,
        'default': 'incumbent',
        'tie': 'incumbent',
        'uses_fields': ['question'],
        'case_specific_lookup_rules': False,
    }


def test_frozen_contract_identity_is_pinned():
    assert identity(contract()) == CONTRACT_IDENTITY
    bind_contract(contract())


def test_public_example_is_taken_from_the_original_user_message():
    prompt = question('compute gcd', 'assert gcd(2,4)==2')
    assert public_example_from_question(prompt) == 'assert gcd(2,4)==2'
    with pytest.raises(ValueError, match='frozen original user message'):
        public_example_from_question('compute gcd')


def test_picker_rejects_extra_keys_and_success_feedback():
    current = view()
    current['task_id'] = 276
    with pytest.raises(ValueError, match='exactly the frozen object'):
        validate_view(current)
    bad = view()
    bad['public_feedback'] = {'passed': True, 'status': 'passed'}
    with pytest.raises(ValueError, match='only called after the public example fails'):
        validate_view(bad)


def test_build_view_skips_picker_when_the_public_example_passes():
    check = lambda code, setup, tests: {'passed': True, 'status': 'passed'}
    assert build_view(question(), '```python\npass\n```', check) is None
    fail = lambda code, setup, tests: {'passed': False, 'status': 'execution-error'}
    built = build_view(question(), '```python\npass\n```', fail)
    assert built['public_feedback'] == {'passed': False, 'status': 'execution-error'}
    extract = build_view(question(), 'def (', fail)
    assert extract['public_feedback']['status'] == 'extraction-error'


def test_nearest_train_jaccard_picks_added_only_on_a_strict_win():
    payload = assets()
    picker = NearestTrainPicker(spec_for(payload), payload)
    added_like = view('added train on string palindromes', 'assert p("aba")==True')
    incumbent_like = view('incumbent train on matrices', 'assert m()==1')
    assert picker.pick(added_like)['choice'] == 'added'
    assert picker.pick(incumbent_like)['choice'] == 'incumbent'
    same = assets([question('same prompt', 'assert f()==1')],
                  [question('same prompt', 'assert f()==1')])
    tied = NearestTrainPicker(spec_for(same), same)
    assert tied.pick(view('same prompt', 'assert f()==1'))['choice'] == 'incumbent'


def test_invalid_picker_output_defaults_to_incumbent():
    payload = assets()
    picker = NearestTrainPicker(spec_for(payload), payload)
    served = serve(picker, {'question': 'bad'})
    assert served['selected'] == 'incumbent'
    assert served['defaulted'] is True
    assert served['error']


def test_nearest_rank_p95_uses_the_frozen_index_rule():
    samples = [float(i) for i in range(38)]
    assert nearest_rank_p95(samples) == samples[math.ceil(0.95 * 38) - 1]
    assert nearest_rank_p95(samples) == 36.0


def test_jaccard_is_the_programming_expert_word_rule_without_the_threshold():
    assert jaccard({'a', 'b'}, {'b', 'c'}) == 1 / 3


def test_decide_records_38_calls_without_tail_answers():
    current = contract()
    payload = assets()
    picker = NearestTrainPicker(spec_for(payload), payload)
    rows = []
    parent = []
    for task_id in current['cpu_screen']['picker_call_task_ids']:
        item = {'id': identity({'dataset': 'synthetic', 'task': task_id}), 'task_id': task_id,
                'messages': [{'role': 'user', 'content': question('unrelated graph traversal %s' % task_id)}],
                'setup': '', 'tests': ['assert f()==1']}
        rows.append(item)
        parent.append({'id': item['id'], 'arm': 'base', 'text': '```python\npass\n```'})
    check = lambda *a: {'passed': False, 'status': 'execution-error'}
    decisions = decide_picker_calls(rows, parent, picker, check, current)
    assert decisions['count'] == 38
    assert [row['task_id'] for row in decisions['decisions']] == current['cpu_screen']['picker_call_task_ids']
    assert all(item['selected'] in ('incumbent', 'added') for item in decisions['decisions'])
    assert all(item['error'] is None and item['overrun'] is False for item in decisions['decisions'])


def view_from_prompt(prompt):
    example = public_example_from_question(prompt)
    return {
        'question': prompt,
        'failed_parent_program': '```python\npass\n```',
        'public_example': example,
        'public_feedback': {'passed': False, 'status': 'execution-error'},
    }


def test_workspace_picker_freeze_binds_the_specified_candidate():
    spec = json.loads((ROOT / 'config/experiments/programming-selector-picker.json').read_text())
    payload = json.loads((ROOT / 'config/experiments/programming-selector-assets.json').read_text())
    freeze = json.loads((ROOT / 'config/experiments/programming-selector-picker-freeze.json').read_text())
    bind_picker_freeze(freeze, spec, payload, contract())
    picker = NearestTrainPicker(spec, payload)
    assert picker.pick(view_from_prompt(payload['added_prompts'][0]))['choice'] == 'added'
    assert picker.pick(view_from_prompt(payload['incumbent_prompts'][0]))['choice'] == 'incumbent'


def test_picker_freeze_rejects_a_changed_asset_hash():
    payload = assets()
    spec = spec_for(payload)
    freeze = {
        'format': FORMAT + '/picker-execution-freeze',
        'contract': CONTRACT_IDENTITY,
        'evaluation_freeze_commit': '8e39b746ab7087e1c1a47c437b644c939f36e739',
        'picker': identity(spec),
        'assets': identity(payload),
        'cpu_screen_performed': False,
        'gpu_launch_authorized': False,
        'files': {},
    }
    bind_picker_freeze(freeze, spec, payload, contract())
    payload['added_prompts'] = [question('changed')]
    with pytest.raises(ValueError, match='does not match the committed artifacts'):
        bind_picker_freeze(freeze, spec, payload, contract())
