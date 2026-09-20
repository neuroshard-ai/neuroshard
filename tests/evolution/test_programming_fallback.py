import copy
import json
from pathlib import Path

import pytest

from neuroshard.evolution import programming_fallback as fallback
from neuroshard.evolution.programming_expert import MBPP_SHA
from neuroshard.evolution.reference_data import identity


def outputs_for(row, *, base, expert, repair, visible_fail=False, repair_ids=None):
    def pack(arm, text, ids, kind='original', generated=False, path='parent', prompt_ids=None):
        prompt_ids = [10] if prompt_ids is None else prompt_ids
        return {'id': row['id'], 'arm': arm, 'text': '```python\n' + text + '\n```',
                'ids': ids, 'seconds': 1.0, 'prompt_kind': kind, 'generated': generated,
                'path': path, 'prompt_ids': prompt_ids, 'input_tokens': len(prompt_ids),
                'output_tokens': len(ids), 'check_seconds': 0.05 if arm == 'base' else 0.0}
    if visible_fail:
        repair_kind, extra_prompt, extra_generated = 'repair', [10, 11, 12], True
        repair_ids = [3, 4] if repair_ids is None else repair_ids
    else:
        repair_kind, extra_prompt, extra_generated = 'unused', [10], False
        repair_ids = [1] if repair_ids is None else repair_ids
    return [
        pack('base', base, [1], prompt_ids=[10]),
        pack('expert', expert, [2], generated=extra_generated, path='expert', prompt_ids=[10]),
        pack('repair', repair, repair_ids, repair_kind, generated=extra_generated,
             path='parent', prompt_ids=extra_prompt),
    ]


def test_keeps_parent_when_public_example_passes_even_if_expert_is_better():
    row = {'id': 'a', 'setup': '', 'tests': ['assert example', 'assert hidden']}
    base = {'text': '```python\nGOOD\n```', 'ids': [1]}
    expert = {'text': '```python\nGOOD\n```', 'ids': [2]}
    check = lambda code, setup, tests: {'passed': tests == ['assert example'] or code.strip() == 'GOOD'}
    choice, chosen = fallback.choose_after_visible_example(base, expert, row, check)
    assert choice == 'base' and chosen is base


def test_uses_expert_only_after_public_example_fails():
    row = {'id': 'a', 'setup': '', 'tests': ['assert example', 'assert hidden']}
    base = {'text': '```python\nBAD\n```', 'ids': [1]}
    expert = {'text': '```python\nGOOD\n```', 'ids': [2]}
    check = lambda code, setup, tests: {'passed': code.strip() == 'GOOD'}
    choice, chosen = fallback.choose_after_visible_example(base, expert, row, check)
    assert choice == 'specialist' and chosen is expert


def test_repair_prompt_contains_failed_program_and_only_the_public_example():
    messages = [{'role': 'user', 'content': 'Solve.\n\nUse this callable interface and behavior:\nassert f(1)==1\n\nReturn only the complete Python code, including needed imports.'}]
    repaired = fallback.repair_messages(messages, 'def f(): return 0', 'assert f(1)==1')
    assert repaired[1]['role'] == 'assistant' and 'return 0' in repaired[1]['content']
    assert 'assert f(1)==1' in repaired[2]['content']
    assert 'assert f(2)' not in repaired[2]['content']


def test_comparison_requires_expert_to_beat_parent_and_equal_budget_repair():
    plan = {'seed': 1, 'bootstrap_samples': 200,
            'gate': {'minimum_code_gain': 0.05, 'maximum_p95_ratio': 1.5, 'maximum_p95_seconds': 90}}
    rows = [{'id': str(i), 'setup': '', 'tests': ['assert example', 'assert hidden']} for i in range(8)]
    outputs = []
    for i, row in enumerate(rows):
        if i < 3:
            outputs += outputs_for(row, base='GOOD', expert='BAD', repair='BAD')
        elif i < 6:
            outputs += outputs_for(row, base='BAD', expert='GOOD', repair='BAD', visible_fail=True)
        elif i == 6:
            outputs += outputs_for(row, base='BAD', expert='GOOD', repair='GOOD', visible_fail=True)
        else:
            outputs += outputs_for(row, base='BAD', expert='BAD', repair='BAD', visible_fail=True)
    check = lambda code, setup, tests: {'passed': code.strip() == 'GOOD'}
    result = fallback.score_comparison(rows, outputs, plan, check)
    assert result['net_vs_base'] == 4
    assert result['net_vs_repair'] == 3
    assert result['passed'] and result['original_final_opened'] is False
    assert result['equal_computation'] is False
    assert result['equal_extra_attempt'] and result['equal_output_token_cap']
    assert result['repair_p95_input_tokens'] > result['fallback_p95_input_tokens']
    failed = copy.deepcopy(outputs)
    for out in failed:
        if out['arm'] == 'expert':
            out['text'] = '```python\nBAD\n```'
    assert not fallback.score_comparison(rows, failed, plan, check)['gates']['gain_vs_repair']


def test_repeated_repair_tokens_are_scored_when_the_generation_was_recorded():
    plan = {'seed': 1, 'bootstrap_samples': 20,
            'gate': {'minimum_code_gain': 0.05, 'maximum_p95_ratio': 1.5, 'maximum_p95_seconds': 90}}
    row = {'id': 'x', 'setup': '', 'tests': ['assert example']}
    outputs = outputs_for(row, base='BAD', expert='GOOD', repair='BAD', visible_fail=True, repair_ids=[1])
    check = lambda code, setup, tests: {'passed': code.strip() == 'GOOD'}
    result = fallback.score_comparison([row], outputs, plan, check)
    assert result['details'][0]['repeated_repair_tokens'] is True
    assert result['details'][0]['scores']['base_repair'] is False
    assert result['net_vs_base'] == 1


def test_comparison_rejects_a_copied_first_attempt_as_the_extra_decode():
    plan = {'seed': 1, 'bootstrap_samples': 20,
            'gate': {'minimum_code_gain': 0.05, 'maximum_p95_ratio': 1.5, 'maximum_p95_seconds': 90}}
    row = {'id': 'x', 'setup': '', 'tests': ['assert example']}
    outputs = outputs_for(row, base='BAD', expert='GOOD', repair='BAD', visible_fail=True)
    outputs[2]['generated'] = False
    with pytest.raises(ValueError, match='generation call'):
        fallback.score_comparison([row], outputs, plan, lambda *a: {'passed': False})
    outputs = outputs_for(row, base='BAD', expert='GOOD', repair='BAD', visible_fail=True)
    outputs[2]['prompt_ids'] = [10]
    outputs[2]['input_tokens'] = 1
    with pytest.raises(ValueError, match='reused the original prompt'):
        fallback.score_comparison([row], outputs, plan, lambda *a: {'passed': False})


def test_opened_development_diagnostic_cannot_admit():
    rows = [{'id': 'a', 'kind': 'code', 'setup': '', 'tests': ['assert example', 'assert hidden']},
            {'id': 'b', 'kind': 'general'}]
    outputs = [
        {'id': 'a', 'arm': 'base', 'text': '```python\nBAD\n```'},
        {'id': 'a', 'arm': 'replacement', 'text': '```python\nGOOD\n```'},
    ]
    check = lambda code, setup, tests: {'passed': code.strip() == 'GOOD'}
    report = fallback.score_opened_development_diagnostic(rows, outputs, check)
    assert report['expert_fallback_correct'] == 1
    assert report['admission_evidence'] is False
    assert report['new_method_not_frozen_before_observation'] is True


def test_leftover_ids_match_the_frozen_list():
    plan = json.loads((Path(__file__).resolve().parents[2]
                       / 'config/experiments/programming-fallback.json').read_text())
    chosen = fallback.leftover_task_ids(
        plan, excluded=plan['comparison']['excluded_parent_pool_tasks'],
        parent_final_tasks=plan['comparison']['parent_final_task_ids'],
        all_pool_tasks=list(range(11, 511)))
    assert chosen == plan['comparison']['task_ids']
    plan['comparison']['task_ids'] = plan['comparison']['task_ids'][1:] + [12]
    with pytest.raises(ValueError, match='leftover'):
        fallback.leftover_task_ids(
            plan, excluded=plan['comparison']['excluded_parent_pool_tasks'],
            parent_final_tasks=plan['comparison']['parent_final_task_ids'],
            all_pool_tasks=list(range(11, 511)))


def test_runtime_rejects_a_reserved_final_task_even_if_preparation_metadata_agrees():
    plan = json.loads((Path(__file__).resolve().parents[2]
                       / 'config/experiments/programming-fallback.json').read_text())
    reserved = plan['comparison']['parent_final_task_ids'][0]
    rows = [{'id': identity({'dataset': MBPP_SHA, 'task': n}), 'task_id': n, 'kind': 'code'}
            for n in plan['comparison']['task_ids']]
    rows[0] = {'id': identity({'dataset': MBPP_SHA, 'task': reserved}),
               'task_id': reserved, 'kind': 'code'}
    prepared_task_ids = [row['task_id'] for row in rows]
    assert reserved in prepared_task_ids
    with pytest.raises(ValueError, match='committed leftover list'):
        fallback.load_comparison_rows(plan, rows)


def test_bind_execution_requires_the_rejected_checkpoint_identity():
    plan = json.loads((Path(__file__).resolve().parents[2]
                       / 'config/experiments/programming-fallback.json').read_text())
    selection = json.loads((Path(__file__).resolve().parents[2]
                            / 'config/experiments/programming-expert-selection.json').read_bytes())
    manifest = {'plan': fallback.PARENT_PLAN, 'selection': fallback.PARENT_SELECTION}
    with pytest.raises(ValueError, match='Expert checkpoint'):
        fallback.bind_execution(plan, selection, manifest, fallback.TOKENIZER)
    with pytest.raises(ValueError, match='Tokenizer'):
        fallback.bind_execution(plan, selection, {'plan': fallback.PARENT_PLAN,
                                                  'selection': fallback.PARENT_SELECTION}, '0' * 64)
    freeze = fallback.execution_freeze(plan, selection, 'commit')
    assert freeze['parent_expert'] == fallback.PARENT_EXPERT
    assert freeze['tokenizer'] == fallback.TOKENIZER
    assert set(freeze['sources']) == set(fallback.EXECUTION_SOURCES)
    assert 'src/neuroshard/evolution/sharded/cached_inference.py' in freeze['sources']
    manifest_path = Path(__file__).resolve().parents[2] / '.neuroshard/programming-expert-20260920/evidence/3/expert/manifest.json'
    if manifest_path.is_file():
        accepted = fallback.bind_execution(
            plan, selection, json.loads(manifest_path.read_bytes()), fallback.TOKENIZER)
        assert accepted['parent_expert'] == fallback.PARENT_EXPERT
