import json
from pathlib import Path

import pytest

from neuroshard.evolution import programming_fallback as fallback
from neuroshard.evolution import programming_growth as growth
from neuroshard.evolution.programming_expert import MBPP_SHA
from neuroshard.evolution.reference_data import identity


ROOT = Path(__file__).resolve().parents[2]


def plan():
    return json.loads((ROOT / 'config/experiments/programming-growth.json').read_text())


def leftover():
    return json.loads((ROOT / 'config/experiments/programming-fallback.json').read_text())


def messages(text, example='assert example'):
    return [{'role': 'user', 'content': text + '\n\nUse this callable interface and behavior:\n'
             + example + '\n\nReturn only the complete Python code, including needed imports.'}]


def row(task_id, text, tests):
    return {'id': identity({'dataset': MBPP_SHA, 'task': task_id}), 'task_id': task_id,
            'kind': 'code', 'messages': messages(text, tests[0]), 'setup': '', 'tests': tests}


def output(row, arm, text, *, generated=True, path='expert', prompt_ids=None, ids=None):
    prompt_ids = [10] if prompt_ids is None else prompt_ids
    return {'id': row['id'], 'arm': arm, 'text': '```python\n' + text + '\n```',
            'ids': ids or [2], 'seconds': 1.0, 'prompt_kind': 'original' if generated else 'unused',
            'generated': generated, 'path': path, 'prompt_ids': prompt_ids,
            'input_tokens': len(prompt_ids), 'output_tokens': len(ids or [2]),
            'check_seconds': 0.05 if arm == 'base' else 0.0}


def test_leftover_ranking_helper_matches_the_committed_preservation_set():
    parent = leftover()
    ranked = fallback.ranked_leftover_tasks(parent['comparison'])
    assert ranked[:32] == parent['comparison']['task_ids']
    assert growth.bind_splits(plan())['preservation_task_ids'] == ranked[:32]
    assert growth.bind_splits(plan())['new_task_ids'] == ranked[32:64]


def test_growth_splits_are_disjoint_from_the_reserved_final_and_incumbent_train():
    current = plan()
    splits = growth.bind_splits(current)
    used = (splits['preservation_task_ids'] + splits['new_task_ids']
            + splits['development_task_ids'] + splits['train_task_ids'])
    excluded = splits['excluded_near_duplicate_train_task_ids']
    ranked = fallback.ranked_leftover_tasks(leftover()['comparison'])
    blocked = set(current['parent_final_task_ids']) | set(current['excluded_parent_pool_tasks'])
    remainder = ranked[96:]
    assert not set(used) & blocked
    assert not set(used + excluded) & set(splits['incumbent_train_task_ids'])
    assert set(remainder) == set(splits['train_task_ids']) | set(excluded)
    assert not set(splits['train_task_ids']) & set(excluded)
    assert splits['required_success_task_ids'] == [399, 201, 169, 505, 11, 478, 258, 412, 115, 70, 292, 249]


def test_unit_task_vector_is_incumbent_plus_added_minus_parent():
    assert growth.task_vector_merge(5, 6, 8, 1, 1) == 9
    assert growth.task_vector_merge(0, 2, 3, 1, 1) == 5
    with pytest.raises(ValueError, match='unit task-vector'):
        growth.task_vector_merge(0, 1, 1, 2, 1)
    assert growth.merge_scales(plan()) == {'incumbent': 1, 'added': 1}


def test_isolation_reuses_the_leftover_gates_on_the_development_slice():
    current = {'seed': 1, 'bootstrap_samples': 200,
               'isolation_gate': {'minimum_code_gain': 0.05, 'maximum_p95_ratio': 1.5,
                                  'maximum_p95_seconds': 90}}
    rows = [row(i, 'task %s' % i, ['assert example', 'assert hidden']) for i in range(8)]
    outputs = []
    for i, item in enumerate(rows):
        if i < 3:
            outputs += [
                output(item, 'base', 'GOOD', path='parent', ids=[1]),
                output(item, 'expert', 'BAD', generated=False, path='parent', ids=[1]),
                output(item, 'repair', 'BAD', generated=False, path='parent', ids=[1]),
            ]
        else:
            outputs += [
                output(item, 'base', 'BAD', path='parent', ids=[1], prompt_ids=[10]),
                output(item, 'expert', 'GOOD', path='expert', ids=[2], prompt_ids=[10]),
                output(item, 'repair', 'BAD', path='parent', ids=[3], prompt_ids=[10, 11]),
            ]
            outputs[-1]['prompt_kind'] = 'repair'
    check = lambda code, setup, tests: {'passed': code.strip() == 'GOOD'}
    result = growth.score_isolation(rows, outputs, current, check)
    assert result['passed'] and result['net_vs_base'] == 5
    assert result['new_answers_opened'] is False
    assert result['admission_evidence'] is False


def test_growth_requires_preserved_successes_and_new_answer_gain():
    current = plan()
    preservation_ids = current['splits']['preservation_task_ids']
    new_ids = current['splits']['new_task_ids']
    required = set(current['splits']['required_success_task_ids'])
    parent_kept = {399, 201, 505, 11, 478, 412, 70, 292}
    specialist_kept = {169, 258, 115, 249}
    assert parent_kept | specialist_kept == required
    preservation = [row(n, 'preserve %s' % n, ['assert example', 'assert hidden']) for n in preservation_ids]
    new_rows = [row(n, 'matrix multiply %s' % n, ['assert example', 'assert hidden']) for n in new_ids]
    outputs = []
    check = lambda code, setup, tests: {'passed': code.strip() == 'GOOD'}

    def unused(item, arm, text):
        packed = output(item, arm, text, path='parent', ids=[1], generated=False, prompt_ids=[10])
        packed['seconds'] = 0.0
        packed['prompt_kind'] = 'unused'
        return packed

    for item in preservation:
        if item['task_id'] in parent_kept:
            outputs += [
                output(item, 'base', 'GOOD', path='parent', ids=[1], prompt_ids=[10]),
                unused(item, growth.INCUMBENT, 'GOOD'),
                unused(item, growth.MERGED, 'GOOD'),
            ]
        elif item['task_id'] in specialist_kept:
            outputs += [
                output(item, 'base', 'BAD', path='parent', ids=[1], prompt_ids=[10]),
                output(item, growth.INCUMBENT, 'GOOD', path='expert', ids=[2], prompt_ids=[10]),
                output(item, growth.MERGED, 'GOOD', path='expert', ids=[3], prompt_ids=[10]),
            ]
        else:
            outputs += [
                output(item, 'base', 'BAD', path='parent', ids=[1], prompt_ids=[10]),
                output(item, growth.INCUMBENT, 'BAD', path='expert', ids=[2], prompt_ids=[10]),
                output(item, growth.MERGED, 'BAD', path='expert', ids=[3], prompt_ids=[10]),
            ]
    for i, item in enumerate(new_rows):
        outputs += [
            output(item, 'base', 'BAD', path='parent', ids=[1], prompt_ids=[10]),
            output(item, growth.INCUMBENT, 'BAD', path='expert', ids=[2], prompt_ids=[10]),
            output(item, growth.MERGED, 'GOOD' if i < 4 else 'BAD', path='expert',
                   ids=[3], prompt_ids=[10]),
        ]
    result = growth.score_growth(preservation, new_rows, outputs, current, check)
    assert result['gates']['preserved_successes']
    assert result['net_vs_baseline_new'] == 4
    assert result['passed'] and result['task_vector_merge']
    assert result['original_final_opened'] is False
    lost = []
    target = next(item['id'] for item in preservation if item['task_id'] == 169)
    for out in outputs:
        item = dict(out)
        if out['arm'] == growth.MERGED and out['id'] == target:
            item['text'] = '```python\nBAD\n```'
        lost.append(item)
    assert not growth.score_growth(
        preservation, new_rows, lost, current, check)['gates']['preserved_successes']


def test_near_duplicate_training_prompts_are_dropped_and_eval_ids_stay_fixed():
    texts = {
        10: 'compute the factorial of a number',
        20: 'compute the factorial of a number',
        21: 'sort a list of unique integers',
    }
    assert growth.near_duplicate_train_exclusions(texts, [20, 21], [10]) == [20]
    current = plan()
    current['splits'] = dict(current['splits'])
    current['splits']['excluded_near_duplicate_train_task_ids'] = []
    with pytest.raises(ValueError, match='training IDs changed'):
        growth.bind_splits(current)


def test_loader_rejects_prepared_metadata_that_swaps_in_a_holdout():
    current = plan()
    holdout = current['splits']['new_task_ids'][0]
    prepared = {'roles': {'train': {
        'file': 'train.jsonl', 'sha256': '00', 'ids': ['x'], 'task_ids': [holdout]}}}
    import importlib.util
    path = ROOT / 'scripts/run_programming_growth.py'
    spec = importlib.util.spec_from_file_location('growth_loader_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match='committed plan'):
        module.read_role(ROOT, prepared, 'train', current)


def test_added_tail_must_still_be_the_parent_before_training():
    growth.require_independent_added_tail({'w': 5}, {'w': 5})
    with pytest.raises(ValueError, match='parent, not the incumbent'):
        growth.require_independent_added_tail({'w': 7}, {'w': 5})
    parent, inc, add = 10, 12, 13
    assert growth.task_vector_merge(parent, inc, add, 1, 1) == 15
    doubled = inc + (add - parent)
    assert growth.task_vector_merge(parent, inc, doubled, 1, 1) == 17


def test_bind_execution_requires_the_incumbent_leftover_checkpoint():
    current = plan()
    selection = json.loads((ROOT / 'config/experiments/programming-expert-selection.json').read_bytes())
    manifest = {'plan': growth.PARENT_PLAN, 'selection': growth.PARENT_SELECTION}
    with pytest.raises(ValueError, match='Incumbent checkpoint'):
        growth.bind_execution(current, selection, manifest, growth.TOKENIZER)
    freeze = growth.execution_freeze(current, selection, 'commit')
    assert freeze['incumbent_expert'] == growth.INCUMBENT_EXPERT
    assert freeze['parent_fallback_plan'] == growth.PARENT_FALLBACK_PLAN
    assert 'src/neuroshard/evolution/programming_growth.py' in freeze['sources']
    assert 'scripts/run_programming_expert.py' in freeze['sources']
    assert 'src/neuroshard/evolution/sharded/cohort_features.py' in freeze['sources']
    assert 'src/neuroshard/evolution/sharded/incremental.py' in freeze['sources']
