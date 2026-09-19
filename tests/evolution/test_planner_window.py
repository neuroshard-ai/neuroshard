"""Consensus-side bounds and duplicate identities need no neural runtime."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from neuroshard.evolution import planner_window as window
from neuroshard.evolution.reference_data import identity


@pytest.fixture
def claim():
    source = Path(__file__).resolve().parents[2]/'config/experiments/owned-planner-results.json'
    initial = json.loads(source.read_bytes())['result']['initial']
    recipe = {'steps': 2, 'learning_rate': .001, 'warmup_steps': 0, 'minimum_lr_ratio': .1,
              'weight_decay': .01, 'clip_norm': 1., 'microbatch': 1, 'schedule': [[0, 1], [1, 0]]}
    initial['binding']['recipe'] = identity(recipe)
    following = copy.deepcopy(initial)
    following.update(step=1, sha256='b'*64, fusion='c'*64)
    profile = {'format': window.FORMAT+'/prescription', 'graph': initial['binding']['graph'],
               'initial': initial, 'row_count': 2, 'recipe': recipe,
               'batches': ['d'*64, 'e'*64], 'numerical_profile': 'f'*64, 'frozen': 'a'*64}
    value = {'format': window.FORMAT, 'prescription': identity(profile), 'checkpoints': [initial, following],
             'updates': [{'step': 1, 'loss': .1, 'gradient_norm': .2, 'planner': following['fusion'],
                          'learning_rate': window.learning_rate(recipe, 0)}]}
    return profile, initial, value


def test_work_identity_ignores_non_numerical_labels_but_binds_consumed_state_and_step_parameters(claim):
    profile, before, value = claim
    result = window.validate(profile, before, value)
    renamed_profile, renamed_before = copy.deepcopy(profile), copy.deepcopy(before)
    renamed_profile['graph'] = '1'*64
    renamed_before['binding']['graph'] = '1'*64
    renamed_before['binding']['rows'] = '2'*64
    renamed_profile['recipe']['schedule'][0] = [1, 0]
    # Actual canonical batch and effective optimizer inputs are the same.
    assert window.work_identity(renamed_profile, renamed_before, profile['batches'][0]) == result['work_ids'][0]
    with pytest.raises(ValueError, match='already been paid'):
        window.validate(profile, before, value, paid=result['work_ids'])
    for change in ('state', 'batch', 'frozen', 'learning_rate', 'microbatch'):
        altered_profile, altered_before = copy.deepcopy(profile), copy.deepcopy(before)
        batch = profile['batches'][0]
        if change == 'state': altered_before['sha256'] = '3'*64
        elif change == 'batch': batch = '4'*64
        elif change == 'frozen': altered_profile['frozen'] = '5'*64
        elif change == 'learning_rate': altered_profile['recipe']['learning_rate'] *= .5
        elif change == 'microbatch': altered_profile['recipe']['microbatch'] = 2
        assert window.work_identity(altered_profile, altered_before, batch) != result['work_ids'][0]


@pytest.mark.parametrize('change', ['cursor', 'rate', 'missing_state', 'binding', 'root', 'boolean_step', 'nan'])
def test_changed_or_incomplete_windows_fail_before_neural_replay(claim, change):
    profile, before, value = claim
    if change == 'cursor': value['checkpoints'][-1]['step'] = 2
    elif change == 'rate': value['updates'][0]['learning_rate'] *= .5
    elif change == 'missing_state': value['checkpoints'].pop()
    elif change == 'binding': value['checkpoints'][-1]['binding']['rows'] = '2'*64
    elif change == 'root': value['updates'][0]['planner'] = '2'*64
    elif change == 'boolean_step': value['updates'][0]['step'] = True
    elif change == 'nan': value['updates'][0]['loss'] = float('nan')
    with pytest.raises(ValueError):
        window.validate(profile, before, value)


def test_batch_identity_depends_on_tokens_targets_and_order_not_document_titles():
    rows = [{'id': 'old document', 'input_ids': [1, 3, 2], 'labels': [-100, 3, 2], 'targets': 2},
            {'id': 'another document', 'input_ids': [1, 4, 2], 'labels': [-100, 4, 2], 'targets': 2}]
    original = window.batch_root(rows, 8, 16)
    renamed = [{**row, 'id': 'changed'} for row in rows]
    assert window.batch_root(renamed, 8, 16) == original
    assert window.batch_root(list(reversed(rows)), 8, 16) != original
    changed = copy.deepcopy(rows)
    changed[0]['labels'][1] = -100
    changed[0]['targets'] = 1
    assert window.batch_root(changed, 8, 16) != original


def test_planner_window_validation_does_not_import_torch():
    subprocess.run([sys.executable, '-c',
        'import sys; from neuroshard.evolution import planner_window; assert "torch" not in sys.modules'],
        check=True, timeout=15)
