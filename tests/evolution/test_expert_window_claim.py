"""Validate real replay records, including relabeling and omitted-state attacks."""
import copy
import json

import pytest

from neuroshard.evolution import expert_checkpoint, expert_window, reference_data as data
from test_expert_window_replay import test_replay_reconstructs_the_actual_five_owner_training_checkpoint as replay_fixture


@pytest.fixture(scope='module')
def records(tmp_path_factory):
    home = tmp_path_factory.mktemp('expert-window-claim')
    replay_fixture(home)
    read = lambda path: json.loads(path.read_bytes())
    parent = read(home/'parent.json')
    window = read(home/'replay/window-000000.json')
    checkpoints = [read(home/'replay'/f'checkpoint-{i:06d}.json') for i in range(5)]
    bank = read(home/'rank-4/features/index.json')
    inputs = {'feature_root': data.identity(bank), 'prepared': data.identity(read(home/'prepared.json')),
              'feature_stages': 3 * sum(len(batch['files']) for batch in bank['batches'])}
    return parent, window, checkpoints, inputs


def validate(parent, window, states, paid=()):
    return expert_window.validate(parent, states[0], window, states,
        [row['batch'] for row in window['steps']], 'b'*64, paid)


def test_real_bounded_window_binds_every_actual_state(records):
    parent, window, states, _ = records
    claim = validate(parent, window, states)
    assert claim['steps'] == 4 and len(set(claim['work_ids'])) == 4
    assert claim['record_root'] == data.identity(window)
    for missing in (states[:1]+states[2:], states[:-1]):
        with pytest.raises(ValueError, match='every intermediate'):
            validate(parent, window, missing)
    forged = copy.deepcopy(window)
    forged['steps'][1]['work_identity'] = 'a'*64
    with pytest.raises(ValueError, match='actual consumed'):
        validate(parent, forged, states)
    batches = [row['batch'] for row in window['steps']]
    batches[2] = 'c'*64
    with pytest.raises(ValueError, match='prescribed batch'):
        expert_window.validate(parent, states[0], window, states, batches, 'b'*64)


def test_renaming_the_job_does_not_make_the_same_updates_payable(records):
    parent, original, states, _ = records
    paid = validate(parent, original, states)['work_ids']
    renamed = []
    for value in states:
        common = expert_checkpoint.unpack(parent, value)
        common['job'] = 'f'*64
        changed = {**value, 'job': common['job']}
        common = expert_checkpoint.reconstruct(parent, changed)
        changed.update(checkpoint=data.identity(common), state_root=common['state_root'])
        renamed.append(changed)
    window = copy.deepcopy(original)
    window['input'], window['output'] = renamed[0], renamed[-1]
    for i, step in enumerate(window['steps']):
        step.update(input_checkpoint=renamed[i]['checkpoint'], output_checkpoint=renamed[i+1]['checkpoint'])
    # All full checkpoint identities changed; all numerical work identities did not.
    assert renamed[0]['checkpoint'] != states[0]['checkpoint']
    assert validate(parent, window, renamed)['work_ids'] == paid
    with pytest.raises(ValueError, match='already been paid'):
        validate(parent, window, renamed, paid)
