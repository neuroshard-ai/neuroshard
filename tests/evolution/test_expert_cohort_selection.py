import json
from types import SimpleNamespace

import pytest

from neuroshard.evolution import cohort_experiment as contract, reference_data as data
from neuroshard.evolution.sharded import cohort_job
from test_expert_cohort_job import inputs


def test_final_requires_committed_eligible_terminal_selection(monkeypatch):
    plan, prepared = {'training': {'steps': 560}}, {'bound': 'inputs'}
    selection = {'format': contract.FORMAT + '/selection', 'eligible': True,
                 'plan': data.identity(plan), 'prepared': data.identity(prepared), 'step': 560}
    monkeypatch.setattr(contract.base, 'committed', lambda path: json.dumps(selection).encode())
    assert contract.final_selection(plan, prepared) == selection
    for key, wrong in [('eligible', False), ('step', 280), ('prepared', '0' * 64), ('plan', '0' * 64)]:
        original = selection[key]
        selection[key] = wrong
        with pytest.raises(ValueError, match='eligible terminal'):
            contract.final_selection(plan, prepared)
        selection[key] = original


def test_wrong_selected_learning_job_fails_before_neural_loading(tmp_path, monkeypatch):
    plan, prepared = inputs(tmp_path)
    second = cohort_job.read(tmp_path / 'first.json')
    data.save(tmp_path / 'second.json', second)
    monkeypatch.setattr(contract, 'validate', lambda: (plan, prepared))
    monkeypatch.setattr(contract, 'final_selection', lambda *args: {
        'checkpoint': data.identity(second), 'graph': '0' * 64})

    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid selection reached neural allocation')

    monkeypatch.setattr(cohort_job, 'Partition', forbidden)
    args = SimpleNamespace(command='final', parent=tmp_path / 'parent.json', first=tmp_path / 'first.json',
                           second=tmp_path / 'second.json')
    with pytest.raises(ValueError, match='selected terminal graph'):
        cohort_job.run(args)
