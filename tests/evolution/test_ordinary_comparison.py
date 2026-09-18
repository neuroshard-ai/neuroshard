"""An equal-budget comparison resumes without one oversized backend call."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import ordinary_comparison
import ordinary_campaign_backend
sys.path.pop(0)

from neuroshard.evolution.reference_data import save


def test_comparison_yields_between_arms_and_keeps_the_control_deadline(tmp_path, monkeypatch):
    job = {'lifecycle': {'candidate_template': {}, 'serving_graph': {},
                         'quality': {'policy_root': 'quality'}}}
    state = {'expert_lifecycle': {'admission': {'active': {'job': job}}},
             'expert_work': {'checkpoint': {}}}
    context = tmp_path/'job'
    context.mkdir()
    save(context/'quality-0.json', {'result': {'decision': {'passed': True}}})
    save(tmp_path/'comparison-growth-start.json',
         {'started': datetime.now(timezone.utc).isoformat()})
    backend = SimpleNamespace(home=tmp_path, cloud=SimpleNamespace(remaining=lambda: None),
        context=lambda job: {'directory': context},
        freeze={'comparison': {'cohort': 'escrow', 'seconds_per_arm': 9000}},
        store=SimpleNamespace(json=lambda key: {}))
    calls = []
    monkeypatch.setattr(ordinary_comparison.life, 'training_expert', lambda template: 'escrow')
    monkeypatch.setattr(ordinary_comparison.life, 'materialize_graph', lambda *args: {})
    monkeypatch.setattr(ordinary_comparison, 'resources', lambda cloud: {})
    monkeypatch.setattr(ordinary_comparison, 'benchmark',
        lambda *args: calls.append(args[-1]) or {'requests': 20})
    assert ordinary_comparison.run(backend, state) == {'pending': True, 'completed_arm': 'growth'}
    assert calls == ['growth']
    folder = tmp_path/'comparison'
    growth = (folder/'growth.json').read_bytes()
    assert not (folder/'control-start.json').exists()
    assert not (folder/'result.json').exists()

    save(context/'claim-prefix-actor-1.json', {'kind': 'expert_features'})
    for step in range(0, 128, 4):
        save(context/f'claim-{step}-actor-1.json',
             {'kind': 'expert_training', 'input_checkpoint': {'step': step}})
    class ControlBoundary(Exception):
        pass
    def start_control(*args):
        raise ControlBoundary
    monkeypatch.setattr(ordinary_campaign_backend, 'Backend', start_control)
    with pytest.raises(ControlBoundary):
        ordinary_comparison.run(backend, state)
    control_start = (folder/'control-start.json').read_bytes()
    with pytest.raises(ControlBoundary):
        ordinary_comparison.run(backend, state)
    assert (folder/'control-start.json').read_bytes() == control_start
    assert (folder/'growth.json').read_bytes() == growth
    assert calls == ['growth']
    assert backend.freeze['comparison']['seconds_per_arm'] == 9000


def test_pending_comparison_does_not_reject_or_admit_the_next_cohort(tmp_path, monkeypatch):
    backend = object.__new__(ordinary_campaign_backend.Backend)
    backend.home = tmp_path
    backend.freeze = {'entries': []}
    backend.probe = lambda state, label: None
    monkeypatch.setattr(ordinary_campaign_backend.ordinary_operation, 'outcome_sequence',
                        lambda *args: {'status': 'next', 'entry': 2})
    monkeypatch.setattr(ordinary_comparison, 'run', lambda *args: {'pending': True, 'completed_arm': 'growth'})
    assert backend.prepare({'state': {}, 'history': []}) == {'job': None}
    assert not (tmp_path/'comparison-stop.json').exists()


def test_expired_comparison_stops_instead_of_restarting_neural_work(tmp_path, monkeypatch):
    job = {'lifecycle': {'candidate_template': {}}}
    state = {'expert_lifecycle': {'admission': {'active': {'job': job}}}}
    folder = tmp_path/'comparison'
    folder.mkdir()
    save(tmp_path/'comparison-growth-start.json', {'started': datetime.now(timezone.utc).isoformat()})
    save(folder/'growth.json', {'preserved': 'completed arm'})
    save(folder/'control-start.json', {'started': datetime.now(timezone.utc).timestamp()-9001})
    def forbidden(*args, **kwargs):
        raise AssertionError('The expired comparison must not start or inspect remote work')
    backend = SimpleNamespace(home=tmp_path, cloud=SimpleNamespace(remaining=forbidden),
        context=lambda job: {'directory': tmp_path/'job'},
        freeze={'comparison': {'cohort': 'conversation', 'seconds_per_arm': 9000}})
    monkeypatch.setattr(ordinary_comparison.life, 'training_expert', lambda graph: 'conversation')
    monkeypatch.setattr(ordinary_campaign_backend, 'Backend', forbidden)
    result = ordinary_comparison.run(backend, state)
    assert result['passed'] is False and result['comparison_completed'] is False
    assert result['failure'] == 'comparison_budget_exhausted'
    assert json.loads((folder/'result.json').read_bytes()) == result
    # Even a caller changing its clock or inputs cannot convert a recorded
    # exhausted arm into a new interval through ordinary retry.
    backend.context = forbidden
    assert ordinary_comparison.run(backend, state) == result


def test_failed_comparison_records_terminal_stop_before_next_admission(tmp_path, monkeypatch):
    backend = object.__new__(ordinary_campaign_backend.Backend)
    backend.home = tmp_path
    backend.freeze = {'entries': []}
    backend.probe = lambda state, label: None
    monkeypatch.setattr(ordinary_campaign_backend.ordinary_operation, 'outcome_sequence',
                        lambda *args: {'status': 'next', 'entry': 2})
    result = {'passed': False, 'comparison_completed': False, 'failure': 'comparison_budget_exhausted'}
    monkeypatch.setattr(ordinary_comparison, 'run', lambda *args: result)
    assert backend.prepare({'state': {}, 'history': []}) == {'job': None}
    assert json.loads((tmp_path/'comparison-stop.json').read_bytes()) == result
