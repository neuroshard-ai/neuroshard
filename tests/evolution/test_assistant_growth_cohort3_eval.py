import copy
import subprocess
import sys

import pytest

from neuroshard.evolution import assistant_growth_cohort3_eval as evaluation
from neuroshard.evolution.modular_reference_execution import ROOT, read, save, sha256

from test_assistant_growth_eval import SETS, episode
from test_assistant_routing import POLICIES

DECLARATION = read(ROOT / evaluation.DECLARATION)
PREVIOUS = {'scheduling': 24, 'cross': 8, 'drafting': 19}


def episodes(solved):
    """``solved`` maps each set to how many of its leading cases the system solves."""
    return {name: [episode(case, i < solved[name]) for i, case in enumerate(cases)] for name, cases in SETS.items()}


def test_the_upgrade_serves_l3_on_drafting_and_keeps_cohort_two_on_scheduling():
    assert evaluation.SYSTEMS == {'upgrade': ('L3', 'L2'), 'previous': ('U1', 'L2')}
    assert evaluation.needed('upgrade') == ['L2', 'L3', 'U1'] and evaluation.needed('previous') == ['L2', 'U1']
    assert evaluation.policies() == {route: read(ROOT / path) for route, path in DECLARATION['policies'].items()}


def test_the_development_gate_compares_both_systems_case_by_case(monkeypatch):
    previous = episodes(PREVIOUS)
    monkeypatch.setattr(evaluation, 'policies', lambda: POLICIES)
    monkeypatch.setattr(evaluation, 'previous_episodes', lambda declaration: previous)
    rows = episodes({**PREVIOUS, 'drafting': 22})
    report = evaluation.assess(DECLARATION, SETS, {'episodes': rows})
    assert report['passed'], report['checks']
    assert report['sets']['drafting']['gained'] == sorted(c['id'] for c in SETS['drafting'][19:22])
    assert (report['sets']['drafting']['upgrade'], report['sets']['drafting']['previous']) == (22, 19)
    short = evaluation.assess(DECLARATION, SETS, {'episodes': episodes({**PREVIOUS, 'drafting': 20})})
    assert short['checks'] == {'drafting': False, 'lost': True, 'p95': True} and not short['passed']
    regressed = evaluation.assess(DECLARATION, SETS, {'episodes': episodes({**PREVIOUS, 'scheduling': 23,
                                                                             'drafting': 23})})
    assert not regressed['checks']['lost'] and regressed['sets']['scheduling']['lost'] == [SETS['scheduling'][23]['id']]
    forged = copy.deepcopy(rows)
    forged['cross'][0]['score']['passed'] = not forged['cross'][0]['score']['passed']
    with pytest.raises(ValueError, match='rescore'):
        evaluation.assess(DECLARATION, SETS, {'episodes': forged})
    shuffled = copy.deepcopy(rows)
    shuffled['drafting'].reverse()
    with pytest.raises(ValueError, match='in order'):
        evaluation.assess(DECLARATION, SETS, {'episodes': shuffled})


def test_the_pinned_previous_system_rescores_under_the_declared_policies():
    previous = evaluation.previous_episodes(DECLARATION)
    report = evaluation.compare(SETS, {'upgrade': previous, 'previous': previous}, evaluation.policies())
    assert {name: row['previous'] for name, row in report.items()} == PREVIOUS
    assert not any(row['gained'] or row['lost'] for row in report.values())


def test_the_previous_episodes_are_pinned_by_digest(tmp_path):
    path = tmp_path / 'previous.json'
    save(path, {'reply': {'episodes': {'drafting': []}}})
    declaration = {'previous': {'development': {'path': str(path), 'sha256': sha256(path)}}}
    assert evaluation.previous_episodes(declaration) == {'drafting': []}
    declaration['previous']['development']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='differ from their pin'):
        evaluation.previous_episodes(declaration)


def test_serving_loads_each_systems_units_behind_the_pinned_gates(monkeypatch):
    calls = {'routed': []}

    def verify(directory, execution, names):
        calls['verified'] = names
        return {'a2': {'arms': {'update': {'gate': 'a2-gate'}}}, 'router': {'gate': {'rule': 'centroid'}}}

    def load_units(load_parent, spec, directory, execution, units):
        calls['loaded'] = units
        return {unit: f'model-{unit}' for unit in units}

    def routed(parent, models, tokenizer, version, a2_gate, turn_gate, cases, *, units, route_policies):
        calls['routed'].append((models, a2_gate, turn_gate, units, route_policies))
        return [{'id': case['id']} for case in cases]

    monkeypatch.setattr(evaluation.stage1_confirmation, 'verify', verify)
    monkeypatch.setattr(evaluation, 'load_units', load_units)
    monkeypatch.setattr(evaluation.stage1_development, 'routed_episodes', routed)
    monkeypatch.setattr(evaluation.reference, 'load_model', lambda directory, name: ('parent', None))
    execution = {'units': {unit: {'trainable_sha256': unit * 8} for unit in ('U1', 'L2', 'L3')}}
    served = evaluation.serve('models', 'tokenizer', execution, 'upgrade', SETS, {})
    assert calls['verified'] == ['L2', 'L3', 'U1'] and calls['loaded'] == ('L3', 'L2')
    assert served['units'] == {'L2': 'L2' * 8, 'L3': 'L3' * 8, 'U1': 'U1' * 8}
    assert served['episodes']['drafting'] == [{'id': c['id']} for c in SETS['drafting']]
    assert all(row == ({'L3': 'model-L3', 'L2': 'model-L2'}, 'a2-gate', {'rule': 'centroid'}, ('L3', 'L2'),
                       evaluation.policies()) for row in calls['routed'])


def test_development_inventory_pins_units_gates_contracts_and_sources():
    from neuroshard.evolution import assistant_growth_eval as stage1
    from test_assistant_growth_baseline import cloud_module

    execution = read(ROOT / evaluation.EXECUTION)
    for name, digest in execution['contracts'].items():
        assert sha256(ROOT / name) == digest
    assert set(execution['contracts']) <= set(execution['sources']) and evaluation.SCRIPT in execution['sources']
    probe = ('import os, sys; import neuroshard.evolution.assistant_growth_cohort3_eval; assert "torch" not in sys.modules; '
             'import neuroshard.evolution.assistant_growth_run, neuroshard.evolution.assistant_experience_run, '
             'neuroshard.evolution.assistant_experience_train, neuroshard.evolution.assistant_selector, '
             'neuroshard.evolution.assistant_serving, neuroshard.evolution.assistant_calendar, '
             'neuroshard.evolution.assistant_calendar_slots; '
             'root = os.path.abspath("src"); print("\\n".join(sorted(os.path.relpath(x.__file__) '
             'for x in list(sys.modules.values()) if getattr(x, "__file__", None) and '
             'os.path.abspath(x.__file__).startswith(root))))')
    imported = subprocess.check_output([sys.executable, '-c', probe], cwd=ROOT, text=True,
                                       env={'PYTHONPATH': str(ROOT / 'src')}).split()
    assert set(imported) <= set(execution['sources'])
    report = read(ROOT / 'config/experiments/assistant-growth-cohort3-report.json')
    assert report['execution_completed'] and execution['units']['L3']['trainable_sha256'] == (
        report['units']['L3']['trainable_sha256'])
    pinned = read(ROOT / stage1.EXECUTION)
    assert {unit: execution['units'][unit] for unit in ('U1', 'L2')} == {unit: pinned['units'][unit] for unit in ('U1', 'L2')}
    assert execution['gates'] == pinned['gates'] and DECLARATION['previous']['development']['path'] in execution['contracts']
    canonical = read(ROOT / 'config/experiments/assistant-workflow-canonical-execution.json')
    assert all(execution[key] == canonical[key] for key in ('packages', 'python', 'required_cpu_flags', 'environment'))
    resources = cloud_module().resources(evaluation.PROFILE)
    assert resources['upload']['files'] == [f'{unit}/{name}' for unit in evaluation.needed('upgrade')
                                            for name in ('manifest.json', 'trainable.safetensors')] + [
        gate['file'] for gate in execution['gates'].values()]
    assert resources['planning_cap_usd'] <= DECLARATION['budget']['development_usd']
    assert execution['prepare_seconds'] + execution['worker_seconds'] + resources['setup_seconds'] + (
        resources['copy_seconds']) + 600 <= resources['hours'] * 3600


def test_importing_cohort_three_development_does_not_load_torch():
    probe = 'import sys, neuroshard.evolution.assistant_growth_cohort3_eval; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', probe], check=True, cwd=ROOT, env={'PYTHONPATH': str(ROOT / 'src')})
