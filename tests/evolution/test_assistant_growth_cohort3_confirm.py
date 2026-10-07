import copy
import subprocess
import sys

import pytest

from neuroshard.evolution import assistant_experience_gate as gate
from neuroshard.evolution import assistant_growth_cohort3_confirm as confirm
from neuroshard.evolution.modular_reference_execution import ROOT, read, save, sha256

from test_assistant_growth_eval import SETS, episode
from test_assistant_routing import POLICIES

DECLARATION = read(ROOT / confirm.DECLARATION)
# The opened development cases stand in for the sealed sets; the thresholds are scaled to them in memory.
SCALED = {**DECLARATION, 'confirmation_gate': {**DECLARATION['confirmation_gate'], 'minimum_net_drafting': 2,
                                              'bootstrap_samples': 500}}
PASSING = {'upgrade-calendar': {'scheduling': 24, 'cross': 8}, 'previous-calendar': {'scheduling': 24, 'cross': 8},
           'upgrade-drafting': {'drafting': 24}, 'previous-drafting': {'drafting': 12}}


def replies(solved, slower=0.0):
    """``solved`` maps each system and set to how many leading cases it solves; the upgrade's turns take ``slower`` more."""
    out = {}
    for system, (role, which) in confirm.SYSTEMS.items():
        episodes = {}
        for name in confirm.SETS[which]:
            rows = [episode(c, i < solved[system][name]) for i, c in enumerate(SETS[name])]
            episodes[name] = [{**row, 'seconds': row['seconds'] + slower} if role == 'upgrade' else row for row in rows]
        out[system] = {'episodes': episodes}
    return out


def test_the_confirmation_gate_from_four_hosts(monkeypatch):
    monkeypatch.setattr(confirm.development, 'policies', lambda: POLICIES)
    report = confirm.assess(SCALED, SETS, replies(PASSING))
    assert report['passed'], report['checks']
    assert report['sets']['drafting']['net'] == 12 and not any(row['lost'] for row in report['sets'].values())
    upgrade = {c['id']: True for c in SETS['drafting']}
    previous = {c['id']: i < 12 for i, c in enumerate(SETS['drafting'])}
    assert report['lower_95_drafting_gain'] == gate.family_bootstrap(SETS['drafting'], upgrade, previous, 500,
                                                                     DECLARATION['confirmation_gate']['bootstrap_seed'])
    assert sum(row['upgrade'] for row in report['per_family'].values()) == 24
    regressed = copy.deepcopy(PASSING)
    regressed['upgrade-calendar']['scheduling'] = 23
    failed = confirm.assess(SCALED, SETS, replies(regressed))
    assert not failed['passed'] and not failed['checks']['lost'] and failed['checks']['net_drafting']
    slower = confirm.assess(SCALED, SETS, replies(PASSING, slower=5.0))
    assert not slower['checks']['p95_ratio'] and slower['checks']['p95'] and not slower['passed']


def test_every_confirmation_episode_is_rescored_in_order_and_on_its_hosts_sets(monkeypatch):
    monkeypatch.setattr(confirm.development, 'policies', lambda: POLICIES)
    rows = replies(PASSING)
    forged = copy.deepcopy(rows)
    score = forged['previous-calendar']['episodes']['scheduling'][0]['score']
    score['passed'] = not score['passed']
    with pytest.raises(ValueError, match='rescore'):
        confirm.assess(SCALED, SETS, forged)
    shuffled = copy.deepcopy(rows)
    shuffled['upgrade-drafting']['episodes']['drafting'].reverse()
    with pytest.raises(ValueError, match='in order'):
        confirm.assess(SCALED, SETS, shuffled)
    extra = copy.deepcopy(rows)
    extra['upgrade-drafting']['episodes']['cross'] = rows['upgrade-calendar']['episodes']['cross']
    with pytest.raises(ValueError, match='other sets'):
        confirm.assess(SCALED, SETS, extra)


def test_sealed_sets_open_only_behind_a_pinned_development_pass(tmp_path, monkeypatch):
    report = tmp_path / 'report.json'
    save(report, {'confirmation_may_open': False, 'report': {'passed': False}})
    monkeypatch.setattr(confirm, 'ROOT', tmp_path)
    execution = {'development_report': {'path': 'report.json', 'sha256': sha256(report)}}
    with pytest.raises(ValueError, match='sealed'):
        confirm.development_pass(execution)
    report.write_text('{"confirmation_may_open": true, "report": {"passed": true}}')
    with pytest.raises(ValueError, match='sealed'):
        confirm.development_pass(execution)
    execution['development_report']['sha256'] = sha256(report)
    assert confirm.development_pass(execution)['report']['passed']


def test_each_host_runs_one_system_with_its_units():
    from test_assistant_growth_baseline import cloud_module

    assert confirm.development.needed('upgrade') == ['L2', 'L3', 'U1']
    assert confirm.development.needed('previous') == ['L2', 'U1']
    cloud = cloud_module()
    assert set(confirm.PROFILES.values()) == set(confirm.SYSTEMS)
    for profile, system in confirm.PROFILES.items():
        command = cloud.remote_command(profile)
        assert command[1].endswith(confirm.SCRIPT) and command[-2:] == ['--system', system]
        assert cloud.UPLOAD_PROFILES[profile] == confirm.UPLOADED
        assert cloud.GRANITE_PROFILES[profile][0] == 'assistant_growth_cohort3_confirm'
    assert sum(cloud.LONG_CPU_PROFILES[profile][1] for profile in confirm.PROFILES) <= (
        DECLARATION['budget']['confirmation_usd'])
    development = cloud.COHORT3_DEVELOPMENT
    assert cloud.LONG_CPU_PROFILES[development][1] <= DECLARATION['budget']['development_usd']
    assert cloud.remote_command(development)[1].endswith(confirm.development.SCRIPT)
    assert cloud.GRANITE_PROFILES[development][0] == 'assistant_growth_cohort3_eval'


def test_importing_cohort_three_confirmation_does_not_load_torch():
    probe = 'import sys, neuroshard.evolution.assistant_growth_cohort3_confirm; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', probe], check=True, cwd=ROOT, env={'PYTHONPATH': str(ROOT / 'src')})


def test_confirmation_inventory_pins_the_development_pass_units_and_sources():
    from test_assistant_growth_baseline import cloud_module

    execution = read(ROOT / confirm.EXECUTION)
    for name, digest in execution['contracts'].items():
        assert sha256(ROOT / name) == digest
    assert set(execution['contracts']) <= set(execution['sources']) and confirm.SCRIPT in execution['sources']
    probe = ('import os, sys; import neuroshard.evolution.assistant_growth_cohort3_confirm; assert "torch" not in sys.modules; '
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
    for name in (confirm.DECLARATION, DECLARATION['sealed']['drafting_data'], DECLARATION['sealed']['scheduling_data']):
        assert name in execution['contracts']
    development = read(ROOT / confirm.development.EXECUTION)
    assert all(execution[key] == development[key] for key in ('packages', 'python', 'required_cpu_flags', 'environment',
                                                               'threads', 'units', 'gates', 'serving'))
    assert confirm.development_pass(execution)['report']['passed']
    cloud = cloud_module()
    gates = [gate['file'] for gate in execution['gates'].values()]
    for profile, system in confirm.PROFILES.items():
        resources = cloud.resources(profile)
        role, _ = confirm.SYSTEMS[system]
        assert resources['upload']['files'] == [f'{unit}/{name}' for unit in confirm.development.needed(role)
                                                for name in ('manifest.json', 'trainable.safetensors')] + gates
        assert execution['prepare_seconds'] + execution['worker_seconds'] + resources['setup_seconds'] + (
            resources['copy_seconds']) + 600 <= resources['hours'] * 3600
    assert sum(cloud.resources(profile)['planning_cap_usd'] for profile in confirm.PROFILES) <= (
        DECLARATION['budget']['confirmation_usd'])
