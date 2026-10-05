import copy
import subprocess
import sys

import pytest

from neuroshard.evolution import assistant_growth_confirm as confirm
from neuroshard.evolution import assistant_workflow as workflow
from neuroshard.evolution.modular_reference_execution import ROOT, save, sha256

from test_assistant_growth_eval import PLAN, SETS, episode
from test_assistant_routing import CALENDAR, DRAFTING, scripted
from test_assistant_workflow import reference_texts

# The opened development cases stand in for the sealed sets; the thresholds are scaled to them in memory.
RULES = {**PLAN['confirmation_gate'], 'minimum_scheduling': 18, 'minimum_per_scheduling_family': 2,
         'minimum_cross': 6, 'minimum_net_vs_previous': 5, 'bootstrap_samples': 500}
SCALED = {**PLAN, 'confirmation_gate': RULES}


def accepted_episode(case, solved, policy):
    texts = (reference_texts(case) if policy is DRAFTING else []) if solved else []
    row = workflow.execute(case, scripted(texts + ['I cannot finish this.'] * 30, []), policy)
    return {**row, 'selected': 'arm', 'selection_seconds': 0.25}


def replies(solved):
    """``solved`` maps each system and set to how many leading cases it solves."""
    out = {}
    for system, (role, which) in confirm.SYSTEMS.items():
        episodes = {}
        for name, _ in confirm.SETS[which]:
            count = solved[system][name]
            if role == 'accepted':
                policy = DRAFTING if which == 'drafting' else CALENDAR
                episodes[name] = [accepted_episode(c, i < count, policy) for i, c in enumerate(SETS[name])]
            else:
                episodes[name] = [episode(c, i < count) for i, c in enumerate(SETS[name])]
        out[system] = {'episodes': episodes}
    return out


PASSING = {'candidate-calendar': {'scheduling': 24, 'cross': 7}, 'accepted-calendar': {'scheduling': 0, 'cross': 0},
           'candidate-drafting': {'drafting': 19}, 'shared-drafting': {'drafting': 16},
           'accepted-drafting': {'drafting': 19}}


def test_the_confirmation_gate_and_the_comparison_from_five_hosts():
    report = confirm.assess(SCALED, SETS, replies(PASSING))
    assert report['passed'], report['checks']
    assert report['correct'] == {'scheduling': 24, 'cross': 7, 'drafting': 19}
    assert report['versus_previous']['net'] == 31 and report['lower_95_gain_vs_previous'] > 0
    assert report['drafting_versus_accepted']['shared-drafting']['lost'] == sorted(c['id'] for c in SETS['drafting'][16:19])
    assert report['separate_retains_better'] and report['accepted_drafting'] == 19
    lost = copy.deepcopy(PASSING)
    lost['candidate-drafting']['drafting'] = 18
    failed = confirm.assess(SCALED, SETS, replies(lost))
    assert not failed['passed'] and not failed['checks']['drafting'] and failed['separate_retains_better']
    weak = copy.deepcopy(PASSING)
    weak['candidate-calendar']['scheduling'] = 15
    assert not confirm.assess(SCALED, SETS, replies(weak))['checks']['scheduling']


def test_every_confirmation_episode_is_rescored_and_in_order():
    rows = replies(PASSING)
    forged = copy.deepcopy(rows)
    score = forged['accepted-calendar']['episodes']['scheduling'][0]['score']
    score['passed'] = not score['passed']
    with pytest.raises(ValueError, match='rescore'):
        confirm.assess(SCALED, SETS, forged)
    shuffled = copy.deepcopy(rows)
    shuffled['candidate-drafting']['episodes']['drafting'].reverse()
    with pytest.raises(ValueError, match='in order'):
        confirm.assess(SCALED, SETS, shuffled)


def test_sealed_sets_open_only_behind_a_pinned_development_pass(tmp_path, monkeypatch):
    report = tmp_path / 'report.json'
    save(report, {'confirmation_may_open': False, 'report': {'passed': False, 'candidate': 'separate_module'}})
    monkeypatch.setattr(confirm, 'ROOT', tmp_path)
    execution = {'development_report': {'path': 'report.json', 'sha256': sha256(report)}}
    with pytest.raises(ValueError, match='sealed'):
        confirm.development_pass(execution)
    report.write_text('{"confirmation_may_open": true, "report": {"passed": true, "candidate": "separate_module"}}')
    with pytest.raises(ValueError, match='sealed'):
        confirm.development_pass(execution)
    execution['development_report']['sha256'] = sha256(report)
    assert confirm.development_pass(execution) == 'separate_module'


def test_each_host_loads_only_its_systems_units():
    assert confirm.units('candidate-calendar', 'separate_module') == ['L2', 'U1']
    assert confirm.units('candidate-drafting', 'separate_update') == ['U1', 'U2']
    assert confirm.units('shared-drafting', 'separate_module') == ['U2']
    assert confirm.units('accepted-calendar', 'separate_module') == confirm.units('accepted-drafting', 'x') == ['U1']
    assert set(confirm.PROFILES.values()) == set(confirm.SYSTEMS)


def test_each_confirmation_profile_runs_one_system_with_its_uploaded_units():
    from test_assistant_growth_baseline import cloud_module

    cloud = cloud_module()
    for profile, system in confirm.PROFILES.items():
        command = cloud.remote_command(profile)
        assert command[1].endswith(confirm.SCRIPT) and command[-2:] == ['--system', system]
        assert cloud.UPLOAD_PROFILES[profile] == confirm.UPLOADED
        assert cloud.GRANITE_PROFILES[profile][0] == 'assistant_growth_confirm'
        assert 5 * cloud.LONG_CPU_PROFILES[profile][1] <= 47


def test_importing_confirmation_does_not_load_torch():
    probe = 'import sys, neuroshard.evolution.assistant_growth_confirm; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', probe], check=True, cwd=ROOT, env={'PYTHONPATH': str(ROOT / 'src')})
