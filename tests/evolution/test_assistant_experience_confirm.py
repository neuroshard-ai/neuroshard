import copy
import importlib.util
import subprocess
import sys

import pytest

from neuroshard.evolution import assistant_experience_confirm as confirm
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution.modular_reference_execution import ROOT, read, save, sha256

from test_assistant_experience_eval import failing
from test_assistant_workflow import execute_fixture

PLAN = read(ROOT / 'config/experiments/assistant-experience-learning.json')


def cloud_module():
    spec = importlib.util.spec_from_file_location('confirmation_cloud', ROOT / 'scripts/modular_reference_cloud.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_confirmation_stays_sealed_without_a_pinned_development_pass(tmp_path):
    for passed in (False, True):
        report = tmp_path / f'report-{passed}.json'
        save(report, {'development_passed': passed})
        execution = {'development_report': {'path': str(report), 'sha256': sha256(report)}}
        if passed:
            assert len(confirm.opened(execution)) == 96
        else:
            with pytest.raises(ValueError, match='sealed'):
                confirm.opened(execution)
    execution['development_report']['sha256'] = 'changed'
    with pytest.raises(ValueError, match='sealed'):
        confirm.opened(execution)


def test_confirmation_gate_rescores_all_three_systems():
    cases = data.cases('confirmation')
    by_family = {f: [c['id'] for c in cases if c['family'] == f] for f in data.FAMILIES}
    parent = {k for f in data.FAMILIES for k in by_family[f][:4]}
    trained = {k for f in data.FAMILIES for k in by_family[f][:11]}

    def rows(successes, routed):
        extra = {'selected': 'arm', 'selection_seconds': 1.0} if routed else {}
        return [{**(execute_fixture(c) if c['id'] in successes else failing(c)), **extra} for c in cases]

    replies = {'parent': {'episodes': rows(parent, False)}, 'update': {'episodes': rows(trained, True)},
               'addition': {'episodes': rows(trained, True)}}
    report = confirm.assess(PLAN, cases, replies)
    assert report['passed'] and report['correct'] == {'parent': 32, 'update': 88, 'addition': 88}
    assert report['selected_arm_episodes'] == {'update': 96, 'addition': 96}
    forged = copy.deepcopy(replies)
    forged['parent']['episodes'][0]['score']['passed'] = not forged['parent']['episodes'][0]['score']['passed']
    with pytest.raises(ValueError, match='rescore'):
        confirm.assess(PLAN, cases, forged)


def test_confirmation_profiles_are_bounded_and_only_routed_systems_upload_arms():
    cloud = cloud_module()
    for profile, system in confirm.PROFILES.items():
        resources = cloud.resources(profile)
        assert not resources['gpu'] and resources['instance_type'] == 'r7i.4xlarge'
        assert (resources['hours'], resources['planning_cap_usd']) == cloud.LONG_CPU_PROFILES[profile]
        assert resources['hours'] * resources['price']['usd_per_hour'] + 3 <= resources['planning_cap_usd']
        command = cloud.remote_command(profile)
        assert command[1].endswith(confirm.SCRIPT) and command[-2:] == ['--system', system]
        assert ('upload' in resources) == (system != 'parent') == (profile in cloud.UPLOAD_PROFILES)


def test_importing_confirmation_does_not_load_torch():
    probe = 'import sys, neuroshard.evolution.assistant_experience_confirm; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', probe], check=True, cwd=ROOT, env={'PYTHONPATH': str(ROOT / 'src')})
