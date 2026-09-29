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


def test_second_confirmation_needs_development_and_a1_and_opens_its_own_frozen_split(tmp_path):
    report = tmp_path / 'report.json'
    save(report, {'development_passed': True, 'development_and_a1_passed': False})
    execution = {'development_report': {'path': str(report), 'sha256': sha256(report),
                                        'requires': ['development_passed', 'development_and_a1_passed']},
                 'split': 'confirmation2', 'data': PLAN['confirmation2_gate']['data']}
    with pytest.raises(ValueError, match='sealed'):
        confirm.opened(execution)
    save(report, {'development_passed': True, 'development_and_a1_passed': True}, exclusive=False)
    execution['development_report']['sha256'] = sha256(report)
    cases = confirm.opened(execution)
    assert len(cases) == PLAN['confirmation2_gate']['cases'] == 192
    assert not {c['id'] for c in cases} & {c['id'] for c in data.cases('confirmation')}
    with pytest.raises(ValueError, match='another split'):
        confirm.opened({**execution, 'split': 'confirmation'})


def test_third_confirmation_split_is_frozen_and_disjoint_from_every_earlier_split():
    from neuroshard.evolution.modular_reference_execution import identity

    manifest = read(ROOT / 'config/experiments/assistant-workflow-data-confirmation3.json')
    cases = data.cases('confirmation3')
    assert manifest['split'] == 'confirmation3' and manifest['count'] == len(cases) == 192
    assert identity(cases) == manifest['sha256'] and [c['id'] for c in cases] == manifest['case_ids']
    assert all(sum(c['family'] == f for c in cases) == 24 for f in data.FAMILIES)
    projects = {t['expected']['project'] for c in cases for t in c['turns']}
    for split in manifest['disjoint_from']:
        earlier = data.cases(split)
        assert not {c['id'] for c in cases} & {c['id'] for c in earlier}
        assert not projects & {t['expected']['project'] for c in earlier for t in c['turns']}


def test_second_confirmation_gate_scales_to_192_episodes():
    cases = data.cases('confirmation2')
    by_family = {f: [c['id'] for c in cases if c['family'] == f] for f in data.FAMILIES}

    def rows(successes, routed):
        extra = {'selected': 'arm', 'selection_seconds': 1.0} if routed else {}
        return [{**(execute_fixture(c) if c['id'] in successes else failing(c)), **extra} for c in cases]

    parent = {k for f in data.FAMILIES for k in by_family[f][:8]}
    for per_family, expected in ((20, True), (19, False)):
        trained = {k for f in data.FAMILIES for k in by_family[f][:per_family]}
        replies = {'parent': {'episodes': rows(parent, False)}, 'update': {'episodes': rows(trained, True)},
                   'addition': {'episodes': rows(trained, True)}}
        report = confirm.assess(PLAN, cases, replies, 'confirmation2_gate')
        assert report['passed'] is expected and report['correct']['addition'] == 8 * per_family


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


def test_confirmation_inventory_pins_the_passing_development_arms_and_runtime():
    execution = read(ROOT / confirm.EXECUTION)
    for name, digest in execution['contracts'].items():
        assert sha256(ROOT / name) == digest
    assert set(execution['contracts']) <= set(execution['sources'])
    report = read(ROOT / execution['development_report']['path'])
    assert report['development_passed'] and report['serving'] == execution['serving'] == 'prefix-cache'
    trained = read(ROOT / f"config/experiments/assistant-experience-round{report['round']}-report.json")
    assert execution['arms'] == {**{arm: {'trainable_sha256': trained['training'][arm]['trainable_sha256']}
                                    for arm in ('update', 'addition')},
                                 'integration_sha256': trained['integration']['integration_sha256']}
    baseline = read(ROOT / 'config/experiments/assistant-workflow-canonical-execution.json')
    assert all(execution[k] == baseline[k] for k in ('packages', 'python', 'required_cpu_flags', 'environment'))
    probe = ('import os, sys; import neuroshard.evolution.assistant_experience_confirm; '
             'import neuroshard.evolution.assistant_experience_run, neuroshard.evolution.assistant_experience_train, '
             'neuroshard.evolution.assistant_selector, neuroshard.evolution.assistant_experience_gate, '
             'neuroshard.evolution.assistant_serving; root = os.path.abspath("src"); '
             'print("\\n".join(sorted(os.path.relpath(x.__file__) for x in list(sys.modules.values()) '
             'if getattr(x, "__file__", None) and os.path.abspath(x.__file__).startswith(root))))')
    imported = subprocess.check_output([sys.executable, '-c', probe], cwd=ROOT, text=True,
                                       env={'PYTHONPATH': str(ROOT / 'src')}).split()
    assert set(imported) <= set(execution['sources'])
    hours = read(ROOT / 'config/experiments/assistant-experience-confirmation-parent-resources.json')['hours']
    assert execution['prepare_seconds'] + execution['worker_seconds'] + 1800 <= hours * 3600
    gate = read(ROOT / 'config/experiments/assistant-experience-learning.json')[execution.get('gate', 'confirmation_gate')]
    assert len(confirm.opened(execution)) == gate['cases']
    if execution.get('gate') == 'confirmation2_gate':
        assert execution['parent_serving'] == 'prefix-cache' and execution['data'] == gate['data']
        assert set(execution['development_report']['requires']) == {'development_passed', 'development_and_a1_passed'}


def test_importing_confirmation_does_not_load_torch():
    probe = 'import sys, neuroshard.evolution.assistant_experience_confirm; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', probe], check=True, cwd=ROOT, env={'PYTHONPATH': str(ROOT / 'src')})
