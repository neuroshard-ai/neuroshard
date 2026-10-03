import copy
import importlib.util
import subprocess
import sys

import pytest

from neuroshard.evolution import assistant_growth_baseline as growth
from neuroshard.evolution import assistant_schedule_data as schedule
from neuroshard.evolution import assistant_workflow as workflow
from neuroshard.evolution.modular_reference_execution import ROOT, read, sha256

from test_assistant_calendar import POLICY, schedule_texts
from test_assistant_workflow import reference_texts, reply

PLAN = read(ROOT / growth.PLAN)


def cloud_module():
    spec = importlib.util.spec_from_file_location('growth_cloud', ROOT / 'scripts/modular_reference_cloud.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def episode(case, solved, routed):
    texts = (schedule_texts(case) if case['id'].startswith('schedule-') else reference_texts(case)) if solved else []
    texts = iter(texts + ['I cannot finish this.'] * 30)
    row = workflow.execute(case, lambda messages, tools: reply(next(texts)), POLICY)
    return {**row, 'selected': 'arm', 'selection_seconds': 1.0} if routed else row


def test_stage0_inventory_pins_contracts_runtime_arms_references_and_every_imported_source():
    execution = read(ROOT / growth.EXECUTION)
    for name, digest in execution['contracts'].items():
        assert sha256(ROOT / name) == digest
    assert set(execution['contracts']) <= set(execution['sources'])
    baseline = read(ROOT / 'config/experiments/assistant-workflow-canonical-execution.json')
    assert all(execution[k] == baseline[k] for k in ('packages', 'python', 'required_cpu_flags', 'environment'))
    third = read(ROOT / PLAN['cohort1']['acceptance'])
    assert execution['arms'] == {'update': {'trainable_sha256': third['candidate']['trainable_sha256']},
                                 'integration_sha256': third['candidate']['integration_sha256']}
    assert (PLAN['cohort1']['trainable_sha256'], PLAN['cohort1']['integration_sha256']) == (
        third['candidate']['trainable_sha256'], third['candidate']['integration_sha256'])
    for pinned in execution['drafting_references'].values():
        assert sha256(ROOT / pinned['path']) == pinned['sha256']
    probe = ('import os, sys; import neuroshard.evolution.assistant_growth_baseline, neuroshard.evolution.assistant_calendar, '
             'neuroshard.evolution.assistant_experience_eval, neuroshard.evolution.assistant_experience_run, '
             'neuroshard.evolution.assistant_experience_train, neuroshard.evolution.assistant_selector, '
             'neuroshard.evolution.assistant_experience_gate, neuroshard.evolution.assistant_serving; '
             'root = os.path.abspath("src"); print("\\n".join(sorted(os.path.relpath(x.__file__) '
             'for x in list(sys.modules.values()) if getattr(x, "__file__", None) and '
             'os.path.abspath(x.__file__).startswith(root))))')
    imported = subprocess.check_output([sys.executable, '-c', probe], cwd=ROOT, text=True,
                                       env={'PYTHONPATH': str(ROOT / 'src')}).split()
    assert set(imported) <= set(execution['sources'])
    hours = read(ROOT / 'config/experiments/assistant-growth-baseline-resources.json')['hours']
    assert execution['prepare_seconds'] + len(growth.SYSTEMS) * execution['worker_seconds'] + 1800 <= hours * 3600
    assert not execution['training_authorized'] and not execution['gpu_launch_authorized']
    assert not PLAN['training_execution_authorized'] and not PLAN['gpu_launch_authorized']


def test_stage0_opens_only_frozen_development_cases():
    sets = growth.opened(PLAN)
    assert {name: len(cases) for name, cases in sets.items()} == {'scheduling': 24, 'cross': 8, 'drafting': 24}
    assert {c['split'] for cases in sets.values() for c in cases} == {'development', 'cross-development'}
    assert not any(c['split'] in schedule.SEALED for cases in sets.values() for c in cases)
    changed = {**PLAN, 'cohort2': {**PLAN['cohort2'], 'data': 'config/experiments/assistant-workflow-data.json'}}
    with pytest.raises(ValueError, match='differ from their frozen split'):
        growth.opened(changed)


def test_assessment_rescores_every_episode_and_reports_room_to_learn_and_the_interface_effect():
    sets = growth.opened(PLAN)
    solved = {'parent': {'scheduling': 4, 'cross': 0, 'drafting': 9}, 'update': {'scheduling': 6, 'cross': 1, 'drafting': 18}}
    replies = {system: {'episodes': {name: [episode(case, i < solved[system][name], system == 'update')
                                            for i, case in enumerate(cases)] for name, cases in sets.items()}}
               for system in growth.SYSTEMS}
    drafting = sets['drafting']
    references = {'parent': [{'id': c['id'], 'score': {'passed': i < 9}} for i, c in enumerate(drafting)],
                  'update': [{'id': c['id'], 'score': {'passed': i < 19}} for i, c in enumerate(drafting)]}
    report = growth.assess(PLAN, sets, replies, references)
    assert report['systems']['update']['scheduling']['correct'] == 6 and report['systems']['parent']['drafting']['correct'] == 9
    assert report['systems']['update']['scheduling']['selected_arm'] == 24
    assert 'selected_arm' not in report['systems']['parent']['scheduling']
    assert report['drafting_interface_effect']['update']['lost'] == [drafting[18]['id']]
    assert report['drafting_interface_effect']['parent']['net'] == 0
    assert report['headroom'] and report['stage1_may_be_declared']
    full = copy.deepcopy(replies)
    full['update']['episodes']['scheduling'] = [episode(case, True, True) for case in sets['scheduling']]
    assert not growth.assess(PLAN, sets, full, references)['headroom']
    forged = copy.deepcopy(replies)
    forged['parent']['episodes']['cross'][0]['score']['passed'] = True
    with pytest.raises(ValueError, match='rescore'):
        growth.assess(PLAN, sets, forged, references)
    missing = copy.deepcopy(replies)
    missing['update']['episodes']['drafting'].pop()
    with pytest.raises(ValueError, match='every growth episode'):
        growth.assess(PLAN, sets, missing, references)


def test_cloud_profile_is_bounded_and_uploads_only_the_update_and_its_gates():
    cloud = cloud_module()
    resources = cloud.resources(growth.PROFILE)
    assert not resources['gpu'] and resources['instance_type'] == 'r7i.4xlarge'
    assert (resources['hours'], resources['planning_cap_usd']) == cloud.LONG_CPU_PROFILES[growth.PROFILE]
    assert resources['hours'] * resources['price']['usd_per_hour'] + 3 <= resources['planning_cap_usd']
    assert cloud.remote_command(growth.PROFILE)[1].endswith(growth.SCRIPT)
    assert cloud.UPLOAD_PROFILES[growth.PROFILE] == growth.UPLOADED
    assert cloud.GRANITE_PROFILES[growth.PROFILE][0] == 'assistant_growth_baseline'
    assert set(resources['upload']['files']) == {'integration.json', 'update-checkpoint/manifest.json',
                                                 'update-checkpoint/trainable.safetensors'}


def test_importing_the_growth_baseline_does_not_load_torch():
    probe = 'import sys, neuroshard.evolution.assistant_growth_baseline; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', probe], check=True, cwd=ROOT, env={'PYTHONPATH': str(ROOT / 'src')})
