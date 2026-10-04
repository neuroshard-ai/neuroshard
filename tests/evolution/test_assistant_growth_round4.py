import copy
import subprocess
import sys

import pytest

from neuroshard.evolution import assistant_growth_round4 as round4
from neuroshard.evolution import assistant_growth_run as growth
from neuroshard.evolution import assistant_schedule_data as schedule
from neuroshard.evolution.modular_reference_execution import ROOT, read, save, sha256

from test_assistant_growth_baseline import cloud_module

DECLARATION = read(ROOT / round4.DECLARATION)


def test_four_times_the_training_cases_each_checked_against_its_frozen_split(tmp_path, monkeypatch):
    stage = round4.plan()
    cases = round4.training_cases(stage)
    assert len(cases) == DECLARATION['experience']['cases'] == 4 * 288
    assert len({case['id'] for case in cases}) == len(cases)
    assert {case['split'] for case in cases} == {'train', 'cross-train', *schedule.GROWTH}
    assert not {case['split'] for case in cases} & {'development', 'cross-development', *schedule.SEALED}
    manifest = read(ROOT / DECLARATION['experience']['data'])
    manifest['splits']['train3']['case_ids'] = manifest['splits']['train3']['case_ids'][::-1]
    save(tmp_path / 'manifest.json', manifest)
    tampered = copy.deepcopy(stage)
    tampered['round4']['experience']['data'] = str(tmp_path / 'manifest.json')
    with pytest.raises(ValueError, match='frozen split'):
        round4.training_cases(tampered)


def test_steps_scale_with_the_demonstrations_and_nothing_else_in_training_changes():
    stage, stage1 = round4.plan(), read(ROOT / growth.PLAN)
    assert stage['training'] == {**stage1['training'], 'steps': 512}
    assert 512 * 4 / 1152 == stage1['training']['steps'] * 4 / 288
    _, learning, _ = growth.contracts(stage)
    assert growth.stage_spec(stage, learning) == {**growth.stage_spec(stage1, learning), 'steps': 512}
    assert DECLARATION['selectors'] == {**DECLARATION['selectors'], 'refit': True, 'failed_ties': False}


def test_the_round_stays_within_the_ceiling_on_one_a10g_host():
    budget = DECLARATION['budget']
    receipts = [read(ROOT / f'config/experiments/assistant-growth-{name}-report.json')['resources_finished']
                for name in ('stage1', 'round2', 'round3')]
    spent = sum(r['conservative_compute_usd'] for r in receipts) + read(
        ROOT / DECLARATION['development_report'])['conservative_compute_usd']
    assert budget['spent_usd'] == round(spent, 2)
    total = budget['spent_usd'] + budget['gpu_usd'] + budget['development_usd'] + budget['confirmation_usd']
    assert total <= read(ROOT / growth.PLAN)['budget']['ceiling_usd'] and f'${total:.2f}' in budget['allowances']
    cloud = cloud_module()
    assert round4.PROFILE in cloud.GPU_PROFILES and cloud.UPLOAD_PROFILES[round4.PROFILE] == growth.UPLOADED
    assert cloud.GRANITE_PROFILES[round4.PROFILE] == ('assistant_growth_round4', cloud.GRANITE_PROFILES[growth.PROFILE][1])
    assert cloud.remote_command(round4.PROFILE)[1].endswith(round4.SCRIPT)


def test_round4_inventory_pins_contracts_sources_and_the_stage1_runtime():
    execution = read(ROOT / round4.EXECUTION)
    for name, digest in execution['contracts'].items():
        assert sha256(ROOT / name) == digest
    assert set(execution['contracts']) <= set(execution['sources'])
    probe = ('import os, sys; import neuroshard.evolution.assistant_growth_round4; assert "torch" not in sys.modules; '
             'import neuroshard.evolution.assistant_experience_run, neuroshard.evolution.assistant_experience_train, '
             'neuroshard.evolution.assistant_selector, neuroshard.evolution.assistant_calendar; '
             'root = os.path.abspath("src"); print("\\n".join(sorted(os.path.relpath(x.__file__) '
             'for x in list(sys.modules.values()) if getattr(x, "__file__", None) and '
             'os.path.abspath(x.__file__).startswith(root))))')
    imported = subprocess.check_output([sys.executable, '-c', probe], cwd=ROOT, text=True,
                                       env={'PYTHONPATH': str(ROOT / 'src')}).split()
    assert set(imported) <= set(execution['sources']) and round4.SCRIPT in execution['sources']
    assert DECLARATION['experience']['data'] in execution['contracts']
    stage1 = read(ROOT / growth.EXECUTION)
    for key in ('packages', 'python', 'gpus', 'environment', 'worker_environment', 'tokenizer_pipeline_sha256',
                'execution', 'drafting_collection'):
        assert execution[key] == stage1[key]
    resources = cloud_module().resources(round4.PROFILE)
    assert [row['instance_type'] for row in resources['candidates']] == ['g5.2xlarge']
    worst = max(row['price']['usd_per_hour'] for row in resources['candidates'])
    assert resources['hours'] * worst + 3 <= resources['planning_cap_usd'] == DECLARATION['budget']['gpu_usd']
    assert execution['worker_seconds'] + resources['setup_seconds'] + 600 <= resources['hours'] * 3600


def test_importing_round4_does_not_load_torch():
    probe = 'import sys, neuroshard.evolution.assistant_growth_round4; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', probe], check=True, cwd=ROOT, env={'PYTHONPATH': str(ROOT / 'src')})
