import gzip
import json
import subprocess
import sys

import pytest

from neuroshard.evolution import assistant_calendar as calendar
from neuroshard.evolution import assistant_experience_train as trainer
from neuroshard.evolution import assistant_growth_round3 as round3
from neuroshard.evolution import assistant_growth_run as growth
from neuroshard.evolution import assistant_schedule_data as schedule
from neuroshard.evolution import assistant_schedule_demonstration as demonstrations
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution import assistant_workspace as workspace
from neuroshard.evolution.modular_reference_execution import ROOT, read

from test_assistant_calendar import POLICY as CALENDAR
from test_assistant_growth_baseline import cloud_module

DECLARATION = read(ROOT / round3.DECLARATION)
TRAINING = [case for split in schedule.TRAINING for case in schedule.cases(split)]


def test_every_training_case_is_solved_within_the_limits_in_the_native_envelope():
    families = {}
    for case in TRAINING:
        item, result = round3.demonstration(case, CALENDAR)
        assert item['complete'] and item['demonstration'] and not item['coached']
        previous = 0
        for round_ in result['rounds']:
            assert round_['generation_count'] - previous <= CALENDAR['limits']['model_turns_per_user_turn']
            previous = round_['generation_count']
        for text in demonstrations.texts(case):
            if '<tool_call>' in text:
                assert text.startswith('<tool_call>\n') and len(workspace.parse_calls(text, calendar.REGISTRY)) <= 2
        teams = {c['call']['arguments']['team'] for c in result['calls'] if c['call']['name'] == 'list_busy'}
        assert teams <= set(calendar.TEAMS)
        names = [c['call']['name'] for c in result['calls']]
        families.setdefault(case['family'], set()).add('shift_date' in names)
        assert names.count('save_meeting') == len([t for t in case['turns'] if calendar.goals(t['expected'])[1]])
    assert families['review'] == families['handoff'] == {True} and families['slot'] == {False}


def test_demonstrations_refuse_every_split_but_training():
    for case in (schedule.cases('development')[0], schedule.cases('integration')[0], data.cases('train')[0]):
        with pytest.raises(ValueError, match='training scheduling cases'):
            demonstrations.texts(case)


def test_demonstrations_become_experience_stored_with_the_calendar_instruction(tmp_path, monkeypatch):
    cases = TRAINING[:2] + [c for c in TRAINING if c['family'] == 'handoff'][:1]
    monkeypatch.setattr(growth, 'scheduling_cases', lambda stage, split: [c for c in cases if c['split'] == split])
    monkeypatch.setattr(trainer, 'encode', lambda tokenizer, item, tools: (item['case_id'], len(tools)))
    stage = {'collection': {'splits': list(schedule.TRAINING)}}
    sequences, report = round3.collect(None, stage, {'scheduling': CALENDAR}, tmp_path)
    assert [s[0] for s in sequences] == [c['id'] for c in cases] and report['complete'] == 3
    stored = [json.loads(line) for line in gzip.open(tmp_path / 'trajectories.jsonl.gz', 'rt')]
    assert all(t['messages'][0]['content'] == CALENDAR['system_instruction'] and t['demonstration'] for t in stored)


def test_a_demonstration_that_fails_the_scorer_stops_the_round(monkeypatch):
    case = TRAINING[0]
    texts = demonstrations.texts(case)
    monkeypatch.setattr(demonstrations, 'texts', lambda case: texts[:-2] + texts[-1:])
    with pytest.raises(ValueError, match='does not solve'):
        round3.demonstration(case, CALENDAR)


def test_the_round_stays_within_the_ceiling_and_runs_like_stage1():
    budget = DECLARATION['budget']
    round2 = read(ROOT / DECLARATION['round2_report'])
    stage1 = read(ROOT / 'config/experiments/assistant-growth-stage1-report.json')
    spent = stage1['resources_finished']['conservative_compute_usd'] + round2['resources_finished']['conservative_compute_usd']
    assert budget['spent_usd'] == round(spent, 2)
    total = budget['spent_usd'] + budget['gpu_usd'] + budget['development_usd'] + budget['confirmation_usd']
    assert total <= read(ROOT / growth.PLAN)['budget']['ceiling_usd'] and f'${total:.2f}' in budget['allowances']
    cloud = cloud_module()
    assert cloud.GPU_PROFILES[round3.PROFILE] == cloud.GPU_PROFILES[growth.PROFILE]
    assert cloud.GRANITE_PROFILES[round3.PROFILE] == ('assistant_growth_round3', cloud.GRANITE_PROFILES[growth.PROFILE][1])
    assert cloud.UPLOAD_PROFILES[round3.PROFILE] == growth.UPLOADED
    assert cloud.remote_command(round3.PROFILE)[1].endswith(round3.SCRIPT)


def test_round3_inventory_pins_contracts_sources_and_the_stage1_runtime():
    from neuroshard.evolution.modular_reference_execution import sha256

    execution = read(ROOT / round3.EXECUTION)
    for name, digest in execution['contracts'].items():
        assert sha256(ROOT / name) == digest
    assert set(execution['contracts']) <= set(execution['sources'])
    probe = ('import os, sys; import neuroshard.evolution.assistant_growth_round3; assert "torch" not in sys.modules; '
             'import neuroshard.evolution.assistant_experience_run, neuroshard.evolution.assistant_experience_train, '
             'neuroshard.evolution.assistant_selector, neuroshard.evolution.assistant_calendar; '
             'root = os.path.abspath("src"); print("\\n".join(sorted(os.path.relpath(x.__file__) '
             'for x in list(sys.modules.values()) if getattr(x, "__file__", None) and '
             'os.path.abspath(x.__file__).startswith(root))))')
    imported = subprocess.check_output([sys.executable, '-c', probe], cwd=ROOT, text=True,
                                       env={'PYTHONPATH': str(ROOT / 'src')}).split()
    assert set(imported) <= set(execution['sources']) and round3.SCRIPT in execution['sources']
    assert DECLARATION['experience']['solver'] in execution['sources']
    stage1 = read(ROOT / growth.EXECUTION)
    for key in ('packages', 'python', 'gpus', 'environment', 'worker_environment', 'tokenizer_pipeline_sha256',
                'execution', 'drafting_collection'):
        assert execution[key] == stage1[key]
    resources = cloud_module().resources(round3.PROFILE)
    worst = max(row['price']['usd_per_hour'] for row in resources['candidates'])
    assert resources['hours'] * worst + 3 <= resources['planning_cap_usd'] == DECLARATION['budget']['gpu_usd']
    assert execution['worker_seconds'] + resources['setup_seconds'] + 600 <= resources['hours'] * 3600
    assert resources['upload']['files'] == read(ROOT / 'config/experiments/assistant-growth-resources.json')['upload']['files']


def test_importing_round3_does_not_load_torch():
    probe = 'import sys, neuroshard.evolution.assistant_growth_round3; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', probe], check=True, cwd=ROOT, env={'PYTHONPATH': str(ROOT / 'src')})
