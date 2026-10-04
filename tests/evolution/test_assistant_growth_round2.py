import gzip
import json
import re
import subprocess
import sys

import pytest

from neuroshard.evolution import assistant_calendar as calendar
from neuroshard.evolution import assistant_experience as experience
from neuroshard.evolution import assistant_experience_train as trainer
from neuroshard.evolution import assistant_growth_round2 as round2
from neuroshard.evolution import assistant_growth_run as growth
from neuroshard.evolution import assistant_rollout as rollout
from neuroshard.evolution import assistant_schedule_data as schedule
from neuroshard.evolution import assistant_workflow as workflow
from neuroshard.evolution.assistant_workflow_data import public_case
from neuroshard.evolution.modular_reference_execution import ROOT, identity, read, sha256

from test_assistant_calendar import POLICY as CALENDAR, schedule_texts
from test_assistant_growth_baseline import cloud_module
from test_assistant_routing import scripted
from test_assistant_workflow import reply

DECLARATION = read(ROOT / round2.DECLARATION)
CASES = schedule.cases('train')[:3]


def natural_rows(path, solved, samples=2, drop=0):
    rows = []
    for case in CASES:
        for sample in range(samples):
            texts = schedule_texts(case) if (case['id'], sample) in solved else []
            result = workflow.execute(case, scripted(texts + ['I cannot finish this.'] * 30, []), CALENDAR)
            rows.append({'case_id': case['id'], 'sample': sample, 'policy_sha256': identity(CALENDAR), 'result': result})
    rows = rows[:len(rows) - drop]
    with gzip.open(path, 'wt', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
    return rows


def stage_for(path, rows, **collection):
    return {'collection': {'splits': ['train', 'cross-train'], 'samples_per_case': 2},
            'round2': {'collection': {**DECLARATION['collection'], 'samples_per_case': 2, **collection,
                                      'natural': {'file': path.name, 'sha256': sha256(path), 'rows_sha256': identity(rows)}},
                       'stop': DECLARATION['stop']}}


@pytest.fixture
def few_cases(monkeypatch):
    monkeypatch.setattr(growth, 'scheduling_cases', lambda stage, split: CASES if split == 'train' else [])


def test_stage1_natural_rollouts_are_reverified_and_decide_coaching(tmp_path, few_cases):
    path = tmp_path / 'stage1-rollouts.jsonl.gz'
    rows = natural_rows(path, {(CASES[0]['id'], 1)})
    found, accepted = round2.natural(stage_for(path, rows), CALENDAR, path)
    assert set(found) == {case['id'] for case in CASES}
    assert [t['case_id'] for t in accepted if t['complete']] == [CASES[0]['id']]
    assert [c['id'] for c in CASES if experience.needs_coaching(c, accepted)] == [c['id'] for c in CASES[1:]]
    forged = stage_for(path, rows)
    forged['round2']['collection']['natural']['rows_sha256'] = 'a' * 64
    with pytest.raises(ValueError, match='stage-1 report records'):
        round2.natural(forged, CALENDAR, path)
    short = tmp_path / 'short.jsonl.gz'
    rows = natural_rows(short, set(), drop=1)
    with pytest.raises(ValueError, match='incomplete'):
        round2.natural(stage_for(short, rows), CALENDAR, short)


def test_only_triggered_cases_are_coached_and_every_verified_success_is_kept(tmp_path, few_cases, monkeypatch):
    uploaded = tmp_path / growth.UPLOADED
    uploaded.mkdir()
    rows = natural_rows(uploaded / 'stage1-rollouts.jsonl.gz', {(CASES[0]['id'], 0)})
    stage = stage_for(uploaded / 'stage1-rollouts.jsonl.gz', rows)
    users = {public_case(case)['user_turns'][0]: case for case in CASES}
    coached = []

    class Batcher:
        def __init__(self, *args, **kwargs):
            self.batches, self.script = [], iter(())

        def respond(self, messages, tools):
            case = users[messages[1]['content']]
            coached.append((case['id'], messages[0]['content'].endswith(stage['round2']['collection']['card'])))
            if len(messages) == 2:
                self.script = iter(schedule_texts(case) if case is CASES[1] else [])
            return reply(next(self.script, 'I cannot finish this.'))

        def close(self):
            pass

    monkeypatch.setattr(round2, 'ROOT', tmp_path)
    monkeypatch.setattr(rollout, 'Batcher', Batcher)
    monkeypatch.setattr(trainer, 'encode', lambda tokenizer, item, tools: item['case_id'])
    sequences, report = round2.collect(None, None, stage, {'scheduling': CALENDAR},
                                       {'max_batch': 4, 'device': 'cpu', 'workers': 1}, tmp_path)
    assert {case_id for case_id, _ in coached} == {CASES[1]['id'], CASES[2]['id']}
    assert all(card for _, card in coached)
    assert report['coached_cases'] == [CASES[1]['id'], CASES[2]['id']]
    assert report['natural_verified'] == 1 and report['coached_verified'] >= 1
    assert report['complete_cases'] == 2 and set(sequences) == {CASES[0]['id'], CASES[1]['id']}
    stored = [json.loads(line) for line in gzip.open(tmp_path / 'trajectories.jsonl.gz', 'rt')]
    assert all(t['messages'][0]['content'] == CALENDAR['system_instruction'] for t in stored)
    assert {t['sample'] for t in stored if t['coached']} <= set(range(2, 4))


def test_the_card_names_no_team_and_the_round_stays_within_the_ceiling():
    card = DECLARATION['collection']['card']
    assert not any(team in card for team in calendar.TEAMS)
    assert 'two' in card and 'ten tool calls' in card
    stage1 = read(ROOT / 'config/experiments/assistant-growth-stage1-report.json')
    assert DECLARATION['collection']['natural']['rows_sha256'] == stage1['collection']['rollouts_sha256']
    budget = DECLARATION['budget']
    assert budget['spent_usd'] == round(stage1['resources_finished']['conservative_compute_usd'], 2)
    total = budget['spent_usd'] + budget['gpu_usd'] + budget['development_usd'] + budget['confirmation_usd']
    assert total <= read(ROOT / growth.PLAN)['budget']['ceiling_usd']
    assert f'${total:.2f}' in budget['allowances'] and DECLARATION['stop']['minimum_complete_cases'] == 32
    assert re.search(r'\$17\b', budget['allowances'])


def test_round2_inventory_pins_contracts_sources_and_the_stage1_runtime():
    execution = read(ROOT / round2.EXECUTION)
    for name, digest in execution['contracts'].items():
        assert sha256(ROOT / name) == digest
    assert set(execution['contracts']) <= set(execution['sources'])
    probe = ('import os, sys; import neuroshard.evolution.assistant_growth_round2; assert "torch" not in sys.modules; '
             'import neuroshard.evolution.assistant_experience_run, neuroshard.evolution.assistant_experience_train, '
             'neuroshard.evolution.assistant_selector, neuroshard.evolution.assistant_calendar; '
             'root = os.path.abspath("src"); print("\\n".join(sorted(os.path.relpath(x.__file__) '
             'for x in list(sys.modules.values()) if getattr(x, "__file__", None) and '
             'os.path.abspath(x.__file__).startswith(root))))')
    imported = subprocess.check_output([sys.executable, '-c', probe], cwd=ROOT, text=True,
                                       env={'PYTHONPATH': str(ROOT / 'src')}).split()
    assert set(imported) <= set(execution['sources']) and round2.SCRIPT in execution['sources']
    stage1 = read(ROOT / growth.EXECUTION)
    for key in ('packages', 'python', 'gpus', 'environment', 'worker_environment', 'tokenizer_pipeline_sha256',
                'execution', 'drafting_collection'):
        assert execution[key] == stage1[key]
    resources = cloud_module().resources(round2.PROFILE)
    worst = max(row['price']['usd_per_hour'] for row in resources['candidates'])
    assert resources['hours'] * worst + 3 <= resources['planning_cap_usd'] == DECLARATION['budget']['gpu_usd']
    assert execution['worker_seconds'] + resources['setup_seconds'] + 600 <= resources['hours'] * 3600
    files = read(ROOT / 'config/experiments/assistant-growth-resources.json')['upload']['files']
    assert resources['upload']['files'] == files + [DECLARATION['collection']['natural']['file']]


def test_the_round2_profile_is_one_gpu_host_like_stage1():
    cloud = cloud_module()
    assert cloud.GPU_PROFILES[round2.PROFILE] == cloud.GPU_PROFILES[growth.PROFILE]
    assert cloud.GRANITE_PROFILES[round2.PROFILE] == ('assistant_growth_round2', cloud.GRANITE_PROFILES[growth.PROFILE][1])
    assert cloud.UPLOAD_PROFILES[round2.PROFILE] == growth.UPLOADED
    assert cloud.remote_command(round2.PROFILE)[1].endswith(round2.SCRIPT)


def test_importing_round2_does_not_load_torch():
    probe = 'import sys, neuroshard.evolution.assistant_growth_round2; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', probe], check=True, cwd=ROOT, env={'PYTHONPATH': str(ROOT / 'src')})
