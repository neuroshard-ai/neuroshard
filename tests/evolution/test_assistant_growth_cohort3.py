import subprocess
import sys

import pytest

from neuroshard.evolution import assistant_drafting_demonstration as solver
from neuroshard.evolution import assistant_growth_cohort3 as cohort3
from neuroshard.evolution import assistant_growth_run as growth
from neuroshard.evolution import assistant_schedule_data as schedule
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution.modular_reference_execution import ROOT, identity, read, sha256

from test_assistant_growth_baseline import cloud_module

DECLARATION = read(ROOT / cohort3.DECLARATION)
DRAFTING = read(ROOT / 'config/experiments/assistant-workflow-policy.json')
SLOTS = read(ROOT / 'config/experiments/assistant-workflow-policy-calendar-slots.json')


def test_demonstrations_solve_every_training_case_within_the_turn_budget_reading_no_draft():
    stage = cohort3.plan()
    _, learning, policies = cohort3.contracts(stage)
    cases = cohort3.training_cases(stage, learning)
    assert len(cases) == DECLARATION['experience']['cases'] == 1248 and len({c['id'] for c in cases}) == 1248
    assert {c['split'] for c in cases} == set(DECLARATION['experience']['splits']) == set(data.TRAINING)
    limit = DRAFTING['limits']['model_turns_per_user_turn']
    for case in cases:
        item, result = cohort3.demonstration(case, policies['drafting'])
        assert item['complete'] and item['demonstration']
        counts = [row['generation_count'] for row in result['rounds']]
        assert all(after - before <= limit for before, after in zip([0] + counts, counts))


def test_independent_calls_share_a_reply_and_only_training_cases_are_solved():
    difference = next(c for c in data.cases('train') if c['family'] == 'difference')
    assert solver.texts(difference)[1].count('read_document') == 2
    scope = next(c for c in data.cases('train') if c['family'] == 'scope')
    assert any('shift_date' in text and 'calculate' in text for text in solver.texts(scope))
    for split in ('integration', 'development', 'confirmation6'):
        with pytest.raises(ValueError, match='demonstrations may only solve training'):
            solver.texts(data.make_case(split, 'copy', 0))
    with pytest.raises(ValueError, match='demonstrations may only solve training'):
        solver.texts(schedule.make_case('train', 'slot', 0))


def test_a_demonstration_that_reads_the_unapproved_draft_stops_the_job(monkeypatch):
    case = next(c for c in data.cases('train') if c['family'] == 'copy')
    project = case['turns'][0]['expected']['project']
    draft = next(d for d in case['world']['documents'] if d['project'] == project and d['status'] == 'draft')
    honest = solver.texts(case)
    forged = [honest[0], honest[1] + '\n' + solver.call('read_document', {'document_id': draft['id']})] + honest[2:]
    assert cohort3.demonstration(case, DRAFTING)[1]['score']['passed']
    monkeypatch.setattr(cohort3.demonstrations, 'texts', lambda _: forged)
    with pytest.raises(ValueError, match='does not solve'):
        cohort3.demonstration(case, DRAFTING)


def test_fresh_sealed_splits_are_frozen_before_training_and_disjoint_from_every_split():
    sealed = DECLARATION['sealed']
    manifest = read(ROOT / sealed['scheduling_data'])
    assert manifest['sealed'] == list(schedule.SEALED3) == [sealed['scheduling'], sealed['cross']]
    earlier = {c['id'] for split in {**schedule.SPLITS, **schedule.GROWTH, **schedule.FRESH}
               for c in schedule.cases(split)}
    for split, frozen in manifest['splits'].items():
        cases = schedule.cases(split)
        assert identity(cases) == frozen['sha256'] and [c['id'] for c in cases] == frozen['case_ids']
        assert not {c['id'] for c in cases} & earlier
        assert all(sum(c['family'] == f for c in cases) == 24 for f in schedule.families(split))
    drafting = read(ROOT / sealed['drafting_data'])
    cases = data.cases(drafting['split'])
    assert drafting['split'] == sealed['drafting'] == 'confirmation6' in data.CONFIRMATIONS and len(cases) == 192
    assert identity(cases) == drafting['sha256'] and [c['id'] for c in cases] == drafting['case_ids']
    assert set(drafting['disjoint_from']) == set(data.SPLITS) - {'confirmation6'}
    others = {c['id'] for split in data.SPLITS if split != 'confirmation6' for c in data.cases(split)}
    assert not {c['id'] for c in cases} & others


def test_one_module_on_top_of_the_accepted_update_with_stage_one_settings():
    stage = cohort3.plan()
    _, learning, policies = cohort3.contracts(stage)
    spec = cohort3.spec(stage, learning)
    assert spec == {**growth.stage_spec(read(ROOT / growth.PLAN), learning), 'steps': 512,
                    'seed': DECLARATION['training']['seed'], 'mixture': {'experience': 6, 'replay': 2}}
    assert DECLARATION['training']['arm'] == 'addition' and (spec['rank'], spec['alpha']) == (16, 32)
    assert sum(spec['mixture'].values()) == spec['gradient_accumulation']
    assert DECLARATION['resource_budget']['parameters'] == 1048576
    assert '1,048,576 parameters' in read(ROOT / growth.PLAN)['training']['module']
    assert policies == {'drafting': DRAFTING, 'scheduling': SLOTS}
    assert [tuple(route) for route in DECLARATION['integration']['routes']] == [('U1', 'drafting'), ('L3', 'drafting')]


def test_previous_development_is_pinned_and_cohort_three_stays_within_the_ceiling():
    previous = DECLARATION['previous']['development']
    assert sha256(ROOT / previous['path']) == previous['sha256']
    budget = DECLARATION['budget']
    first = sum(read(ROOT / f'config/experiments/assistant-growth-{name}-report.json')['resources_finished'][
        'conservative_compute_usd'] for name in ('stage1', 'round2', 'round3', 'round4', 'router2', 'router3',
                                                 'router4', 'round5'))
    second = sum(read(ROOT / f'config/experiments/assistant-growth-{name}-report.json')['conservative_compute_usd']
                 for name in ('development', 'development4', 'development5', 'development6', 'confirmation',
                              'development7', 'confirmation2'))
    assert budget['spent_usd'] == round(first + second, 2)
    total = budget['spent_usd'] + budget['gpu_usd'] + budget['development_usd'] + budget['confirmation_usd']
    assert total <= budget['ceiling_usd'] == 150 and f'${total:.2f}' in budget['allowances']
    cloud = cloud_module()
    resources = cloud.resources(cohort3.PROFILE)
    assert resources['gpu'] and resources['planning_cap_usd'] == budget['gpu_usd'] and resources['hours'] == 5
    assert cloud.UPLOAD_PROFILES[cohort3.PROFILE] == growth.UPLOADED
    assert cloud.GRANITE_PROFILES[cohort3.PROFILE][0] == 'assistant_growth_cohort3'
    assert cloud.remote_command(cohort3.PROFILE)[1].endswith(cohort3.SCRIPT)


def test_cohort_three_inventory_pins_contracts_sources_and_the_stage_one_runtime():
    execution = read(ROOT / cohort3.EXECUTION)
    for name, digest in execution['contracts'].items():
        assert sha256(ROOT / name) == digest
    assert set(execution['contracts']) <= set(execution['sources']) and cohort3.SCRIPT in execution['sources']
    probe = ('import os, sys; import neuroshard.evolution.assistant_growth_cohort3; assert "torch" not in sys.modules; '
             'import neuroshard.evolution.assistant_experience_run, neuroshard.evolution.assistant_experience_train, '
             'neuroshard.evolution.assistant_experience, neuroshard.evolution.assistant_selector, '
             'neuroshard.evolution.assistant_calendar, neuroshard.evolution.assistant_calendar_slots; '
             'root = os.path.abspath("src"); print("\\n".join(sorted(os.path.relpath(x.__file__) '
             'for x in list(sys.modules.values()) if getattr(x, "__file__", None) and '
             'os.path.abspath(x.__file__).startswith(root))))')
    imported = subprocess.check_output([sys.executable, '-c', probe], cwd=ROOT, text=True,
                                       env={'PYTHONPATH': str(ROOT / 'src')}).split()
    assert set(imported) <= set(execution['sources'])
    for name in (cohort3.DECLARATION, DECLARATION['sealed']['drafting_data'], DECLARATION['sealed']['scheduling_data'],
                 DECLARATION['experience']['policy'], *DECLARATION['policies'].values()):
        assert name in execution['contracts']
    stage1 = read(ROOT / growth.EXECUTION)
    for key in ('packages', 'python', 'gpus', 'environment', 'worker_environment', 'tokenizer_pipeline_sha256',
                'execution', 'drafting_collection'):
        assert execution[key] == stage1[key]
    resources = cloud_module().resources(cohort3.PROFILE)
    assert execution['worker_seconds'] + resources['setup_seconds'] + resources['copy_seconds'] + 600 <= (
        resources['hours'] * 3600)


def test_importing_cohort_three_does_not_load_torch():
    probe = 'import sys, neuroshard.evolution.assistant_growth_cohort3; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', probe], check=True, cwd=ROOT, env={'PYTHONPATH': str(ROOT / 'src')})
