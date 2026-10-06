import copy
import json
import subprocess
import sys

import pytest

from neuroshard.evolution import assistant_calendar as calendar
from neuroshard.evolution import assistant_calendar_slots as slots
from neuroshard.evolution import assistant_growth_round5 as round5
from neuroshard.evolution import assistant_growth_run as growth
from neuroshard.evolution import assistant_routing as routing
from neuroshard.evolution import assistant_schedule_data as schedule
from neuroshard.evolution import assistant_schedule_demonstration as demonstrations
from neuroshard.evolution import assistant_workflow as workflow
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution.modular_reference_execution import ROOT, identity, read, save, sha256

from test_assistant_growth_baseline import cloud_module
from test_assistant_workflow import reply

DECLARATION = read(ROOT / round5.DECLARATION)
SLOTS = read(ROOT / DECLARATION['interface']['policy'])
CALENDAR = read(ROOT / 'config/experiments/assistant-workflow-policy-calendar.json')
DRAFTING = read(ROOT / 'config/experiments/assistant-workflow-policy.json')


def meeting_turns(splits):
    """Every meeting turn of non-sealed splits with the tool's arguments the turn implies."""
    for split in splits:
        for case in schedule.cases(split):
            busy = {entry['team']: entry['busy'] for entry in case['world']['calendars']}
            for turn in case['turns']:
                _, goal = calendar.goals(turn['expected'])
                if goal:
                    bound = demonstrations.BOUND.search(turn['user'])
                    yield busy, goal, calendar.minutes(bound.group(1)) if bound else calendar.DAY_START


def test_the_first_window_is_the_generators_earliest_start_on_every_open_split():
    checked = 0
    for busy, goal, bound in meeting_turns(('train', 'cross-train', 'integration', 'cross-integration', 'development',
                                            'cross-development', *schedule.GROWTH)):
        day = {team: busy[team][goal['date']] for team in goal['attendees']}
        windows = slots.free_windows(day, goal['attendees'], goal['duration_minutes'], bound)
        assert windows and windows[0][0] == goal['start_time']
        assert all(calendar.minutes(b) - calendar.minutes(a) >= goal['duration_minutes'] for a, b in windows)
        checked += 1
    assert checked > 1800


def test_the_tool_validates_its_arguments_and_only_the_new_interface_offers_it():
    case = schedule.make_case('train', 'three', 0)
    world = slots.Workspace({'documents': case['world']['documents'], 'calendars': case['world']['calendars']})
    teams = [entry['team'] for entry in case['world']['calendars']][:2]
    day = sorted(case['world']['calendars'][0]['busy'])[0]

    def ask(**changes):
        return world.execute({'name': 'free_slots', 'arguments': {'teams': teams, 'date': day, 'duration_minutes': 30,
                                                                  'not_before': '09:00', **changes}})

    result = ask()
    assert result['teams'] == sorted(teams) and result['date'] == day and result['free']
    for bad in ({'teams': ['marketing team']}, {'teams': teams + teams[:1]}, {'duration_minutes': 20},
                {'duration_minutes': 300}, {'not_before': '17:00'}, {'not_before': '9am'}, {'date': '2030-01-01'}):
        assert 'error' in ask(**bad)
    assert ask(not_before='16:30')['free'] in ([], [['16:30', '17:00']])
    text = '<tool_call>\n' + json.dumps({'name': 'free_slots', 'arguments': {
        'teams': teams, 'date': day, 'duration_minutes': 30, 'not_before': '09:00'}}) + '\n</tool_call>'
    assert slots.parse_calls(text)[0]['name'] == 'free_slots'
    with pytest.raises(ValueError):
        calendar.parse_calls(text)
    assert workflow.interface(SLOTS) is slots and workflow.interface(CALENDAR) is calendar


def test_the_new_policy_changes_only_its_interface_and_how_meetings_are_found():
    assert SLOTS['interface'] == slots.INTERFACE and CALENDAR['interface'] == calendar.INTERFACE
    assert {k: v for k, v in SLOTS.items() if k not in ('interface', 'system_instruction')} == {
        k: v for k, v in CALENDAR.items() if k not in ('interface', 'system_instruction')}
    before, after = CALENDAR['system_instruction'], SLOTS['system_instruction']
    shared = before[:before.index('For meetings')]
    assert after.startswith(shared) and 'free_slots' in after and 'free_slots' not in before
    assert [t['function']['name'] for t in slots.TOOLS] == [t['function']['name'] for t in calendar.TOOLS] + ['free_slots']


def test_demonstrations_under_the_tool_solve_training_cases_and_book_the_first_window():
    stage = round5.plan()
    cases = [c for c in round5.training_cases(stage)]
    chosen = [next(c for c in cases if c['family'] == family and c['split'] == split)
              for split in ('train', 'cross-train', 'train3') for family in schedule.families(split)]
    for case in chosen:
        item, result = round5.demonstration(case, SLOTS)
        assert result['score']['passed'] and item['demonstration'] and round5.first_windows(result)
        assert any(row['call']['name'] == 'free_slots' for row in result['calls'])
        assert not any(row['call']['name'] == 'list_busy' for row in result['calls'])
    forged = copy.deepcopy(round5.demonstration(chosen[0], SLOTS)[1])
    saved = next(row for row in forged['calls'] if row['call']['name'] == 'save_meeting')
    saved['call']['arguments']['start_time'] = '16:45'
    assert not round5.first_windows(forged)
    with pytest.raises(ValueError, match='demonstrations may only solve training'):
        demonstrations.slot_texts(schedule.make_case('development', 'slot', 0))


def test_round_four_cases_and_steps_with_only_the_scheduling_policy_changed():
    stage, stage1 = round5.plan(), read(ROOT / growth.PLAN)
    cases = round5.training_cases(stage)
    assert len(cases) == DECLARATION['experience']['cases'] == 1152 and len({c['id'] for c in cases}) == 1152
    assert not {c['split'] for c in cases} & {'development', 'cross-development', *schedule.SEALED, *schedule.SEALED2}
    assert stage['training'] == {**stage1['training'], 'steps': 512}
    _, learning, policies = round5.contracts(stage)
    _, _, before = growth.contracts(stage1)
    assert policies == {**before, 'scheduling': SLOTS} and policies['drafting'] == DRAFTING
    assert round5.ROUTE_RUNS == growth.ROUTE_RUNS + (('U1', 'scheduling'),)


def test_a_routed_conversation_runs_the_tool_in_the_shared_workspace_and_rescores():
    case = next(c for c in schedule.cases('cross-train') if c['family'] == 'handoff')
    texts = iter(demonstrations.slot_texts(case))

    def respond(messages, tools):
        return reply(next(texts))

    routes = {'drafting': (respond, DRAFTING), 'scheduling': (respond, SLOTS)}
    result = routing.execute(case, routes, lambda turn, user: ['drafting', 'scheduling'][turn])
    assert result['score']['passed'] and result['score']['routes'] == ['drafting', 'scheduling']
    assert any(row['call']['name'] == 'free_slots' for row in result['calls'])
    assert routing.score(case, result, {'drafting': DRAFTING, 'scheduling': SLOTS}) == result['score']
    assert routing.workspace([DRAFTING, CALENDAR]) is calendar and routing.workspace([DRAFTING, SLOTS]) is slots


def test_fresh_sealed_splits_are_frozen_before_training_and_disjoint_from_every_split():
    manifest = read(ROOT / DECLARATION['sealed']['scheduling_data'])
    assert manifest['sealed'] == list(schedule.SEALED2) == [DECLARATION['sealed']['scheduling'],
                                                            DECLARATION['sealed']['cross']]
    earlier = {c['id'] for split in {**schedule.SPLITS, **schedule.GROWTH} for c in schedule.cases(split)}
    for split, frozen in manifest['splits'].items():
        cases = schedule.cases(split)
        assert identity(cases) == frozen['sha256'] and [c['id'] for c in cases] == frozen['case_ids']
        assert not {c['id'] for c in cases} & earlier
        assert all(sum(c['family'] == f for c in cases) == 24 for f in schedule.families(split))
    drafting = read(ROOT / DECLARATION['sealed']['drafting_data'])
    cases = data.cases(drafting['split'])
    assert drafting['split'] == DECLARATION['sealed']['drafting'] == 'confirmation5' and len(cases) == 192
    assert identity(cases) == drafting['sha256'] and [c['id'] for c in cases] == drafting['case_ids']
    assert set(drafting['disjoint_from']) == set(data.SPLITS) - {'confirmation5'}
    others = {c['id'] for split in data.SPLITS if split != 'confirmation5' for c in data.cases(split)}
    assert not {c['id'] for c in cases} & others


def test_round_five_stays_within_the_raised_ceiling():
    budget = DECLARATION['budget']
    reports = [read(ROOT / f'config/experiments/assistant-growth-{name}-report.json')
               for name in ('stage1', 'round2', 'round3', 'round4', 'router2', 'router3', 'router4')]
    spent = sum(r['resources_finished']['conservative_compute_usd'] for r in reports) + sum(
        read(ROOT / f'config/experiments/assistant-growth-{name}-report.json')['conservative_compute_usd']
        for name in ('development', 'development4', 'development5', 'development6', 'confirmation'))
    assert budget['spent_usd'] == round(spent, 2)
    total = budget['spent_usd'] + budget['gpu_usd'] + budget['development_usd'] + budget['confirmation_usd']
    assert total <= budget['ceiling_usd'] == 150 and f'${total:.2f}' in budget['allowances']
    cloud = cloud_module()
    resources = cloud.resources(round5.PROFILE)
    assert resources['gpu'] and resources['planning_cap_usd'] == budget['gpu_usd'] and resources['hours'] == 5
    assert cloud.UPLOAD_PROFILES[round5.PROFILE] == growth.UPLOADED
    assert cloud.GRANITE_PROFILES[round5.PROFILE][0] == 'assistant_growth_round5'
    assert cloud.remote_command(round5.PROFILE)[1].endswith(round5.SCRIPT)


def test_round_five_inventory_pins_contracts_sources_and_the_stage_one_runtime():
    execution = read(ROOT / round5.EXECUTION)
    for name, digest in execution['contracts'].items():
        assert sha256(ROOT / name) == digest
    assert set(execution['contracts']) <= set(execution['sources']) and round5.SCRIPT in execution['sources']
    probe = ('import os, sys; import neuroshard.evolution.assistant_growth_round5; assert "torch" not in sys.modules; '
             'import neuroshard.evolution.assistant_experience_run, neuroshard.evolution.assistant_experience_train, '
             'neuroshard.evolution.assistant_selector, neuroshard.evolution.assistant_calendar, '
             'neuroshard.evolution.assistant_calendar_slots; '
             'root = os.path.abspath("src"); print("\\n".join(sorted(os.path.relpath(x.__file__) '
             'for x in list(sys.modules.values()) if getattr(x, "__file__", None) and '
             'os.path.abspath(x.__file__).startswith(root))))')
    imported = subprocess.check_output([sys.executable, '-c', probe], cwd=ROOT, text=True,
                                       env={'PYTHONPATH': str(ROOT / 'src')}).split()
    assert set(imported) <= set(execution['sources'])
    for name in (DECLARATION['interface']['policy'], DECLARATION['sealed']['scheduling_data'],
                 DECLARATION['sealed']['drafting_data'], DECLARATION['experience']['data']):
        assert name in execution['contracts']
    stage1 = read(ROOT / growth.EXECUTION)
    for key in ('packages', 'python', 'gpus', 'environment', 'worker_environment', 'tokenizer_pipeline_sha256',
                'execution', 'drafting_collection'):
        assert execution[key] == stage1[key]
    resources = cloud_module().resources(round5.PROFILE)
    assert execution['worker_seconds'] + resources['setup_seconds'] + resources['copy_seconds'] + 600 <= (
        resources['hours'] * 3600)


def test_importing_round_five_does_not_load_torch():
    probe = 'import sys, neuroshard.evolution.assistant_growth_round5; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', probe], check=True, cwd=ROOT, env={'PYTHONPATH': str(ROOT / 'src')})
