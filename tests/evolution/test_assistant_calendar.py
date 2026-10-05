import copy
import json

import pytest

from neuroshard.evolution import assistant_calendar as calendar
from neuroshard.evolution import assistant_schedule_data as schedule
from neuroshard.evolution import assistant_workflow as workflow
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution import assistant_workspace as drafting
from neuroshard.evolution.modular_reference_execution import ROOT, identity, read

from test_assistant_workflow import envelope, execute_fixture, reference_texts, reply

POLICY = read(ROOT / 'config/experiments/assistant-workflow-policy-calendar.json')
DRAFTING = read(ROOT / 'config/experiments/assistant-workflow-policy.json')


def schedule_texts(case):
    # Goal-directed fixture ONLY validates the environment and budgets, never a model result.
    outputs = []
    for turn in case['turns']:
        draft, meeting = calendar.goals(turn['expected'])
        if draft and (not outputs or turn is case['turns'][0]):
            outputs.append(envelope('list_documents', {'project': draft['project']}))
            outputs.append('\n'.join(envelope('read_document', {'document_id': key}) for key in draft['source_ids']))
            outputs.append(envelope('save_draft', draft))
        if meeting:
            if meeting['source_ids']:
                outputs.append(envelope('list_documents', {'project': meeting['project']}))
                outputs.append('\n'.join(envelope('read_document', {'document_id': key}) for key in meeting['source_ids']))
            days = [meeting['date']]
            if case['family'] == 'day' and turn is case['turns'][0]:
                first = case['turns'][0]['user'].split(' from ')[1][:10]
                days = [d for d in case_window(case) if first <= d <= meeting['date']]
            calls = [envelope('list_busy', {'team': team, 'date': day}) for day in days for team in meeting['attendees']]
            outputs += ['\n'.join(calls[i:i + 2]) for i in range(0, len(calls), 2)]
            outputs.append(envelope('save_meeting', meeting))
        outputs.append('Saved locally.')
    return outputs


def case_window(case):
    return sorted(case['world']['calendars'][0]['busy'])


def run(case, texts=None, policy=POLICY):
    texts = iter(texts or schedule_texts(case))
    return workflow.execute(case, lambda messages, tools: reply(next(texts)), policy)


def brute_force(case, meeting, not_before='09:00'):
    """The earliest common free start, searched minute by minute from the public calendars alone."""
    busy = {c['team']: c['busy'][meeting['date']] for c in case['world']['calendars']}
    start = calendar.minutes(not_before)
    while start + meeting['duration_minutes'] <= calendar.DAY_END:
        end = start + meeting['duration_minutes']
        if not any(calendar.minutes(a) < end and start < calendar.minutes(b)
                   for team in meeting['attendees'] for a, b in busy[team]):
            return calendar.clock(start)
        start += 1
    return None


def test_every_scheduling_and_cross_case_is_solvable_within_the_conversation_budgets():
    for split in ('development', 'cross-development', 'integration', 'cross-integration'):
        for case in schedule.cases(split):
            result = run(case)
            assert result['score']['passed'], (case['id'], result['rounds'])
            for round_ in result['rounds']:
                assert round_['generation_count'] - (0 if round_ is result['rounds'][0] else
                                                     result['rounds'][0]['generation_count']) <= 6


def test_expected_meetings_are_the_earliest_common_free_starts_of_the_public_calendars():
    for split in schedule.SPLITS:
        for case in schedule.cases(split):
            for turn in case['turns']:
                _, meeting = calendar.goals(turn['expected'])
                if not meeting:
                    continue
                bound = turn['user'].split('starting no earlier than ')[1][:5] if 'no earlier than' in turn['user'] else '09:00'
                assert meeting['start_time'] == brute_force(case, meeting, bound), case['id']
                assert meeting['start_time'] != '09:00' or turn is not case['turns'][0], case['id']
            if case['family'] == 'day':
                first = case['turns'][0]['user'].split(' from ')[1][:10]
                _, meeting = calendar.goals(case['turns'][0]['expected'])
                for day in [d for d in case_window(case) if first <= d < meeting['date']]:
                    assert brute_force(case, {**meeting, 'date': day}) is None, case['id']


def test_wrong_slots_extra_writes_and_unread_citations_do_not_pass():
    case = schedule.make_case('development', 'slot', 0)
    goal = case['turns'][0]['expected']['meeting']
    texts = schedule_texts(case)
    wrong = dict(goal, start_time=calendar.clock(calendar.minutes(goal['start_time']) + 30))
    assert not run(case, texts[:-2] + [envelope('save_meeting', wrong), 'Saved.'])['score']['passed']
    extra = texts[:-1] + [envelope('save_meeting', {**goal, 'project': goal['project'] + ' Annex'}), 'Saved.']
    result = run(case, extra)
    assert set(result['rounds'][0]['snapshot']['meetings']) == {goal['project'], goal['project'] + ' Annex'}
    assert not result['score']['passed']
    world = calendar.Workspace(case['world'])
    document = case['world']['documents'][0]['id']
    assert 'error' in world.execute({'name': 'save_meeting', 'arguments': {**goal, 'source_ids': [document]}})
    assert 'error' in world.execute({'name': 'save_meeting', 'arguments': {**goal, 'attendees': ['nobody']}})
    assert 'error' in world.execute({'name': 'save_meeting', 'arguments': {**goal, 'start_time': '16:45'}})
    assert 'error' in world.execute({'name': 'save_meeting', 'arguments': {**goal, 'duration_minutes': 50}})
    assert 'error' in world.execute({'name': 'list_busy', 'arguments': {'team': goal['attendees'][0], 'date': '2099-01-01'}})
    assert world.execute({'name': 'add_minutes', 'arguments': {'time': '16:30', 'minutes': 45}}) == {'time': '17:15'}
    assert 'error' in world.execute({'name': 'add_minutes', 'arguments': {'time': '23:30', 'minutes': 45}})
    assert world.execute({'name': 'save_meeting', 'arguments': goal})['meeting']['revision'] == 1
    assert world.execute({'name': 'save_meeting', 'arguments': goal})['meeting']['revision'] == 2
    with pytest.raises(ValueError, match='public documents and calendars'):
        calendar.Workspace({**case['world'], 'expected': goal})


def test_drafting_conversations_score_alike_under_both_interfaces_and_a_meeting_is_an_extra_write():
    for case in data.cases('development')[:8]:
        assert run(case, reference_texts(case))['score'] == execute_fixture(case)['score']
    case = data.make_case('development', 'copy', 0)
    goal = case['turns'][0]['expected']
    texts = reference_texts(case)
    stray = {'project': goal['project'], 'attendees': ['design team'], 'date': goal['due_date'], 'start_time': '10:00',
             'duration_minutes': 30, 'source_ids': []}
    result = run(case, texts[:-1] + [envelope('save_meeting', stray), 'Saved.'])
    assert result['calls'][-1]['result'] == {'error': 'invalid tool arguments or unavailable workspace object'}
    assert result['score']['passed']
    assert POLICY['system_instruction'].startswith(DRAFTING['system_instruction'])
    assert {k: v for k, v in POLICY.items() if k not in ('interface', 'system_instruction', 'rendering')} == {
        k: v for k, v in DRAFTING.items() if k not in ('system_instruction', 'rendering')}


def test_the_policy_selects_the_interface_and_drafting_keeps_its_tools():
    assert workflow.interface(DRAFTING) is drafting and workflow.interface(POLICY) is calendar
    assert calendar.TOOLS[:len(drafting.TOOLS)] == drafting.TOOLS
    assert [t['function']['name'] for t in calendar.TOOLS[len(drafting.TOOLS):]] == ['list_busy', 'add_minutes',
                                                                                     'save_meeting']
    with pytest.raises(ValueError, match='unknown workspace interface'):
        workflow.interface({**POLICY, 'interface': 'other/1'})
    with pytest.raises(ValueError, match='unknown tool'):
        drafting.parse_calls(envelope('list_busy', {'team': 'design team', 'date': '2027-01-01'}))
    assert calendar.parse_calls(envelope('list_busy', {'team': 'design team', 'date': '2027-01-01'}))


def test_scheduling_splits_are_frozen_and_disjoint_from_each_other_and_from_every_drafting_split():
    manifest = read(ROOT / 'config/experiments/assistant-schedule-data.json')
    assert list(manifest['splits']) == list(schedule.SPLITS) and manifest['sealed'] == list(schedule.SEALED)
    projects, ids = {}, {}
    for split, frozen in manifest['splits'].items():
        cases = schedule.cases(split)
        assert identity(cases) == frozen['sha256'] and [c['id'] for c in cases] == frozen['case_ids']
        assert all(sum(c['family'] == f for c in cases) == schedule.SPLITS[split][1] for f in schedule.families(split))
        ids[split] = {c['id'] for c in cases}
        projects[split] = {c['world']['documents'][0]['project'].removesuffix(' Annex') for c in cases}
    for split in data.SPLITS:
        cases = data.cases(split)
        ids[split] = {c['id'] for c in cases}
        projects[split] = {t['expected']['project'] for c in cases for t in c['turns']}
    names = sorted(ids)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            assert not ids[left] & ids[right] and not projects[left] & projects[right], (left, right)
    retention = read(ROOT / 'config/experiments/assistant-workflow-data-confirmation4.json')
    cases = data.cases('confirmation4')
    assert identity(cases) == retention['sha256'] and [c['id'] for c in cases] == retention['case_ids']
    assert len(cases) == 192 and all(sum(c['family'] == f for c in cases) == 24 for f in data.FAMILIES)


def test_a_calendar_world_needs_valid_busy_times():
    case = schedule.make_case('development', 'three', 1)
    broken = copy.deepcopy(case['world'])
    day = next(iter(broken['calendars'][0]['busy']))
    broken['calendars'][0]['busy'][day] = [['12:00', '13:00'], ['12:30', '14:00']]
    with pytest.raises(ValueError, match='sorted, disjoint'):
        calendar.Workspace(broken)
    broken['calendars'][0]['busy'][day] = [['08:30', '09:30']]
    with pytest.raises(ValueError, match='sorted, disjoint'):
        calendar.Workspace(broken)
    assert json.dumps(calendar.Workspace(case['world']).snapshot()['meetings']) == '{}'
