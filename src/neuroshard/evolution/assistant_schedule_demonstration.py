"""Correct solutions of training scheduling cases, written as the accepted version's own replies.

Every reply is one or two tool calls in the native envelope the accepted version emits,
or a one-sentence confirmation. A plan's due date comes from shift_date on the start date
and review interval the plan states. Busy times are listed two calls per reply, for each
date the user asks about, until the expected date. The expected draft and meeting are
saved once each. Under the free-slot interface each of those dates is asked for its common
windows instead. Only training cases are solved, and a demonstration becomes experience
only after it passes the scorer in the calendar workspace.
"""

import json
import re

from neuroshard.evolution import assistant_calendar as calendar
from neuroshard.evolution import assistant_schedule_data as schedule

PLAN = re.compile(r'starts on (\d{4}-\d{2}-\d{2})\..*The review interval is (\d+) calendar days', re.S)
FIRST_DATE = re.compile(r'from (\d{4}-\d{2}-\d{2}) onward')
BOUND = re.compile(r'no earlier than (\d{2}:\d{2})')


def call(name, arguments):
    return '<tool_call>\n' + json.dumps({'name': name, 'arguments': arguments}) + '\n</tool_call>'


def plan_steps(case, project, plan_id):
    """List the project's documents, read the cited plan, and shift its start date by its review interval."""
    content = next(d['content'] for d in case['world']['documents'] if d['id'] == plan_id)
    start, interval = PLAN.search(content).groups()
    return [call('list_documents', {'project': project}), call('read_document', {'document_id': plan_id}),
            call('shift_date', {'start_date': start, 'days': int(interval)})]


def busy_calls(turn, meeting, days):
    """Under the first calendar interface: every attendee's busy times on each date."""
    return [call('list_busy', {'team': team, 'date': day}) for day in days for team in meeting['attendees']]


def slot_calls(turn, meeting, days):
    """Under the free-slot interface: the attendees' common windows on each date, after any bound the user names."""
    bound = BOUND.search(turn['user'])
    return [call('free_slots', {'teams': meeting['attendees'], 'date': day, 'duration_minutes': meeting['duration_minutes'],
                                'not_before': bound.group(1) if bound else '09:00'}) for day in days]


def texts(case, lookups=busy_calls):
    """The demonstration's replies, in order, for every turn of one training case.

    ``lookups(turn, meeting, days)`` are the calls that find the meeting's time, made two per reply.
    """
    if not case.get('capability') or case['split'] not in schedule.TRAINING:
        raise ValueError('demonstrations may only solve training scheduling cases')
    window = sorted(case['world']['calendars'][0]['busy'])
    outputs = []
    for index, turn in enumerate(case['turns']):
        draft, meeting = calendar.goals(turn['expected'])
        if draft and index == 0:
            outputs += plan_steps(case, draft['project'], draft['source_ids'][0])
            outputs.append(call('save_draft', draft))
            outputs.append(f"Draft saved for {draft['project']}, addressed to the {draft['recipient']}, due "
                           f"{draft['due_date']}, with a total of {draft['total']}.")
        if meeting:
            if meeting['source_ids']:
                outputs += plan_steps(case, meeting['project'], meeting['source_ids'][0])
            days = [meeting['date']]
            if case['family'] == 'day' and index == 0:
                first = FIRST_DATE.search(turn['user']).group(1)
                days = [day for day in window if first <= day <= meeting['date']]
            found = lookups(turn, meeting, days)
            outputs += ['\n'.join(found[i:i + 2]) for i in range(0, len(found), 2)]
            outputs.append(call('save_meeting', meeting))
            outputs.append(f"Meeting draft saved for {meeting['project']}: {meeting['duration_minutes']} minutes with "
                           f"the {' and the '.join(meeting['attendees'])} on {meeting['date']} at "
                           f"{meeting['start_time']}.")
    return outputs


def slot_texts(case):
    """The same solutions under the free-slot interface."""
    return texts(case, slot_calls)
