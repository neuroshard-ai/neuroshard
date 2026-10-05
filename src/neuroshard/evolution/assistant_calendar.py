"""The drafting workspace with team calendars: the interface of the scheduling cohort.

Every drafting tool, document check and draft rule is unchanged. Calendars add three
tools: a team's busy times on one date, exact clock arithmetic, and a meeting draft
saved locally; nothing is booked or sent. Times are 24-hour HH:MM within working
hours. Expected outcomes stay with the scorer, which also accepts drafting goals, and
a world may hold no calendars, so drafting conversations run unchanged under this
interface: the same documents and goals, with the calendar tools offered.
"""

import copy
import re

from neuroshard.evolution import assistant_workspace as drafting
from neuroshard.evolution.modular_reference_execution import identity

INTERFACE = 'workspace-calendar/1'
DAY_START, DAY_END = 9 * 60, 17 * 60
TEAMS = ('operations team', 'design team', 'service team', 'logistics team', 'review board', 'finance team')
LIST = {'type': 'array', 'items': drafting.STRING}
MEETING_FIELDS = {'project': drafting.STRING, 'attendees': LIST, 'date': drafting.STRING,
                  'start_time': drafting.STRING, 'duration_minutes': drafting.INTEGER, 'source_ids': LIST}
TOOLS = drafting.TOOLS + [
    drafting.tool('list_busy', "List one team's busy times on one date as [start, end) pairs of 24-hour HH:MM "
                  'times within working hours (09:00-17:00).', {'team': drafting.STRING, 'date': drafting.STRING}),
    drafting.tool('add_minutes', 'Add a signed number of minutes to a 24-hour HH:MM time on the same day.',
                  {'time': drafting.STRING, 'minutes': drafting.INTEGER}),
    drafting.tool('save_meeting', "Create or replace this project's meeting draft. This saves a local draft only; "
                  'nothing is booked or sent. Attendees are team names. Cite any documents used by ID; each must '
                  'have been read in this conversation.', MEETING_FIELDS),
]
REGISTRY = {item['function']['name']: item['function']['parameters'] for item in TOOLS}


def minutes(value):
    if not isinstance(value, str) or not re.fullmatch(r'([01]\d|2[0-3]):[0-5]\d', value):
        raise ValueError('time must be HH:MM')
    return int(value[:2]) * 60 + int(value[3:])


def clock(total):
    if type(total) is not int or not 0 <= total < 24 * 60:
        raise ValueError('time outside the day')
    return f'{total // 60:02d}:{total % 60:02d}'


def busy_intervals(intervals):
    """Validated busy times of one team on one date: sorted, disjoint and within working hours."""
    if not isinstance(intervals, list) or len(intervals) > 12:
        raise ValueError('invalid busy list')
    previous = DAY_START
    for pair in intervals:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError('invalid busy interval')
        start, end = minutes(pair[0]), minutes(pair[1])
        if not previous <= start < end <= DAY_END:
            raise ValueError('busy intervals must be sorted, disjoint and within working hours')
        previous = end
    return intervals


def parse_calls(text):
    return drafting.parse_calls(text, REGISTRY)


class Workspace(drafting.Workspace):
    registry = REGISTRY

    def __init__(self, public_world):
        if set(public_world) not in ({'documents'}, {'documents', 'calendars'}):
            raise ValueError('workspace accepts public documents and calendars only, not scoring metadata')
        super().__init__({'documents': public_world['documents']})
        self.calendars = {}
        for calendar in copy.deepcopy(public_world.get('calendars', [])):
            if (set(calendar) != {'team', 'busy'} or calendar['team'] not in TEAMS or calendar['team'] in self.calendars
                    or not isinstance(calendar['busy'], dict) or not 1 <= len(calendar['busy']) <= 31):
                raise ValueError('invalid calendar inventory')
            for day, intervals in calendar['busy'].items():
                if drafting.iso_date(day).isoformat() != day:
                    raise ValueError('calendar dates must be ISO dates')
                busy_intervals(intervals)
            self.calendars[calendar['team']] = calendar['busy']
        if 'calendars' in public_world and not self.calendars:
            raise ValueError('a calendar world needs at least one calendar')
        self.meetings = {}
        self.world_root = identity(public_world)

    def snapshot(self):
        return {**super().snapshot(), 'meetings': copy.deepcopy(self.meetings)}

    def _apply(self, name, args):
        if name == 'list_busy':
            busy = self.calendars[args['team']]
            day = drafting.iso_date(args['date']).isoformat()
            if day not in busy:
                raise ValueError('date outside the calendar')
            return {'team': args['team'], 'date': day, 'busy': copy.deepcopy(busy[day])}
        if name == 'add_minutes':
            if abs(args['minutes']) > 24 * 60:
                raise ValueError('minute offset bound exceeded')
            return {'time': clock(minutes(args['time']) + args['minutes'])}
        if name == 'save_meeting':
            attendees, sources = args['attendees'], args['source_ids']
            if (not 1 <= len(args['project']) <= 120 or not 1 <= len(attendees) <= 4
                    or len(set(attendees)) != len(attendees) or not set(attendees) <= set(self.calendars)
                    or len(sources) > 4 or len(set(sources)) != len(sources) or not set(sources) <= self.read_ids):
                raise ValueError('invalid meeting fields or unread evidence')
            day = drafting.iso_date(args['date']).isoformat()
            start, duration = minutes(args['start_time']), args['duration_minutes']
            if not 15 <= duration <= 240 or duration % 15 or start < DAY_START or start + duration > DAY_END:
                raise ValueError('meeting outside working hours or of an invalid length')
            if not any(d['project'] == args['project'] for d in self.documents.values()):
                raise ValueError('project is absent from this workspace')
            old = self.meetings.get(args['project'])
            meeting = {**copy.deepcopy(args), 'date': day, 'attendees': sorted(attendees), 'source_ids': sorted(sources),
                       'revision': old['revision'] + 1 if old else 1}
            self.meetings[args['project']] = meeting
            return {'saved': True, 'meeting': copy.deepcopy(meeting)}
        return super()._apply(name, args)


def goals(expected):
    """A round's expected draft and meeting; a drafting goal is a draft alone."""
    if set(expected) <= {'draft', 'meeting'} and expected:
        return expected.get('draft'), expected.get('meeting')
    return expected, None


def score_round(snapshot, expected, terminated, final_text):
    """Outcome scoring: exactly the expected draft and meeting, and no other write."""
    if not terminated or not isinstance(final_text, str) or not final_text.strip():
        return False
    draft, meeting = goals(expected)
    drafts, meetings = snapshot['drafts'], snapshot['meetings']
    if set(drafts) != ({draft['project']} if draft else set()) or set(meetings) != ({meeting['project']} if meeting else set()):
        return False
    if draft and identity({key: drafts[draft['project']][key] for key in drafting.DRAFT_FIELDS}) != identity(
            {**draft, 'source_ids': sorted(draft['source_ids'])}):
        return False
    return not meeting or identity({key: meetings[meeting['project']][key] for key in MEETING_FIELDS}) == identity(
        {**meeting, 'attendees': sorted(meeting['attendees']), 'source_ids': sorted(meeting['source_ids'])})


def replay_transcript(public_world, calls):
    workspace = Workspace(public_world)
    for row in calls:
        if workspace.execute(row['call']) != row['result']:
            raise ValueError('tool transcript result differs from deterministic execution')
    return workspace.snapshot()
