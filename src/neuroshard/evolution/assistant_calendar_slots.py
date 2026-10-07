"""The calendar workspace with a free-slot tool: the interface of A3 round 5.

Every calendar and drafting tool, rule and scorer of the first calendar interface is
unchanged. One tool is added: the windows on one date in which every listed team is free
for at least a given length, earliest first, starting on the quarter hour no earlier than
a given time. It computes from the public calendars only; the model still chooses the
teams, date, length and bound, and saves the meeting itself.
"""

from neuroshard.evolution import assistant_calendar as calendar
from neuroshard.evolution import assistant_workspace as drafting
from neuroshard.evolution.assistant_calendar import (  # noqa: F401  (one interface surface)
    DAY_END, DAY_START, MEETING_FIELDS, TEAMS, clock, goals, minutes, score_round,
)

INTERFACE = 'workspace-calendar/2'
TOOLS = calendar.TOOLS + [
    drafting.tool('free_slots', 'List the windows on one date in which every listed team is free for at least '
                  'duration_minutes, earliest first, as [start, end) pairs of 24-hour HH:MM times within working '
                  'hours (09:00-17:00). Windows start on the quarter hour, no earlier than not_before.',
                  {'teams': calendar.LIST, 'date': drafting.STRING, 'duration_minutes': drafting.INTEGER,
                   'not_before': drafting.STRING}),
]
REGISTRY = {item['function']['name']: item['function']['parameters'] for item in TOOLS}


def free_windows(busy, teams, duration, not_before=DAY_START):
    """Quarter-hour-aligned windows, earliest first, in which no team in ``busy`` is busy for ``duration`` minutes.

    ``busy`` maps each team to its [start, end) HH:MM pairs on one date. A window's start is the
    first quarter hour at or after both its free gap and ``not_before``.
    """
    taken = sorted((minutes(a), minutes(b)) for team in teams for a, b in busy[team])
    windows, cursor = [], DAY_START
    for start, end in taken + [(DAY_END, DAY_END)]:
        begin = max(cursor, not_before)
        begin += -begin % 15
        if start - begin >= duration:
            windows.append([clock(begin), clock(start)])
        cursor = max(cursor, end)
    return windows


def parse_calls(text):
    return drafting.parse_calls(text, REGISTRY)


class Workspace(calendar.Workspace):
    registry = REGISTRY

    def _apply(self, name, args):
        if name == 'free_slots':
            teams, duration = args['teams'], args['duration_minutes']
            if (not 1 <= len(teams) <= 4 or len(set(teams)) != len(teams) or not set(teams) <= set(self.calendars)
                    or not 15 <= duration <= 240 or duration % 15):
                raise ValueError('invalid teams or meeting length')
            day = drafting.iso_date(args['date']).isoformat()
            bound = minutes(args['not_before'])
            if not DAY_START <= bound < DAY_END or any(day not in self.calendars[team] for team in teams):
                raise ValueError('time outside working hours or date outside the calendar')
            busy = {team: self.calendars[team][day] for team in teams}
            return {'teams': sorted(teams), 'date': day, 'duration_minutes': duration, 'not_before': args['not_before'],
                    'free': free_windows(busy, teams, duration, bound)}
        return super()._apply(name, args)


def replay_transcript(public_world, calls):
    workspace = Workspace(public_world)
    for row in calls:
        if workspace.execute(row['call']) != row['result']:
            raise ValueError('tool transcript result differs from deterministic execution')
    return workspace.snapshot()
