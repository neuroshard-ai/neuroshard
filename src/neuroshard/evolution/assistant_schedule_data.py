"""Fictional scheduling workflows for the calendar workspace: the second cohort of repeated growth.

Each case keeps the drafting grammar's delivery plans for its project and adds the busy
times of four teams over two weeks. Primitive families book one meeting; compound
families apply a follow-up; cross families also need the drafting skills, taking the
date from a plan or saving a draft first. Expected meetings are computed from the
calendars and never enter a request. Splits have their own seeds, so their projects and
IDs are disjoint from every drafting split.
"""

from datetime import date, timedelta
import random

from neuroshard.evolution.assistant_calendar import DAY_END, DAY_START, TEAMS, clock, minutes
from neuroshard.evolution.modular_reference_execution import identity

FAMILIES = ('slot', 'after', 'three', 'day', 'longer', 'invite', 'move', 'swap')
PRIMITIVE = FAMILIES[:4]
CROSS = ('review', 'handoff')
SPLITS = {'train': (21000, 32), 'integration': (22000, 8), 'development': (23000, 3), 'confirmation': (24000, 24),
          'cross-train': (25000, 16), 'cross-integration': (26000, 4), 'cross-development': (27000, 4),
          'cross-confirmation': (28000, 24)}
SEALED = ('confirmation', 'cross-confirmation')
TRAINING = ('train', 'cross-train')
WINDOW = 14
DURATIONS = (30, 45, 60, 90)
BOUNDS = (11 * 60, 12 * 60 + 30, 13 * 60, 14 * 60)
PURPOSES = ('planning meeting', 'status review', 'kickoff', 'risk review')
RECIPIENTS = ('operations team', 'design team', 'service team', 'logistics team')
SAVE = ' Save a meeting draft only and do not send anything.'


def families(split):
    return CROSS if split.startswith('cross-') else FAMILIES


def day_busy(rng):
    """Two to four busy blocks on the half hour, merged, sorted and within working hours."""
    taken = set()
    for _ in range(rng.randint(2, 4)):
        length = rng.choice((1, 2, 3, 4))
        start = rng.randrange(0, (DAY_END - DAY_START) // 30 - length + 1)
        taken.update(range(start, start + length))
    blocks = []
    for slot in sorted(taken):
        if blocks and blocks[-1][1] == slot:
            blocks[-1][1] = slot + 1
        else:
            blocks.append([slot, slot + 1])
    return [[clock(DAY_START + 30 * a), clock(DAY_START + 30 * b)] for a, b in blocks]


def blocked(rng, duration):
    """Busy times for two teams that leave no common free slot of ``duration`` minutes all day."""
    split = DAY_START + 30 * rng.randrange(4, 13)
    gap = 30 if duration > 30 and rng.random() < .5 else 0
    return [[clock(DAY_START), clock(split)]], [[clock(split + gap), clock(DAY_END)]]


def free(busy, attendees, day, start, duration):
    end = start + duration
    return DAY_START <= start and end <= DAY_END and not any(
        minutes(a) < end and start < minutes(b) for team in attendees for a, b in busy[team][day])


def earliest(busy, attendees, day, duration, not_before=DAY_START):
    """The earliest quarter-hour start at which every attendee is free for the whole meeting, or None."""
    for start in range(not_before, DAY_END - duration + 1, 15):
        if free(busy, attendees, day, start, duration):
            return start
    return None


def meeting(project, attendees, day, start, duration, sources=()):
    return {'project': project, 'attendees': sorted(attendees), 'date': day, 'start_time': clock(start),
            'duration_minutes': duration, 'source_ids': sorted(sources)}


def make_case(split, family, index):
    if split not in SPLITS or family not in families(split) or type(index) is not int or not 0 <= index < 40:
        raise ValueError('unknown scheduling split or family')
    number = SPLITS[split][0] + families(split).index(family) * 40 + index
    rng = random.Random(number)
    project = f"{rng.choice(['Cedar', 'Harbor', 'Meadow', 'Orchard', 'Summit', 'Willow'])} {number}"
    sibling = project + ' Annex'
    recipient = rng.choice(RECIPIENTS)
    start = date(2027, 1, 1) + timedelta(days=rng.randrange(160))
    quantity, interval = rng.randrange(25, 80), rng.randrange(3, 10)
    documents = []

    def document(owner, revision, status, day, count):
        value = {'id': 'doc-' + identity([owner, revision, status, 'schedule', split])[:10], 'project': owner,
                 'title': owner + ' delivery plan', 'revision': revision, 'status': status,
                 'content': f'The plan for {owner} starts on {day.isoformat()}. '
                            f'The baseline quantity is {count} units. '
                            f'The review interval is {interval} calendar days. '
                            'Use the status and revision in the document metadata to choose a version.'}
        documents.append(value)
        return value

    document(project, 1, 'approved', start - timedelta(days=7), quantity - rng.randrange(3, 9))
    current = document(project, 2, 'approved', start, quantity)
    document(project, 3, 'draft', start + timedelta(days=12), quantity + 25)
    document(sibling, 2, 'approved', start + timedelta(days=2), quantity + 9)
    rng.shuffle(documents)
    others = [team for team in TEAMS if team != recipient]
    teams = [recipient] + rng.sample(others, 3)
    a, b, c = teams[0], teams[1], teams[2]
    window = [(start + timedelta(days=offset)).isoformat() for offset in range(WINDOW)]
    duration, purpose = rng.choice(DURATIONS), rng.choice(PURPOSES)
    # Busy times sit on the half hour, so only an extension past a half-hour boundary can force a move.
    extension = 30 * (1 + index % 2)
    due = (start + timedelta(days=interval)).isoformat()
    day = due if family in CROSS else window[rng.randrange(0, WINDOW - 4)]

    def ok(busy):
        first = earliest(busy, [a, b], day, duration)
        if family == 'day':
            gap = 1 + index % 2
            return (all(earliest(busy, [a, b], window[window.index(day) + k], duration) is None for k in range(gap))
                    and earliest(busy, [a, b], window[window.index(day) + gap], duration) not in (None, DAY_START))
        if first in (None, DAY_START):
            return False
        if family == 'after':
            bound = BOUNDS[index % len(BOUNDS)]
            later = earliest(busy, [a, b], day, duration, bound)
            return first < bound and later not in (None, bound)
        if family in ('three', 'invite'):
            joined = earliest(busy, [a, b, c], day, duration)
            return joined is not None and (joined != first or index % 2)
        if family == 'longer':
            longer = earliest(busy, [a, b], day, duration + extension)
            return longer is not None and (longer != first or index % 2)
        if family == 'move':
            return earliest(busy, [a, b], window[window.index(day) + 1], duration) is not None
        if family == 'swap':
            swapped = earliest(busy, [a, c], day, duration)
            return swapped is not None and (swapped != first or index % 2)
        return True

    for _ in range(5000):
        busy = {team: {d: day_busy(rng) for d in window} for team in teams}
        if family == 'day':
            for k in range(1 + index % 2):
                busy[a][window[window.index(day) + k]], busy[b][window[window.index(day) + k]] = blocked(rng, duration)
        if ok(busy):
            break
    else:
        raise ValueError('no calendar satisfies the scheduling family')
    world = {'documents': documents, 'calendars': [{'team': team, 'busy': busy[team]} for team in teams]}
    book = f'Schedule a {duration}-minute {purpose} for {project} with the {a} and the {b} on {day}, '
    pair = [a, b]
    if family == 'after':
        bound = BOUNDS[index % len(BOUNDS)]
        text = book + f'starting no earlier than {clock(bound)}, at the earliest such time both teams are free.'
        goal = meeting(project, pair, day, earliest(busy, pair, day, duration, bound), duration)
    elif family == 'three':
        text = (f'Schedule a {duration}-minute {purpose} for {project} with the {a}, the {b} and the {c} on {day}, '
                'at the earliest time all three teams are free.')
        goal = meeting(project, [a, b, c], day, earliest(busy, [a, b, c], day, duration), duration)
    elif family == 'day':
        booked = window[window.index(day) + 1 + index % 2]
        text = (f'Schedule a {duration}-minute {purpose} for {project} with the {a} and the {b} on the first date from '
                f'{day} onward when both teams have a free {duration}-minute slot, at the earliest such time.')
        goal = meeting(project, pair, booked, earliest(busy, pair, booked, duration), duration)
    elif family == 'review':
        text = (f'Schedule a {duration}-minute delivery review for {project} with the {a} and the {b} on the due date '
                'of the latest approved delivery plan, which is one review interval after its start date. Use the '
                'earliest time both teams are free and cite the plan you used.')
        goal = meeting(project, pair, due, earliest(busy, pair, due, duration), duration, [current['id']])
    elif family == 'handoff':
        text = (f'Create a draft for {project}, addressed to the {a}. Use the latest approved delivery plan, not an '
                'unapproved revision. The due date is one review interval after the plan start date. Keep the '
                'baseline quantity as the total. Save a draft only, cite the documents used, and do not send anything.')
        goal = {'draft': {'project': project, 'recipient': a, 'due_date': due, 'total': quantity,
                          'source_ids': [current['id']]}}
    else:
        text = book + 'at the earliest time both teams are free.'
        goal = meeting(project, pair, day, earliest(busy, pair, day, duration), duration)
    turns = [{'user': text + ('' if family == 'handoff' else SAVE),
              'expected': goal if family == 'handoff' else {'meeting': goal}}]
    if family == 'longer':
        longer = duration + extension
        turns.append({'user': f'Make that meeting {extension} minutes longer on the same date, at the earliest '
                              'time both teams are free for the whole meeting.',
                      'expected': {'meeting': meeting(project, pair, day, earliest(busy, pair, day, longer), longer)}})
    elif family == 'invite':
        turns.append({'user': f'Also invite the {c}, at the earliest time that date when all three teams are free.',
                      'expected': {'meeting': meeting(project, [a, b, c], day,
                                                      earliest(busy, [a, b, c], day, duration), duration)}})
    elif family == 'move':
        moved = window[window.index(day) + 1]
        turns.append({'user': 'Move it to the next day, at the earliest time both teams are free.',
                      'expected': {'meeting': meeting(project, pair, moved, earliest(busy, pair, moved, duration),
                                                      duration)}})
    elif family == 'swap':
        turns.append({'user': f'Replace the {b} with the {c}, at the earliest time that date when the {a} and the {c} '
                              'are both free.',
                      'expected': {'meeting': meeting(project, [a, c], day, earliest(busy, [a, c], day, duration),
                                                      duration)}})
    elif family == 'handoff':
        turns.append({'user': f'Now schedule a {duration}-minute handoff for {project} with the {a} and the {b} on that '
                              "draft's due date, at the earliest time both teams are free. Keep the draft unchanged."
                              + SAVE,
                      'expected': {**goal, 'meeting': meeting(project, pair, due, earliest(busy, pair, due, duration),
                                                              duration)}})
    return {'id': 'schedule-' + identity([split, family, index])[:12], 'split': split, 'family': family,
            'block': family, 'primitive': family in PRIMITIVE, 'capability': 'cross' if family in CROSS else 'scheduling',
            'world': world, 'turns': turns}


def cases(split):
    if split not in SPLITS:
        raise ValueError('unknown scheduling split')
    result = [make_case(split, family, i) for family in families(split) for i in range(SPLITS[split][1])]
    random.Random(SPLITS[split][0]).shuffle(result)
    return result
