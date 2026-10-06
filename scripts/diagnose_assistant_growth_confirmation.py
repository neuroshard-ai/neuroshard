#!/usr/bin/env python3
"""Classify the A3 candidate's failed sealed confirmation episodes from the published result; no model runs."""

import argparse
from collections import Counter
import json
from pathlib import Path

from neuroshard.evolution import assistant_calendar as calendar
from neuroshard.evolution import assistant_schedule_data as schedule
from neuroshard.evolution.modular_reference_execution import ROOT, read, sha256

RESULT = 'config/experiments/assistant-growth-confirmation-candidate-calendar-result.json'
OUTPUT = 'config/experiments/assistant-growth-confirmation-diagnostic.json'


def failed_turn(row):
    successes = row['score']['round_successes']
    return next((turn for turn, ok in enumerate(successes) if not ok), len(successes))


def wrong_start(case, row, turn, goal, saved):
    """How a saved meeting on the right date misses the expected start, from the public calendars."""
    busy = {entry['team']: entry['busy'] for entry in case['world']['calendars']}
    start, duration, day = calendar.minutes(saved['start_time']), saved['duration_minutes'], saved['date']
    previous = row['rounds'][turn - 1]['call_count'] if turn else 0
    calls = row['calls'][previous:row['rounds'][turn]['call_count']]
    checked = {(c['call']['arguments'].get('team'), c['call']['arguments'].get('date'))
               for c in calls if c['call']['name'] == 'list_busy'}
    unchecked = sorted(team for team in saved['attendees'] if (team, day) not in checked)
    if schedule.free(busy, saved['attendees'], day, start, duration):
        kind = 'later than the earliest free slot' if start > calendar.minutes(goal['start_time']) else 'earlier than expected'
    elif any(calendar.minutes(a) <= start < calendar.minutes(b) for team in saved['attendees'] for a, b in busy[team][day]):
        kind = 'starts inside a busy block'
    else:
        kind = 'free at the start but runs into a busy block'
    return kind, unchecked


def classify(case, row):
    turn = failed_turn(row)
    if turn >= len(row['rounds']):
        return turn, 'stopped before this turn', []
    round_ = row['rounds'][turn]
    if round_['failure']:
        return turn, round_['failure'], []
    _, goal = calendar.goals(case['turns'][turn]['expected'])
    saved = round_['snapshot']['meetings'].get(goal['project']) if goal else None
    if goal is None:
        return turn, 'draft turn failed', []
    if saved is None:
        return turn, 'no meeting saved for the project', []
    def differs(key):
        return sorted(saved[key]) != sorted(goal[key]) if isinstance(goal[key], list) else saved[key] != goal[key]

    wrong = [key for key in ('date', 'start_time', 'duration_minutes', 'attendees', 'source_ids') if differs(key)]
    if wrong == ['start_time']:
        kind, unchecked = wrong_start(case, row, turn, goal, saved)
        return turn, 'wrong start time: ' + kind, unchecked
    return turn, 'wrong ' + ', '.join(wrong) if wrong else 'meeting right but another write or no final text', []


def diagnose():
    result = read(ROOT / RESULT)
    cases = {case['id']: case for split in schedule.SEALED for case in schedule.cases(split)}
    rows, by_family, kinds, unchecked_turns = [], {}, Counter(), 0
    for name in ('scheduling', 'cross'):
        for row in result['reply']['episodes'][name]:
            case = cases[row['id']]
            family = by_family.setdefault(case['family'], {'episodes': 0, 'failed': 0, 'failures': Counter()})
            family['episodes'] += 1
            if row['score']['passed']:
                continue
            turn, kind, unchecked = classify(case, row)
            family['failed'] += 1
            family['failures'][f'turn {turn}: {kind}'] += 1
            kinds[kind] += 1
            unchecked_turns += bool(unchecked)
            rows.append({'id': row['id'], 'set': name, 'family': case['family'], 'turn': turn, 'kind': kind,
                         'unchecked_attendees': unchecked})
    return {'format': 'neuroshard-assistant-growth-confirmation-diagnostic/1',
            'source': {'result': RESULT, 'sha256': sha256(ROOT / RESULT), 'splits': list(schedule.SEALED)},
            'failed': {'scheduling': sum(r['set'] == 'scheduling' for r in rows), 'cross': sum(r['set'] == 'cross' for r in rows)},
            'kinds': dict(kinds.most_common()), 'failures_with_an_unchecked_attendee': unchecked_turns,
            'families': {name: {**value, 'failures': dict(value['failures'].most_common())}
                         for name, value in by_family.items()},
            'episodes': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / OUTPUT)
    args = parser.parse_args()
    report = diagnose()
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: report[key] for key in ('failed', 'kinds', 'failures_with_an_unchecked_attendee')}, indent=1))
