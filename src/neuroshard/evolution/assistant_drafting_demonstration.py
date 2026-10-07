"""Correct solutions of training drafting cases, written as the accepted version's own replies.

Every reply is one or two tool calls in the native envelope the accepted version emits, or
a one-sentence confirmation. A turn that needs a plan lists the project's documents and
reads its latest approved revision; it also reads approved revision 1 when the user asks
for the change from it, and reads revision 1 instead when the user switches to it. Calls
that do not depend on each other share a reply: two reads, or a date shift and a
calculation. The expected draft is saved once per turn. Only training cases are solved,
and a demonstration becomes experience only after it passes the scorer within the
policy's limits.
"""

from datetime import date, timedelta
import json
import re

from neuroshard.evolution import assistant_workflow_data as data

PLAN = re.compile(r'starts on (\d{4}-\d{2}-\d{2})\. The baseline quantity is (\d+) units\. '
                  r'The review interval is (\d+) calendar days\.')
ADDITION = re.compile(r'[Aa]dd (\d+) units to the baseline quantity|the baseline quantity plus (\d+) units'
                      r'|a total (\d+) units above the baseline quantity')
EXTRA_DAYS = re.compile(r'Also push the due date (\d+) calendar days later\.')
EXTRA_UNITS = re.compile(r'Also add (\d+) more units to the total\.')
OFFSET = re.compile(r'Move that due date (\d+) calendar days later\.')
DOUBLE = 'Double the baseline quantity'
# Families whose first turn sets the due date one review interval after the start.
INTERVAL = ('date', 'latest', 'reschedule', 'scope')


def call(name, arguments):
    return '<tool_call>\n' + json.dumps({'name': name, 'arguments': arguments}) + '\n</tool_call>'


def plan(document):
    start, quantity, interval = PLAN.search(document['content']).groups()
    return start, int(quantity), int(interval)


def approved(documents, project, revision=None):
    """The latest approved revision of a project's plan, or the approved revision named."""
    rows = [d for d in documents if d['project'] == project and d['status'] == 'approved'
            and (revision is None or d['revision'] == revision)]
    return max(rows, key=lambda d: d['revision'])


def shift(day, days):
    return (date.fromisoformat(day) + timedelta(days=days)).isoformat()


def computed(start, days, total, operations):
    """The calls that take a start date and a quantity to the due date and total, two per reply when independent.

    ``days`` are successive shifts of the date; ``operations`` successive (operation, right)
    calculations on the total. The first of each shares a reply, then the second of each.
    """
    replies, due = [], start
    for level in range(max(len(days), len(operations))):
        calls = []
        if level < len(days):
            calls.append(call('shift_date', {'start_date': due, 'days': days[level]}))
            due = shift(due, days[level])
        if level < len(operations):
            operation, right = operations[level]
            calls.append(call('calculate', {'operation': operation, 'left': total, 'right': right}))
            total = {'add': total + right, 'subtract': total - right, 'multiply': total * right}[operation]
        replies.append('\n'.join(calls))
    return replies, due, total


def addition(user):
    match = ADDITION.search(user)
    return int(next(group for group in match.groups() if group))


def extras(user):
    days, units = EXTRA_DAYS.search(user), EXTRA_UNITS.search(user)
    return [int(days.group(1))] if days else [], [('add', int(units.group(1)))] if units else []


def texts(case):
    """The demonstration's replies, in order, for every turn of one training drafting case."""
    if case['split'] not in data.TRAINING or case.get('capability'):
        raise ValueError('demonstrations may only solve training drafting cases')
    documents = case['world']['documents']
    family = case['family']
    outputs, state = [], {}
    for index, turn in enumerate(case['turns']):
        expected, user = turn['expected'], turn['user']
        project = expected['project']
        replies = []
        if index == 0:
            latest = approved(documents, project)
            start, quantity, interval = plan(latest)
            reads = [latest] + ([approved(documents, project, 1)] if family == 'difference' else [])
            replies += [call('list_documents', {'project': project}),
                        '\n'.join(call('read_document', {'document_id': d['id']}) for d in reads)]
            if family == 'difference':
                operations = [('subtract', plan(reads[1])[1])]
            elif family == 'sum' and DOUBLE in user:
                operations = [('multiply', 2)]
            elif family in ('copy', 'date'):
                operations = []
            else:
                state['addition'] = addition(user)
                operations = [('add', state['addition'])]
            more_days, more_units = extras(user)
            steps, due, total = computed(start, ([interval] if family in INTERVAL else []) + more_days, quantity,
                                         operations + more_units)
            state['interval'] = interval
        elif family == 'recipient':
            steps, due, total = [], state['due'], state['total']
        elif family == 'reschedule':
            steps, due, total = computed(state['due'], [int(OFFSET.search(user).group(1))], state['total'], [])
        else:
            if family == 'scope':
                source = approved(documents, project + ' Annex')
                replies.append(call('list_documents', {'project': source['project']}))
            else:
                source = approved(documents, project, 1)
            replies.append(call('read_document', {'document_id': source['id']}))
            start, quantity, _ = plan(source)
            _, more_units = extras(user)
            steps, due, total = computed(start, [state['interval']], quantity,
                                         [('add', state['addition'])] + more_units)
        if (due, total) != (expected['due_date'], expected['total']):
            raise ValueError(f"the solver does not reach {case['id']} turn {index}")
        replies += steps + [call('save_draft', expected),
                            f"Draft saved for {project}, addressed to the {expected['recipient']}, due "
                            f"{expected['due_date']}, with a total of {expected['total']}."]
        outputs += replies
        state.update(due=due, total=total)
    return outputs
