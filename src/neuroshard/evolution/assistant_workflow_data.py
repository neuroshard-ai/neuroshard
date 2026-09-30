"""Public fictional workspaces, with evaluation goals kept out of tool execution.

The grammar defines executable office workflows, not an external benchmark.
Splits use disjoint projects and document IDs, with separately sampled values. Training and
integration demonstrations are explicit; the model runner never loads them.
"""

from datetime import date, timedelta
import random

from neuroshard.evolution.modular_reference_execution import identity

FAMILIES = ('copy', 'date', 'sum', 'difference', 'latest', 'recipient', 'reschedule', 'scope')
SPLITS = {'train': (4100, 32), 'integration': (5200, 8),
          'development': (6300, None), 'confirmation': (7400, 12), 'confirmation2': (8500, 24),
          'confirmation3': (9600, 24), 'train2': (10700, 32), 'train3': (11800, 32),
          'compose1': (12900, 40), 'compose2': (14000, 40)}
# A fresh confirmation reuses the confirmation correction grammar with new values.
CONFIRMATIONS = ('confirmation', 'confirmation2', 'confirmation3')
# Further training cases for growing verified experience; the training grammar with new values.
GROWTH = ('train2', 'train3')
# Compositional practice: training cases whose instructions carry one extra operation. Development and
# confirmation hold out (family, extra-operation kind) pairs, so only pairs neither holds out appear here:
# the single-turn families take any kind in their first turn; latest and scope take a total change in
# their correction; recipient and reschedule are left out, since every kind is held out for them.
COMPOSE = ('compose1', 'compose2')
COMPOSE_FAMILIES = FAMILIES[:4] + ('latest', 'scope')
TRAINING = ('train', *GROWTH, *COMPOSE)


def make_case(split, family, index):
    if split not in SPLITS or family not in FAMILIES or type(index) is not int or index < 0:
        raise ValueError('unknown workflow split or family')
    number = SPLITS[split][0] + FAMILIES.index(family) * 40 + index
    rng = random.Random(number)
    project = f"{rng.choice(['Cedar', 'Harbor', 'Meadow', 'Orchard', 'Summit', 'Willow'])} {number}"
    sibling = project + ' Annex'
    recipient = rng.choice(['operations team', 'design team', 'service team', 'logistics team'])
    changed_recipient = 'review board'
    start = date(2027, 1, 1) + timedelta(days=rng.randrange(160))
    quantity, addition, duration = rng.randrange(25, 80), rng.randrange(5, 16), rng.randrange(3, 10)
    old_quantity, old_start = quantity - rng.randrange(3, 9), start - timedelta(days=7)
    documents = []

    def document(owner, revision, status, day, count):
        identifier = 'doc-' + identity([owner, revision, status, split])[:10]
        value = {'id': identifier, 'project': owner, 'title': owner + ' delivery plan',
                 'revision': revision, 'status': status,
                 'content': f'The plan for {owner} starts on {day.isoformat()}. '
                            f'The baseline quantity is {count} units. '
                            f'The review interval is {duration} calendar days. '
                            'Use the status and revision in the document metadata to choose a version.'}
        documents.append(value)
        return value

    old = document(project, 1, 'approved', old_start, old_quantity)
    current = document(project, 2, 'approved', start, quantity)
    document(project, 3, 'draft', start + timedelta(days=12), quantity + 25)
    other = document(sibling, 2, 'approved', start + timedelta(days=2), quantity + 9)
    rng.shuffle(documents)
    due = start.isoformat()
    total = quantity
    if family in ('date', 'latest', 'reschedule', 'scope'):
        due = (start + timedelta(days=duration)).isoformat()
    if family in ('sum', 'latest', 'recipient', 'reschedule', 'scope'):
        total = quantity + addition
    multiply = family == 'sum' and index % 2 == 1
    if multiply:
        total = quantity * 2
    if family == 'difference':
        total = quantity - old_quantity
    sources = [current['id']] + ([old['id']] if family == 'difference' else [])
    first = {'project': project, 'recipient': recipient, 'due_date': due,
             'total': total, 'source_ids': sorted(sources)}
    base = f'Create a draft for {project}, addressed to the {recipient}. Use the latest approved delivery plan, not an unapproved revision. '
    instructions = {
        'copy': 'Use the plan start date as the due date and its baseline quantity as the total.',
        'date': 'The due date is one review interval after the plan start date. Keep the baseline quantity as the total.',
        'sum': ('Use the plan start date as the due date. Double the baseline quantity for the total.' if multiply
                else f'Use the plan start date as the due date. Add {addition} units to the baseline quantity for the total.'),
        'difference': 'Use the latest approved start date as the due date. The total should be the increase in quantity from approved revision 1 to the latest approved revision.',
        'latest': f'Set the due date one review interval after the start, and the total to the baseline quantity plus {addition} units.',
        'recipient': f'Keep the start date as the due date and add {addition} units to the baseline quantity.',
        'reschedule': f'The due date is one review interval after the start; the total is the baseline quantity plus {addition} units.',
        'scope': f'Use a due date one review interval after the start and a total {addition} units above the baseline quantity.',
    }
    extra = ''
    amount = 3 + (index // 3) % 3
    if split in COMPOSE and family in FAMILIES[:4]:
        kind = ('date', 'total', 'recipient')[index % 3]
        if kind == 'date':
            extra = f' Also push the due date {amount} calendar days later.'
            first['due_date'] = (date.fromisoformat(first['due_date']) + timedelta(days=amount)).isoformat()
        elif kind == 'total':
            extra = f' Also add {amount} more units to the total.'
            first['total'] += amount
        else:
            extra = f' Also address it to the finance team instead of the {recipient}.'
            first['recipient'] = 'finance team'
    turns = [{'user': base + instructions[family] + extra
              + ' Save a draft only, cite the documents used, and do not send anything.', 'expected': first}]
    if family in ('latest', 'recipient', 'reschedule', 'scope'):
        second = dict(first)
        if family == 'latest':
            correction = 'Actually use approved revision 1 for this draft instead. Keep the same recipient, review-interval rule, and extra quantity.'
            second.update(due_date=(old_start + timedelta(days=duration)).isoformat(),
                          total=old_quantity + addition, source_ids=[old['id']])
        elif family == 'recipient':
            correction = f'Change the recipient to the {changed_recipient}. Keep the date, total and cited source unchanged.'
            second['recipient'] = changed_recipient
        elif family == 'reschedule':
            offset = 2 + index % 4
            correction = f'Move that due date {offset} calendar days later. Keep the recipient, total and cited source unchanged.'
            second['due_date'] = (date.fromisoformat(first['due_date']) + timedelta(days=offset)).isoformat()
        else:
            correction = f'Keep this draft under {project} and keep its recipient, but use {sibling}\'s latest approved plan for the date and quantity instead. Apply the same review-interval and extra-unit rules.'
            second.update(due_date=(start + timedelta(days=2 + duration)).isoformat(),
                          total=quantity + 9 + addition, source_ids=[other['id']])
        if split in COMPOSE:
            correction += f' Also add {amount} more units to the total.'
            second['total'] += amount
        # Development and confirmation hold out different conjunctions of learned
        # operations. They are not merely new names under identical corrections.
        if split == 'development' or split in CONFIRMATIONS:
            correction = correction.replace('Keep the date, total and cited source unchanged.',
                                             'Keep the cited source unchanged.')
            correction = correction.replace('Keep the recipient, total and cited source unchanged.',
                                             'Keep the cited source unchanged.')
        if split in CONFIRMATIONS:
            correction = correction.replace('Keep the same recipient, review-interval rule, and extra quantity.',
                                             'Keep the same review-interval rule and extra quantity.')
            correction = correction.replace('and keep its recipient, but use', 'but use')
        if split == 'development':
            if family in ('latest', 'scope'):
                correction += ' Also move the resulting due date one calendar day later.'
                second['due_date'] = (date.fromisoformat(second['due_date']) + timedelta(days=1)).isoformat()
            elif family == 'recipient':
                correction += ' Also increase the total by two units.'
                second['total'] += 2
            else:
                correction += f' Also address it to the {changed_recipient}.'
                second['recipient'] = changed_recipient
        elif split in CONFIRMATIONS:
            if family in ('latest', 'scope'):
                correction += f' Also change the recipient to the {changed_recipient}.'
                second['recipient'] = changed_recipient
            elif family == 'recipient':
                correction += ' Also move the existing due date two calendar days later.'
                second['due_date'] = (date.fromisoformat(second['due_date']) + timedelta(days=2)).isoformat()
            else:
                correction += ' Also double the current total.'
                second['total'] *= 2
        turns.append({'user': correction, 'expected': second})
    return {'id': 'workflow-' + identity([split, family, index])[:12], 'split': split,
            'family': family, 'block': family, 'primitive': family in FAMILIES[:4],
            'world': {'documents': documents}, 'turns': turns}


def cases(split):
    per_family = SPLITS[split][1]
    result = []
    for family in (COMPOSE_FAMILIES if split in COMPOSE else FAMILIES):
        count = per_family if per_family else (2 if family in FAMILIES[:4] else 4)
        result.extend(make_case(split, family, i) for i in range(count))
    random.Random(SPLITS[split][0]).shuffle(result)
    return result


def public_case(case):
    """No ID, split, family, target or reference trajectory enters an agent request."""
    return {'world': case['world'], 'user_turns': [turn['user'] for turn in case['turns']]}
