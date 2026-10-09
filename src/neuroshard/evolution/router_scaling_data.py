"""Fictional capabilities for a router scaling study: does per-turn routing survive many units?

The accepted assistant routes each user turn to one of two units. Growth means more
units, and every added unit is another class the router must separate from all earlier
ones. This module authors twelve further workspace capabilities as turn grammars, so the
study can add them one at a time after the two real ones and measure what happens.

The grammars are deliberately hard in the ways a growing assistant would be:

- several capabilities share vocabulary with the real ones (an invoice draft and a
  drafting request, a room booking and a meeting, a plan approval and a plan citation);
- some follow-ups are generic and shared ("make that two days later"), so the message
  alone cannot tell which unit should serve it;
- cross cases open in one capability and continue in another;
- each capability has *fit* phrasings, used to fit routers, and *unseen* phrasings,
  used only to measure how routing generalises to wording no router saw.

Everything is authored and fictional. Nothing here is a workspace a unit can execute, and
no sealed split of the real grammars is read: the real anchors use training and
integration splits only.
"""

import random

from neuroshard.evolution.modular_reference_execution import identity

PROJECTS = ('Harbor', 'Summit', 'Cedar', 'Orchard', 'Willow', 'Granite', 'Meadow', 'Beacon', 'Falcon', 'Juniper')
TEAMS = ('operations team', 'design team', 'service team', 'logistics team', 'finance team', 'review board')
PEOPLE = ('Ada Brook', 'Ravi Stone', 'Mei Lund', 'Tomas Reyes', 'Ines Varga', 'Kofi Mensah', 'Lena Ortiz')
ROOMS = ('Atlas', 'Birch', 'Comet', 'Delta', 'Ember')
CITIES = ('Northport', 'Easton', 'Lakeside', 'Westfield', 'Riverton')
ITEMS = ('pallet wrap', 'cable ties', 'label rolls', 'safety gloves', 'packing foam')
CATEGORIES = ('travel', 'equipment', 'catering', 'software', 'training')

# Shared follow-ups: the same words can continue several capabilities.
GENERIC = (
    'Move that {n} calendar days later.',
    'Actually make it {n} days earlier.',
    'Change the amount to {amount} instead.',
    'Use the {team} instead.',
    'Undo that and keep the original.',
)


def _date(rng):
    return f'2027-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}'


def _time(rng):
    return f'{rng.randint(8, 16):02d}:{rng.choice(("00", "15", "30", "45"))}'


def _slots(rng):
    return {'project': f'{rng.choice(PROJECTS)} {rng.randint(1000, 9999)}', 'team': rng.choice(TEAMS),
            'team2': rng.choice(TEAMS), 'person': rng.choice(PEOPLE), 'room': rng.choice(ROOMS),
            'city': rng.choice(CITIES), 'city2': rng.choice(CITIES), 'item': rng.choice(ITEMS),
            'category': rng.choice(CATEGORIES), 'date': _date(rng), 'time': _time(rng), 'time2': _time(rng),
            'n': rng.randint(1, 9), 'amount': rng.randint(20, 900), 'hours': rng.choice((1.5, 2, 3.5, 4, 6)),
            'minutes': rng.choice((30, 45, 60, 90)), 'rev': rng.randint(2, 6), 'quarter': rng.randint(1, 4),
            'phone': f'555-{rng.randint(1000, 9999)}', 'level': rng.choice(('low', 'normal', 'high', 'urgent'))}


# Each capability: (fit openings, unseen openings, own follow-ups). Generic follow-ups are shared.
CAPABILITIES = {
    'expenses': (
        ('Record an expense of {amount} credits for {project} under {category}, dated {date}.',
         'Log a {category} expense for {project}: {amount} credits on {date}. Keep it as a pending claim.'),
        ('I spent {amount} credits on {category} for {project} on {date}, can you file that?',
         'expense claim pls, {project}, {category}, {amount} credits, {date}'),
        ('Attach the receipt reference R-{amount} to that expense.', 'Mark that claim as reimbursable.'),
    ),
    'inventory': (
        ('Check the stock of {item} for {project} and reorder if fewer than {amount} remain.',
         'Look up how many {item} the {project} warehouse holds and flag a reorder below {amount}.'),
        ('Are we running low on {item} at {project}? Order more if it is under {amount}.',
         'stock check: {item}, {project}, threshold {amount}'),
        ('Reserve {n} of those for the {team}.', 'Raise the reorder threshold to {amount}.'),
    ),
    'invoices': (
        ('Prepare an invoice draft for {project} addressed to the {team}, totalling {amount} credits. Do not send it.',
         'Create an invoice for {project} to the {team} for {amount} credits, due {date}. Save it as a draft only.'),
        ('Could you bill the {team} {amount} credits for {project}? Leave it unsent.',
         'invoice draft -> {team}, {project}, {amount} credits, due {date}'),
        ('Add a line item of {n} hours of support to that invoice.', 'Set the payment terms to {n} weeks.'),
    ),
    'reminders': (
        ('Set a reminder for {person} on {date} at {time} to review the {project} plan.',
         'Remind me on {date} at {time} to call the {team} about {project}.'),
        ('Can you ping {person} at {time} on {date} about {project}?',
         "don't let me forget {project} on {date}, {time}"),
        ('Repeat that reminder every week.', 'Snooze it by {minutes} minutes.'),
    ),
    'tickets': (
        ('Open a support ticket for {project} with {level} priority: the export to the {team} failed on {date}.',
         'File a {level} priority issue for {project}; the {team} cannot open the shared folder.'),
        ("The {team}'s dashboard for {project} is broken again, please raise it as {level}.",
         'bug: {project} sync failing since {date}, prio {level}'),
        ('Assign that ticket to {person}.', 'Escalate it to urgent and notify the {team}.'),
    ),
    'summaries': (
        ('Summarize the {project} status report into three bullet points for the {team}.',
         'Write a short summary of the latest {project} review notes, at most five sentences.'),
        ('What are the main points of the {project} report? Keep it brief.',
         'tl;dr of the {project} notes for the {team} please'),
        ('Make that summary shorter, two bullets only.', 'Add the open risks at the end.'),
    ),
    'contacts': (
        ('Update the phone number of {person} to {phone} in the {project} contact list.',
         'Add {person} from the {team} to the contacts for {project}.'),
        ("{person}'s number changed to {phone}, fix it in the {project} directory.",
         'contact update: {person}, {phone}, {project}'),
        ('Also set their email to the {team} alias.', 'Remove their old number entirely.'),
    ),
    'travel': (
        ('Book a train for {person} from {city} to {city2} on {date}, departing after {time}.',
         'Find a flight for {person} from {city} to {city2} on {date} and hold the cheapest seat.'),
        ('{person} needs to get from {city} to {city2} on {date}, sometime after {time}. Sort out tickets?',
         'trip: {person}, {city} -> {city2}, {date}'),
        ('Add a hotel for {n} nights near the station.', 'Switch to a return ticket.'),
    ),
    'timesheets': (
        ('Log {hours} hours on {project} for {person} on {date}.',
         'Record {hours} hours of {category} work by {person} on {project}, dated {date}.'),
        ('{person} worked {hours} hours on {project} on {date}, please put that in the timesheet.',
         'timesheet: {person} / {project} / {hours}h / {date}'),
        ('Mark those hours as billable.', 'Split them evenly across two days.'),
    ),
    'approvals': (
        ('Approve revision {rev} of the delivery plan for {project} and notify the {team}.',
         'Reject revision {rev} of the {project} delivery plan; the quantities do not match.'),
        ('Revision {rev} of the {project} plan looks fine to me, can you sign it off?',
         'approve {project} plan rev {rev}'),
        ('Add a note that the {team} must confirm the quantities.', 'Withdraw that approval.'),
    ),
    'search': (
        ('Find all documents for {project} that mention the {team}.',
         'Search the {project} folder for files changed after {date}.'),
        ('Where are the {project} files that talk about {item}?',
         'look for {project} docs about {category}'),
        ('Only show the ones from the last {n} weeks.', 'Sort those by most recently edited.'),
    ),
    'rooms': (
        ('Reserve room {room} for the {team} on {date} from {time} to {time2}.',
         'Book the {room} room on {date} at {time} for {minutes} minutes for {project}.'),
        ('Is room {room} free on {date} at {time}? Grab it for the {team} if so.',
         'room {room}, {date}, {time}-{time2}, {project}'),
        ('Add a projector to that booking.', 'Change it to room {room} instead.'),
    ),
}

REAL = ('drafting', 'scheduling')
SYNTHETIC = tuple(CAPABILITIES)

# Unseen phrasings of the two real capabilities, authored for this study only: the router
# must still send earlier cohorts' requests home when users word them differently. Labels
# hold by construction: a draft request needs drafting, a meeting request needs scheduling.
REAL_UNSEEN = {
    'drafting': (
        ('Could you put together a draft for {project} to the {team}? Go by the newest approved delivery plan.',
         'draft for {project} -> {team}, latest approved plan, due on the plan start, cite sources, do not send',
         'I need the {project} handoff written up for the {team} from the approved plan; keep it unsent.'),
        ('Push that due date back {n} days.', 'Same draft, but address it to the {team}.'),
    ),
    'scheduling': (
        ('When can the {team} and the {team2} both meet for {minutes} minutes on {date}? Book the first slot.',
         'mtg: {project}, {team} + {team2}, {date}, {minutes} min, earliest common slot, draft only',
         'Find a {minutes}-minute window on {date} that suits the {team} and the {team2} for {project} and hold it.'),
        ('Bring the {team} into that meeting as well.', 'Make it {minutes} minutes instead, same day.'),
    ),
}
# The order in which the study adds capabilities after the two real ones; confusable ones are spread out.
ORDER = REAL + ('invoices', 'tickets', 'rooms', 'expenses', 'approvals', 'reminders', 'summaries', 'travel',
                'inventory', 'timesheets', 'contacts', 'search')

# What each unit would declare about itself when it joins: a short card, written without
# reusing any request template. The prototype router is fitted on these alone.
DESCRIPTIONS = {
    'drafting': ('Writes and saves unsent document drafts for a project from its approved delivery plans.',
                 'Composes a handoff draft to a team, citing the plan revision it used.',
                 'Edits a saved draft: due dates, totals, recipients.'),
    'scheduling': ('Books meetings at times when every attendee team is free.',
                   'Finds a common free slot in team calendars and saves a meeting draft.',
                   'Changes a booked meeting: attendees, length, earliest start.'),
    'invoices': ('Bills a client or team: creates unsent invoices with amounts and due dates.',
                 'Adds line items and payment terms to an invoice.',
                 'Prepares billing documents for project work.'),
    'tickets': ('Opens and tracks support issues when something is broken.',
                'Reports a failure with a priority and assigns it to a person.',
                'Escalates bugs and outages to the responsible team.'),
    'rooms': ('Reserves physical meeting rooms and equipment for a time.',
              'Books a named room on a date for a team.',
              'Checks whether a room is available and holds it.'),
    'expenses': ('Files spending claims with a category, amount and date.',
                 'Records money a person spent for reimbursement.',
                 'Attaches receipts and marks claims reimbursable.'),
    'approvals': ('Signs off or rejects a revision of a plan.',
                  'Records an approval decision and notifies the team.',
                  'Withdraws or annotates an earlier sign-off.'),
    'reminders': ('Sets personal reminders and notifications for a given time.',
                  'Pings a person later so they do not forget something.',
                  'Repeats or snoozes a reminder.'),
    'summaries': ('Condenses reports and notes into short bullet points.',
                  'Gives the main points of a long document briefly.',
                  'Shortens or extends an existing summary.'),
    'travel': ('Arranges trips: trains, flights and hotels between cities.',
               'Buys or holds tickets for a traveller on a date.',
               'Adds accommodation or return legs to a trip.'),
    'inventory': ('Tracks warehouse stock levels and reorders supplies.',
                  'Counts items on hand and flags low stock.',
                  'Reserves stock for a team or changes reorder thresholds.'),
    'timesheets': ('Logs hours a person worked on a project.',
                   'Records time entries by day and marks them billable.',
                   'Corrects or splits recorded work hours.'),
    'contacts': ('Maintains the directory of people: phone numbers and emails.',
                 'Adds or updates a person in a contact list.',
                 'Removes outdated contact details.'),
    'search': ('Finds files and documents in a folder by topic or date.',
               'Looks up which documents mention something.',
               'Filters and sorts search results.'),
}
SPLITS = {'fit': (61000, 48), 'test': (62000, 24), 'unseen': (63000, 24)}


def _turns(rng, capability, phrasing):
    fit, unseen, own = CAPABILITIES[capability]
    slots = _slots(rng)
    pool = fit if phrasing == 'fit' else unseen
    template = rng.randrange(len(pool))
    turns = [(capability, pool[template].format(**slots))]
    for _ in range(rng.choice((0, 1, 1, 2))):
        follow = own if rng.random() < 0.6 else GENERIC
        turns.append((capability, rng.choice(follow).format(**_slots(rng))))
    return turns, template


def make_case(split, capability, index):
    """One fictional conversation: its user turns, the capability each turn needs, and its opening template."""
    seed, _ = SPLITS[split]
    rng = random.Random(f'{seed}:{capability}:{index}')
    turns, template = _turns(rng, capability, 'unseen' if split == 'unseen' else 'fit')
    case = {'id': f'{split}-{capability}-{index:03d}', 'split': split, 'capability': capability,
            'user_turns': [text for _, text in turns], 'labels': [label for label, _ in turns], 'template': template}
    case['sha256'] = identity(case)
    return case


def make_cross(split, first, second, index):
    """A conversation that opens in ``first`` and continues with an opening request of ``second``."""
    seed, _ = SPLITS[split]
    rng = random.Random(f'{seed}:cross:{first}:{second}:{index}')
    phrasing = 'unseen' if split == 'unseen' else 'fit'
    opening, template = _turns(rng, first, phrasing)
    follow, _ = _turns(rng, second, phrasing)
    turns = opening[:1] + [(second, 'Next, ' + follow[0][1][0].lower() + follow[0][1][1:])]
    case = {'id': f'{split}-cross-{first}-{second}-{index:03d}', 'split': split, 'capability': f'{first}+{second}',
            'user_turns': [text for _, text in turns], 'labels': [label for label, _ in turns], 'template': template}
    case['sha256'] = identity(case)
    return case


def cases(split, capabilities=SYNTHETIC, cross_per_pair=0):
    """Every case of a split for the given synthetic capabilities, in a fixed order."""
    if split not in SPLITS:
        raise ValueError(f'unknown router scaling split: {split}')
    unknown = set(capabilities) - set(CAPABILITIES)
    if unknown:
        raise ValueError(f'unknown synthetic capabilities: {sorted(unknown)}')
    count = SPLITS[split][1]
    result = [make_case(split, capability, i) for capability in capabilities for i in range(count)]
    for first in capabilities:
        for second in capabilities:
            if first != second:
                result.extend(make_cross(split, first, second, i) for i in range(cross_per_pair))
    return result


def real_unseen_cases(count=24):
    """Authored unseen-phrasing conversations of drafting and scheduling, for the ``unseen`` set only."""
    seed, _ = SPLITS['unseen']
    result = []
    for capability, (openings, follow_ups) in REAL_UNSEEN.items():
        for index in range(count):
            rng = random.Random(f'{seed}:real:{capability}:{index}')
            turns = [rng.choice(openings).format(**_slots(rng))]
            turns += [rng.choice(follow_ups).format(**_slots(rng)) for _ in range(rng.choice((0, 1, 1)))]
            case = {'id': f'unseen-real-{capability}-{index:03d}', 'split': 'unseen', 'capability': capability,
                    'user_turns': turns, 'labels': [capability] * len(turns)}
            case['sha256'] = identity(case)
            result.append(case)
    return result


def real_cases(split, limit=None, seed=64000):
    """Drafting and scheduling turns from the real grammars' training or integration splits, never a sealed one.

    A drafting case's turns all need drafting. A scheduling or cross case's turn needs scheduling
    when its expected outcome includes a meeting, the rule the accepted router's check uses.
    """
    from neuroshard.evolution import assistant_calendar as calendar
    from neuroshard.evolution import assistant_schedule_data as schedule
    from neuroshard.evolution import assistant_workflow_data as drafting
    from neuroshard.evolution.assistant_workflow_data import public_case

    sources = {'fit': (('train', drafting), ('train', schedule), ('cross-train', schedule)),
               'test': (('integration', drafting), ('integration', schedule), ('cross-integration', schedule))}
    if split not in sources:
        raise ValueError('real anchors exist only for the fit and test splits')
    result = []
    for name, module in sources[split]:
        if name in getattr(module, 'SEALED', ()) or 'confirmation' in name or 'development' in name:
            raise ValueError('the router scaling study never reads sealed or development splits')
        chosen = module.cases(name)
        if limit is not None:
            chosen = random.Random(f'{seed}:{name}:{module.__name__}').sample(chosen, min(limit, len(chosen)))
        for case in chosen:
            public = public_case(case)
            if module is drafting:
                labels = ['drafting'] * len(public['user_turns'])
            else:
                labels = ['scheduling' if case.get('capability') and calendar.goals(spec['expected'])[1] is not None
                          else 'drafting' for spec in case['turns']]
            kind = 'drafting' if module is drafting else ('cross' if name.startswith('cross') else 'scheduling')
            result.append({'id': f'real-{name}-{case["id"]}', 'split': split, 'capability': kind,
                           'user_turns': list(public['user_turns']), 'labels': labels})
    return result
