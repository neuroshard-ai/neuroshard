"""Authored, deterministic document-grounded tasks for the general route.

Each case is one workspace of 10-14 fictional personal and work documents and one or two user
turns. Documents carry distractors on purpose: a superseded itinerary next to the current one,
last year's policy next to this year's, two invoices from one vendor. Questions are lookups, sums
across documents, date offsets, and questions the documents cannot answer. Expected outcomes
(the facts the answer must state and the documents it must cite) never enter the workspace.

Phrasings: ``train`` (router fitting and future learning) and ``development`` share templates with new
values; the sealed ``confirmation`` uses templates disjoint from both, so it measures new wording too.
``generated-train``, ``generated-development`` and the sealed ``confirmation4`` take their wording from
the Bedrock-written sets in ``assistant_phrasings`` (three disjoint template sets); the workspace, the
facts and the expected outcome are built here exactly as for every other split.
"""

from datetime import date, timedelta
import random

from neuroshard.evolution.modular_reference_execution import identity

SPLITS = {'train': (90000, 12), 'development': (91000, 6), 'confirmation': (92000, 12),
          'confirmation2': (96000, 12), 'confirmation3': (98000, 12),
          'generated-train': (93000, 60), 'generated-development': (93500, 6), 'confirmation4': (99000, 12),
          'generated-train-extra': (93900, 0), 'generated-development2': (93600, 6),
          'generated-train-styles': (94100, 30), 'generated-development3': (94200, 6), 'confirmation5': (99700, 12),
          'generated-train-styles2': (94500, 30), 'generated-development4': (94600, 6), 'confirmation6': (99600, 12),
          'generated-train-styles3': (94700, 40), 'generated-development5': (94800, 6), 'confirmation7': (99400, 12)}
# Further training workspaces for chosen families (L7): return-flight not-found with outbound lookups
# beside it, so the extra experience contrasts the two rather than teaching not-found alone.
EXTRA = {'generated-train-extra': {'not-found-return': 60, 'flight-time': 30}}
CITIES = ('Lisbon', 'Osaka', 'Denver', 'Tallinn', 'Montreal', 'Nairobi', 'Hobart', 'Valparaiso')
VENDORS = ('Brightline Print', 'Northwind Cleaning', 'Juniper IT', 'Harbor Freight Co', 'Cobalt Catering')
PEOPLE = ('Ada Brook', 'Ravi Stone', 'Mei Lund', 'Tomas Reyes', 'Ines Varga', 'Kofi Mensah')
PRODUCTS = ('Aster X2 speaker', 'Lumen desk lamp', 'Kestrel e-bike', 'Nimbus air purifier')


def _iso(day):
    return day.isoformat()


def _world(rng, split, index):
    """A workspace and the facts the questions draw on."""
    base = date(2027, 1, 1) + timedelta(days=rng.randrange(300))
    docs, facts = [], {}

    def add(title, content):
        key = 'doc-' + identity([split, index, len(docs), title])[:8]
        docs.append({'id': key, 'title': title, 'content': content})
        return key

    city, home = rng.sample(CITIES, 2)
    old_flight, new_flight = f'NA{rng.randint(100, 999)}', f'NA{rng.randint(100, 999)}'
    depart = base + timedelta(days=rng.randint(5, 40))
    old_time, new_time = f'{rng.randint(6, 11):02d}:{rng.choice(("05", "20", "35", "50"))}', \
        f'{rng.randint(12, 21):02d}:{rng.choice(("10", "25", "40", "55"))}'
    hotel, code = rng.choice(('Hotel Mira', 'The Quayside', 'Casa Lume', 'Pinewood Inn')), \
        ''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') for _ in range(6))
    facts['trip'] = {'city': city, 'flight': new_flight, 'time': new_time, 'date': _iso(depart), 'hotel': hotel,
                     'code': code, 'nights': rng.randint(2, 6)}
    facts['trip']['old'] = add(f'Itinerary {city} (superseded)',
                               f'Superseded on {_iso(depart - timedelta(days=9))}. Flight {old_flight} from {home} '
                               f'to {city} on {_iso(depart)} departing {old_time}. Do not use this itinerary.')
    facts['trip']['doc'] = add(f'Itinerary {city}',
                               f'Current itinerary. Flight {new_flight} from {home} to {city} on {_iso(depart)}, '
                               f'departing {new_time}. Hotel: {hotel}, {facts["trip"]["nights"]} nights, '
                               f'confirmation code {code}.')
    year = depart.year
    old_days, days = rng.randint(15, 20), rng.randint(21, 30)
    limit, window = rng.choice((50, 75, 100, 150)), rng.choice((14, 21, 30, 45))
    facts['policy'] = {'days': days, 'limit': limit, 'window': window}
    add(f'Leave and expenses policy {year - 1}',
        f'Policy for {year - 1}. Annual leave: {old_days} days. Meal expenses up to {limit + 25} per day. '
        f'Claims must be filed within {window + 15} days of the expense.')
    facts['policy']['doc'] = add(f'Leave and expenses policy {year}',
                                 f'Policy for {year}, replacing the {year - 1} policy. Annual leave: {days} days. '
                                 f'Meal expenses up to {limit} per day. Claims must be filed within {window} days '
                                 'of the expense date.')
    vendor = rng.choice(VENDORS)
    first, second = rng.randint(120, 900), rng.randint(120, 900)
    months = sorted(rng.sample(range(1, 13), 2))
    facts['invoices'] = {'vendor': vendor, 'first': first, 'second': second, 'sum': first + second,
                         'months': months, 'ids': []}
    for month, amount in zip(months, (first, second)):
        number = f'INV-{rng.randint(1000, 9999)}'
        facts['invoices']['ids'].append(add(
            f'Invoice {number}', f'{vendor}. Invoice {number} for services in {date(year, month, 1):%B %Y}. '
                                 f'Amount due: {amount}. Status: unpaid.'))
    paid = add(f'Invoice INV-{rng.randint(1000, 9999)}',
               f'{vendor}. Services in {date(year - 1, 12, 1):%B %Y}. Amount: {rng.randint(120, 900)}. Status: paid.')
    facts['invoices']['paid'] = paid
    rent, notice = rng.randint(900, 2600), rng.choice((30, 60, 90))
    start = base - timedelta(days=rng.randint(30, 200))
    facts['lease'] = {'rent': rent, 'notice': notice, 'start': _iso(start)}
    facts['lease']['doc'] = add('Apartment lease',
                                f'Lease starting {_iso(start)}. Monthly rent {rent}, due on the 1st. Either party '
                                f'may end the lease with {notice} days written notice. Deposit: {2 * rent}.')
    product = rng.choice(PRODUCTS)
    battery, weight = rng.randint(8, 40), rng.randint(2, 30)
    facts['product'] = {'name': product, 'battery': battery, 'weight': weight}
    facts['product']['doc'] = add(f'{product} specification',
                                  f'{product}. Battery life: {battery} hours. Weight: {weight / 10:.1f} kg. '
                                  'Warranty: two years.')
    owner, decision_day = rng.choice(PEOPLE), base + timedelta(days=rng.randint(1, 20))
    facts['meeting'] = {'owner': owner, 'deadline': _iso(decision_day)}
    facts['meeting']['doc'] = add('Team meeting notes',
                                  f'Decisions: move the launch review to {_iso(decision_day)}; {owner} owns the '
                                  'vendor shortlist. Next meeting in two weeks.')
    add('Grocery list', 'Oats, lentils, lemons, coffee, rice, spinach.')
    add('Book notes', 'Chapter 3 argues that small, frequent reviews beat rare, large ones.')
    rng.shuffle(docs)
    return {'documents': docs}, facts


# Each question kind: templates for development, templates for confirmation, and its expected outcome.
def _kinds(facts):
    trip, policy, inv, lease, product, meeting = (facts[k] for k in ('trip', 'policy', 'invoices', 'lease',
                                                                     'product', 'meeting'))
    claim_day = date.fromisoformat(trip['date']) + timedelta(days=2)
    return {
        'flight-time': ((f'What time does my flight to {trip["city"]} leave?',
                         f'When do I depart for {trip["city"]}?'),
                        (f'Remind me of my departure time for the {trip["city"]} trip.',
                         f'flight to {trip["city"]}: what time?'),
                        {'facts': [trip['time']], 'source_ids': [trip['doc']]}),
        'hotel-code': ((f'What is the confirmation code for my hotel in {trip["city"]}?',
                        f'I need my {trip["city"]} hotel booking code.'),
                       (f'Which code do I give at check-in in {trip["city"]}?',
                        f'hotel confirmation for {trip["city"]} please'),
                       {'facts': [trip['code']], 'source_ids': [trip['doc']]}),
        'leave-days': (('How many days of annual leave do I get this year?',
                        f'What is my annual leave allowance for {date.fromisoformat(trip["date"]).year}?'),
                       ('How much vacation am I entitled to under the current policy?',
                        'current leave entitlement, in days?'),
                       {'facts': [str(policy['days'])], 'source_ids': [policy['doc']]}),
        'invoice-sum': ((f'How much do I owe {inv["vendor"]} in total across the unpaid invoices?',
                         f'Add up what is still unpaid to {inv["vendor"]}.'),
                        (f'What is my outstanding balance with {inv["vendor"]}?',
                         f'total unpaid for {inv["vendor"]}?'),
                        {'facts': [str(inv['sum'])], 'source_ids': sorted(inv['ids'])}),
        'claim-deadline': ((f'I bought a meal on {claim_day.isoformat()}. By what date must I file the claim?',
                            f'What is the last day to claim an expense dated {claim_day.isoformat()}?'),
                           (f'Deadline for an expense claim from {claim_day.isoformat()}?',
                            f'If I spent money on {claim_day.isoformat()}, when is the claim due at the latest?'),
                           {'facts': [(claim_day + timedelta(days=policy['window'])).isoformat()],
                            'source_ids': [policy['doc']]}),
        'notice': (('How many days of notice do I need to give to end my lease?',
                    'What is the notice period on my apartment lease?'),
                   ('If I want to move out, how much notice does the lease require?',
                    'lease termination notice, how long?'),
                   {'facts': [str(lease['notice'])], 'source_ids': [lease['doc']]}),
        'deposit': (('How large is the deposit on my apartment?', 'What deposit did the lease set?'),
                    ('How much is the security deposit for my flat?', 'apartment deposit amount?'),
                    {'facts': [str(2 * lease['rent'])], 'source_ids': [lease['doc']]}),
        'battery': ((f'What is the battery life of the {product["name"]}?',
                     f'How long does the {product["name"]} run on a charge?'),
                    (f'How many hours does the {product["name"]} battery last?',
                     f'{product["name"]} battery hours?'),
                    {'facts': [str(product['battery'])], 'source_ids': [product['doc']]}),
        'owner': (('Who owns the vendor shortlist?', 'Which person took the vendor shortlist in the meeting?'),
                  ('Who was assigned the vendor shortlist?', 'vendor shortlist owner?'),
                  {'facts': [meeting['owner']], 'source_ids': [meeting['doc']]}),
        'not-found-insurance': (('What is my car insurance policy number?',
                                 'Find my car insurance policy number.'),
                                ('Which number is on my vehicle insurance policy?',
                                 'car insurance policy no.?'),
                                {'not_found': True}),
        'not-found-return': ((f'When is my return flight from {trip["city"]}?',
                              f'What time do I fly back from {trip["city"]}?'),
                             (f'Remind me when I leave {trip["city"]} to come home.',
                              f'return trip from {trip["city"]}: date and time?'),
                             {'not_found': True}),
    }


def _phrasings2(facts):
    """A third phrasing per kind, for the second sealed confirmation; disjoint from every earlier split."""
    trip, inv, product = facts['trip'], facts['invoices'], facts['product']
    claim_day = date.fromisoformat(trip['date']) + timedelta(days=2)
    return {
        'flight-time': (f'Check my documents: at what hour is my outbound flight to {trip["city"]}?',
                        f'Looking at my travel docs, what is the scheduled departure time to {trip["city"]}?'),
        'hotel-code': (f'Please pull up the booking reference for where I am staying in {trip["city"]}.',
                       f'What reservation code did the {trip["city"]} hotel give me?'),
        'leave-days': ('Under this year\'s rules, how many days off am I allowed in total?',
                       'Per my current employer policy, what is the annual leave figure?'),
        'invoice-sum': (f'Sum the amounts I have not yet paid {inv["vendor"]}.',
                        f'Across all open invoices from {inv["vendor"]}, what do I owe altogether?'),
        'claim-deadline': (f'A meal expense from {claim_day.isoformat()}: what is the final date I can still claim it?',
                           f'Work out my expense claim cutoff for something I paid on {claim_day.isoformat()}.'),
        'notice': ('To terminate my rental agreement, what notice period applies?',
                   'How far in advance must I tell my landlord I am leaving?'),
        'deposit': ('What security deposit is written into my rental agreement?',
                    'Tell me the deposit figure from my apartment contract.'),
        'battery': (f'According to the spec sheet, how long does the {product["name"]} last on battery?',
                    f'What runtime per charge does the {product["name"]} have?'),
        'owner': ('From the meeting notes, who is responsible for the vendor shortlist?',
                  'Whose task is the vendor shortlist, per our last meeting?'),
        'not-found-insurance': ('Look up my auto insurance policy number in my files.',
                                'Do my documents list a policy number for my car cover?'),
        'not-found-return': (f'On which date and at what time is my flight home from {trip["city"]}?',
                             f'When does my return leg from {trip["city"]} depart?'),
    }


def _phrasings3(facts):
    """A fourth phrasing per kind, for the third sealed confirmation; disjoint from every earlier split."""
    trip, inv, product = facts['trip'], facts['invoices'], facts['product']
    claim_day = date.fromisoformat(trip['date']) + timedelta(days=2)
    return {
        'flight-time': (f'Quick one: going to {trip["city"]}, my plane takes off when?',
                        f'Find the departure time on my booking for {trip["city"]}.'),
        'hotel-code': (f'What do I quote at the front desk in {trip["city"]}?',
                       f'Get me the hotel confirmation number for the {trip["city"]} stay.'),
        'leave-days': ('How much paid leave does the policy give me now?',
                       'Check the newest policy for my number of vacation days.'),
        'invoice-sum': (f'What is the combined total of {inv["vendor"]}\'s unpaid bills?',
                        f'I want one figure: everything outstanding to {inv["vendor"]}.'),
        'claim-deadline': (f'I have a receipt from {claim_day.isoformat()}. Until when can I submit it for reimbursement?',
                           f'For a cost incurred on {claim_day.isoformat()}, find the latest submission date for the claim.'),
        'notice': ('Before I move out, how many days of warning does my landlord need?',
                   'What does my lease say about the notice period?'),
        'deposit': ('How much did I put down as a deposit on the flat?',
                    'Find the deposit amount in my lease.'),
        'battery': (f'Battery-wise, how many hours do I get from the {product["name"]}?',
                    f'Look up the {product["name"]} battery rating in hours.'),
        'owner': ('Who did we put in charge of the vendor shortlist?',
                  'Check the notes: which colleague handles the vendor shortlist?'),
        'not-found-insurance': ('What policy number is on my car insurance?',
                                'Search my files for the vehicle insurance policy ID.'),
        'not-found-return': (f'What time is my return flight out of {trip["city"]}?',
                             f'Pull up the flight I take back home from {trip["city"]}.'),
    }


KINDS = ('flight-time', 'hotel-code', 'leave-days', 'invoice-sum', 'claim-deadline', 'notice', 'deposit',
         'battery', 'owner', 'not-found-insurance', 'not-found-return')


def _fill(template, facts):
    """A generated template with this workspace's values; only placeholders the kind declares appear."""
    claim_day = date.fromisoformat(facts['trip']['date']) + timedelta(days=2)
    return template.format(city=facts['trip']['city'], vendor=facts['invoices']['vendor'], day=claim_day.isoformat(),
                           product=facts['product']['name'])


def make_case(split, kind, index):
    seed, _ = SPLITS[split]
    rng = random.Random(f'{seed}:{kind}:{index}')
    world, facts = _world(rng, split, f'{kind}-{index}')
    development, confirmation, expected = _kinds(facts)[kind]
    if split in ('generated-train', 'generated-development', 'confirmation4', 'generated-train-extra',
                 'generated-development2', 'generated-train-styles', 'generated-development3', 'confirmation5',
                 'generated-train-styles2', 'generated-development4', 'confirmation6', 'generated-train-styles3',
                 'generated-development5', 'confirmation7'):
        from neuroshard.evolution import assistant_phrasings as phrasings  # only the generated splits need it

        rows = phrasings.load('documents', split)[kind]
        index_in_rows = index + (60 if split == 'generated-train-extra' else 0)  # templates generated-train used least
        return {'id': f'docs-{split}-{kind}-{index:02d}', 'split': split, 'family': kind, 'world': world,
                'turns': [{'user': _fill(rows[index_in_rows % len(rows)], facts), 'expected': expected}]}
    templates = (_phrasings3(facts)[kind] if split == 'confirmation3' else _phrasings2(facts)[kind] if split == 'confirmation2'
                 else confirmation if split == 'confirmation' else development)
    case = {'id': f'docs-{split}-{kind}-{index:02d}', 'split': split, 'family': kind, 'world': world,
            'turns': [{'user': templates[index % len(templates)], 'expected': expected}]}
    return case


def cases(split):
    if split not in SPLITS:
        raise ValueError('unknown documents split')
    if split in EXTRA:
        return [make_case(split, kind, index) for kind, count in EXTRA[split].items() for index in range(count)]
    return [make_case(split, kind, index) for kind in KINDS for index in range(SPLITS[split][1])]


def public_case(case):
    return {'world': case['world'], 'user_turns': [turn['user'] for turn in case['turns']]}
