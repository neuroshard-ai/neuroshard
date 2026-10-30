"""Development tool-use items. ``fixture/1`` items share one fixture (with deliberate distractors: two
customers named Jordan Lee*, two contacts named Priya, a shipped order that cannot be refunded, an
out-of-stock product, a calendar clash). ``documents/2`` items use the served documents route's tools over
a small set of personal documents, including a superseded insurance policy and no return flight.
"""

FIXTURE = {
    'customers': [{'id': 'C-101', 'name': 'Jordan Lee', 'email': 'jordan.lee@example.com'},
                  {'id': 'C-102', 'name': 'Maria Gomez', 'email': 'maria.g@example.com'},
                  {'id': 'C-103', 'name': 'Jordan Leeds', 'email': 'jleeds@example.com'}],
    'orders': [{'id': 'O-5001', 'customer_id': 'C-101', 'date': '2026-10-02', 'status': 'delivered',
                'total_cents': 4599, 'items': ['desk lamp']},
               {'id': 'O-5002', 'customer_id': 'C-101', 'date': '2026-10-20', 'status': 'shipped',
                'total_cents': 2599, 'items': ['phone case']},
               {'id': 'O-5003', 'customer_id': 'C-102', 'date': '2026-09-15', 'status': 'delivered',
                'total_cents': 12000, 'refunded_cents': 2000, 'items': ['blender']},
               {'id': 'O-5004', 'customer_id': 'C-103', 'date': '2026-10-11', 'status': 'delivered',
                'total_cents': 899, 'items': ['usb cable']}],
    'rates': {'EUR': {'USD': 1.08, 'GBP': 0.86}, 'USD': {'JPY': 150.0}},
    'weather': [{'city': 'Lisbon', 'date': '2026-11-02', 'high_c': 19, 'low_c': 13, 'conditions': 'light rain'},
                {'city': 'Lisbon', 'date': '2026-11-03', 'high_c': 21, 'low_c': 14, 'conditions': 'sunny'},
                {'city': 'Oslo', 'date': '2026-11-02', 'high_c': 4, 'low_c': -1, 'conditions': 'snow'},
                {'city': 'Oslo', 'date': '2026-11-03', 'high_c': 3, 'low_c': -2, 'conditions': 'cloudy'}],
    'contacts': [{'id': 'P-1', 'name': 'Priya Shah'}, {'id': 'P-2', 'name': 'Priya Nair'},
                 {'id': 'P-3', 'name': 'Tom Becker'}],
    'events': [{'id': 'E-1', 'title': 'Standup', 'date': '2026-11-04', 'start': '09:00', 'end': '09:30'},
               {'id': 'E-2', 'title': 'Dentist', 'date': '2026-11-04', 'start': '11:00', 'end': '12:00'},
               {'id': 'E-3', 'title': 'Team lunch', 'date': '2026-11-04', 'start': '12:30', 'end': '13:30'}],
    'products': [{'sku': 'SKU-1', 'name': 'Trail running shoes', 'price_cents': 8900, 'stock': 4, 'tags': 'shoes'},
                 {'sku': 'SKU-2', 'name': 'Road running shoes', 'price_cents': 7500, 'stock': 0, 'tags': 'shoes'},
                 {'sku': 'SKU-3', 'name': 'Hiking boots', 'price_cents': 12900, 'stock': 2, 'tags': 'shoes'},
                 {'sku': 'SKU-4', 'name': 'Running socks (3 pack)', 'price_cents': 1500, 'stock': 30, 'tags': 'socks'}],
}

DOCUMENTS = {'documents': [
    {'id': 'doc-1a2b3c4d', 'title': 'Gym membership', 'content': 'FitLab membership number FL-88213. Monthly fee 39 '
     'euros, billed on the 5th of each month. Cancellation requires 30 days notice from the day you ask.'},
    {'id': 'doc-2b3c4d5e', 'title': 'Car insurance 2026', 'content': 'Policy CI-55120 with Northwind Insurance, '
     'current. Renewal date 2027-01-14. Excess 250 euros per claim.'},
    {'id': 'doc-3c4d5e6f', 'title': 'Car insurance 2025 (expired)', 'content': 'Policy CI-41007 with Northwind '
     'Insurance, replaced by CI-55120. Renewal date 2026-01-14. Excess 400 euros per claim.'},
    {'id': 'doc-4d5e6f70', 'title': 'Electricity bill October', 'content': 'Account EL-3009. Amount due 84.20 euros '
     'by 2026-11-12.'},
    {'id': 'doc-5e6f7081', 'title': 'Water bill October', 'content': 'Account WA-771. Amount due 31.55 euros by '
     '2026-11-20.'},
    {'id': 'doc-6f708192', 'title': 'Flight to Porto', 'content': 'Booking QX7R2. Outbound flight TP 1947 on '
     '2026-12-18, departs Lisbon 07:25, arrives Porto 08:20. One way.'},
]}


def fixture_item(expect):
    return {'kind': 'tools', 'toolset': 'fixture/1', 'expect': expect}


def documents_item(expect):
    return {'kind': 'tools', 'toolset': 'documents/2', 'expect': expect}


CITE = ' Cite the document ID you relied on.'
TOOL_USE = [
    ('lookup', 'What is the status of order O-5002?',
     fixture_item({'required_calls': ['get_order'], 'final_contains': ['shipped']})),
    ('state-change', 'Please refund Jordan Lee\'s desk lamp order in full; it arrived broken. His email is '
                     'jordan.lee@example.com.',
     fixture_item({'required_calls': ['issue_refund'], 'refunds': [{'order_id': 'O-5001', 'amount_cents': 4599}]})),
    ('refusal', 'Refund the phone case that Jordan Lee (jordan.lee@example.com) ordered.',
     fixture_item({'refunds': [], 'final_contains': [['shipped', 'not been delivered', 'not delivered',
                                                       'not yet delivered', 'hasn t been delivered']]})),
    ('conversion', 'How much is 250 euros in US dollars today?',
     fixture_item({'required_calls': ['convert_currency'], 'final_contains': [['270', '270.00']]})),
    ('conversion', 'Convert 40 British pounds to euros at today\'s rate.',
     fixture_item({'required_calls': ['convert_currency'], 'final_contains': ['46.51']})),
    ('lookup', 'Will I need an umbrella in Lisbon on 2 November 2026?',
     fixture_item({'required_calls': ['get_weather'], 'final_contains': ['rain']})),
    ('multi-step', 'On 3 November 2026, which city has the higher daily high, Lisbon or Oslo, and by how many '
                   'degrees Celsius?', fixture_item({'final_contains': ['Lisbon', '18']})),
    ('state-change', 'Text Tom that I am running 10 minutes late.',
     fixture_item({'messages': [{'contact_id': 'P-3', 'contains': ['10', 'late']}]})),
    ('ambiguity', 'Send Priya a message saying the report is ready.',
     fixture_item({'messages': [], 'final_contains': ['Shah', 'Nair']})),
    ('conflict', 'Book a 30-minute call with the bank on 4 November 2026 at 11:30.',
     fixture_item({'events_created': [], 'final_contains': [['dentist', 'conflict', 'clash', 'overlap', 'busy']]})),
    ('state-change', 'Add "Gym" to my calendar on 2026-11-04 from 14:00 to 15:00.',
     fixture_item({'events_created': [{'title': 'Gym', 'date': '2026-11-04', 'start': '14:00', 'end': '15:00'}]})),
    ('search', 'Find me running shoes that are actually in stock and tell me the price in dollars.',
     fixture_item({'required_calls': ['search_products'], 'final_contains': ['Trail', '89']})),
    ('multi-step', 'I want three packs of running socks and the trail running shoes. What is the total in dollars?',
     fixture_item({'final_contains': ['134']})),
    ('lookup', 'Maria Gomez says her blender order was only partly refunded. How much has been refunded so far, and '
               'how much more could still be refunded, in dollars? Do not issue any refund.',
     fixture_item({'refunds': [], 'final_contains': ['20', '100']})),
    ('documents', 'What is the excess on my current car insurance?' + CITE,
     documents_item({'facts': ['250'], 'source_ids': ['doc-2b3c4d5e']})),
    ('documents', 'How much do I owe in total for the October electricity and water bills?' + ' Cite the document '
                  'IDs you relied on.', documents_item({'facts': ['115.75'], 'source_ids': ['doc-4d5e6f70',
                                                                                         'doc-5e6f7081']})),
    ('documents', 'If I ask to cancel my gym membership today, 2026-10-30, on what date (YYYY-MM-DD) does the '
                  'notice period end?' + CITE, documents_item({'facts': ['2026-11-29'], 'source_ids': ['doc-1a2b3c4d']})),
    ('documents', 'When does my return flight from Porto leave?' + CITE, documents_item({'not_found': True})),
    ('documents', 'What is my FitLab membership number?' + CITE,
     documents_item({'facts': ['FL-88213'], 'source_ids': ['doc-1a2b3c4d']})),
    ('documents', 'When is my car insurance up for renewal?' + CITE,
     documents_item({'facts': ['2027-01-14'], 'source_ids': ['doc-2b3c4d5e']})),
]
