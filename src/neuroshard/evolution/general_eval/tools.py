"""Deterministic fixture tools for the tool-use area. Every tool reads or changes only the item's fixture.

Two tool sets:

- ``fixture/1``: a small, generic personal-and-work toolset (customers and orders with refunds, currency
  conversion at fixed rates, weather forecasts, contacts and messages that are recorded and never sent,
  a calendar, a product catalogue, exact arithmetic). It tests general function calling. The served
  ``a4-general`` cannot be handed caller-defined tools, so for the served system these items run only
  in the declared approximation (see docs/GENERAL_CAPABILITY_EVAL.md).
- ``documents/2``: the served documents route's own tools and workspace
  (``assistant_documents``, interface ``workspace-documents/2``), over the item's fixture documents.
  These run unchanged through the served route.

Calls use the native Granite envelope parsed by ``assistant_workspace.parse_calls``, as every served
route does. Nothing here touches a network or a file.
"""

import copy
from decimal import ROUND_HALF_EVEN, Decimal

from neuroshard.evolution import assistant_workspace as base

STRING, INTEGER, NUMBER = {'type': 'string'}, {'type': 'integer'}, {'type': 'number'}
TOOLS = [
    base.tool('find_customer', 'Find customers whose name or email contains the query (case-insensitive).',
              {'query': STRING}),
    base.tool('list_orders', 'List every order of one customer ID, newest first.', {'customer_id': STRING}),
    base.tool('get_order', 'Get one order by its ID, including status, total and amount already refunded.',
              {'order_id': STRING}),
    base.tool('issue_refund', 'Refund part or all of a delivered order, in cents. This changes the order.',
              {'order_id': STRING, 'amount_cents': INTEGER, 'reason': STRING}),
    base.tool('convert_currency', 'Convert an amount between currencies at today\'s fixed rate; rounded to cents.',
              {'amount': NUMBER, 'from_currency': STRING, 'to_currency': STRING}),
    base.tool('get_weather', 'Forecast for a city on a date (YYYY-MM-DD): high and low in Celsius and conditions.',
              {'city': STRING, 'date': STRING}),
    base.tool('find_contact', 'Find contacts whose name contains the query.', {'query': STRING}),
    base.tool('send_message', 'Send a short text message to one contact ID.', {'contact_id': STRING, 'text': STRING}),
    base.tool('list_events', 'List calendar events on a date (YYYY-MM-DD), with start and end times (HH:MM).',
              {'date': STRING}),
    base.tool('create_event', 'Create a calendar event; refused if it overlaps an existing event.',
              {'title': STRING, 'date': STRING, 'start': STRING, 'end': STRING}),
    base.tool('search_products', 'Search the product catalogue by keyword; returns SKU, name, price in cents and '
              'stock.', {'query': STRING}),
    base.tool('calculate', 'Exact arithmetic on two numbers: add, subtract, multiply or divide.',
              {'operation': {'type': 'string', 'enum': ['add', 'subtract', 'multiply', 'divide']},
               'left': NUMBER, 'right': NUMBER}),
]
REGISTRY = {item['function']['name']: item['function']['parameters'] for item in TOOLS}
STATE_CHANGING = ('issue_refund', 'send_message', 'create_event')


def _minutes(value):
    hours, minutes = value.split(':')
    if len(hours) != 2 or len(minutes) != 2 or not (0 <= int(hours) < 24 and 0 <= int(minutes) < 60):
        raise ValueError('time must be HH:MM')
    return int(hours) * 60 + int(minutes)


def _number(value):
    value = Decimal(str(value))
    return int(value) if value == value.to_integral_value() else float(value)


class FixtureWorld:
    """The ``fixture/1`` tools over one item's fixture. ``changes`` records every effect a tool had."""

    def __init__(self, fixture):
        self.data = copy.deepcopy(fixture)
        for key in ('customers', 'orders', 'contacts', 'events', 'products', 'weather'):
            self.data.setdefault(key, [])
        self.data.setdefault('rates', {})
        for order in self.data['orders']:
            order.setdefault('refunded_cents', 0)
        self.events, self.changes = [], {'refunds': [], 'messages': [], 'events_created': []}

    def execute(self, call):
        try:
            if set(call) != {'name', 'arguments'} or call['name'] not in REGISTRY:
                raise ValueError('unknown tool')
            base._check_value(call['arguments'], REGISTRY[call['name']])
            result = getattr(self, '_' + call['name'])(**call['arguments'])
        except (ValueError, KeyError, TypeError, ArithmeticError) as error:
            result = {'error': str(error) or type(error).__name__}
        self.events.append({'call': copy.deepcopy(call), 'result': copy.deepcopy(result)})
        return result

    def snapshot(self):
        return {'changes': copy.deepcopy(self.changes), 'events': copy.deepcopy(self.events)}

    def _order(self, order_id):
        for order in self.data['orders']:
            if order['id'] == order_id:
                return order
        raise KeyError(f'no order {order_id}')

    def _find_customer(self, query):
        q = query.strip().lower()
        if not q:
            raise ValueError('empty query')
        return {'customers': [c for c in self.data['customers'] if q in c['name'].lower() or q in c['email'].lower()]}

    def _list_orders(self, customer_id):
        rows = [o for o in self.data['orders'] if o['customer_id'] == customer_id]
        return {'orders': sorted(copy.deepcopy(rows), key=lambda o: o['date'], reverse=True)}

    def _get_order(self, order_id):
        return copy.deepcopy(self._order(order_id))

    def _issue_refund(self, order_id, amount_cents, reason):
        order = self._order(order_id)
        if order['status'] != 'delivered':
            raise ValueError(f'order {order_id} is {order["status"]}; only delivered orders can be refunded')
        if not 0 < amount_cents <= order['total_cents'] - order['refunded_cents']:
            raise ValueError('refund must be positive and at most the unrefunded total')
        if not reason.strip():
            raise ValueError('a refund needs a reason')
        order['refunded_cents'] += amount_cents
        self.changes['refunds'].append({'order_id': order_id, 'amount_cents': amount_cents})
        return {'refunded': True, 'order_id': order_id, 'refunded_cents_total': order['refunded_cents']}

    def _convert_currency(self, amount, from_currency, to_currency):
        source, target = from_currency.upper(), to_currency.upper()
        if source == target:
            rate = Decimal(1)
        elif source in self.data['rates'] and target in self.data['rates'][source]:
            rate = Decimal(str(self.data['rates'][source][target]))
        elif target in self.data['rates'] and source in self.data['rates'][target]:
            rate = 1 / Decimal(str(self.data['rates'][target][source]))
        else:
            raise ValueError(f'no rate from {source} to {target}')
        value = (Decimal(str(amount)) * rate).quantize(Decimal('0.01'), rounding=ROUND_HALF_EVEN)
        return {'amount': float(value), 'currency': target}

    def _get_weather(self, city, date):
        for row in self.data['weather']:
            if row['city'].lower() == city.strip().lower() and row['date'] == date:
                return {k: row[k] for k in ('city', 'date', 'high_c', 'low_c', 'conditions')}
        raise KeyError(f'no forecast for {city} on {date}')

    def _find_contact(self, query):
        q = query.strip().lower()
        if not q:
            raise ValueError('empty query')
        return {'contacts': [c for c in self.data['contacts'] if q in c['name'].lower()]}

    def _send_message(self, contact_id, text):
        if not any(c['id'] == contact_id for c in self.data['contacts']):
            raise KeyError(f'no contact {contact_id}')
        if not 1 <= len(text) <= 1000:
            raise ValueError('message must be 1-1000 characters')
        self.changes['messages'].append({'contact_id': contact_id, 'text': text})
        return {'sent': True}

    def _list_events(self, date):
        base.iso_date(date)
        rows = [e for e in self.data['events'] if e['date'] == date]
        return {'events': sorted(copy.deepcopy(rows), key=lambda e: e['start'])}

    def _create_event(self, title, date, start, end):
        base.iso_date(date)
        begin, finish = _minutes(start), _minutes(end)
        if not title.strip() or finish <= begin:
            raise ValueError('an event needs a title and an end after its start')
        for event in self.data['events']:
            if event['date'] == date and begin < _minutes(event['end']) and _minutes(event['start']) < finish:
                raise ValueError(f'overlaps "{event["title"]}" {event["start"]}-{event["end"]}')
        event = {'id': f'evt-new-{len(self.changes["events_created"]) + 1}', 'title': title, 'date': date,
                 'start': start, 'end': end}
        self.data['events'].append(event)
        self.changes['events_created'].append({k: event[k] for k in ('title', 'date', 'start', 'end')})
        return {'created': True, 'event_id': event['id']}

    def _search_products(self, query):
        terms = [t for t in query.lower().split() if t]
        if not terms:
            raise ValueError('empty query')
        rows = [p for p in self.data['products'] if all(t in (p['name'] + ' ' + p.get('tags', '')).lower()
                                                        for t in terms)]
        return {'products': [{k: p[k] for k in ('sku', 'name', 'price_cents', 'stock')} for p in rows]}

    def _calculate(self, operation, left, right):
        a, b = Decimal(str(left)), Decimal(str(right))
        if operation == 'divide' and b == 0:
            raise ValueError('division by zero')
        value = {'add': a + b, 'subtract': a - b, 'multiply': a * b, 'divide': a / b if b else None}[operation]
        return {'value': _number(value.quantize(Decimal('0.000001')) if operation == 'divide' else value)}


def toolset(name, fixture):
    """``(tool schemas, workspace, call parser)`` for a declared tool set."""
    if name == 'fixture/1':
        return TOOLS, FixtureWorld(fixture), lambda text: base.parse_calls(text, REGISTRY)
    if name == 'documents/2':
        from neuroshard.assistant import interfaces

        interface = interfaces.interface({'interface': 'workspace-documents/2'})
        return interface.TOOLS, interface.Workspace(fixture), interface.parse_calls
    raise ValueError(f'unknown tool set {name!r}')
