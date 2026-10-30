"""Program scorers for every area. A scorer sees the item's hidden ``scoring`` block and the system's output.

Each ``score_*`` returns ``{'passed': bool, 'detail': ...}``. All rules are declared here, before any run,
and apply identically to every system; a rescore from recorded outputs must reproduce the recorded result.

Answer extraction (reasoning): the prompt asks for a final line ``Answer: <value>``. The last such line is
the answer; when there is none, a numeric item falls back to the last number in the reply (the same rule as
``assistant_chat_data``'s ``number`` constraint), and other types fail.
"""

from decimal import Decimal, InvalidOperation
from fractions import Fraction
import itertools
import json
import re
import unicodedata

from neuroshard.evolution import assistant_chat_data as chat_data

ANSWER_LINE = re.compile(r'^[\s>*_#-]*answer[\s*_]*[:：]\s*(.+?)\s*$', re.IGNORECASE | re.MULTILINE)
NUMBER = re.compile(r'-?\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?%?')  # a comma only groups thousands
PLAN_VALIDATORS = ('schedule', 'order', 'assignment', 'budget', 'route')
CHAT_KINDS = ('bullets', 'max_words', 'json_keys', 'include', 'exclude', 'ends_with', 'lowercase', 'number')
EXTRA_KINDS = ('min_words', 'paragraphs', 'starts_with', 'include_all', 'exclude_all', 'no_char', 'lines',
               'title_line', 'max_sentence_words')


# ---------------------------------------------------------------- shared text helpers

def normalize(text):
    text = unicodedata.normalize('NFKC', text).lower()
    text = re.sub(r'(?<=\d),(?=\d{3})', '', text)
    text = re.sub(r"[^\w\s.%-]|(?<!\d)\.|\.(?!\d)|_", ' ', text)
    return ' '.join(text.split())


def parse_number(text):
    """A number written as 1,234.5 / $12 / 3/4 / 40% (compared as 40; the item states its unit); else None."""
    raw = text.strip().replace(',', '').replace('$', '').rstrip('.').rstrip('%')
    try:
        return Fraction(raw) if '/' in raw else Fraction(Decimal(raw))
    except (ValueError, ZeroDivisionError, InvalidOperation):
        return None


def final_answer(text):
    lines = ANSWER_LINE.findall(text or '')
    return lines[-1].strip().strip('*_`').strip() if lines else None


# ---------------------------------------------------------------- reasoning

def score_answer(spec, text):
    answer = spec['answer']
    stated = final_answer(text)
    if answer['type'] == 'number':
        source = stated if stated is not None else (text or '')
        candidates = NUMBER.findall(source)[-1:]
        value = parse_number(candidates[0]) if candidates else None
        target, tolerance = Fraction(Decimal(str(answer['value']))), Fraction(Decimal(str(answer.get('tolerance', 0))))
        ok = value is not None and abs(value - target) <= tolerance
        return {'passed': ok, 'detail': {'extracted': None if value is None else str(value), 'line': stated is not None}}
    if stated is None:
        return {'passed': False, 'detail': {'extracted': None, 'line': False}}
    if answer['type'] == 'choice':
        letter = re.match(r'\(?([A-Ha-h])\)?(?:[\s.):]|$)', stated)
        got = letter[1].upper() if letter else None
        return {'passed': got == answer['value'], 'detail': {'extracted': got, 'line': True}}
    got = normalize(stated)
    ok = any(got == normalize(a) for a in answer['accepted'])
    return {'passed': ok, 'detail': {'extracted': got, 'line': True}}


# ---------------------------------------------------------------- coding

def extract_code(text, entry_point):
    """The last fenced block that defines ``entry_point``; else the whole reply if it defines it."""
    blocks = re.findall(r'```[ \t]*(?:python|py|python3)?[ \t]*\n(.*?)```', text or '', re.DOTALL | re.IGNORECASE)
    pattern = re.compile(rf'^\s*(?:def|class)\s+{re.escape(entry_point)}\s*[(:]', re.MULTILINE)
    for block in reversed(blocks):
        if pattern.search(block):
            return block
    if blocks:
        return None
    return text if text and pattern.search(text) else None


def score_code(spec, text, run=None):
    from neuroshard.evolution.general_eval import sandbox

    code = extract_code(text, spec['entry_point'])
    if code is None:
        return {'passed': False, 'detail': {'error': 'no code defining ' + spec['entry_point']}}
    result = (run or sandbox.run_python)(code, spec['tests'], timeout=spec.get('timeout', 10))
    return {'passed': bool(result['passed']), 'detail': result}


# ---------------------------------------------------------------- writing constraints

def sentences(text):
    return [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]


def check_constraint(constraint, text):
    kind = constraint['kind']
    if kind in CHAT_KINDS:
        return chat_data.check(constraint, text)
    if kind == 'min_words':
        return chat_data.word_count(text) >= constraint['n']
    if kind == 'paragraphs':
        return len([p for p in re.split(r'\n\s*\n', text.strip()) if p.strip()]) == constraint['n']
    if kind == 'lines':
        return len([line for line in text.strip().splitlines() if line.strip()]) == constraint['n']
    if kind == 'starts_with':
        body = re.sub(r'^[\s*_"\'“#]+', '', text)
        return body.lower().startswith(constraint['phrase'].lower())
    if kind == 'include_all':
        return all(chat_data.check({'kind': 'include', 'word': w}, text) for w in constraint['words'])
    if kind == 'exclude_all':
        return all(chat_data.check({'kind': 'exclude', 'word': w}, text) for w in constraint['words'])
    if kind == 'no_char':
        return constraint['char'] not in text
    if kind == 'title_line':
        first = text.strip().splitlines()[0] if text.strip() else ''
        title = re.sub(r'^#+\s*|\*\*', '', first).strip()
        return bool(title) and len(title.split()) <= constraint.get('max_words', 8) and not title.endswith('.')
    if kind == 'max_sentence_words':
        return bool(sentences(text)) and all(chat_data.word_count(s) <= constraint['n'] for s in sentences(text))
    raise ValueError(f'unknown constraint {kind}')


def score_constraints(spec, text):
    results = [(c['kind'], bool(text and text.strip()) and check_constraint(c, text)) for c in spec['constraints']]
    return {'passed': all(ok for _, ok in results), 'detail': {'constraints': results}}


# ---------------------------------------------------------------- factual

def contains(haystack, needle):
    """``needle`` appears as whole words; a numeric needle matches any number in the text of equal value."""
    if re.fullmatch(r'-?\d+(?:\.\d+)?', needle.strip()):
        target = parse_number(needle)
        return any(parse_number(n) == target for n in NUMBER.findall(haystack or ''))
    return f' {normalize(needle)} ' in f' {normalize(haystack)} '


def score_short_answer(spec, text):
    stated = final_answer(text) or text or ''
    hit = [a for a in spec['accepted'] if contains(stated, a)]
    wrong = [r for r in spec.get('rejected', ()) if contains(stated, r)]
    return {'passed': bool(hit) and not wrong, 'detail': {'accepted_hit': hit, 'rejected_hit': wrong}}


# ---------------------------------------------------------------- planning

def extract_json(text):
    """The last fenced JSON block, else the last balanced top-level object or array in the reply."""
    fenced = re.findall(r'```(?:json)?\s*\n?(.*?)```', text or '', re.DOTALL)
    for body in reversed(fenced):
        try:
            return json.loads(body)
        except ValueError:
            continue
    decoder, found, index, text = json.JSONDecoder(), None, 0, text or ''
    while index < len(text):
        if text[index] in '{[':
            try:
                found, end = decoder.raw_decode(text[index:])
                index += end  # skip the values nested inside this one
                continue
            except ValueError:
                pass
        index += 1
    return found


def _minutes(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{1,2}:\d{2}', value.strip()):
        raise ValueError(f'bad time {value!r}')
    hours, minutes = value.strip().split(':')
    if not (0 <= int(hours) < 24 and 0 <= int(minutes) < 60):
        raise ValueError(f'bad time {value!r}')
    return int(hours) * 60 + int(minutes)


def validate_schedule(spec, plan):
    tasks = spec['tasks']
    if not isinstance(plan, dict) or set(plan) != set(tasks):
        return ['the plan must give a start time for exactly the listed tasks']
    problems, start = [], {}
    for name, value in plan.items():
        try:
            start[name] = _minutes(value)
        except ValueError as error:
            return [str(error)]
    end = {name: start[name] + tasks[name]['duration'] for name in tasks}
    window = [_minutes(t) for t in spec['window']]
    for name, task in tasks.items():
        if start[name] < window[0] or end[name] > window[1]:
            problems.append(f'{name} is outside the day window')
        if 'earliest' in task and start[name] < _minutes(task['earliest']):
            problems.append(f'{name} starts before {task["earliest"]}')
        if 'deadline' in task and end[name] > _minutes(task['deadline']):
            problems.append(f'{name} ends after {task["deadline"]}')
        for before in task.get('after', ()):
            if start[name] < end[before]:
                problems.append(f'{name} starts before {before} ends')
    blocks = [(start[n], end[n], n) for n in tasks] + [(_minutes(b['start']), _minutes(b['end']), b['name'])
                                                       for b in spec.get('busy', ())]
    for (a0, a1, a), (b0, b1, b) in itertools.combinations(blocks, 2):
        if a0 < b1 and b0 < a1 and not (a in spec.get('parallel_ok', ()) or b in spec.get('parallel_ok', ())):
            problems.append(f'{a} overlaps {b}')
    return problems


def validate_order(spec, plan):
    items = spec['items']
    if not isinstance(plan, list) or sorted(map(str, plan)) != sorted(items) or len(plan) != len(items):
        return ['the plan must list every item exactly once']
    where = {name: index for index, name in enumerate(plan)}
    problems = [f'{a} must come before {b}' for a, b in spec.get('before', ()) if where[a] > where[b]]
    if 'first' in spec and plan[0] != spec['first']:
        problems.append(f'{spec["first"]} must be first')
    if 'last' in spec and plan[-1] != spec['last']:
        problems.append(f'{spec["last"]} must be last')
    for a, b in spec.get('not_adjacent', ()):
        if abs(where[a] - where[b]) == 1:
            problems.append(f'{a} and {b} must not be adjacent')
    return problems


def validate_assignment(spec, plan):
    slots, people = spec['slots'], spec['people']
    if not isinstance(plan, dict) or set(plan) != set(slots):
        return ['the plan must staff exactly the listed slots']
    problems, load = [], {p: 0 for p in people}
    for slot, need in slots.items():
        staff = plan[slot]
        if not isinstance(staff, list) or len(set(staff)) != len(staff) or len(staff) != need['count']:
            problems.append(f'{slot} needs exactly {need["count"]} distinct people')
            continue
        for person in staff:
            if person not in people:
                problems.append(f'{person} is not on the team')
                continue
            load[person] += 1
            if slot in people[person].get('unavailable', ()):
                problems.append(f'{person} is unavailable for {slot}')
        for skill in need.get('requires', ()):
            if not any(skill in people.get(p, {}).get('skills', ()) for p in staff):
                problems.append(f'{slot} needs someone with {skill}')
        for a, b in spec.get('apart', ()):
            if a in staff and b in staff:
                problems.append(f'{a} and {b} must not share {slot}')
    for person, info in people.items():
        if load[person] > info.get('max', len(slots)):
            problems.append(f'{person} has more than {info["max"]} slots')
        if load[person] < info.get('min', 0):
            problems.append(f'{person} has fewer than {info["min"]} slots')
    return problems


def _budget_value(spec, chosen):
    return sum(spec['options'][name]['value'] for name in chosen)


def _budget_feasible(spec, chosen):
    problems = []
    if sum(spec['options'][name]['cost'] for name in chosen) > spec['budget']:
        problems.append('over budget')
    problems += [f'{m} is required' for m in spec.get('must', ()) if m not in chosen]
    problems += [f'{a} and {b} exclude each other' for a, b in spec.get('exclusive', ()) if a in chosen and b in chosen]
    problems += [f'{a} needs {b}' for a, b in spec.get('requires', ()) if a in chosen and b not in chosen]
    return problems


def best_budget_value(spec):
    names = sorted(spec['options'])
    return max(_budget_value(spec, chosen) for r in range(len(names) + 1)
               for chosen in itertools.combinations(names, r) if not _budget_feasible(spec, chosen))


def validate_budget(spec, plan):
    if not isinstance(plan, list) or len(set(map(str, plan))) != len(plan) or not set(plan) <= set(spec['options']):
        return ['the plan must be a list of distinct listed options']
    problems = _budget_feasible(spec, plan)
    if not problems and spec.get('optimal') and _budget_value(spec, plan) < best_budget_value(spec):
        problems.append('a feasible choice with more total value exists')
    return problems


def _route_time(spec, path):
    return sum(spec['times'][a][b] for a, b in zip(path, path[1:]))


def validate_route(spec, plan):
    stops = spec['stops']
    if not isinstance(plan, list) or not plan or plan[0] != spec['start'] or plan[-1] != spec.get('end', spec['start']):
        return [f'the route must start at {spec["start"]} and end at {spec.get("end", spec["start"])}']
    middle = plan[1:-1]
    if sorted(middle) != sorted(stops) or len(set(middle)) != len(middle):
        return ['the route must visit every listed stop exactly once']
    problems = [f'{a} must be visited before {b}' for a, b in spec.get('before', ()) if middle.index(a) > middle.index(b)]
    total = _route_time(spec, plan)
    if 'limit' in spec and total > spec['limit']:
        problems.append(f'route takes {total}, over the limit of {spec["limit"]}')
    if not problems and spec.get('optimal'):
        best = min(_route_time(spec, [spec['start'], *order, spec.get('end', spec['start'])])
                   for order in itertools.permutations(stops)
                   if all(order.index(a) < order.index(b) for a, b in spec.get('before', ())))
        if total > best:
            problems.append(f'route takes {total}; {best} is possible')
    return problems


VALIDATE = {'schedule': validate_schedule, 'order': validate_order, 'assignment': validate_assignment,
            'budget': validate_budget, 'route': validate_route}


def score_plan(spec, text):
    plan = extract_json(text)
    if plan is None:
        return {'passed': False, 'detail': {'problems': ['no JSON plan in the reply']}}
    if spec.get('key') and isinstance(plan, dict) and spec['key'] in plan:
        plan = plan[spec['key']]
    try:
        problems = VALIDATE[spec['validator']](spec, plan)
    except (KeyError, TypeError, ValueError) as error:
        problems = [f'malformed plan: {type(error).__name__}: {error}']
    return {'passed': not problems, 'detail': {'problems': problems[:10]}}


# ---------------------------------------------------------------- tool use

def score_tools(spec, final_text, terminated, snapshot, toolset):
    """``fixture/1``: the effects on the fixture must equal the expected ones exactly (no extra refund, message or
    event), required tools must have been used, and the final reply must state the expected facts.
    ``documents/2``: the served documents scorer (facts stated, exactly the needed sources cited, or not-found
    reported), unchanged."""
    expect = spec['expect']
    if toolset == 'documents/2':
        from neuroshard.evolution import assistant_documents as documents

        ok = documents.score_round(snapshot, expect, terminated, final_text, 2)
        return {'passed': bool(ok), 'detail': {'read_ids': snapshot.get('read_ids'), 'answer': snapshot.get('answer'),
                                               'not_found': snapshot.get('not_found')}}
    problems = []
    if not terminated or not (final_text or '').strip():
        problems.append('no final reply')
    changes = snapshot['changes']
    used = [event['call']['name'] for event in snapshot['events']]
    for name in expect.get('required_calls', ()):
        if name not in used:
            problems.append(f'never called {name}')
    if changes['refunds'] != expect.get('refunds', []):
        problems.append(f'refunds {changes["refunds"]} != expected')
    want_messages = expect.get('messages', [])
    if len(changes['messages']) != len(want_messages):
        problems.append(f'{len(changes["messages"])} messages sent, expected {len(want_messages)}')
    else:
        for got, want in zip(changes['messages'], want_messages):
            if got['contact_id'] != want['contact_id'] or not all(contains(got['text'], s) for s in want['contains']):
                problems.append(f'message to {got["contact_id"]} does not match')
    want_events = expect.get('events_created', [])
    if len(changes['events_created']) != len(want_events):
        problems.append(f'{len(changes["events_created"])} events created, expected {len(want_events)}')
    else:
        for got, want in zip(changes['events_created'], want_events):
            if any(got[k] != want[k] for k in ('date', 'start', 'end')) or not contains(got['title'], want['title']):
                problems.append(f'event {got} does not match')
    for fact in expect.get('final_contains', ()):
        alternatives = fact if isinstance(fact, list) else [fact]
        if not any(contains(final_text or '', alt) for alt in alternatives):
            problems.append(f'final reply does not state {alternatives}')
    return {'passed': not problems, 'detail': {'problems': problems, 'calls': used}}


# ---------------------------------------------------------------- spec checks

def check_spec(item):
    spec, kind = item['scoring'], item['scoring']['kind']
    try:
        if kind == 'answer':
            answer = spec['answer']
            if answer['type'] == 'number':
                Decimal(str(answer['value']))
            elif answer['type'] == 'choice':
                assert re.fullmatch(r'[A-H]', answer['value'])
            elif answer['type'] == 'text':
                assert answer['accepted'] and all(isinstance(a, str) for a in answer['accepted'])
            else:
                raise ValueError('answer type')
        elif kind == 'code':
            assert re.fullmatch(r'[A-Za-z_]\w*', spec['entry_point']) and spec['tests'].strip()
            assert 'reference' in spec, 'a coding item carries a reference solution that passes its tests'
        elif kind == 'constraints':
            assert spec['constraints']
            for c in spec['constraints']:
                assert c['kind'] in CHAT_KINDS + EXTRA_KINDS, c['kind']
        elif kind == 'short_answer':
            assert spec['accepted'] and not set(map(normalize, spec['accepted'])) & set(
                map(normalize, spec.get('rejected', ())))
        elif kind == 'plan':
            assert spec['validator'] in PLAN_VALIDATORS
            assert 'reference' in spec, 'a planning item carries a reference plan that validates'
        elif kind == 'tools':
            assert spec['toolset'] in ('fixture/1', 'documents/2') and isinstance(spec['expect'], dict)
    except (AssertionError, KeyError, TypeError, InvalidOperation) as error:
        raise ValueError(f'{item["id"]}: invalid {kind} scoring spec: {error!r}') from None
