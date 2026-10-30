"""Run a system over items, score every episode, and write the report.

Two kinds of system:

- ``ModelSystem(generate)``: a model behind ``generate(messages, tools) -> reply`` where ``messages`` is a
  chat list (``system``/``user``/``assistant``/``tool`` roles), ``tools`` is a list of JSON tool schemas or
  None, and ``reply`` is the generated text or a dict with ``text`` and ``terminated`` (and optionally
  ``token_ids``), the record format of ``assistant_serving.cached_responder``. The harness owns the tool
  loop: it parses native Granite tool calls, executes them on the item's fixture and feeds results back.
  This runs a local transformers model, a stub, or anything else with that callable.
- ``ServedSystem``: the served assistant as users get it. Each item is one ``assistant.public.Session``
  with the version's route policies; every user turn is routed by ``route(text, opening)`` and answered
  by that route's responder, with the route's own system prompt, tools and limits. ``a4-general`` binds
  the router and the units (see ``served_a4_general``); the network's ``Conversation`` is the same Session
  behind paid shards. Caller-defined tools cannot be given to a served route, so ``fixture/1`` tool items
  are recorded as unsupported for a served system (scored as failures, counted separately).

``run`` returns ``(report, transcripts)``. The report holds counts, intervals, per-item outcomes and cost
fields; transcripts hold every generation so ``rescore`` can recompute every result without a model.
"""

import copy
import json
import time

from neuroshard.evolution.general_eval import items as schema
from neuroshard.evolution.general_eval import scorers, stats
from neuroshard.evolution.general_eval import tools as toolsets

REPORT_FORMAT = 'neuroshard-general-eval-report/1'
DEFAULT_SYSTEM_PROMPT = 'You are a helpful, accurate and concise assistant.'


def as_reply(value):
    if isinstance(value, str):
        return {'text': value, 'terminated': True}
    if not isinstance(value, dict) or not isinstance(value.get('text'), str):
        raise TypeError('generate must return text or a dict with text')
    return {**value, 'terminated': bool(value.get('terminated', True))}


class ModelSystem:
    kind = 'model'

    def __init__(self, generate, name, system_prompt=DEFAULT_SYSTEM_PROMPT, max_steps=8, max_calls=10):
        self.generate, self.name, self.system_prompt = generate, name, system_prompt
        self.max_steps, self.max_calls = max_steps, max_calls

    def describe(self):
        return {'kind': self.kind, 'name': self.name, 'system_prompt': self.system_prompt,
                'max_steps': self.max_steps, 'max_calls': self.max_calls}

    def _call(self, messages, tools, generations):
        reply = as_reply(self.generate(copy.deepcopy(messages), copy.deepcopy(tools)))
        generations.append(reply)
        return reply

    def episode(self, item):
        messages = [{'role': 'system', 'content': self.system_prompt}] if self.system_prompt else []
        generations, final, terminated, snapshot, failure = [], '', False, None, None
        tool_spec = item['scoring'].get('toolset') if item['area'] == 'tool_use' else None
        tools, world, parse = toolsets.toolset(tool_spec, item['fixture']) if tool_spec else (None, None, None)
        for turn in item['turns']:
            messages.append({'role': 'user', 'content': turn['user']})
            final, terminated, used = '', False, 0
            for _ in range(self.max_steps if tools else 1):
                reply = self._call(messages, tools, generations)
                if not reply['terminated']:
                    failure = 'generation did not terminate'
                    break
                messages.append({'role': 'assistant', 'content': reply['text']})
                if not tools:
                    final, terminated = reply['text'], True
                    break
                try:
                    calls = parse(reply['text'])
                except (ValueError, TypeError, RecursionError):
                    messages.append({'role': 'tool', 'content': json.dumps({'error': 'invalid tool-call syntax'})})
                    continue
                if not calls:
                    final, terminated = reply['text'], True
                    break
                if used + len(calls) > self.max_calls:
                    failure = 'tool-call budget exhausted'
                    break
                for call in calls:
                    used += 1
                    messages.append({'role': 'tool', 'content': json.dumps(world.execute(call), sort_keys=True)})
            if not terminated:
                failure = failure or 'step budget exhausted'
                break
        if world is not None:
            snapshot = world.snapshot()
        return {'final_text': final, 'terminated': terminated, 'snapshot': snapshot, 'failure': failure,
                'route': None, 'generations': generations}


class ServedSystem:
    """``route(text, opening_text) -> route name``; ``responder(route) -> respond(messages, tools)`` for one episode
    (a fresh cache per episode, as ``cached_responder`` requires)."""
    kind = 'served'

    def __init__(self, policies, route, responder, name, modules, entry_policy='drafting'):
        self.policies, self.route, self.responder = policies, route, responder
        self.name, self.modules, self.entry_policy = name, tuple(modules), entry_policy

    def describe(self):
        from neuroshard.evolution.modular_reference_execution import identity

        return {'kind': self.kind, 'name': self.name, 'modules': list(self.modules),
                'policies_sha256': {k: identity(v) for k, v in sorted(self.policies.items())}}

    def episode(self, item):
        tool_spec = item['scoring'].get('toolset') if item['area'] == 'tool_use' else None
        if tool_spec == 'fixture/1':
            return {'final_text': '', 'terminated': False, 'snapshot': None, 'route': None, 'generations': [],
                    'calls': [], 'failure': 'unsupported: the served version accepts no caller-defined tools'}
        world = item['fixture'] if tool_spec == 'documents/2' else {'documents': []}
        return self.converse(world, [turn['user'] for turn in item['turns']])

    def converse(self, world, users):
        """One conversation exactly as ``network.Conversation`` holds it: one Session, a route per turn, one
        responder per route for the whole conversation."""
        from neuroshard.assistant.public import Session

        session = Session(self.policies[self.entry_policy], world, name=self.name, modules=self.modules,
                          route_policies=self.policies)
        responders, generations, routes, result = {}, [], [], None
        opening = users[0]
        for user in users:
            route = self.route(user, opening)
            routes.append(route)
            if route not in responders:
                holder = {}

                def respond(messages, tools, route=route, holder=holder):
                    # Built on first use, so a Session that cannot build its workspace fails first, as in serving.
                    if 'inner' not in holder:
                        holder['inner'] = self.responder(route)
                    return as_reply(holder['inner'](messages, tools))

                responders[route] = respond
            try:
                result = session.turn(user, responders[route], route=route)
            except ValueError as error:
                if route in ('drafting', 'scheduling') and not world.get('documents'):
                    # A general request the router sent to A3's calendar workspace, which needs the user's files.
                    return {'final_text': '', 'terminated': False, 'snapshot': None, 'route': route, 'routes': routes,
                            'generations': generations, 'calls': copy.deepcopy(session.calls),
                            'failure': f'routed to {route} without a workspace: {error}'}
                raise
            generations += [as_reply(r) for r in result['responses']]
            if result['failure']:
                break
        return {'final_text': result['final_text'], 'terminated': bool(result['completed']),
                'snapshot': result['snapshot'], 'failure': result['failure'], 'route': routes[-1], 'routes': routes,
                'generations': generations, 'calls': copy.deepcopy(session.calls)}


def score(item, outcome):
    spec, text = item['scoring'], outcome['final_text'] if outcome['terminated'] else ''
    kind = spec['kind']
    if outcome.get('failure', '') and str(outcome.get('failure')).startswith('unsupported'):
        return {'passed': False, 'detail': {'unsupported': outcome['failure']}}
    if kind == 'answer':
        return scorers.score_answer(spec, text)
    if kind == 'code':
        return scorers.score_code(spec, text)
    if kind == 'constraints':
        return scorers.score_constraints(spec, text)
    if kind == 'short_answer':
        return scorers.score_short_answer(spec, text)
    if kind == 'plan':
        return scorers.score_plan(spec, text)
    snapshot = outcome['snapshot']
    if spec['toolset'] == 'documents/2' and (snapshot is None or 'read_ids' not in snapshot):
        return {'passed': False, 'detail': {'problems': [f'not answered on the documents route ({outcome.get("route")})']}}
    return scorers.score_tools(spec, text, outcome['terminated'], snapshot, spec['toolset'])


def _tokens(generations):
    counted = [len(g['token_ids']) for g in generations if isinstance(g.get('token_ids'), list)]
    return sum(counted) if counted else None


def run(items, system, *, split, split_digest=None, commitment=None, cost=None, publish_items=True, progress=None,
        stop=None):
    """Every item once, in order; one failing episode is recorded as a failed item, never retried.

    ``progress(done, total, row, transcript)`` sees each item as it finishes; ``stop()`` is asked before each item
    and ends the run early (the report then records ``stopped_early`` and covers only the items run)."""
    started = time.monotonic()
    rows, transcripts, isolation, stopped = [], [], set(), False
    for index, item in enumerate(items):
        if stop is not None and stop():
            stopped = True
            break
        began = time.monotonic()
        try:
            outcome = system.episode(item)
            error = None
        except Exception as exc:  # noqa: BLE001 - a system error is a failed item, recorded
            outcome = {'final_text': '', 'terminated': False, 'snapshot': None, 'route': None, 'generations': [],
                       'failure': f'system error: {type(exc).__name__}: {exc}'}
            error = outcome['failure']
        seconds = time.monotonic() - began
        result = score(item, outcome)
        if item['scoring']['kind'] == 'code' and 'isolation' in result['detail']:
            isolation.add(tuple(result['detail']['isolation']))
        failure = outcome.get('failure') or ''
        rows.append({'id': item['id'], 'area': item['area'], 'family': item['family'], 'passed': result['passed'],
                     'toolset': item['scoring'].get('toolset'),
                     'unsupported': failure.startswith('unsupported'), 'error': error, 'route': outcome.get('route'),
                     'seconds': round(seconds, 3), 'generations': len(outcome['generations']),
                     'generated_tokens': _tokens(outcome['generations'])})
        transcripts.append({'id': item['id'], 'outcome': outcome, 'score': result})
        if progress:
            progress(index + 1, len(items), rows[-1], transcripts[-1])
    report = summarise(rows, system.describe(), split, split_digest, commitment, cost,
                       time.monotonic() - started, isolation)
    report.update(items_planned=len(items), stopped_early=stopped)
    if publish_items:
        report['per_item'] = rows
    return report, transcripts


def summarise(rows, system, split, split_digest, commitment, cost, wall_seconds, isolation=()):
    areas = {}
    for area in schema.AREAS:
        mine = [r for r in rows if r['area'] == area]
        if not mine:
            continue
        passed = sum(r['passed'] for r in mine)
        families, routes = {}, {}
        for r in mine:
            hit, total = families.get(r['family'], (0, 0))
            families[r['family']] = (hit + r['passed'], total + 1)
            routes[str(r['route'])] = routes.get(str(r['route']), 0) + 1
        low, high = stats.wilson(passed, len(mine))
        areas[area] = {'passed': passed, 'cases': len(mine), 'rate': round(passed / len(mine), 4),
                       'wilson95': [round(low, 4), round(high, 4)],
                       'families': {k: f'{h}/{t}' for k, (h, t) in sorted(families.items())},
                       'unsupported': sum(r['unsupported'] for r in mine), 'errors': sum(bool(r['error']) for r in mine),
                       'routes': dict(sorted(routes.items()))}
        toolsets = sorted({r.get('toolset') for r in mine if r.get('toolset')})
        if toolsets:
            areas[area]['by_toolset'] = {}
            for name in toolsets:
                part = [r for r in mine if r.get('toolset') == name]
                hit = sum(r['passed'] for r in part)
                areas[area]['by_toolset'][name] = {
                    'passed': hit, 'cases': len(part), 'unsupported': sum(r['unsupported'] for r in part),
                    'wilson95': [round(x, 4) for x in stats.wilson(hit, len(part))]}
    tokens = [r['generated_tokens'] for r in rows if r['generated_tokens'] is not None]
    cost = dict(cost or {})
    usd_per_hour = cost.get('usd_per_hour')
    cost.update(wall_seconds=round(wall_seconds, 3), model_seconds=round(sum(r['seconds'] for r in rows), 3),
                generations=sum(r['generations'] for r in rows), generated_tokens=sum(tokens) if tokens else None,
                compute_usd=round(usd_per_hour * wall_seconds / 3600, 4) if usd_per_hour is not None else None)
    cost.setdefault('judge_calls', 0)
    cost.setdefault('judge_usd', 0.0)
    rates = [a['rate'] for a in areas.values()]
    return {'format': REPORT_FORMAT, 'system': system, 'split': split, 'split_digest': split_digest,
            'commitment': commitment, 'items': len(rows), 'areas': areas,
            'overall': {'passed': sum(r['passed'] for r in rows), 'cases': len(rows),
                        'macro_rate': round(sum(rates) / len(rates), 4) if rates else None},
            'sandbox_isolation': [list(layers) for layers in sorted(isolation)], 'cost': cost,
            'baseline_claim': False}


def rescore(items, transcripts):
    """Recompute every recorded result from its transcript (re-running code); raises on any disagreement."""
    by_id = {item['id']: item for item in items}
    for row in transcripts:
        again = score(by_id[row['id']], row['outcome'])
        if again['passed'] != row['score']['passed']:
            raise ValueError(f'rescore differs: {row["id"]}')
    return sum(row['score']['passed'] for row in transcripts)
