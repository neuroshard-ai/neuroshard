"""Episodes for the general route: the user's documents with tools, and tool-free chat with checkable constraints.

``documents_episode`` runs a model through the documents workspace exactly as the routed workspace
runner does: up to the policy's model turns, at most two native tool calls per response, the tool
budget, and a final reply. ``chat_episode`` is one response, scored by its constraints. Both record
every generation, so a transcript can be rescored without the model.
"""

import copy
import json
import re
import time

from neuroshard.evolution import assistant_chat_data as chat_data
from neuroshard.evolution import assistant_documents as documents
from neuroshard.evolution.assistant_documents_data import public_case
from neuroshard.evolution.modular_reference_execution import identity


def version_of(policy):
    return documents.INTERFACES.index(policy['interface']) + 1


def documents_episode(case, respond, policy):
    public = public_case(case)
    version = version_of(policy)
    world = documents.Workspace(public['world'], version)
    tools = documents.tools(version)
    history = [{'role': 'system', 'content': policy['system_instruction']}]
    calls, rounds, generations = [], [], []
    started = time.monotonic()
    for turn, user in enumerate(public['user_turns']):
        history.append({'role': 'user', 'content': user})
        final, completed, failure, used = '', False, None, 0
        for _ in range(policy['limits']['model_turns_per_user_turn']):
            generated = respond(copy.deepcopy(history), copy.deepcopy(tools))
            generations.append({**generated, 'request_sha256': identity({'messages': history, 'tools': tools})})
            if not generated['terminated']:
                failure = 'generation did not terminate within its token/input budget'
                break
            history.append({'role': 'assistant', 'content': generated['text']})
            try:
                proposed = documents.parse_calls(generated['text'], version)
            except (ValueError, TypeError, RecursionError):
                history.append({'role': 'tool', 'content': json.dumps({'error': 'invalid tool-call syntax or schema'})})
                continue
            if not proposed:
                final, completed = generated['text'], bool(generated['text'].strip())
                break
            if used + len(proposed) > policy['limits']['tool_calls_per_user_turn']:
                failure = 'tool-call budget exhausted'
                break
            for call in proposed:
                result = world.execute(call)
                calls.append({'call': call, 'result': result})
                used += 1
                history.append({'role': 'tool', 'content': json.dumps(result, ensure_ascii=False, sort_keys=True)})
        if not completed and failure is None:
            failure = 'model-turn budget exhausted'
        snapshot = world.snapshot()
        rounds.append({'completed': completed, 'final_text': final, 'failure': failure, 'snapshot': snapshot,
                       'passed': documents.score_round(snapshot, case['turns'][turn]['expected'], completed, final,
                                                       version)})
        if failure:
            break
    return {'id': case['id'], 'family': case['family'], 'version': version, 'rounds': rounds, 'calls': calls,
            'generations': generations,
            'passed': len(rounds) == len(case['turns']) and all(r['passed'] for r in rounds),
            'seconds': time.monotonic() - started, 'tool_calls': len(calls),
            'model_calls': sum(g.get('executed', True) for g in generations)}


class Calculator:
    """The chat route's only tool: exact integer arithmetic, recorded like any workspace call."""

    TOOLS = [documents.TOOLS[2]]
    REGISTRY = {item['function']['name']: item['function']['parameters'] for item in TOOLS}

    def __init__(self):
        self.events = []

    def execute(self, call):
        from neuroshard.evolution import assistant_workspace as base

        try:
            if set(call) != {'name', 'arguments'} or call['name'] not in self.REGISTRY:
                raise ValueError('unknown tool')
            base._check_value(call['arguments'], self.REGISTRY[call['name']])
            result = base.Workspace._apply(base.Workspace.__new__(base.Workspace), call['name'], call['arguments'])
        except (ValueError, KeyError, OverflowError, TypeError):
            result = {'error': 'invalid tool arguments'}
        self.events.append({'call': copy.deepcopy(call), 'result': copy.deepcopy(result)})
        return result


RULE = re.compile(r'^(include|exclude|ends_with|max_words|bullets|json_keys):(.+)$|^lowercase$')


def rule_constraint(rule):
    """One ``check_text`` rule as the checker's constraint; the model writes the rules it reads in the request."""
    match = RULE.match(rule.strip())
    if not match:
        raise ValueError(f'unknown rule {rule!r}')
    if match.group(0) == 'lowercase':
        return {'kind': 'lowercase'}
    kind, value = match.group(1), match.group(2).strip().strip('"\'')
    if not value:
        raise ValueError('empty rule value')
    if kind in ('max_words', 'bullets'):
        return {'kind': kind, 'n': int(value)}
    if kind == 'json_keys':
        return {'kind': 'json_keys', 'keys': [k.strip() for k in value.split(',') if k.strip()]}
    if kind == 'ends_with':
        return {'kind': 'ends_with', 'phrase': value, 'markup': 'strip'}
    return {'kind': kind, 'word': value}


def violation(constraint, text):
    """What a draft gets wrong about one rule, specifically enough to revise it."""
    kind = constraint['kind']
    if kind == 'exclude':
        hits = re.findall(rf'\b{re.escape(constraint["word"])}\b', text, re.IGNORECASE)
        return f'the word "{constraint["word"]}" appears {len(hits)} time(s); rewrite those sentences without it'
    if kind == 'include':
        return f'the word "{constraint["word"]}" does not appear'
    if kind == 'ends_with':
        return f'the text must end with exactly "{constraint["phrase"]}" and nothing after it'
    if kind == 'max_words':
        return f'{chat_data.word_count(text)} words; the limit is {constraint["n"]}'
    if kind == 'bullets':
        return f'{len(chat_data.bullets(text))} bullet points; exactly {constraint["n"]} are required'
    if kind == 'json_keys':
        return f'the text must be only a JSON object whose keys are exactly {", ".join(constraint["keys"])}'
    return 'the text contains capital letters'


class ChatTools(Calculator):
    """Calculator plus ``check_text``: the model checks its own draft against the rules it read in the request."""

    TOOLS = [documents.TOOLS[2], {'type': 'function', 'function': {
        'name': 'check_text',
        'description': 'Check a draft reply against the user\'s requirements before sending it. Rules, one per item: '
                       'include:WORD, exclude:WORD, ends_with:PHRASE, max_words:N, bullets:N, json_keys:a,b,c, '
                       'lowercase. Returns every rule the draft breaks; revise and check again until it passes.',
        'parameters': {'type': 'object', 'properties': {'text': {'type': 'string'},
                                                         'rules': {'type': 'array', 'items': {'type': 'string'}}},
                       'required': ['text', 'rules'], 'additionalProperties': False}}}]
    REGISTRY = {item['function']['name']: item['function']['parameters'] for item in TOOLS}

    def execute(self, call):
        if not (isinstance(call, dict) and call.get('name') == 'check_text'):
            return super().execute(call)
        from neuroshard.evolution import assistant_workspace as base

        try:
            if set(call) != {'name', 'arguments'}:
                raise ValueError('unknown tool')
            base._check_value(call['arguments'], self.REGISTRY['check_text'])
            text, rules = call['arguments']['text'], call['arguments']['rules']
            if not 1 <= len(text) <= 8000 or not 1 <= len(rules) <= 6:
                raise ValueError('a draft and one to six rules')
            constraints = [rule_constraint(rule) for rule in rules]
            broken = [f'{rule}: {violation(c, text)}' for rule, c in zip(rules, constraints) if not chat_data.check(c, text)]
            result = {'passes': not broken, 'violations': broken}
        except (ValueError, KeyError, TypeError):
            result = {'error': 'invalid check_text arguments; rules look like exclude:the or max_words:40'}
        self.events.append({'call': copy.deepcopy(call), 'result': copy.deepcopy(result)})
        return result


def calculator_chat_episode(case, respond, policy):
    """Chat with tools (a calculator, and under chat-tools/1 also check_text) until a final reply, scored by the
    case's constraints, which the model never receives."""
    from neuroshard.evolution import assistant_workspace as base

    started = time.monotonic()
    kit = ChatTools if policy.get('interface') == 'chat-tools/1' else Calculator
    tool = kit()
    history = [{'role': 'system', 'content': policy['system_instruction']},
               {'role': 'user', 'content': case['turns'][0]['user']}]
    generations, final, terminated, used = [], '', False, 0
    for _ in range(policy['limits']['model_turns_per_user_turn']):
        generated = respond(copy.deepcopy(history), copy.deepcopy(kit.TOOLS))
        generations.append(generated)
        if not generated['terminated']:
            break
        history.append({'role': 'assistant', 'content': generated['text']})
        try:
            proposed = base.parse_calls(generated['text'], kit.REGISTRY)
        except (ValueError, TypeError, RecursionError):
            history.append({'role': 'tool', 'content': json.dumps({'error': 'invalid tool-call syntax or schema'})})
            continue
        if not proposed:
            final, terminated = generated['text'], True
            break
        if used + len(proposed) > policy['limits']['tool_calls_per_user_turn']:
            break
        for call in proposed:
            used += 1
            history.append({'role': 'tool', 'content': json.dumps(tool.execute(call), sort_keys=True)})
    return {'id': case['id'], 'family': case['family'], 'generations': generations, 'final_text': final,
            'terminated': terminated, 'calls': tool.events,
            'passed': chat_data.score(case['turns'][0]['expected'], final, terminated),
            'seconds': time.monotonic() - started}


def chat_episode(case, respond, policy):
    if policy.get('interface') in ('chat-calculator/1', 'chat-tools/1'):
        return calculator_chat_episode(case, respond, policy)
    started = time.monotonic()
    messages = [{'role': 'system', 'content': policy['system_instruction']},
                {'role': 'user', 'content': case['turns'][0]['user']}]
    generated = respond(copy.deepcopy(messages), None)
    text = generated['text'] if generated['terminated'] else ''
    return {'id': case['id'], 'family': case['family'], 'generations': [generated], 'final_text': generated['text'],
            'terminated': generated['terminated'],
            'passed': chat_data.score(case['turns'][0]['expected'], text, generated['terminated']),
            'seconds': time.monotonic() - started}


def rescore(cases, rows, kind):
    """Rescore recorded episodes from their outcomes alone; raises on any disagreement."""
    by_id = {case['id']: case for case in cases}
    for row in rows:
        case = by_id[row['id']]
        if kind == 'chat':
            passed = chat_data.score(case['turns'][0]['expected'], row['final_text'] if row['terminated'] else '',
                                     row['terminated'])
        else:
            version = row.get('version', 1)
            replay = documents.replay_transcript(case['world'], row['calls'], version)
            last = row['rounds'][-1]
            passed = (len(row['rounds']) == len(case['turns'])
                      and documents.score_round(replay, case['turns'][-1]['expected'], last['completed'],
                                                last['final_text'], version))
        if passed != row['passed']:
            raise ValueError(f'rescore differs: {row["id"]}')
    return sum(row['passed'] for row in rows)


def summary(rows):
    families = {}
    for row in rows:
        hit, total = families.get(row['family'], (0, 0))
        families[row['family']] = (hit + row['passed'], total + 1)
    return {'passed': sum(r['passed'] for r in rows), 'cases': len(rows),
            'families': {k: f'{h}/{t}' for k, (h, t) in sorted(families.items())}}
