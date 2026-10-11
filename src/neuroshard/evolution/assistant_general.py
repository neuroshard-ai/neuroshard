"""Episodes for the general route: the user's documents with tools, and tool-free chat with checkable constraints.

``documents_episode`` runs a model through the documents workspace exactly as the routed workspace
runner does: up to the policy's model turns, at most two native tool calls per response, the tool
budget, and a final reply. ``chat_episode`` is one response, scored by its constraints. Both record
every generation, so a transcript can be rescored without the model.
"""

import copy
import json
import time

from neuroshard.evolution import assistant_chat_data as chat_data
from neuroshard.evolution import assistant_documents as documents
from neuroshard.evolution.assistant_documents_data import public_case
from neuroshard.evolution.modular_reference_execution import identity


def documents_episode(case, respond, policy):
    public = public_case(case)
    world = documents.Workspace(public['world'])
    history = [{'role': 'system', 'content': policy['system_instruction']}]
    calls, rounds, generations = [], [], []
    started = time.monotonic()
    for turn, user in enumerate(public['user_turns']):
        history.append({'role': 'user', 'content': user})
        final, completed, failure, used = '', False, None, 0
        for _ in range(policy['limits']['model_turns_per_user_turn']):
            generated = respond(copy.deepcopy(history), copy.deepcopy(documents.TOOLS))
            generations.append({**generated, 'request_sha256': identity({'messages': history, 'tools': documents.TOOLS})})
            if not generated['terminated']:
                failure = 'generation did not terminate within its token/input budget'
                break
            history.append({'role': 'assistant', 'content': generated['text']})
            try:
                proposed = documents.parse_calls(generated['text'])
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
                       'passed': documents.score_round(snapshot, case['turns'][turn]['expected'], completed, final)})
        if failure:
            break
    return {'id': case['id'], 'family': case['family'], 'rounds': rounds, 'calls': calls, 'generations': generations,
            'passed': len(rounds) == len(case['turns']) and all(r['passed'] for r in rounds),
            'seconds': time.monotonic() - started, 'tool_calls': len(calls),
            'model_calls': sum(g.get('executed', True) for g in generations)}


def chat_episode(case, respond, policy):
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
            replay = documents.replay_transcript(case['world'], row['calls'])
            last = row['rounds'][-1]
            passed = (len(row['rounds']) == len(case['turns'])
                      and documents.score_round(replay, case['turns'][-1]['expected'], last['completed'],
                                                last['final_text']))
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
