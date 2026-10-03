"""Complete multi-turn assistant execution and independent deterministic scoring.

The responder receives only conversation and tool definitions. Tool execution
receives only the public workspace. Expected outcomes are read after execution.
This runtime is reusable by frozen, updated and expanded model candidates.
"""

import copy
import json
import time

from neuroshard.evolution import assistant_workspace as sandbox
from neuroshard.evolution.assistant_workflow_data import public_case
from neuroshard.evolution.modular_reference_execution import identity


def execute(case, respond, policy, *, _rescore=True):
    public = public_case(case)
    world = sandbox.Workspace(public['world'])
    messages = [{'role': 'system', 'content': policy['system_instruction']}]
    calls, rounds, generations = [], [], []
    started = time.monotonic()
    for user in public['user_turns']:
        messages.append({'role': 'user', 'content': user})
        round_started = time.monotonic()
        final_text, completed, failure = '', False, None
        used_calls = 0
        for _ in range(policy['limits']['model_turns_per_user_turn']):
            generated = respond(copy.deepcopy(messages), copy.deepcopy(sandbox.TOOLS))
            generated = {**generated, 'request_sha256': identity({'messages': messages, 'tools': sandbox.TOOLS})}
            generations.append(generated)
            if not generated['terminated']:
                failure = 'generation did not terminate within its token/input budget'
                break
            text = generated['text']
            messages.append({'role': 'assistant', 'content': text})
            try:
                proposed = sandbox.parse_calls(text)
            except (ValueError, TypeError, RecursionError):
                messages.append({'role': 'tool', 'content': json.dumps({'error': 'invalid tool-call syntax or schema'})})
                continue
            if not proposed:
                final_text, completed = text, bool(text.strip())
                break
            if used_calls + len(proposed) > policy['limits']['tool_calls_per_user_turn']:
                failure = 'tool-call budget exhausted'
                break
            for call in proposed:
                response = world.execute(call)
                calls.append({'call': call, 'result': response})
                used_calls += 1
                messages.append({'role': 'tool', 'content': json.dumps(response, ensure_ascii=False, sort_keys=True)})
        if not completed and failure is None:
            failure = 'model-turn budget exhausted'
        rounds.append({'completed': completed, 'final_text': final_text, 'failure': failure,
                       'snapshot': world.snapshot(), 'call_count': len(calls),
                       'generation_count': len(generations), 'seconds': time.monotonic() - round_started})
        if failure:
            break
    result = {'id': case['id'], 'case_sha256': identity(case), 'public_input_sha256': identity(public),
              'policy_sha256': identity(policy), 'calls': calls, 'rounds': rounds, 'generations': generations,
              'messages': messages, 'seconds': time.monotonic() - started,
              'input_tokens': sum(len(g['input_token_ids']) for g in generations if g.get('executed', True)),
              'output_tokens': sum(len(g['token_ids']) for g in generations),
              'model_calls': sum(g.get('executed', True) for g in generations),
              'model_attempts': len(generations), 'tool_calls': len(calls), 'prose_quality_evaluated': False}
    if _rescore:
        result['score'] = score(case, result, policy)
    return result


def score(case, result, policy):
    if (result['id'] != case['id'] or result['case_sha256'] != identity(case)
            or result['public_input_sha256'] != identity(public_case(case))
            or result['policy_sha256'] != identity(policy)):
        raise ValueError('assistant workflow binding differs')
    index = 0

    def recorded_response(messages, tools):
        nonlocal index
        if index >= len(result['generations']):
            raise ValueError('missing recorded model response')
        generated = result['generations'][index]
        index += 1
        if generated['request_sha256'] != identity({'messages': messages, 'tools': tools}):
            raise ValueError('model request does not match conversation and tool state')
        return generated

    reproduced = execute(case, recorded_response, policy, _rescore=False)
    if index != len(result['generations']):
        raise ValueError('unused model responses in transcript')
    if reproduced['calls'] != result['calls'] or reproduced['messages'] != result['messages']:
        raise ValueError('tool actions differ from recorded model responses')
    if (len(reproduced['rounds']) != len(result['rounds'])
            or any(any(left[key] != right[key] for key in ('completed', 'final_text', 'failure',
                        'snapshot', 'call_count', 'generation_count'))
                   for left, right in zip(reproduced['rounds'], result['rounds']))):
        raise ValueError('workspace rounds differ from model/tool transcript')
    if len(result['rounds']) > len(case['turns']):
        raise ValueError('too many conversation rounds')
    successes = []
    previous_calls = previous_generations = 0
    for index, turn in enumerate(result['rounds']):
        count, generations = turn['call_count'], turn['generation_count']
        if (not previous_calls <= count <= len(result['calls'])
                or not previous_generations < generations <= len(result['generations'])
                or count - previous_calls > policy['limits']['tool_calls_per_user_turn']
                or generations - previous_generations > policy['limits']['model_turns_per_user_turn']):
            raise ValueError('invalid workflow counters or budget')
        replay = sandbox.replay_transcript(case['world'], result['calls'][:count])
        if identity(replay) != identity(turn['snapshot']):
            raise ValueError('workspace state does not match tool transcript')
        last = result['generations'][generations - 1]
        if turn['completed'] and (not last['terminated'] or turn['final_text'] != last['text']
                                  or sandbox.parse_calls(last['text']) or turn['failure']):
            raise ValueError('workflow completion differs from model response')
        successes.append(sandbox.score_round(replay, case['turns'][index]['expected'],
                                               turn['completed'], turn['final_text']))
        previous_calls, previous_generations = count, generations
    if (previous_calls != len(result['calls']) or previous_generations != len(result['generations'])
            or result['model_attempts'] != len(result['generations'])
            or result['model_calls'] != sum(g.get('executed', True) for g in result['generations'])
            or result['tool_calls'] != len(result['calls'])
            or result['input_tokens'] != sum(len(g['input_token_ids']) for g in result['generations'] if g.get('executed', True))
            or result['output_tokens'] != sum(len(g['token_ids']) for g in result['generations'])):
        raise ValueError('workflow accounting differs')
    return {'round_successes': successes, 'passed': len(successes) == len(case['turns']) and all(successes),
            'prose_quality_evaluated': False, 'external_effects': False}


def replay_matches(first, second):
    return (first['calls'] == second['calls'] and first['score'] == second['score']
            and len(first['rounds']) == len(second['rounds'])
            and all(all(a[k] == b[k] for k in ('completed', 'final_text', 'snapshot', 'failure'))
                    for a, b in zip(first['rounds'], second['rounds']))
            and len(first['generations']) == len(second['generations'])
            and all(all(a[k] == b[k] for k in ('prompt_sha256', 'input_token_ids', 'token_ids', 'text', 'terminated'))
                    for a, b in zip(first['generations'], second['generations'])))
