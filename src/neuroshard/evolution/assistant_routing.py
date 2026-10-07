"""Conversations served turn by turn: each user turn by one learning unit with its own tool set.

A route is a responder and the policy it is served under: its instruction, tools and
limits. One calendar workspace runs every call and keeps the history, so a conversation
can switch routes at a follow-up. A route's model is shown, and may call, only its own
tools. The scorer replays the recorded responses route by route, with the same checks
as the single-route scorer.
"""

import copy
import json
import time

from neuroshard.evolution import assistant_calendar as calendar
from neuroshard.evolution import assistant_workflow as workflow
from neuroshard.evolution.assistant_workflow_data import public_case
from neuroshard.evolution.modular_reference_execution import identity


def workspace(policies):
    """The calendar interface whose workspace runs every route's calls: the free-slot one if any route offers it."""
    from neuroshard.evolution import assistant_calendar_slots as slots

    return slots if any(policy.get('interface') == slots.INTERFACE for policy in policies) else calendar


def execute(case, routes, select, *, _rescore=True):
    """``routes`` maps a name to (respond, policy); ``select(turn, user)`` names the route of each user turn."""
    public = public_case(case)
    world = workspace([policy for _, policy in routes.values()]).Workspace(public['world'])
    history, calls, rounds, generations = [], [], [], []
    started = time.monotonic()
    for turn, user in enumerate(public['user_turns']):
        choosing = time.monotonic()
        route = select(turn, user)
        selection = time.monotonic() - choosing
        respond, policy = routes[route]
        tools = workflow.interface(policy)
        history.append({'role': 'user', 'content': user})
        round_started = time.monotonic()
        final_text, completed, failure = '', False, None
        used_calls = 0
        for _ in range(policy['limits']['model_turns_per_user_turn']):
            messages = [{'role': 'system', 'content': policy['system_instruction']}] + history
            generated = respond(copy.deepcopy(messages), copy.deepcopy(tools.TOOLS))
            generated = {**generated, 'request_sha256': identity({'messages': messages, 'tools': tools.TOOLS})}
            generations.append(generated)
            if not generated['terminated']:
                failure = 'generation did not terminate within its token/input budget'
                break
            text = generated['text']
            history.append({'role': 'assistant', 'content': text})
            try:
                proposed = tools.parse_calls(text)
            except (ValueError, TypeError, RecursionError):
                history.append({'role': 'tool', 'content': json.dumps({'error': 'invalid tool-call syntax or schema'})})
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
                history.append({'role': 'tool', 'content': json.dumps(response, ensure_ascii=False, sort_keys=True)})
        if not completed and failure is None:
            failure = 'model-turn budget exhausted'
        rounds.append({'route': route, 'selection_seconds': selection, 'completed': completed, 'final_text': final_text,
                       'failure': failure, 'snapshot': world.snapshot(), 'call_count': len(calls),
                       'generation_count': len(generations), 'seconds': time.monotonic() - round_started})
        if failure:
            break
    result = {'id': case['id'], 'case_sha256': identity(case), 'public_input_sha256': identity(public),
              'policies_sha256': {name: identity(policy) for name, (_, policy) in sorted(routes.items())},
              'calls': calls, 'rounds': rounds, 'generations': generations, 'messages': history,
              'seconds': time.monotonic() - started, 'selection_seconds': sum(r['selection_seconds'] for r in rounds),
              'input_tokens': sum(len(g['input_token_ids']) for g in generations if g.get('executed', True)),
              'output_tokens': sum(len(g['token_ids']) for g in generations),
              'model_calls': sum(g.get('executed', True) for g in generations),
              'model_attempts': len(generations), 'tool_calls': len(calls), 'prose_quality_evaluated': False}
    if _rescore:
        result['score'] = score(case, result, {name: policy for name, (_, policy) in routes.items()})
    return result


def single(case, respond, policy):
    """One route for the whole conversation, in the calendar workspace."""
    return execute(case, {'single': (respond, policy)}, lambda turn, user: 'single')


def rollouts(jobs, respond, *, workers, progress=None):
    """Execute (case, policy, sample) jobs concurrently on one route each; each episode stays sequential."""
    from concurrent.futures import ThreadPoolExecutor
    import threading

    lock, done = threading.Lock(), [0]

    def run(job):
        case, policy, sample = job
        result = single(case, respond, policy)
        if progress:
            with lock:
                done[0] += 1
                progress(done[0], len(jobs))
        return {'case_id': case['id'], 'sample': sample, 'policy_sha256': identity(policy), 'result': result}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, jobs))


def score(case, result, policies):
    """Replay the recorded responses route by route, then score each round's outcome."""
    if (result['id'] != case['id'] or result['case_sha256'] != identity(case)
            or result['public_input_sha256'] != identity(public_case(case))
            or result['policies_sha256'] != {name: identity(policy) for name, policy in sorted(policies.items())}):
        raise ValueError('routed workflow binding differs')
    chosen = [row['route'] for row in result['rounds']]
    if not set(chosen) <= set(policies) or len(chosen) > len(case['turns']):
        raise ValueError('routed rounds name unknown routes or too many turns')
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

    def select(turn, user):
        if turn >= len(chosen):
            raise ValueError('a replayed conversation ran past its recorded rounds')
        return chosen[turn]

    routes = {name: (recorded_response, policy) for name, policy in policies.items()}
    reproduced = execute(case, routes, select, _rescore=False)
    if index != len(result['generations']):
        raise ValueError('unused model responses in transcript')
    if reproduced['calls'] != result['calls'] or reproduced['messages'] != result['messages']:
        raise ValueError('tool actions differ from recorded model responses')
    if (len(reproduced['rounds']) != len(result['rounds'])
            or any(any(left[key] != right[key] for key in ('route', 'completed', 'final_text', 'failure', 'snapshot',
                                                            'call_count', 'generation_count'))
                   for left, right in zip(reproduced['rounds'], result['rounds']))):
        raise ValueError('workspace rounds differ from model/tool transcript')
    successes = []
    previous_calls = previous_generations = 0
    for turn, row in enumerate(result['rounds']):
        policy = policies[row['route']]
        count, generations = row['call_count'], row['generation_count']
        if (not previous_calls <= count <= len(result['calls'])
                or not previous_generations < generations <= len(result['generations'])
                or count - previous_calls > policy['limits']['tool_calls_per_user_turn']
                or generations - previous_generations > policy['limits']['model_turns_per_user_turn']):
            raise ValueError('invalid workflow counters or budget')
        replay = workspace(policies.values()).replay_transcript(case['world'], result['calls'][:count])
        if identity(replay) != identity(row['snapshot']):
            raise ValueError('workspace state does not match tool transcript')
        last = result['generations'][generations - 1]
        if row['completed'] and (not last['terminated'] or row['final_text'] != last['text']
                                 or workflow.interface(policy).parse_calls(last['text']) or row['failure']):
            raise ValueError('workflow completion differs from model response')
        successes.append(calendar.score_round(replay, case['turns'][turn]['expected'], row['completed'], row['final_text']))
        previous_calls, previous_generations = count, generations
    if (previous_calls != len(result['calls']) or previous_generations != len(result['generations'])
            or result['model_attempts'] != len(result['generations'])
            or result['model_calls'] != sum(g.get('executed', True) for g in result['generations'])
            or result['tool_calls'] != len(result['calls'])
            or result['input_tokens'] != sum(len(g['input_token_ids']) for g in result['generations'] if g.get('executed', True))
            or result['output_tokens'] != sum(len(g['token_ids']) for g in result['generations'])):
        raise ValueError('workflow accounting differs')
    return {'round_successes': successes, 'passed': len(successes) == len(case['turns']) and all(successes),
            'routes': chosen, 'prose_quality_evaluated': False, 'external_effects': False}


def turn_ids(tokenizer, policy, user):
    """Token IDs of one user message rendered alone after the policy's instruction and tools."""
    messages = [{'role': 'system', 'content': policy['system_instruction']}, {'role': 'user', 'content': user}]
    prompt = tokenizer.apply_chat_template(messages, tools=workflow.interface(policy).TOOLS, add_generation_prompt=True,
                                           tokenize=False)
    return tokenizer(prompt, add_special_tokens=False)['input_ids']


def turn_feature(model, tokenizer, policy, user, device):
    """Frozen parent final-layer state at the generation boundary after one user message."""
    import torch

    ids = torch.tensor([turn_ids(tokenizer, policy, user)], device=device)
    with torch.no_grad():
        return model.model(input_ids=ids).last_hidden_state[0, -1].float().cpu().tolist()


def message_prefix(model, tokenizer, policy, device):
    """The tokens every turn shares, the instruction, tools and user header, run once into a key/value cache."""
    import torch

    first, second = turn_ids(tokenizer, policy, 'Schedule'), turn_ids(tokenizer, policy, 'Create')
    common = 0
    for left, right in zip(first, second):
        if left != right:
            break
        common += 1
    if not 0 < common < min(len(first), len(second)):
        raise ValueError('turns share no prefix before the user message')
    ids = first[:common]
    with torch.no_grad():
        cache = model.model(input_ids=torch.tensor([ids], device=device), use_cache=True).past_key_values
    return {'ids': ids, 'cache': cache}


def message_feature(model, tokenizer, policy, user, device, prefix):
    """Mean frozen-parent final-layer state over one user message and the reply header, after the cached prefix."""
    import torch

    ids = turn_ids(tokenizer, policy, user)
    if ids[:len(prefix['ids'])] != prefix['ids'] or len(ids) == len(prefix['ids']):
        raise ValueError('a turn does not extend the shared prefix')
    try:
        with torch.no_grad():
            states = model.model(input_ids=torch.tensor([ids[len(prefix['ids']):]], device=device),
                                 past_key_values=prefix['cache'], use_cache=True).last_hidden_state[0]
        return states.float().mean(dim=0).cpu().tolist()
    finally:
        prefix['cache'].crop(len(prefix['ids']))


def turn_key(case_id, turn):
    return f'{case_id}#{turn}'


def turn_targets(drafting, scheduling, turns, failed_ties=True):
    """Per-turn targets from single-route outcomes: 1 selects scheduling, 0 keeps the incumbent drafting route.

    A turn selects scheduling only where that route succeeded more often, weighted by the
    difference; a tie counts for the drafting route with the weight of one episode. With
    ``failed_ties`` false, a turn that both routes always failed is no example at all.
    ``drafting`` and ``scheduling`` map a case to its runs' round successes; ``turns`` to its user-turn count.
    """
    rows = {}
    for case_id in sorted(turns):
        runs = (drafting[case_id], scheduling[case_id])
        if not all(runs):
            raise ValueError('a route has no integration outcome for a case')
        for turn in range(turns[case_id]):
            rates = [sum(bool(run[turn]) if turn < len(run) else False for run in outcomes) / len(outcomes)
                     for outcomes in runs]
            if not failed_ties and not any(rates):
                continue
            difference = rates[1] - rates[0]
            weight = abs(difference) or 1 / max(len(outcomes) for outcomes in runs)
            rows[turn_key(case_id, turn)] = (1.0 if difference > 0 else 0.0, weight)
    return rows
