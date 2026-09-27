"""Verified self-experience for the workspace assistant.

Rollouts come from the model being trained, in the deterministic workspace.
The frozen scorer replays each transcript against the recorded responses, so a
trajectory is accepted on its outcome, not on trust in whoever produced it.
Coached rollouts are stored without their card. Only training goals are read.
"""

import copy
import json

from neuroshard.evolution import assistant_workflow as workflow
from neuroshard.evolution import assistant_workspace as sandbox
from neuroshard.evolution.modular_reference_execution import identity

TRAINING_SPLITS = ('train',)


def coached(policy, card):
    if not isinstance(card, str) or not card.strip():
        raise ValueError('coaching card must be non-empty text')
    return {**copy.deepcopy(policy), 'system_instruction': policy['system_instruction'] + '\n\n' + card}


def rejected_turns(messages):
    """Assistant messages whose envelope or any call produced a tool error."""
    rejected = set()
    for index, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        following = index + 1
        while following < len(messages) and messages[following]['role'] == 'tool':
            try:
                if 'error' in json.loads(messages[following]['content']):
                    rejected.add(index)
            except (ValueError, TypeError):
                rejected.add(index)
            following += 1
    return rejected


def trajectory(case, result, executed_policy, train_policy, *, sample, coaching=False):
    """Leading verified rounds as a trainable conversation, or None when round one fails."""
    if case['split'] not in TRAINING_SPLITS:
        raise ValueError('experience may only be collected on training goals')
    if coaching != (executed_policy['system_instruction'] != train_policy['system_instruction']):
        raise ValueError('coaching flag does not match the executed instruction')
    score = workflow.score(case, result, executed_policy)
    rounds = 0
    for success in score['round_successes']:
        if not success:
            break
        rounds += 1
    if not rounds:
        return None
    messages = copy.deepcopy(result['messages'])
    users = [i for i, m in enumerate(messages) if m['role'] == 'user']
    messages = messages[:users[rounds] if rounds < len(users) else len(messages)]
    if messages[0] != {'role': 'system', 'content': executed_policy['system_instruction']}:
        raise ValueError('transcript does not begin with the executed instruction')
    messages[0] = {'role': 'system', 'content': train_policy['system_instruction']}
    rejected = rejected_turns(messages)
    trainable = [m['role'] == 'assistant' and i not in rejected for i, m in enumerate(messages)]
    generations = sum(m['role'] == 'assistant' for m in messages)
    return {'case_id': case['id'], 'family': case['family'], 'sample': sample, 'coached': coaching,
            'rounds': rounds, 'complete': rounds == len(case['turns']), 'messages': messages,
            'trainable': trainable, 'model_calls': generations, 'rejected_turns': len(rejected),
            'tools_sha256': identity(sandbox.TOOLS), 'executed_policy_sha256': identity(executed_policy),
            'train_policy_sha256': identity(train_policy), 'transcript_sha256': identity(result['messages'])}


def assistant_texts(item):
    return [m['content'] for m in item['messages'] if m['role'] == 'assistant']


def select(trajectories, per_case):
    """Distinct accepted trajectories: more verified rounds, then fewer model calls, then sample."""
    chosen, seen = {}, set()
    for item in sorted(trajectories, key=lambda t: (t['case_id'], -t['rounds'], t['model_calls'],
                                                     t['coached'], t['sample'])):
        key = identity([item['case_id'], assistant_texts(item)])
        if key in seen or len(chosen.setdefault(item['case_id'], [])) >= per_case:
            continue
        seen.add(key)
        chosen[item['case_id']].append(item)
    return [item for case_id in sorted(chosen) for item in chosen[case_id]]


def needs_coaching(case, accepted):
    """A training case without an uncoached trajectory that completes every round."""
    return not any(t['case_id'] == case['id'] and t['complete'] and not t['coached'] for t in accepted)


def near_policy(trajectories, nll):
    """Keep coached trajectories no less likely under the parent than every natural success."""
    natural = [nll[t['transcript_sha256']] for t in trajectories if not t['coached']]
    if not natural:
        return [t for t in trajectories if not t['coached']], None
    ceiling = max(natural)
    return [t for t in trajectories if not t['coached'] or nll[t['transcript_sha256']] <= ceiling], ceiling


def summary(cases, attempts, accepted):
    by_family = {}
    for case in cases:
        rows = [t for t in accepted if t['case_id'] == case['id']]
        family = by_family.setdefault(case['family'], {'cases': 0, 'complete': 0, 'coached_only': 0})
        family['cases'] += 1
        family['complete'] += any(t['complete'] for t in rows)
        family['coached_only'] += bool(rows) and all(t['coached'] for t in rows)
    return {'cases': len(cases), 'rollouts': attempts, 'accepted': len(accepted),
            'complete': sum(t['complete'] for t in accepted), 'coached': sum(t['coached'] for t in accepted),
            'cases_without_experience': sorted(c['id'] for c in cases if not any(t['case_id'] == c['id'] for t in accepted)),
            'by_family': by_family}
