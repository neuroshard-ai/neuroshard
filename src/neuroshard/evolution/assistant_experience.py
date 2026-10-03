"""Verified self-experience for the workspace assistant.

Rollouts come from the model being trained, in the deterministic workspace.
The frozen scorer replays each transcript against the recorded responses, so a
trajectory is accepted on its outcome, not on trust in whoever produced it.
Coached rollouts are stored without their card. Only training goals are read.
"""

import copy
import json
import re

from neuroshard.evolution import assistant_workflow as workflow
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution import assistant_workspace as sandbox
from neuroshard.evolution.modular_reference_execution import identity

TRAINING_SPLITS = data.TRAINING


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


def first_read(messages):
    """Index and text of the first assistant message that reads a document."""
    for index, message in enumerate(messages):
        if message['role'] == 'assistant':
            try:
                calls = sandbox.parse_calls(message['content'])
            except (ValueError, TypeError):
                continue
            if any(call['name'] == 'read_document' for call in calls):
                return index, calls
    return None, None


def decision_pairs(case, rollouts, policy, per_case):
    """Verified version-choice preferences from natural rollouts of one training case.

    Chosen: the first read opens the latest approved revision and round one passes.
    Rejected: the first read opens an older approved revision and round one fails.
    Both share every earlier message, so the pair isolates that one decision.
    """
    if case['split'] not in TRAINING_SPLITS:
        raise ValueError('preferences may only be built from training goals')
    project = case['turns'][0]['expected']['project']
    approved = [d for d in case['world']['documents'] if d['project'] == project and d['status'] == 'approved']
    latest = max(approved, key=lambda d: d['revision'])['id']
    chosen, rejected = {}, {}
    for row in sorted(rollouts, key=lambda r: r['sample']):
        if row['case_id'] != case['id'] or row['policy_sha256'] != identity(policy):
            continue
        result = row['result']
        if workflow.score(case, result, policy) != result['score']:
            raise ValueError('rollout outcome does not re-verify')
        index, calls = first_read(result['messages'])
        if index is None:
            continue
        opened = {call['arguments']['document_id'] for call in calls if call['name'] == 'read_document'}
        prefix, text = result['messages'][:index], result['messages'][index]['content']
        key = identity(prefix)
        if opened == {latest} and result['score']['round_successes'][0]:
            chosen.setdefault(key, {}).setdefault(text, row['sample'])
        elif opened and latest not in opened and opened <= {d['id'] for d in approved} \
                and not result['score']['round_successes'][0]:
            rejected.setdefault(key, {}).setdefault(text, (row['sample'], prefix))
    pairs = []
    for key in sorted(set(chosen) & set(rejected)):
        for good, good_sample in sorted(chosen[key].items(), key=lambda item: item[1]):
            for bad, (bad_sample, prefix) in sorted(rejected[key].items(), key=lambda item: item[1][0]):
                pairs.append({'case_id': case['id'], 'messages': copy.deepcopy(prefix), 'chosen': good,
                              'rejected': bad, 'samples': [good_sample, bad_sample]})
    return pairs[:per_case]


def decision(message):
    """What a message decides: free-text replies are equivalent, tool calls compare by their parsed calls."""
    if message['role'] != 'assistant':
        return message['role'], message['content']
    try:
        calls = sandbox.parse_calls(message['content'])
    except (ValueError, TypeError):
        return 'invalid', message['content']
    return ('calls', identity(calls)) if calls else ('reply',)


def divergence_pairs(case, rollouts, policy, per_case, *, aligned=False):
    """Verified preferences at the first message where a success and a failure of one case differ.

    The chosen message is a valid tool call from a fully successful rollout; the
    rejected message comes from a rollout whose round containing it failed. By
    default both share every earlier message exactly. ``aligned`` compares
    decisions instead, so differently worded replies do not end the shared
    prefix; the success's messages then form the context for both continuations.
    """
    if case['split'] not in TRAINING_SPLITS:
        raise ValueError('preferences may only be built from training goals')
    rows = [r for r in rollouts if r['case_id'] == case['id'] and r['policy_sha256'] == identity(policy)]
    for row in rows:
        if workflow.score(case, row['result'], policy) != row['result']['score']:
            raise ValueError('rollout outcome does not re-verify')
    successes = [r for r in rows if r['result']['score']['passed']]
    failures = [r for r in rows if not r['result']['score']['passed']]
    pairs = {}
    for good in successes:
        for bad in failures:
            left, right = good['result']['messages'], bad['result']['messages']
            same = (lambda a, b: decision(a) == decision(b)) if aligned else (lambda a, b: a == b)
            index = next((i for i, (a, b) in enumerate(zip(left, right)) if not same(a, b)), None)
            if index is None or left[index]['role'] != 'assistant' or right[index]['role'] != 'assistant':
                continue
            try:
                if not sandbox.parse_calls(left[index]['content']):
                    continue
            except (ValueError, TypeError):
                continue
            turn = sum(m['role'] == 'user' for m in left[:index]) - 1
            if bad['result']['score']['round_successes'][turn:turn + 1] != [False]:
                continue
            key = identity([left[:index], left[index]['content'], right[index]['content']])
            pairs.setdefault(key, {'case_id': case['id'], 'messages': copy.deepcopy(left[:index]), 'turn': turn,
                                   'chosen': left[index]['content'], 'rejected': right[index]['content'],
                                   'samples': [good['sample'], bad['sample']]})
    return sorted(pairs.values(), key=lambda p: (p['turn'], len(p['messages']), p['samples']))[:per_case]


def envelope(calls):
    return '\n'.join('<tool_call>\n' + json.dumps({'name': c['name'], 'arguments': c['arguments']}) + '\n</tool_call>'
                     for c in calls)


def wrong_read(case, result):
    """First message that reads a document outside its round's goal sources, with a repaired call.

    Returns (message index, generation index, repaired text) or None. Training goals
    only locate the error; the repair reads goal sources not yet read in that round.
    """
    if case['split'] not in TRAINING_SPLITS:
        raise ValueError('repairs may only use training goals')
    turn, read, generation = -1, set(), 0
    for index, message in enumerate(result['messages']):
        if message['role'] == 'user':
            turn, read = turn + 1, set()
            continue
        if message['role'] != 'assistant':
            continue
        generation += 1
        try:
            calls = sandbox.parse_calls(message['content'])
        except (ValueError, TypeError):
            continue
        opened = [c['arguments']['document_id'] for c in calls if c['name'] == 'read_document']
        goal = sorted(case['turns'][turn]['expected']['source_ids'])
        if any(d not in goal for d in opened):
            missing = [d for d in goal if d not in read and d not in opened]
            repaired = []
            for call in calls:
                if call['name'] == 'read_document' and call['arguments']['document_id'] not in goal:
                    if not missing:
                        return None
                    call = {'name': 'read_document', 'arguments': {'document_id': missing.pop(0)}}
                repaired.append(call)
            return index, generation - 1, envelope(repaired)
        read.update(opened)
    return None


def plan_start(document):
    """The start date a delivery-plan document states."""
    found = re.search(r'starts on (\d{4}-\d{2}-\d{2})', document['content'])
    return found.group(1) if found else None


def wrong_date_base(case, result):
    """First shift_date call that starts from a date its round cannot justify, with a repaired call.

    Returns (message index, generation index, repaired text) or None. A base is
    justified if it is the start date of one of the round's goal sources, a date an
    earlier shift_date returned, or the goal due date of an earlier round. The repair
    starts the same shift from the round's goal-source start date. Training goals only
    locate the error.
    """
    if case['split'] not in TRAINING_SPLITS:
        raise ValueError('repairs may only use training goals')
    documents = {d['id']: d for d in case['world']['documents']}
    turn, generation, returned, earlier = -1, 0, set(), set()
    for index, message in enumerate(result['messages']):
        if message['role'] == 'user':
            turn += 1
            if turn:
                earlier.add(case['turns'][turn - 1]['expected']['due_date'])
            continue
        if message['role'] == 'tool':
            try:
                value = json.loads(message['content'])
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict) and isinstance(value.get('date'), str):
                returned.add(value['date'])
            continue
        if message['role'] != 'assistant':
            continue
        generation += 1
        try:
            calls = sandbox.parse_calls(message['content'])
        except (ValueError, TypeError):
            continue
        starts = {plan_start(documents[d]) for d in case['turns'][turn]['expected']['source_ids'] if d in documents}
        allowed = starts | returned | earlier
        for position, call in enumerate(calls):
            if call['name'] == 'shift_date' and call['arguments'].get('start_date') not in allowed:
                if len(starts) != 1 or None in starts:
                    return None
                repaired = [dict(c, arguments={**c['arguments'], 'start_date': next(iter(starts))}) if i == position else c
                            for i, c in enumerate(calls)]
                return index, generation - 1, envelope(repaired)
    return None


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
