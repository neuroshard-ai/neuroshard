"""The network decides what to learn next: sealed probes of the serving version, and recipes for its weak spots.

``probe`` runs the serving model on a beacon-sealed evaluation (``sealed_evaluation``) and reports its pass
rate per family. Any agent can run it, at any drand round, and anyone can recompute the same report.

``weakest`` picks the families below a target rate; ``recipe`` turns them into a training recipe:
verified demonstrations of fresh beacon-seeded cases of exactly those families (the reference solver
writes and checks every one), mixed with the serving model's own replay so earlier behaviour is kept,
on the route's unit arm. The recipe and its start state are what ``recipe_propose`` names; owners vote it
into the growth queue, and the growth pool opens it as a job when usage has paid for it.

The probe's cases and the recipe's cases come from different beacon rounds and different purposes, so a
recipe never trains on the cases that measured the weakness, nor on the cases that will seal its job.
"""

import hashlib
import json
import random

from neuroshard.evolution import sealed_evaluation as sealed


def probe(run, cases, route):
    """Pass rate per family of the serving model on one sealed set: ``run(case) -> passed``."""
    by = {}
    for case in cases['sets'][route]:
        hit, total = by.get(case['family'], (0, 0))
        by[case['family']] = (hit + bool(run(case)), total + 1)
    return {'round': cases['round'], 'cases_sha256': cases['sha256'], 'route': route,
            'families': {family: {'passed': h, 'cases': t, 'rate': h / t} for family, (h, t) in sorted(by.items())}}


def weakest(report, target=0.9, limit=3):
    """The families below ``target``, weakest first; ties by name, so every agent picks the same ones."""
    rows = sorted((v['rate'], name) for name, v in report['families'].items() if v['rate'] < target)
    return [name for _, name in rows[:limit]]


def demonstrations(families, randomness, per_family=60, purpose='recipe'):
    """Fresh documents cases of the given families, seeded for training (never the probe's or a seal's cases)."""
    from neuroshard.evolution import assistant_documents_data as docs

    pool, out = sealed.templates('documents'), []
    for family in families:
        for index in range(per_family):
            rng = random.Random(sealed.seed_of(randomness, f'{purpose}:{family}:{index}'))
            world, facts = docs._world(rng, purpose, f'{family}-{index}')
            expected = docs._kinds(facts)[family][2]
            template = pool[family][rng.randrange(len(pool[family]))]
            out.append({'id': f'docs-{purpose}-{family}-{index:02d}', 'split': 'generated-train', 'family': family,
                        'world': world, 'turns': [{'user': docs._fill(template, facts), 'expected': expected}]})
    return out


def recipe(report, randomness, policy, spec, model_root, encode, per_family=60, replay_per_family=8, target=0.9):
    """A training recipe for the documents route's weak families, every demonstration solved and checked.

    Experience is ``per_family`` fresh cases of each weak family; replay is ``replay_per_family`` fresh cases of
    every other family, so the unit keeps what already works. ``encode(trajectory) -> sequence`` tokenizes under
    the route's tools. Returns None if nothing is below target: a strong network does not grow for its own sake.
    """
    families = weakest(report, target)
    if not families:
        return None
    others = sorted(set(report['families']) - set(families))
    return {'format': 'neuroshard-growth-recipe/1', 'route': 'documents', 'families': families, 'others': others,
            'probe': {'round': report['round'], 'cases_sha256': report['cases_sha256']},
            'randomness': randomness, 'per_family': per_family, 'replay_per_family': replay_per_family,
            'arm': 'addition', 'spec': spec, 'model_root': model_root, 'base': 'granite-parent',
            'sources': {'experience': recipe_from(families, randomness, policy, encode, per_family),
                        'replay': recipe_from(others, randomness, policy, encode, replay_per_family, 'replay')},
            'derived_from': hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()}


# Chat: no solver writes a chat reply, so experience is the serving model's own replies on fresh cases, kept
# only when the deterministic constraint checker passes them (and every tool result recomputes). The recipe
# carries that evidence unencoded, so an owner regenerates each case from the beacon, rescores each reply,
# re-executes each tool call, and re-encodes it to the exact training sequence before voting yes.

def chat_cases(families, randomness, per_family=60, purpose='recipe'):
    """Fresh chat cases of the given kinds, seeded for training (never a probe's or a seal's cases)."""
    from neuroshard.evolution import assistant_chat_data as chat_data

    pool, out = sealed.templates('chat'), []
    for family in families:
        for index in range(per_family):
            rng = random.Random(sealed.seed_of(randomness, f'{purpose}-chat:{family}:{index}'))
            topic = sealed.TOPIC_WORDS[rng.randrange(len(sealed.TOPIC_WORDS))]
            rows = {k: [v[rng.randrange(len(v))]] for k, v in pool.items()}
            text, constraints = chat_data._generated_task(family, rng, topic, rows, index)
            out.append({'id': f'chat-{purpose}-{family}-{index:02d}', 'split': 'generated-train', 'family': family,
                        'turns': [{'user': text, 'expected': {'constraints': constraints}}]})
    return out


def chat_item(case, row, policy):
    """The training item of one passing chat episode: system, user, then every assistant and tool message."""
    from neuroshard.evolution import assistant_general as general
    from neuroshard.evolution import assistant_workspace as base

    kit = general.ChatTools if policy.get('interface') == 'chat-tools/1' else general.Calculator
    messages = [{'role': 'system', 'content': policy['system_instruction']},
                {'role': 'user', 'content': case['turns'][0]['user']}]
    calls = iter(row.get('calls', []))
    for generated in row['generations']:
        messages.append({'role': 'assistant', 'content': generated['text']})
        for _ in base.parse_calls(generated['text'], kit.REGISTRY) if row.get('calls') is not None else ():
            messages.append({'role': 'tool', 'content': json.dumps(next(calls)['result'], sort_keys=True)})
    return {'case_id': case['id'], 'family': case['family'], 'messages': messages,
            'trainable': [m['role'] == 'assistant' for m in messages]}


def chat_experience(cases, responders, policy):
    """For each case, the first passing episode over ``responders`` (greedy first, then sampled), as items."""
    from neuroshard.evolution import assistant_general as general

    items = []
    for case in cases:
        for make in responders:
            row = general.chat_episode(case, make(case), policy)
            if row['passed']:
                try:
                    items.append(chat_item(case, row, policy))
                except (ValueError, TypeError, RecursionError, StopIteration):
                    continue
                break
    return items


def chat_recipe(report, randomness, policy, spec, model_root, encode, responders, per_family=60, replay_per_family=8,
                target=0.9, minimum_items=8):
    """A chat recipe from the serving model's own checked replies: weak kinds as experience, the others as replay."""
    families = weakest(report, target)
    if not families:
        return None
    others = sorted(set(report['families']) - set(families))
    evidence = chat_experience(chat_cases(families, randomness, per_family), responders, policy)
    if len(evidence) < minimum_items:
        return None  # too few verified replies to learn from: wait for a better model or a later round
    kept = chat_experience(chat_cases(others, randomness, replay_per_family, 'replay'), responders, policy)
    return {'format': 'neuroshard-growth-recipe/1', 'route': 'chat', 'families': families, 'others': others,
            'probe': {'round': report['round'], 'cases_sha256': report['cases_sha256']},
            'randomness': randomness, 'per_family': per_family, 'replay_per_family': replay_per_family,
            'evidence': {'experience': evidence, 'replay': kept},
            'arm': 'addition', 'spec': spec, 'model_root': model_root, 'base': 'granite-parent',
            'sources': {'experience': [encode(item) for item in evidence], 'replay': [encode(item) for item in kept]},
            'derived_from': hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()}


def verify_chat(recipe, policy, encode):
    """An owner's check of a chat recipe: every item is a fresh case of a named family, its reply passes the
    checker, every tool result recomputes, and it encodes to exactly the training sequence the recipe names."""

    if not isinstance(recipe.get('evidence'), dict) or set(recipe['evidence']) != {'experience', 'replay'}:
        return False
    for part, families, count, purpose in (('experience', recipe['families'], recipe['per_family'], 'recipe'),
                                           ('replay', recipe['others'], recipe['replay_per_family'], 'replay')):
        cases = {c['id']: c for c in chat_cases(families, recipe['randomness'], count, purpose)}
        if not chat_items_hold(recipe['evidence'][part], recipe['sources'][part], cases, policy, encode):
            return False
    return True


def chat_items_hold(evidence, sequences, cases, policy, encode):
    from neuroshard.evolution import assistant_chat_data as chat_data
    from neuroshard.evolution import assistant_general as general
    from neuroshard.evolution import assistant_workspace as base

    if len(evidence) != len(sequences) or len({e['case_id'] for e in evidence}) != len(evidence):
        return False
    tools = policy.get('interface') in ('chat-calculator/1', 'chat-tools/1')
    kit = general.ChatTools if policy.get('interface') == 'chat-tools/1' else general.Calculator
    for item, sequence in zip(evidence, sequences):
        case, messages = cases.get(item['case_id']), item['messages']
        if case is None or messages[:2] != [{'role': 'system', 'content': policy['system_instruction']},
                                            {'role': 'user', 'content': case['turns'][0]['user']}]:
            return False
        if not tools and len(messages) != 3:
            return False
        tool, pending = kit(), []
        for message in messages[2:]:
            if message['role'] == 'assistant':
                if pending:
                    return False
                try:
                    pending = list(base.parse_calls(message['content'], kit.REGISTRY)) if tools else []
                except (ValueError, TypeError, RecursionError):
                    return False
            elif message['role'] == 'tool':
                if not pending or json.dumps(tool.execute(pending.pop(0)), sort_keys=True) != message['content']:
                    return False
            else:
                return False
        if pending or messages[-1]['role'] != 'assistant':
            return False
        if not chat_data.score(case['turns'][0]['expected'], messages[-1]['content'], True):
            return False
        if item['trainable'] != [m['role'] == 'assistant' for m in messages] or encode(item) != sequence:
            return False
    return True


def verify_documents(recipe, policy, encode):
    """An owner's check of a documents recipe: rebuild every demonstration from the beacon and re-encode it."""
    try:
        return recipe['sources'] == {
            'experience': recipe_from(recipe['families'], recipe['randomness'], policy, encode, recipe['per_family']),
            'replay': recipe_from(recipe['others'], recipe['randomness'], policy, encode, recipe['replay_per_family'],
                                  'replay')}
    except (KeyError, ValueError, TypeError):
        return False


def recipe_from(families, randomness, policy, encode, per_family, purpose='recipe'):
    from neuroshard.evolution import assistant_documents_demonstration as demo

    return [encode(demo.trajectory(case, demo.demonstration(case, policy), policy))
            for case in demonstrations(families, randomness, per_family, purpose)]
