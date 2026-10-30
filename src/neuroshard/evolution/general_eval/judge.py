"""Blind pairwise judging for what a program cannot check (writing quality, open answers), with calibration.

Protocol (declared in docs/GENERAL_CAPABILITY_EVAL.md):

1. For every judged prompt, the two systems' final replies are shown as "Response 1" and "Response 2" in an
   order drawn from a seeded RNG, then again in the swapped order. The judge never sees system names,
   routes, tool calls or which system is the candidate.
2. The judge answers with exactly one of ``1``, ``2`` or ``tie`` under the pinned rubric. A prompt scores
   1 for the candidate if it wins both orders, 0 if it loses both, and 0.5 otherwise (a tie or an order-
   dependent verdict). Order inconsistency is reported as the judge's position-bias rate.
3. A calibration sample of prompts is also rated by humans under the same rubric. The judge may be used
   for a gate only if its agreement with the human majority and Cohen's kappa meet the declared minimums,
   and its position-consistency rate is at least the declared minimum.

``judge(prompt_text) -> '1' | '2' | 'tie'`` is any callable: a pinned model behind an API, a local model,
or a scripted stub in tests. Its identity, prompt digest and every verdict are recorded.
"""

import hashlib
import random
import re

RUBRICS = {'writing/1': (
    'You are comparing two assistant replies to the same request. Judge which reply better serves the user: '
    'it does what was asked, is accurate, clear, well organised, appropriately concise, and its tone fits the '
    'request. Formatting constraints in the request are checked separately by a program; do not reward or '
    'punish length for its own sake. Do not favour a reply for its position. Answer with exactly one token: '
    '1, 2, or tie.')}

TEMPLATE = '{rubric}\n\n[Request]\n{request}\n\n[Response 1]\n{first}\n\n[Response 2]\n{second}\n\n[Verdict]'


def prompt(rubric, request, first, second):
    return TEMPLATE.format(rubric=RUBRICS[rubric], request=request, first=first, second=second)


def verdict(text):
    match = re.fullmatch(r'\s*(1|2|tie)\s*\.?\s*', (text or '').lower())
    if not match:
        raise ValueError(f'judge answered {text!r}; expected 1, 2 or tie')
    return match[1]


def pairwise(items, candidate, incumbent, judge, seed):
    """``candidate``/``incumbent``: {item id: final reply}. Returns per-item scores and verdicts."""
    rng, rows = random.Random(seed), []
    for item in items:
        if 'judged' not in item:
            continue
        request, rubric = item['turns'][-1]['user'], item['judged']['rubric']
        a, b = candidate[item['id']], incumbent[item['id']]
        candidate_first = rng.random() < 0.5
        orders = [(a, b, 'candidate') if candidate_first else (b, a, 'incumbent')]
        orders.append((orders[0][1], orders[0][0], 'incumbent' if candidate_first else 'candidate'))
        wins, verdicts = [], []
        for first, second, first_is in orders:
            text = prompt(rubric, request, first, second)
            said = verdict(judge(text))
            verdicts.append({'first': first_is, 'verdict': said,
                             'prompt_sha256': hashlib.sha256(text.encode()).hexdigest()})
            if said == 'tie':
                wins.append(None)
            else:
                wins.append((said == '1') == (first_is == 'candidate'))
        score = 1.0 if wins == [True, True] else 0.0 if wins == [False, False] else 0.5
        consistent = wins[0] == wins[1]
        rows.append({'id': item['id'], 'score': score, 'consistent': consistent, 'verdicts': verdicts})
    return rows


def summary(rows, samples=10000, seed=0):
    """Mean preference for the candidate with a percentile bootstrap 95% interval, and position consistency."""
    if not rows:
        return {'prompts': 0}
    scores = [r['score'] for r in rows]
    rng = random.Random(seed)
    means = sorted(sum(scores[rng.randrange(len(scores))] for _ in scores) / len(scores) for _ in range(samples))
    return {'prompts': len(rows), 'preference': sum(scores) / len(scores),
            'lower95': means[int(0.025 * samples)], 'upper95': means[int(0.975 * samples) - 1],
            'position_consistency': sum(r['consistent'] for r in rows) / len(rows), 'judge_calls': 2 * len(rows)}


def cohen_kappa(left, right, labels=('1', '2', 'tie')):
    n = len(left)
    if n == 0 or n != len(right):
        raise ValueError('need two equal, non-empty label lists')
    observed = sum(a == b for a, b in zip(left, right)) / n
    expected = sum((left.count(k) / n) * (right.count(k) / n) for k in labels)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def calibration(judge_labels, human_labels, minimum_agreement=0.75, minimum_kappa=0.5):
    """``human_labels``: per prompt, the list of human verdicts (majority taken; ties among humans -> 'tie')."""
    majority = []
    for votes in human_labels:
        counts = {k: votes.count(k) for k in set(votes)}
        top = max(counts.values())
        winners = [k for k, v in counts.items() if v == top]
        majority.append(winners[0] if len(winners) == 1 else 'tie')
    agreement = sum(a == b for a, b in zip(judge_labels, majority)) / len(majority)
    kappa = cohen_kappa(list(judge_labels), majority)
    return {'prompts': len(majority), 'agreement': agreement, 'kappa': kappa,
            'usable': agreement >= minimum_agreement and kappa >= minimum_kappa}
