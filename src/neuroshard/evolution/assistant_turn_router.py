"""A K-route turn router for a growing assistant, with a fallback and an acceptance check.

The [router scaling study](../../../docs/ROUTER_SCALING.md) found that the accepted
two-route centroid rule loses accuracy as units are added, while a multinomial logistic
router over the turn's message state *and* the conversation's opening message state keeps
it on familiar wording. It also found that new wording is the bottleneck, that a
confidence threshold removes most remaining errors at a small abstention cost, and that an
added unit can silently displace an earlier unit's turns. This module packages those three
findings, without serving anything:

- ``fit``: the logistic router over ``message ⊕ opening`` features, fitted full-batch with
  a fixed seed in float64, so one runtime always produces the same router file.
- ``calibrate``: the confidence threshold below which a turn goes to a declared fallback
  route (the parent, or a clarifying reply). It is chosen on held-out folds of conversations,
  never on the turns it is judged by, as the largest-coverage threshold whose held-out kept
  accuracy meets a declared target.
- ``admit``: the acceptance rule for adding a unit. Every earlier route's recall, on a
  declared check set that should include paraphrases, may fall by at most a declared margin,
  and the new route must reach a declared recall.

Routing a turn is a dot product over a JSON file of floats, so any party holding the file
and the feature can recompute the decision. Features are supplied by the caller; this module
does not compute model states.
"""

import numpy as np

from neuroshard.evolution.modular_reference_execution import identity

RULE = 'logistic-context'


def _digest(router):
    return identity({key: value for key, value in router.items() if key != 'sha256'})


def context_feature(message, opening):
    """The routed feature of one turn: its message state followed by the conversation's opening message state."""
    if len(message) != len(opening):
        raise ValueError('message and opening features differ in width')
    return [float(value) for value in message] + [float(value) for value in opening]


def conversation_features(turn_features):
    """``context_feature`` for each turn of one conversation, given its per-turn message states in order."""
    if not turn_features:
        raise ValueError('a conversation has at least one turn')
    return [context_feature(message, turn_features[0]) for message in turn_features]


def _normalised(matrix, mean, epsilon):
    x = np.asarray(matrix, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), epsilon)


def fit(features, rows, *, epsilon=1e-6, steps=300, learning_rate=0.05, weight_decay=1e-3, seed=0, fallback=None):
    """The router from ``features[key]`` (context features) and ``rows[key] = (route, weight)``.

    ``fallback`` names the route a low-confidence turn is sent to; it need not be a fitted
    route. The threshold starts at zero, routing every turn, until ``calibrate`` sets it.
    """
    import torch

    keys = sorted(rows)
    if not keys or set(keys) - set(features):
        raise ValueError('every routed turn needs a feature')
    routes = sorted({rows[key][0] for key in keys})
    if len(routes) < 2:
        raise ValueError('a router needs at least two routes')
    raw = np.array([features[key] for key in keys], dtype=np.float64)
    mean = raw.mean(axis=0)
    x = torch.tensor(_normalised(raw, mean, epsilon))
    y = torch.tensor([routes.index(rows[key][0]) for key in keys])
    w = torch.tensor([float(rows[key][1]) for key in keys], dtype=torch.float64)
    if (w <= 0).any():
        raise ValueError('routing weights are positive')
    torch.manual_seed(seed)
    weight = torch.zeros(x.shape[1], len(routes), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(len(routes), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.AdamW([weight, bias], lr=learning_rate, weight_decay=weight_decay)
    for _ in range(steps):
        loss = (torch.nn.functional.cross_entropy(x @ weight + bias, y, reduction='none') * w).sum() / w.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    router = {'rule': RULE, 'routes': routes, 'width': int(raw.shape[1]), 'mean': mean.tolist(), 'epsilon': epsilon,
              'weight': weight.detach().tolist(), 'bias': bias.detach().tolist(), 'threshold': 0.0,
              'fallback': fallback,
              'recipe': {'steps': steps, 'learning_rate': learning_rate, 'weight_decay': weight_decay, 'seed': seed},
              'fitted_turns': len(keys)}
    router['sha256'] = _digest(router)
    return router


def verify(router):
    """Refuse a router file whose digest, rule or shapes do not hold."""
    if router.get('rule') != RULE or router.get('sha256') != _digest(router):
        raise ValueError('router file differs from its digest or rule')
    width, count = router['width'], len(router['routes'])
    if (len(router['mean']) != width or len(router['weight']) != width
            or any(len(row) != count for row in router['weight']) or len(router['bias']) != count):
        raise ValueError('router shapes are inconsistent')
    if router['threshold'] > 0 and router['fallback'] is None:
        raise ValueError('a router that abstains must name its fallback')
    return router


def probabilities(router, matrix):
    """Route probabilities for each row of context features."""
    x = _normalised(matrix, router['mean'], router['epsilon'])
    logits = x @ np.asarray(router['weight']) + np.asarray(router['bias'])
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _margins(router, matrix):
    p = np.sort(probabilities(router, matrix), axis=1)
    return p[:, -1] - p[:, -2]


def route_many(router, matrix):
    """Each row's route: the most probable one, or the fallback when the top-two probability gap is below the threshold."""
    if len(matrix) == 0:
        return []
    p = probabilities(router, matrix)
    ordered = np.sort(p, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    picked = [router['routes'][index] for index in np.argmax(p, axis=1)]
    return [router['fallback'] if gap < router['threshold'] else route for route, gap in zip(picked, margin)]


def route(router, feature):
    return route_many(router, [feature])[0]


def _folds(rows, folds):
    cases = sorted({key.rsplit('#', 1)[0] for key in rows})
    if len(cases) < folds:
        raise ValueError('calibration needs at least one conversation per fold')
    return {case: index % folds for index, case in enumerate(cases)}


def calibrate(router, features, rows, *, target, folds=4, fallback=None, min_coverage=0.5, **recipe):
    """Set the abstention threshold from held-out folds of conversations.

    Each fold is scored by a router refitted without it under the router's own recipe. The
    threshold is the smallest top-two probability gap such that the held-out turns at or
    above it are routed correctly at least ``target`` of the time; turns below it go to
    ``fallback``. If that keeps fewer than ``min_coverage`` of held-out turns, the target is
    unreachable and every turn goes to the fallback: the parent keeps serving rather than a
    router that is reliable only on a sliver of traffic. Returns the router and its held-out curve.
    """
    fallback = fallback if fallback is not None else router['fallback']
    if fallback is None:
        raise ValueError('calibration needs a fallback route')
    if not 0 < target <= 1:
        raise ValueError('the target is a kept accuracy in (0, 1]')
    recipe = {**router['recipe'], 'epsilon': router['epsilon'], **recipe}
    fold = _folds(rows, folds)
    margins, correct = [], []
    for k in range(folds):
        train = {key: row for key, row in rows.items() if fold[key.rsplit('#', 1)[0]] != k}
        held = sorted(key for key in rows if fold[key.rsplit('#', 1)[0]] == k)
        if len({row[0] for row in train.values()}) < 2 or not held:
            continue
        partial = fit({key: features[key] for key in train}, train, **recipe)
        matrix = [features[key] for key in held]
        p = probabilities(partial, matrix)
        picked = [partial['routes'][index] for index in np.argmax(p, axis=1)]
        ordered = np.sort(p, axis=1)
        margins.extend((ordered[:, -1] - ordered[:, -2]).tolist())
        correct.extend(route == rows[key][0] for route, key in zip(picked, held))
    if not margins:
        raise ValueError('no fold could be scored')
    gaps_all, correct_all = np.asarray(margins), np.asarray(correct)
    order = np.argsort(-gaps_all, kind='stable')
    gaps, right = gaps_all[order], np.cumsum(correct_all[order])
    kept_accuracy = right / np.arange(1, len(gaps) + 1)
    # A threshold keeps every turn whose gap is at least it, so only the last index of each
    # run of equal gaps is a candidate. Take the largest coverage whose kept accuracy meets the target.
    ends = [i for i in range(len(gaps)) if i + 1 == len(gaps) or gaps[i + 1] != gaps[i]]
    meeting = [i for i in ends if kept_accuracy[i] >= target and (i + 1) / len(gaps) >= min_coverage]
    reachable = bool(meeting)
    if not reachable:
        threshold = 2.0  # probability gaps lie in [0, 1]: abstain on every turn; JSON has no infinity
    elif meeting[-1] == len(gaps) - 1:
        threshold = 0.0
    else:
        threshold = float(gaps[meeting[-1]])
    curve = [{'coverage': float(share), 'kept_accuracy': float(kept_accuracy[max(int(share * len(gaps)) - 1, 0)])}
             for share in (1.0, 0.95, 0.9, 0.8, 0.7)]
    calibrated = {key: value for key, value in router.items() if key != 'sha256'}
    calibrated.update(threshold=threshold, fallback=fallback,
                      calibration={'target': target, 'folds': folds, 'min_coverage': min_coverage,
                                   'held_out_turns': len(gaps),
                                   'held_out_accuracy': float(correct_all.mean()), 'target_reachable': reachable,
                                   'held_out_coverage': float((gaps_all >= threshold).mean()),
                                   'held_out_kept_accuracy': (float(correct_all[gaps_all >= threshold].mean())
                                                              if reachable else None),
                                   'curve': curve})
    calibrated['sha256'] = _digest(calibrated)
    return verify(calibrated), calibrated['calibration']


def recall(router, features, rows):
    """Per-route recall of the labelled turns; a turn sent to the fallback counts as missed."""
    keys = sorted(rows)
    chosen = dict(zip(keys, route_many(router, [features[key] for key in keys])))
    result = {}
    for name in sorted({rows[key][0] for key in keys}):
        mine = [key for key in keys if rows[key][0] == name]
        result[name] = {'turns': len(mine), 'recall': sum(chosen[key] == name for key in mine) / len(mine),
                        'abstained': sum(chosen[key] == router['fallback'] for key in mine) / len(mine)}
    return result


def sign_test(lost, gained):
    """One-sided exact sign test: the probability of at least ``lost`` losses among the discordant turns if losing and gaining were equally likely."""
    from math import comb

    total = lost + gained
    if total == 0:
        return 1.0
    return sum(comb(total, k) for k in range(lost, total + 1)) / 2 ** total


def admit(previous, candidate, features, rows, *, added, margin, minimum, alpha=0.05, min_turns=30):
    """Whether ``candidate`` may replace ``previous`` after adding the route ``added``.

    ``rows`` is the declared check set over every route, which should include paraphrases of
    earlier routes. On the same check turns, an earlier route fails when its recall falls by
    more than ``margin`` and the turns it lost significantly outnumber the turns it gained
    (one-sided exact sign test at ``alpha``), so a drop of a turn or two in a small check set
    does not block growth while a real displacement does. An earlier route with fewer than
    ``min_turns`` check turns cannot be verified and also fails. ``added`` must reach
    ``minimum`` recall. Returns the decision and every route's before/after record.
    """
    verify(previous)
    verify(candidate)
    if added in previous['routes'] or added not in candidate['routes']:
        raise ValueError('the candidate must add exactly the declared route')
    if set(previous['routes']) - set(candidate['routes']):
        raise ValueError('a candidate may not drop an earlier route')
    keys = sorted(rows)
    before_routes = dict(zip(keys, route_many(previous, [features[key] for key in keys])))
    after_routes = dict(zip(keys, route_many(candidate, [features[key] for key in keys])))
    routes = {}
    for name in sorted({rows[key][0] for key in keys}):
        mine = [key for key in keys if rows[key][0] == name]
        after = sum(after_routes[key] == name for key in mine) / len(mine)
        entry = {'turns': len(mine), 'after': after}
        if name != added:
            before = sum(before_routes[key] == name for key in mine) / len(mine)
            lost = sum(before_routes[key] == name and after_routes[key] != name for key in mine)
            gained = sum(before_routes[key] != name and after_routes[key] == name for key in mine)
            p = sign_test(lost, gained)
            entry.update(before=before, drop=before - after, lost=lost, gained=gained, p_value=p,
                         failed=len(mine) < min_turns or (before - after > margin and p < alpha),
                         unverifiable=len(mine) < min_turns)
        routes[name] = entry
    if added not in routes:
        raise ValueError('the check set has no turn for the added route')
    missing = sorted(set(previous['routes']) - set(routes))
    failures = sorted(name for name, entry in routes.items() if entry.get('failed')) + missing
    added_recall = routes[added]['after']
    return {'admitted': not failures and added_recall >= minimum, 'added': added, 'added_recall': added_recall,
            'margin': margin, 'minimum': minimum, 'alpha': alpha, 'min_turns': min_turns,
            'earlier_routes_failing': failures, 'earlier_routes_unchecked': missing, 'routes': routes,
            'previous_sha256': previous['sha256'], 'candidate_sha256': candidate['sha256']}
