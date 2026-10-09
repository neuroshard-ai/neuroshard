"""K-way turn routing and a growth simulation that adds one routed unit at a time.

The accepted router separates two routes: the nearer weighted class mean of mean-centred,
normalised parent features (`assistant_selector.fit_centroid`). An ever-growing assistant
needs the same decision among many units, and must not lose earlier decisions when a unit
is added. This module generalises the centroid rule to K routes, adds a multinomial
logistic baseline under the same features, and measures, as routes are added in a declared
order:

- turn and episode accuracy, and the lowest per-route recall;
- the most confused pair of routes;
- *route retention*: of the held-out turns an earlier router sent correctly, how many the
  router with one more route now sends elsewhere;
- the same on phrasings no router was fitted on.

It is a measurement, not a gate: it trains no unit, opens no sealed split, and grants no
checklist credit. Features come from an encoder supplied by the caller.
"""

import numpy as np

from neuroshard.evolution.modular_reference_execution import identity

STRATEGIES = ('message', 'with-opening', 'episode')


def _rows(matrix, epsilon):
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=-1, keepdims=True), epsilon)


def _digest(gate):
    return identity({key: value for key, value in gate.items() if key != 'sha256'})


def fit_centroids(features, rows, epsilon=1e-6, mean=None):
    """The nearer weighted class mean among K routes: ``rows`` maps a key to (route, weight).

    ``mean`` fixes the centring vector, so a router refitted with more routes centres the same
    way as before; by default it is the mean of the fitted features, as in the two-route rule.
    """
    keys = sorted(rows)
    if set(keys) - set(features):
        raise ValueError('every routed turn needs a feature')
    if not keys:
        raise ValueError('a router needs at least one turn')
    raw = np.array([features[key] for key in keys], dtype=np.float64)
    centre = raw.mean(axis=0) if mean is None else np.asarray(mean, dtype=np.float64)
    x = _rows(raw - centre, epsilon)
    weights = np.array([rows[key][1] for key in keys], dtype=np.float64)
    labels = [rows[key][0] for key in keys]
    order = sorted({label for label, weight in zip(labels, weights) if weight > 0})
    means = {}
    for route in order:
        mask = np.array([label == route for label in labels]) & (weights > 0)
        means[route] = _rows((x[mask] * weights[mask, None]).sum(axis=0), epsilon).tolist()
    gate = {'rule': 'centroids', 'routes': order, 'mean': centre.tolist(), 'epsilon': epsilon, 'means': means}
    gate['sha256'] = _digest(gate)
    return gate


def fit_logistic(features, rows, epsilon=1e-6, steps=300, learning_rate=0.05, weight_decay=1e-3, seed=0):
    """A multinomial logistic router on the same centred, normalised features; full batch, fixed seed."""
    import torch

    keys = sorted(rows)
    order = sorted({rows[key][0] for key in keys})
    raw = torch.tensor([features[key] for key in keys], dtype=torch.float64)
    mean = raw.mean(dim=0)
    x = torch.nn.functional.normalize(raw - mean, dim=1, eps=epsilon)
    y = torch.tensor([order.index(rows[key][0]) for key in keys])
    w = torch.tensor([rows[key][1] for key in keys], dtype=torch.float64)
    torch.manual_seed(seed)
    weight = torch.zeros(x.shape[1], len(order), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(len(order), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.AdamW([weight, bias], lr=learning_rate, weight_decay=weight_decay)
    for _ in range(steps):
        losses = torch.nn.functional.cross_entropy(x @ weight + bias, y, reduction='none')
        loss = (losses * w).sum() / w.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    gate = {'rule': 'logistic', 'routes': order, 'mean': mean.tolist(), 'epsilon': epsilon,
            'weight': weight.detach().tolist(), 'bias': bias.detach().tolist()}
    gate['sha256'] = _digest(gate)
    return gate


def score_matrix(gate, matrix):
    """Each route's score for each row of features: cosine to its class mean (plus any bias), or its logit."""
    x = _rows(np.asarray(matrix, dtype=np.float64) - np.asarray(gate['mean']), gate['epsilon'])
    if gate['rule'] == 'centroids':
        means = np.array([gate['means'][route] for route in gate['routes']])
        biases = np.array([gate.get('biases', {}).get(route, 0.0) for route in gate['routes']])
        return x @ means.T + biases
    if gate['rule'] == 'logistic':
        return x @ np.asarray(gate['weight']) + np.asarray(gate['bias'])
    raise ValueError(f'unknown router rule: {gate["rule"]}')


def scores(gate, feature):
    """Each route's score for one feature."""
    return dict(zip(gate['routes'], score_matrix(gate, [feature])[0].tolist()))


def choose_many(gate, matrix):
    """The highest-scoring route per row; an exact tie goes to the route listed first."""
    if len(matrix) == 0:
        return []
    return [gate['routes'][index] for index in np.argmax(score_matrix(gate, matrix), axis=1)]


def choose(gate, feature):
    return choose_many(gate, [feature])[0]


def with_biases(gate, biases):
    """A centroid router whose route scores are raised by ``biases``: the K-route form of the calibrated shift."""
    if gate['rule'] != 'centroids' or set(biases) - set(gate['routes']):
        raise ValueError('biases apply to a centroid router and name its routes')
    moved = {key: value for key, value in gate.items() if key != 'sha256'}
    moved['biases'] = {route: float(biases.get(route, 0.0)) for route in gate['routes']}
    moved['sha256'] = _digest(moved)
    return moved


def turn_rows(cases, routes):
    """Labelled turns of the cases whose every turn needs one of ``routes``: key -> (route, 1.0)."""
    allowed = set(routes)
    rows = {}
    for case in cases:
        if set(case['labels']) <= allowed:
            for turn, label in enumerate(case['labels']):
                rows[f'{case["id"]}#{turn}'] = (label, 1.0)
    return rows


def decisions(gate, cases, features, routes, strategy):
    """Where the router sends each turn of the eligible cases under a strategy.

    ``message`` and ``with-opening`` route every turn from its own feature (``features`` is
    keyed by strategy); ``episode`` routes the whole conversation from its first turn, as the
    A2 gate does.
    """
    allowed = set(routes)
    eligible = [case for case in cases if set(case['labels']) <= allowed]
    if strategy == 'episode':
        firsts = choose_many(gate, [features['message'][f'{case["id"]}#0'] for case in eligible])
        return {f'{case["id"]}#{turn}': route for case, route in zip(eligible, firsts)
                for turn in range(len(case['labels']))}
    keys = [f'{case["id"]}#{turn}' for case in eligible for turn in range(len(case['labels']))]
    return dict(zip(keys, choose_many(gate, [features[strategy][key] for key in keys])))


def evaluate(chosen, cases):
    """Turn and episode accuracy, per-route recall and the most confused pair for a set of decisions."""
    labels = {f'{case["id"]}#{turn}': label for case in cases for turn, label in enumerate(case['labels'])}
    keys = sorted(chosen)
    if not keys:
        return {'turns': 0}
    right = sum(chosen[key] == labels[key] for key in keys)
    recall, confusion = {}, {}
    for key in keys:
        truth = labels[key]
        hit, seen = recall.get(truth, (0, 0))
        recall[truth] = (hit + (chosen[key] == truth), seen + 1)
        if chosen[key] != truth:
            pair = f'{truth}->{chosen[key]}'
            confusion[pair] = confusion.get(pair, 0) + 1
    episodes = [case for case in cases if f'{case["id"]}#0' in chosen]
    passed = sum(all(chosen[f'{case["id"]}#{t}'] == label for t, label in enumerate(case['labels']))
                 for case in episodes)
    per_route = {route: hit / seen for route, (hit, seen) in sorted(recall.items())}
    worst = max(confusion.items(), key=lambda item: (item[1], item[0])) if confusion else None
    return {'turns': len(keys), 'turn_accuracy': right / len(keys), 'episodes': len(episodes),
            'episode_accuracy': passed / len(episodes), 'min_recall': min(per_route.values()),
            'min_recall_route': min(per_route, key=lambda route: (per_route[route], route)),
            'recall': per_route, 'errors': len(keys) - right,
            'worst_confusion': {'pair': worst[0], 'turns': worst[1]} if worst else None}


def retention(previous, current, cases):
    """Of the turns ``previous`` sent correctly, how many ``current`` now sends elsewhere."""
    labels = {f'{case["id"]}#{turn}': label for case in cases for turn, label in enumerate(case['labels'])}
    kept = [key for key in previous if previous[key] == labels[key]]
    lost = sorted(key for key in kept if current.get(key) != labels[key])
    return {'previously_correct': len(kept), 'lost': len(lost), 'lost_turns': lost[:20]}


def growth(order, fit_cases, eval_sets, features, *, rule='centroids', strategies=STRATEGIES,
           centre='refit', start=2):
    """Add routes one at a time in ``order``; at each size, fit on ``fit_cases`` and evaluate every set.

    ``features[strategy][key]`` holds each turn's feature; fitting always uses the turn's own
    strategy feature (``message`` for ``episode``). ``centre`` is ``refit`` (re-centre on the
    fitted turns each time) or ``frozen`` (keep the centring of the first router).
    """
    if centre not in ('refit', 'frozen'):
        raise ValueError('centre is refit or frozen')
    steps, previous, mean = [], {}, {}
    for size in range(start, len(order) + 1):
        routes = order[:size]
        step = {'routes': size, 'added': routes[-1], 'strategies': {}}
        for strategy in strategies:
            fitted_with = 'message' if strategy == 'episode' else strategy
            rows = turn_rows(fit_cases, routes)
            if rule == 'centroids':
                gate = fit_centroids(features[fitted_with], rows, mean=mean.get(strategy) if centre == 'frozen' else None)
                mean.setdefault(strategy, gate['mean'])
            else:
                gate = fit_logistic(features[fitted_with], rows)
            report = {'fit_turns': len(rows), 'gate_sha256': gate['sha256']}
            for name, cases in eval_sets.items():
                chosen = decisions(gate, cases, features, routes, strategy)
                report[name] = evaluate(chosen, cases)
                if (strategy, name) in previous:
                    report[name]['retention'] = retention(previous[strategy, name], chosen, cases)
                previous[strategy, name] = chosen
            step['strategies'][strategy] = report
        steps.append(step)
    return {'order': list(order), 'rule': rule, 'centre': centre, 'steps': steps}


def summary_rows(result, set_name):
    """Flat rows for a table: one per (routes, strategy) on one evaluation set."""
    rows = []
    for step in result['steps']:
        for strategy, report in step['strategies'].items():
            measured = report[set_name]
            if not measured.get('turns'):
                continue
            kept = measured.get('retention')
            rows.append({'routes': step['routes'], 'added': step['added'], 'strategy': strategy,
                         'turn_accuracy': measured['turn_accuracy'], 'episode_accuracy': measured['episode_accuracy'],
                         'min_recall': measured['min_recall'], 'min_recall_route': measured['min_recall_route'],
                         'worst_confusion': (measured['worst_confusion'] or {}).get('pair'),
                         'lost': kept['lost'] if kept else None,
                         'previously_correct': kept['previously_correct'] if kept else None})
    return rows
