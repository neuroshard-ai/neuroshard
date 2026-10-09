import random

import pytest

from neuroshard.evolution import router_scaling as scaling
from neuroshard.evolution import router_scaling_data as data
from neuroshard.evolution import router_scaling_features as encoders
from neuroshard.evolution import assistant_selector as selector


def clusters(routes, per_route=12, width=24, noise=0.2, seed=3):
    """Separable turns: each route raises its own block of a shared offset vector."""
    rng = random.Random(seed)
    features, rows = {}, {}
    for r, route in enumerate(routes):
        for i in range(per_route):
            vector = [3.0 + rng.uniform(-noise, noise) for _ in range(width)]
            vector[r % width] += 1.0
            key = f'{route}-{i:02d}#0'
            features[key], rows[key] = vector, (route, 1.0)
    return features, rows


def test_two_route_centroids_agree_with_the_accepted_rule():
    features, rows = clusters(('drafting', 'scheduling'), per_route=20, noise=0.4)
    binary = {key: (1.0 if route == 'scheduling' else 0.0, weight) for key, (route, weight) in rows.items()}
    accepted = selector.fit_centroid(features, binary, 1e-6)
    gate = scaling.fit_centroids(features, rows)
    probe = random.Random(9)
    for _ in range(200):
        x = [3.0 + probe.uniform(-0.6, 0.6) for _ in range(24)]
        x[probe.randrange(2)] += probe.uniform(0, 1)
        assert (scaling.choose(gate, x) == 'scheduling') == selector.choose(accepted, x)


def test_centroids_separate_many_routes_and_ties_go_to_the_first_route():
    routes = tuple(f'unit{i:02d}' for i in range(10))
    features, rows = clusters(routes)
    gate = scaling.fit_centroids(features, rows)
    assert gate['routes'] == sorted(routes)
    assert all(scaling.choose(gate, features[key]) == route for key, (route, _) in rows.items())
    assert scaling.choose(gate, gate['mean']) == gate['routes'][0]
    assert gate['sha256'] == scaling._digest(gate)


def test_biases_move_only_turns_within_the_bias():
    features, rows = clusters(('a', 'b'), noise=0.0)
    gate = scaling.fit_centroids(features, rows)
    middle = [(x + y) / 2 for x, y in zip(features['a-00#0'], features['b-00#0'])]
    middle[0] += 0.01
    margin = scaling.scores(gate, middle)['a'] - scaling.scores(gate, middle)['b']
    assert margin > 0 and scaling.choose(gate, middle) == 'a'
    assert scaling.choose(scaling.with_biases(gate, {'b': margin * 0.9}), middle) == 'a'
    assert scaling.choose(scaling.with_biases(gate, {'b': margin * 1.1}), middle) == 'b'
    with pytest.raises(ValueError):
        scaling.with_biases(gate, {'c': 1.0})


def test_logistic_router_fits_separable_routes():
    features, rows = clusters(('a', 'b', 'c', 'd'))
    gate = scaling.fit_logistic(features, rows, steps=200)
    assert all(scaling.choose(gate, features[key]) == route for key, (route, _) in rows.items())


def test_frozen_centring_keeps_the_first_mean():
    order = ('a', 'b', 'c')
    features, _ = clusters(order, per_route=6)
    cases = [{'id': key[:-2], 'labels': [key[0]], 'user_turns': ['x']} for key in sorted(features)]
    strategies = {'message': features, 'with-opening': features}
    frozen = scaling.growth(order, cases, {'test': cases}, strategies, strategies=('message',), centre='frozen')
    refit = scaling.growth(order, cases, {'test': cases}, strategies, strategies=('message',))
    assert len(frozen['steps']) == len(refit['steps']) == 2
    assert frozen['steps'][1]['strategies']['message']['gate_sha256'] != refit['steps'][1]['strategies']['message']['gate_sha256']
    assert refit['steps'][1]['strategies']['message']['test']['turn_accuracy'] == 1.0


def test_evaluation_counts_episodes_recall_confusion_and_retention():
    cases = [{'id': 'x', 'labels': ['a', 'b']}, {'id': 'y', 'labels': ['a']}, {'id': 'z', 'labels': ['b', 'b']}]
    chosen = {'x#0': 'a', 'x#1': 'a', 'y#0': 'a', 'z#0': 'b', 'z#1': 'b'}
    report = scaling.evaluate(chosen, cases)
    assert report['turn_accuracy'] == 4 / 5 and report['episode_accuracy'] == 2 / 3
    assert report['recall'] == {'a': 1.0, 'b': 2 / 3} and report['worst_confusion'] == {'pair': 'b->a', 'turns': 1}
    later = {**chosen, 'y#0': 'c'}
    kept = scaling.retention(chosen, later, cases)
    assert kept['previously_correct'] == 4 and kept['lost'] == 1 and kept['lost_turns'] == ['y#0']


def test_episode_strategy_routes_every_turn_from_the_opening():
    cases = [{'id': 'x', 'labels': ['a', 'b']}]
    features, _ = clusters(('a', 'b'), per_route=1, noise=0.0)
    gate = scaling.fit_centroids(features, {'a-00#0': ('a', 1.0), 'b-00#0': ('b', 1.0)})
    per_turn = {'x#0': features['a-00#0'], 'x#1': features['b-00#0']}
    assert scaling.decisions(gate, cases, {'message': per_turn}, ('a', 'b'), 'message') == {'x#0': 'a', 'x#1': 'b'}
    assert scaling.decisions(gate, cases, {'message': per_turn}, ('a', 'b'), 'episode') == {'x#0': 'a', 'x#1': 'a'}
    assert scaling.decisions(gate, cases, {'message': per_turn}, ('a',), 'message') == {}


def test_synthetic_cases_are_deterministic_disjoint_and_labelled():
    fit, test, unseen = (data.cases(split, cross_per_pair=1) for split in ('fit', 'test', 'unseen'))
    assert fit == data.cases('fit', cross_per_pair=1)
    texts = [{text for case in cases for text in case['user_turns'][:1]} for cases in (fit, test, unseen)]
    assert not texts[0] & texts[2] and not texts[1] & texts[2]
    for case in fit + test + unseen:
        assert len(case['labels']) == len(case['user_turns']) >= 1
        assert set(case['labels']) <= set(data.SYNTHETIC)
    crosses = [case for case in fit if '+' in case['capability']]
    assert len(crosses) == len(data.SYNTHETIC) * (len(data.SYNTHETIC) - 1)
    assert all(case['labels'][0] != case['labels'][1] for case in crosses)
    assert set(data.ORDER) == set(data.REAL) | set(data.SYNTHETIC) and len(data.ORDER) == len(set(data.ORDER))
    with pytest.raises(ValueError):
        data.cases('confirmation')


def test_real_anchors_read_only_training_and_integration_splits():
    fit = data.real_cases('fit', limit=4)
    assert {case['capability'] for case in fit} == {'drafting', 'scheduling', 'cross'}
    assert all(set(case['labels']) <= set(data.REAL) for case in fit)
    assert any('scheduling' in case['labels'] for case in fit)
    with pytest.raises(ValueError):
        data.real_cases('unseen')


def test_hashed_features_and_strategies():
    a, b = encoders.hashed('Book room Atlas on 2027-01-02'), encoders.hashed('Book room Atlas on 2028-11-30')
    assert a == b and abs(sum(v * v for v in a) - 1.0) < 1e-9
    cases = [{'id': 'x', 'user_turns': ['Open a ticket', 'Assign it to Ada'], 'labels': ['tickets', 'tickets']}]
    features = encoders.strategy_features(cases, lambda texts: [encoders.hashed(t, 8) for t in texts])
    assert features['message']['x#1'] == encoders.hashed('Assign it to Ada', 8)
    assert features['with-opening']['x#1'] == encoders.hashed('Assign it to Ada', 8) + encoders.hashed('Open a ticket', 8)
