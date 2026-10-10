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


def single_turn_cases(features):
    return [{'id': key[:-2], 'labels': [key.split('-')[0]], 'user_turns': ['x']} for key in sorted(features)]


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
    cases = single_turn_cases(features)
    strategies = {'message': features, 'with-opening': features}
    frozen = scaling.growth(order, cases, {'test': cases}, strategies, strategies=('message',), centre='frozen')
    refit = scaling.growth(order, cases, {'test': cases}, strategies, strategies=('message',))
    assert len(frozen['steps']) == len(refit['steps']) == 2
    assert frozen['final_gates']['message']['mean'] != refit['final_gates']['message']['mean']
    assert refit['steps'][1]['strategies']['message']['test']['turn_accuracy'] == 1.0
    interval = refit['steps'][1]['strategies']['message']['test']['interval']
    assert interval['turn_accuracy'] == [1.0, 1.0]


def test_evaluation_counts_episodes_recall_confusion_and_retention():
    cases = [{'id': 'x', 'labels': ['a', 'b']}, {'id': 'y', 'labels': ['a']}, {'id': 'z', 'labels': ['b', 'b']}]
    chosen = {'x#0': 'a', 'x#1': 'a', 'y#0': 'a', 'z#0': 'b', 'z#1': 'b'}
    report = scaling.evaluate(chosen, cases)
    assert report['turn_accuracy'] == 4 / 5 and report['episode_accuracy'] == 2 / 3
    assert report['recall'] == {'a': 1.0, 'b': 2 / 3} and report['worst_confusion'] == {'pair': 'b->a', 'turns': 1}
    later = {**chosen, 'y#0': 'c'}
    kept = scaling.retention(chosen, later, cases)
    assert kept['previously_correct'] == 4 and kept['lost'] == 1 and kept['lost_turns'] == ['y#0']


def test_bootstrap_resamples_whole_conversations():
    cases = [{'id': f'c{i}', 'labels': ['a', 'a']} for i in range(40)]
    chosen = {f'c{i}#{t}': ('a' if i % 2 == 0 else ('a' if t == 0 else 'b')) for i in range(40) for t in range(2)}
    interval = scaling.bootstrap(chosen, cases, draws=500, seed=1)
    low, high = interval['episode_accuracy']
    assert low < 0.5 < high and 0.3 < low and high < 0.7
    assert interval['turn_accuracy'][0] < 0.75 < interval['turn_accuracy'][1]
    assert scaling.bootstrap(chosen, cases, draws=500, seed=1) == interval
    assert scaling.bootstrap({}, cases) is None


def test_coverage_abstains_on_the_least_confident_turns_first():
    features, rows = clusters(('a', 'b'), per_route=20, noise=0.0)
    gate = scaling.fit_centroids(features, rows)
    hard = [(x + y) / 2 for x, y in zip(features['a-00#0'], features['b-00#0'])]
    hard[1] += 0.001
    features = {**features, 'a-99#0': hard}
    cases = single_turn_cases(features)
    curve = scaling.coverage(gate, cases, {'message': features}, ('a', 'b'), 'message', quantiles=(0.0, 0.05))
    assert curve[0]['kept'] == 1.0 and curve[0]['errors_kept'] == 1
    assert curve[1]['kept'] < 1.0 and curve[1]['errors_kept'] == 0 and curve[1]['kept_accuracy'] == 1.0
    assert scaling.coverage(gate, cases, {'message': features}, ('a', 'b'), 'episode') is None


def test_description_prototypes_share_the_fitted_centring_and_blend():
    features, rows = clusters(('a', 'b'), per_route=10, noise=0.1)
    fitted = scaling.fit_centroids(features, rows)
    cards = {'a': ['alpha'], 'b': ['beta']}
    lookup = {'alpha': features['a-00#0'], 'beta': features['b-00#0']}
    described = scaling.prototypes(cards, lambda texts: [lookup[t] for t in texts], mean=fitted['mean'])
    assert described['mean'] == fitted['mean'] and described['routes'] == ['a', 'b']
    assert all(scaling.choose(described, features[key]) == route for key, (route, _) in rows.items())
    zero = scaling.blend(fitted, described, 0.0)['means']
    assert all(max(abs(x - y) for x, y in zip(zero[r], fitted['means'][r])) < 1e-12 for r in zero)
    halfway = scaling.blend(fitted, described, 0.5)
    assert halfway['blend'] == 0.5 and halfway['sha256'] == scaling._digest(halfway)
    with pytest.raises(ValueError):
        scaling.blend(fitted, scaling.fit_centroids(features, rows, mean=[0.0] * 24), 0.5)


def test_paired_bootstrap_measures_the_difference_on_the_same_turns():
    cases = [{'id': f'c{i}', 'labels': ['a']} for i in range(50)]
    worse = {f'c{i}#0': 'a' if i < 25 else 'b' for i in range(50)}
    better = {f'c{i}#0': 'a' if i < 45 else 'b' for i in range(50)}
    result = scaling.paired_bootstrap(worse, better, cases, draws=500, seed=2)
    assert result['difference'] == 0.4 and 0.2 < result['interval'][0] < 0.4 < result['interval'][1]
    same = scaling.paired_bootstrap(worse, worse, cases, draws=100)
    assert same['difference'] == 0.0 and same['interval'] == [0.0, 0.0]
    with pytest.raises(ValueError):
        scaling.paired_bootstrap(worse, {'c0#0': 'a'}, cases)


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
    assert set(data.DESCRIPTIONS) == set(data.ORDER)
    with pytest.raises(ValueError):
        data.cases('confirmation')


def test_descriptions_reuse_no_request_template():
    templates = [t for fit, unseen, own in data.CAPABILITIES.values() for t in fit + unseen + own]
    templates += [t for openings, follow in data.REAL_UNSEEN.values() for t in openings + follow]
    for cards in data.DESCRIPTIONS.values():
        for card in cards:
            assert all(card.lower()[:30] not in template.lower() for template in templates)


def test_real_anchors_read_only_training_and_integration_splits():
    fit = data.real_cases('fit', limit=4)
    assert {case['capability'] for case in fit} == {'drafting', 'scheduling', 'cross'}
    assert all(set(case['labels']) <= set(data.REAL) for case in fit)
    assert any('scheduling' in case['labels'] for case in fit)
    with pytest.raises(ValueError):
        data.real_cases('unseen')


def test_real_unseen_phrasings_differ_from_the_real_grammars():
    unseen = data.real_unseen_cases(8)
    assert {case['capability'] for case in unseen} == set(data.REAL)
    assert all(set(case['labels']) == {case['capability']} for case in unseen)
    real = {text for case in data.real_cases('fit', limit=16) + data.real_cases('test') for text in case['user_turns']}
    assert not real & {text for case in unseen for text in case['user_turns']}


def test_hashed_features_and_strategies():
    a, b = encoders.hashed('Book room Atlas on 2027-01-02'), encoders.hashed('Book room Atlas on 2028-11-30')
    assert a == b and abs(sum(v * v for v in a) - 1.0) < 1e-9
    cases = [{'id': 'x', 'user_turns': ['Open a ticket', 'Assign it to Ada'], 'labels': ['tickets', 'tickets']}]
    features = encoders.strategy_features(cases, lambda texts: [encoders.hashed(t, 8) for t in texts])
    assert features['message']['x#1'] == encoders.hashed('Assign it to Ada', 8)
    assert features['with-opening']['x#1'] == encoders.hashed('Assign it to Ada', 8) + encoders.hashed('Open a ticket', 8)
