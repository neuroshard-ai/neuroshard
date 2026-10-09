import json
import random

import pytest

from neuroshard.evolution import assistant_turn_router as router


def conversations(routes, per_route=16, width=12, noise=0.25, seed=5, follow_up=True):
    """Conversations per route: an opening that names its route and, optionally, a generic follow-up.

    The follow-up's message state is the same for every route, so only the opening tells
    which unit it needs; a message-only router cannot route it.
    """
    rng = random.Random(seed)
    generic = [rng.uniform(-1, 1) for _ in range(width)]
    features, rows = {}, {}
    for r, name in enumerate(routes):
        for i in range(per_route):
            opening = [3.0 + rng.uniform(-noise, noise) for _ in range(width)]
            opening[r % width] += 1.5
            turns = [opening] + ([[g + rng.uniform(-0.05, 0.05) for g in generic]] if follow_up else [])
            for t, feature in enumerate(router.conversation_features(turns)):
                key = f'{name}-{i:02d}#{t}'
                features[key], rows[key] = feature, (name, 1.0)
    return features, rows


def test_context_features_put_the_opening_after_each_message():
    turns = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert router.conversation_features(turns) == [[1.0, 2.0, 1.0, 2.0], [3.0, 4.0, 1.0, 2.0], [5.0, 6.0, 1.0, 2.0]]
    with pytest.raises(ValueError):
        router.context_feature([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        router.conversation_features([])


def test_the_router_routes_generic_follow_ups_by_their_opening():
    features, rows = conversations(('drafting', 'scheduling', 'invoices', 'rooms'))
    fitted = router.verify(router.fit(features, rows, fallback='parent'))
    assert fitted['routes'] == ['drafting', 'invoices', 'rooms', 'scheduling'] and fitted['threshold'] == 0.0
    assert all(router.route(fitted, features[key]) == name for key, (name, _) in rows.items())
    assert router.recall(fitted, features, rows)['rooms'] == {'turns': 32, 'recall': 1.0, 'abstained': 0.0}


def test_fitting_is_deterministic_and_the_file_round_trips():
    features, rows = conversations(('a', 'b', 'c'), per_route=8)
    first, second = router.fit(features, rows), router.fit(features, rows)
    assert first == second and first['sha256'] == router._digest(first)
    loaded = router.verify(json.loads(json.dumps(first)))
    assert router.route_many(loaded, list(features.values())) == router.route_many(first, list(features.values()))


def test_a_changed_or_inconsistent_router_file_is_refused():
    features, rows = conversations(('a', 'b'), per_route=6)
    fitted = router.fit(features, rows)
    with pytest.raises(ValueError):
        router.verify({**fitted, 'bias': [0.0, 1.0]})
    broken = {**fitted, 'bias': [0.0]}
    broken['sha256'] = router._digest(broken)
    with pytest.raises(ValueError):
        router.verify(broken)
    abstaining = {**fitted, 'threshold': 0.5, 'fallback': None}
    abstaining['sha256'] = router._digest(abstaining)
    with pytest.raises(ValueError):
        router.verify(abstaining)
    with pytest.raises(ValueError):
        router.fit(features, {key: ('a', 1.0) for key in rows})


def test_calibration_abstains_only_where_held_out_turns_are_unreliable():
    features, rows = conversations(('a', 'b', 'c'), per_route=20, noise=0.25, follow_up=False)
    # Ten ambiguous conversations: an opening halfway between routes a and b, labelled a or b alternately.
    rng = random.Random(1)
    for i in range(10):
        vector = [3.0 + rng.uniform(-0.05, 0.05) for _ in range(12)]
        vector[0] += 0.75
        vector[1] += 0.75
        key = f'z-{i:02d}#0'
        features[key], rows[key] = router.context_feature(vector, vector), ('a' if i % 2 else 'b', 1.0)
    fitted = router.fit(features, rows)
    calibrated, report = router.calibrate(fitted, features, rows, target=0.95, fallback='parent')
    assert report['target_reachable'] and 0 < calibrated['threshold'] < 1
    assert report['held_out_kept_accuracy'] >= 0.95 and 0.7 < report['held_out_coverage'] < 1.0
    assert report['held_out_accuracy'] < 0.95
    chosen = dict(zip(sorted(rows), router.route_many(calibrated, [features[k] for k in sorted(rows)])))
    ambiguous = [chosen[f'z-{i:02d}#0'] for i in range(10)]
    clean = [chosen[key] == name for key, (name, _) in rows.items() if not key.startswith('z-')]
    assert ambiguous.count('parent') >= 5 and sum(clean) >= 0.9 * len(clean)
    assert calibrated['sha256'] == router._digest(calibrated) and calibrated['sha256'] != fitted['sha256']
    strict, strict_report = router.calibrate(fitted, features, rows, target=0.99, fallback='parent')
    assert strict['threshold'] > calibrated['threshold']
    assert strict_report['held_out_coverage'] < report['held_out_coverage']


def test_calibration_keeps_every_turn_when_held_out_routing_is_perfect_and_refuses_bad_inputs():
    features, rows = conversations(('a', 'b'), per_route=12, follow_up=False)
    fitted = router.fit(features, rows, fallback='parent')
    calibrated, report = router.calibrate(fitted, features, rows, target=1.0)
    assert calibrated['threshold'] == 0.0 and report['held_out_coverage'] == 1.0
    with pytest.raises(ValueError):
        router.calibrate(router.fit(features, rows), features, rows, target=0.9)
    with pytest.raises(ValueError):
        router.calibrate(fitted, features, rows, target=0.0)


def test_an_unreachable_target_abstains_on_everything():
    rng = random.Random(4)
    features, rows = {}, {}
    for i in range(24):
        vector = [rng.uniform(-1, 1) for _ in range(6)]
        key = f'c{i:02d}#0'
        features[key], rows[key] = router.context_feature(vector, vector), ('a' if rng.random() < 0.5 else 'b', 1.0)
    calibrated, report = router.calibrate(router.fit(features, rows), features, rows, target=1.0, fallback='parent')
    assert not report['target_reachable'] and calibrated['threshold'] == 2.0
    assert set(router.route_many(calibrated, list(features.values()))) == {'parent'}
    json.dumps(calibrated, allow_nan=False)


def test_admission_refuses_a_unit_that_displaces_an_earlier_route():
    features, rows = conversations(('drafting', 'scheduling', 'rooms'), per_route=16, follow_up=False)
    earlier = {key: row for key, row in rows.items() if row[0] != 'rooms'}
    previous = router.fit({key: features[key] for key in earlier}, earlier)
    good = router.fit(features, rows)
    gate = {'margin': 0.02, 'minimum': 0.9, 'min_turns': 10}
    decision = router.admit(previous, good, features, rows, added='rooms', **gate)
    assert decision['admitted'] and decision['added_recall'] == 1.0 and not decision['earlier_routes_failing']
    # A candidate fitted with rooms labels on half the scheduling conversations steals scheduling turns.
    stolen = {key: (('rooms', 1.0) if row[0] == 'scheduling' and int(key.split('-')[1][:2]) % 2 else row)
              for key, row in rows.items()}
    bad = router.fit(features, stolen)
    refused = router.admit(previous, bad, features, rows, added='rooms', **gate)
    assert not refused['admitted'] and refused['earlier_routes_failing'] == ['scheduling']
    assert refused['routes']['scheduling']['lost'] > 0 == refused['routes']['scheduling']['gained']
    assert refused['routes']['scheduling']['p_value'] < 0.05
    with pytest.raises(ValueError):
        router.admit(previous, good, features, rows, added='drafting', **gate)
    with pytest.raises(ValueError):
        router.admit(good, previous, features, rows, added='rooms', **gate)
    unverifiable = router.admit(previous, good, features, rows, added='rooms', margin=0.02, minimum=0.9, min_turns=17)
    assert not unverifiable['admitted'] and unverifiable['earlier_routes_failing'] == ['drafting', 'scheduling']
    partial = {key: row for key, row in rows.items() if row[0] != 'drafting'}
    unchecked = router.admit(previous, good, features, partial, added='rooms', **gate)
    assert not unchecked['admitted'] and unchecked['earlier_routes_unchecked'] == ['drafting']


def test_sign_test_and_noise_tolerance():
    assert router.sign_test(0, 0) == 1.0 and router.sign_test(3, 0) == 0.125
    assert router.sign_test(5, 0) == 1 / 32 and abs(router.sign_test(2, 2) - 11 / 16) < 1e-12
    # One lost turn of 16 exceeds a 2% margin but is not evidence of displacement.
    features, rows = conversations(('a', 'b', 'c'), per_route=16, follow_up=False)
    earlier = {key: row for key, row in rows.items() if row[0] != 'c'}
    previous = router.fit({key: features[key] for key in earlier}, earlier)
    nudged = dict(rows)
    nudged['a-00#0'] = ('c', 1.0)
    candidate = router.fit(features, nudged, steps=600)
    decision = router.admit(previous, candidate, features, rows, added='c', margin=0.02, minimum=0.9, min_turns=10)
    entry = decision['routes']['a']
    assert entry['lost'] == 1 and entry['drop'] > 0.02 and entry['p_value'] == 0.5
    assert not entry['failed'] and decision['admitted']
