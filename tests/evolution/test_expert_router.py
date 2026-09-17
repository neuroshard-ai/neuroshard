import copy

import pytest

from neuroshard.evolution import expert_router as router
from neuroshard.evolution.reference_data import identity


def samples():
    rows = []
    for name, directions in {
        'parent': [(8, 1, 0), (7, 0, 1), (9, -1, 0)],
        'directory': [(0, 9, 1), (1, 7, 0), (-1, 8, 1)],
        'protocol': [(0, 0, 8), (1, 0, 7), (-1, 1, 9)],
    }.items():
        for index, values in enumerate(directions):
            rows.append({'id': identity({'route': name, 'training': index}),
                         'route': name, 'features': router.normalize(list(values))})
    return rows


def fit(rows=None, **options):
    return router.fit(samples() if rows is None else rows,
                      embedding_root='1' * 64, tokenizer_root='2' * 64, **options)


def test_unlabeled_queries_select_learned_routes_and_input_order_cannot_tune_model():
    model = fit()
    assert model == fit(list(reversed(samples())))
    for name, vector in [('parent', [15, 1, 1]), ('directory', [1, 15, 1]), ('protocol', [1, 1, 15])]:
        selected = router.select(model, router.normalize(vector))
        assert selected['route'] == name and selected['confident']
        assert selected['router'] == identity(model)


def test_ambiguity_and_out_of_distribution_revert_to_supported_parent():
    model = fit(prototypes_per_route=1)
    model['minimum_margin'] = 2**40
    result = router.select(model, router.normalize([0, 100, 0]))
    assert result['nearest'] == 'directory' and result['route'] == 'parent'
    model['minimum_margin'] = 0
    model['maximum_distance'] = 0
    assert router.select(model, router.normalize([1, 1, -8]))['route'] == 'parent'


def test_new_capacity_keeps_old_supported_routes_on_their_training_distribution():
    original = fit()
    extra = [{'id': identity({'new': i}), 'route': 'new_cohort',
              'features': router.normalize([-10, i - 1, 0])} for i in range(3)]
    expanded = fit(samples() + extra)
    for point in samples():
        assert router.select(expanded, point['features'])['route'] == router.select(original, point['features'])['route']
    assert router.select(expanded, router.normalize([-15, 0, 0]))['route'] == 'new_cohort'


def test_fit_rejects_duplicate_rows_and_accidental_answer_fields():
    rows = samples()
    with pytest.raises(ValueError, match='Duplicate'):
        fit(rows + [rows[0]])
    rows[0]['answer'] = 'A held-out reference must never enter routing'
    with pytest.raises(ValueError, match='prompt features'):
        fit(rows)


@pytest.mark.parametrize('bad', [[True, 0, 1], [0, 0, 0], [0.0, 1, 2], [router.SCALE + 1, 1, 2]])
def test_noninteger_or_invalid_features_are_rejected(bad):
    with pytest.raises(ValueError):
        router.select(fit(), bad)


def test_models_bind_embedding_tokenizer_and_feature_dimensions():
    model = fit()
    for key in ('embedding_root', 'tokenizer_root', 'training_root'):
        bad = copy.deepcopy(model)
        bad[key] = None
        with pytest.raises(ValueError):
            router.validate(bad)
    with pytest.raises(ValueError, match='dimensions'):
        router.select(model, [1, 2])
    with pytest.raises(ValueError, match='fallback'):
        fit(fallback='untrained')


def test_integer_normalization_has_no_small_vector_or_signed_rounding_bias():
    positive = router.normalize([1, 1])
    negative = router.normalize([-1, -1])
    assert positive == [11585, 11585]
    assert negative == [-value for value in positive]
    assert router.normalize([1000, 1000]) == positive
    assert [router.rounded_ratio(x, 2) for x in (-5, -3, -1, 1, 3, 5)] == [-2, -2, 0, 0, 2, 2]


def test_discriminative_router_learns_unlabeled_decisions_and_preserves_fallback():
    rows = samples()
    model = router.fit_classifier(rows, fit())
    assert model == router.fit_classifier(list(reversed(rows)), fit())
    for name, values in [('parent', [20, 2, 1]), ('directory', [2, 20, 1]), ('protocol', [2, 1, 20])]:
        result = router.select(model, router.normalize(values))
        assert result['route'] == name and result['predicted'] == name
        assert result['margin'] > 0
    model['minimum_margin'] = 2**40
    result = router.select(model, router.normalize([0, 15, 1]))
    assert result['predicted'] == 'directory' and result['route'] == 'parent'
    model['minimum_margin'] = 0
    model['maximum_distance'] = 0
    assert router.select(model, router.normalize([0, 15, 1]))['route'] == 'parent'


def test_classifier_cannot_substitute_observations_or_malformed_weights():
    rows = samples()
    prototype = fit(rows)
    rows[0]['route'] = 'protocol'
    with pytest.raises(ValueError, match='committed'):
        router.fit_classifier(rows, prototype)
    model = router.fit_classifier(samples(), prototype)
    for weights in (None, [], {'parent': [1, 2, 3]}):
        altered = copy.deepcopy(model)
        altered['classifier']['weights'] = weights
        with pytest.raises(ValueError, match='classifier'):
            router.validate(altered)


def test_balanced_fitting_and_support_radius_use_only_committed_training_inputs():
    rows = samples()
    # Unequal numbers of paraphrases must not suppress a small supported class.
    rows += [{**row, 'id': identity({'paraphrase': index, 'source': row['id']})}
             for index in range(4) for row in samples() if row['route'] == 'directory']
    prototype = router.calibrate_support(rows, fit(rows, prototypes_per_route=1))
    for row in rows:
        assert min(router.distance(row['features'], center)
                   for center in prototype['prototypes'][row['route']]) <= prototype['maximum_distance']
    trained = router.fit_classifier(rows, prototype, balance_classes=True)
    assert trained == router.fit_classifier(list(reversed(rows)), prototype, balance_classes=True)
    assert trained['classifier']['method'] == 'integer-balanced-averaged-margin-perceptron-v1'
    for row in samples():
        assert router.select(trained, row['features'])['route'] == row['route']
    with pytest.raises(ValueError, match='committed training'):
        router.calibrate_support(rows[:-1], prototype)


def test_isolated_growth_preserves_the_accepted_router_and_routes_unlabeled_inputs():
    previous = router.fit_classifier(samples(), fit())
    frozen = copy.deepcopy(previous)
    extra = [{'id': identity(['growth-training', i]), 'route': 'fresh',
              'features': router.normalize([-10, i-1, 0])} for i in range(3)]
    candidate = router.append_route(previous, samples()+extra, 'fresh')
    assert previous == frozen and candidate['base'] == frozen
    assert candidate == router.append_route(previous, list(reversed(samples()+extra)), 'fresh')
    assert router.select(candidate, router.normalize([-15, 0, 0]))['route'] == 'fresh'
    for row in samples():
        decision = router.select(candidate, row['features'])
        assert decision['route'] == router.select(previous, row['features'])['route']
        assert decision['base'] == router.select(previous, row['features'], eligible=set(previous['prototypes']))
    # A later gate appends capacity without changing either earlier classifier.
    next_rows = [{'id': identity(['later-growth', i]), 'route': 'later',
                  'features': router.normalize([0, -10, i-1])} for i in range(3)]
    later = router.append_route(candidate, samples()+extra+next_rows, 'later')
    assert later['base'] == frozen and later['additions'][:-1] == candidate['additions']
    assert router.select(later, router.normalize([0, -15, 0]))['route'] == 'later'
    assert router.select(later, router.normalize([-15, 0, 0]), eligible=set(candidate['prototypes'])) == {
        **router.select(candidate, router.normalize([-15, 0, 0]), eligible=set(candidate['prototypes'])),
        'router': identity(later)}


def test_growth_rejects_changed_features_unknown_labels_and_recursive_model_trees():
    previous = fit()
    extra = [{'id': identity(['growth-training', i]), 'route': 'fresh',
              'features': router.normalize([-10, i-1, 0])} for i in range(3)]
    candidate = router.append_route(previous, samples()+extra, 'fresh')
    changed = copy.deepcopy(candidate)
    changed['additions'][0]['gate']['tokenizer_root'] = '4'*64
    with pytest.raises(ValueError, match='replace earlier routes or features'):
        router.validate(changed)
    changed = copy.deepcopy(candidate)
    changed['base'] = candidate
    with pytest.raises(ValueError, match='flat bounded'):
        router.validate(changed)
    with pytest.raises(ValueError, match='known prompt routes'):
        router.append_route(previous, [*samples(), {**extra[0], 'answer': 'leaked target'}], 'fresh')
    with pytest.raises(ValueError, match='new bounded capacity'):
        router.append_route(previous, samples(), 'protocol')
