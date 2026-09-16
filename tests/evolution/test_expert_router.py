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
