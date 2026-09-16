import pytest

from neuroshard.evolution.fusion_score import components, interval, score


def test_shared_knowledge_is_one_resampling_unit_even_through_a_third_document():
    rows = [{'groups': ['person:a', 'topic:b']}, {'groups': ['topic:b', 'person:c']},
            {'groups': ['person:c']}, {'groups': ['person:d']}]
    groups = components(rows)
    assert groups[0] == groups[1] == groups[2] != groups[3]
    result = interval(rows, [1., 1., 1., -1.], 77)
    assert result['mean'] == .5 and result['groups'] == 2
    assert result['lower_95_one_sided'] == -1


def test_loss_improvement_cannot_rescue_failed_answers_or_retention():
    rows = [{'id': kind, 'kind': kind, 'groups': [kind]} for kind in
            ('directory', 'protocol', 'mixed', 'structured', 'general')]
    answers = {row['id']: {arm: {'correct': arm == 'fusion'} for arm in ('hub', 'fusion', 'ablation')}
               for row in rows}
    losses = {row['id']: {'hub': 1., 'fusion': .5, 'ablation': 1.} for row in rows}
    rule = {'specialist_gain': .15, 'require_positive_lower_bound': True, 'mixed_accuracy': .25,
            'structured_drop': 0., 'general_loss_ucb': .02}
    assert score(rows, answers, losses, rule, 1)['passed']
    answers['mixed']['fusion']['correct'] = False
    assert not score(rows, answers, losses, rule, 1)['passed']
    answers['mixed']['fusion']['correct'] = True
    losses['general']['fusion'] = 1.03
    assert not score(rows, answers, losses, rule, 1)['passed']
    with pytest.raises(ValueError):
        score(rows, {}, losses, rule, 1)
