import pytest

from neuroshard.evolution import expert_scope, expert_router
from test_expert_router import fit, samples


SCOPES = {'directory': {'input_root': '1'*64, 'subjects': ['Mara Vale', 'Ben Ford']},
          'protocol': {'input_root': '2'*64, 'subjects': ['NeuroShard', 'NEURO']}}
ROUTES = ['parent', 'directory', 'protocol']


@pytest.mark.parametrize('question,expected', [
    ('Why does the sky appear blue?', ['parent']),
    ('What is Mara Vale\'s profession?', ['directory', 'parent']),
    ('What package installs NeuroShard?', ['parent', 'protocol']),
    ('What package installs NeuroShardX?', ['parent']),
    ('What is Mara Valerian\'s profession?', ['parent']),
    ('MARA\nVALE and neuroshard', ROUTES),
])
def test_finite_domains_keep_unknown_inputs_on_general_model(question, expected):
    assert expert_scope.eligible(question, SCOPES, ROUTES, 'parent') == sorted(expected)


def test_scope_cannot_disable_fallback_or_contain_answer_payloads():
    with pytest.raises(ValueError, match='fallback'):
        expert_scope.validate({'parent': SCOPES['directory']}, ROUTES, 'parent')
    with pytest.raises(ValueError, match='input root'):
        expert_scope.validate({'directory': {**SCOPES['directory'], 'answers': ['Sofia']}}, ROUTES, 'parent')
    with pytest.raises(ValueError, match='Duplicate'):
        expert_scope.validate({'directory': {'input_root': '1'*64, 'subjects': ['Ben Ford', 'BEN FORD']}}, ROUTES, 'parent')


@pytest.mark.parametrize('linear', [False, True])
def test_ineligible_winner_cannot_capture_request(linear):
    model = fit()
    if linear:
        model = expert_router.fit_classifier(samples(), model)
    features = expert_router.normalize([0, 15, 0])
    assert expert_router.select(model, features)['route'] == 'directory'
    observed = expert_router.select(model, features, eligible=['parent'])
    assert observed['route'] == 'parent' and not observed['confident']
    assert observed['eligible'] == ['parent']
    assert set(observed['distances']) == set(ROUTES)
    with pytest.raises(ValueError, match='fallback'):
        expert_router.select(model, features, eligible=['directory'])


def test_scope_masks_before_argmax_and_preserves_other_specialists():
    model = fit()
    # Make the unsupported expert win unconditionally. Selection must still
    # choose among all eligible alternatives, not simply discard its output.
    features = expert_router.normalize([0, 0, 15])
    model['prototypes']['directory'] = [features]
    assert expert_router.select(model, features, eligible=['parent', 'protocol'])['route'] == 'protocol'


def test_structured_model_requires_its_actual_input_shape():
    scopes = {'structured': {'input_root': '3'*64, 'subjects': ['records'],
                             'record_fields': [['id', 'status']]}}
    routes = ['parent', 'structured']
    for invalid in ['What are world records?', 'Records: [broken JSON]',
                    'Records: [{"id":"A","answer":"ready"}]',
                    'Records: [{"id":"A","status":"ready"},{"other":1}]']:
        assert expert_scope.eligible(invalid, scopes, routes, 'parent') == ['parent']
    valid = 'Filter records: [{"id":"A","status":"ready"}]'
    assert expert_scope.eligible(valid, scopes, routes, 'parent') == routes
