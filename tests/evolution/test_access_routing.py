import copy

import pytest

from neuroshard.evolution import access_routing, expert_router
from neuroshard.evolution.reference_data import identity


def test_outcomes_prefer_preservation_and_leave_missing_knowledge_unlabeled():
    label = lambda old, new: access_routing.outcome_label(old, new,
        earlier_route='protocol', candidate_route='planner')
    assert label(True, True) == 'protocol'
    assert label(True, False) == 'protocol'
    assert label(False, True) == 'planner'
    assert label(False, False) is None
    with pytest.raises(ValueError, match='measured'):
        label(None, True)


def test_c_input_unwrapping_requires_the_training_contract():
    assert access_routing.ordinary_c_question(
        'NeuroShard research protocol: Explain a training window. Provide only the answer.') == 'Explain a training window.'
    with pytest.raises(ValueError, match='Unknown'):
        access_routing.ordinary_c_question('An undeclared wrapper: question')


def test_only_training_outcomes_can_change_selector_targets():
    rows = [{'id': 'train', 'route': 'planner', 'question': 'q'},
            {'id': 'unknown-knowledge', 'route': 'planner', 'question': 'other'}]
    measured = [{'id': 'train', 'earlier_route': 'protocol', 'existing_correct': True, 'candidate_correct': True},
                {'id': 'unknown-knowledge', 'earlier_route': 'parent', 'existing_correct': False, 'candidate_correct': False}]
    assert access_routing.apply_outcomes(rows, measured) == [{**rows[0], 'route': 'protocol'}]
    with pytest.raises(ValueError, match='fitting input'):
        access_routing.apply_outcomes(rows, [{**measured[0], 'id': 'held-out-diagnostic'}])


def test_training_excludes_diagnostic_questions_and_ignores_reference_answers():
    sources = {'earlier': {}}
    selection = {'training': []}
    for route, text, extra in [
        ('protocol', 'NeuroShard 0.4.0 — What client command displays status? Give just the answer.', {}),
        ('directory', 'Where is Ada Vale from?', {'task': {'name': 'Ada Vale', 'attribute': 'city'}}),
        ('parent', 'Explain rainbows.', {}),
        ('parent', 'THE EXPOSED QUESTION?', {}),
    ]:
        key = identity(text)
        sources['earlier'][key] = {'id': key, 'messages': [{'role': 'user', 'content': text},
            {'role': 'assistant', 'content': 'unused reference'}], **extra}
        selection['training'].append({'id': key, 'file': 'earlier', 'route': route})
    c = [{'id': identity('c'), 'stratum': 'single', 'topics': ['topic'], 'messages': [
        {'role': 'user', 'content': 'NeuroShard research protocol: Explain a training window. Provide only the answer.'},
        {'role': 'assistant', 'content': 'unused C target'}]}]
    rows, report = access_routing.training_rows(selection, sources, c, ['the exposed question'])
    assert all(access_routing.question_key(row['question']) != 'the exposed question' for row in rows)
    changed = copy.deepcopy(c)
    changed[0]['messages'][-1]['content'] = 'A completely different answer'
    assert access_routing.training_rows(selection, sources, changed, ['the exposed question']) == (rows, report)
    assert report['omitted']
    assert {row['route'] for row in rows} == {'directory', 'protocol', 'parent', 'planner'}


def test_candidate_keeps_base_and_never_exposes_guard_internal_route():
    vectors = {'parent': [-16384, 0], 'protocol': [16384, 0],
               'directory': [0, 16384], 'planner': [0, -16384]}
    samples = [{'id': identity(name), 'route': name, 'features': value}
               for name, value in vectors.items() if name != 'planner']
    # Four examples are the lower fitting bound.
    samples.append({'id': identity('extra-parent'), 'route': 'parent', 'features': vectors['parent']})
    base = expert_router.fit(samples, embedding_root=identity('embedding'), tokenizer_root=identity('tokenizer'))
    original = copy.deepcopy(base)
    rows = [{'id': identity(name), 'route': name, 'question': name} for name in vectors]
    candidate = access_routing.fit(base, rows, vectors.__getitem__, epochs=8)
    assert base == original and candidate['base'] == original
    assert candidate['fallback_guard']['fallback'] == 'parent'
    for value in vectors.values():
        result = expert_router.select(candidate, value)
        assert result['route'] in vectors
        assert 'fallback_guard' in result
    changed = copy.deepcopy(candidate)
    changed['fallback_guard']['tokenizer_root'] = identity('wrong')
    with pytest.raises(ValueError, match='features'):
        expert_router.validate(changed)


def test_standalone_expert_contract_does_not_forward_another_part_of_the_request():
    from neuroshard.evolution.sharded.planned_graph import PlannedGraphNetwork
    service = object.__new__(PlannedGraphNetwork)
    service.config = {'expert_prompts': {'planner': {'prefix': 'Domain: ', 'suffix': ' Answer briefly.',
                                                     'context': 'standalone'}}, 'general_instruction': ''}
    conversation = [{'role': 'user', 'content': 'First unrelated question and a planner question.'}]
    messages = service.answer_messages('planner', 'Explain a training window.', conversation)
    assert messages == [{'role': 'user', 'content': 'Domain: Explain a training window. Answer briefly.'}]
    general = service.answer_messages('interpreter', 'Explain a training window.', conversation)
    assert 'First unrelated question' in general[-1]['content']
