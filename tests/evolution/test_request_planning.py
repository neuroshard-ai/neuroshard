from types import SimpleNamespace

import pytest

from neuroshard.evolution import request_planning as policy
from neuroshard.evolution.sharded.planned_graph import PlannedGraphNetwork


@pytest.mark.parametrize('utterance', [
    'Where is the Aurora telescope located?',
    'How many samples fit in a sensor packet?',
    'What command starts the renderer?',
])
def test_complete_single_questions_are_not_rewritten(utterance):
    assert policy.direct_question([{'role': 'user', 'content': utterance}]) == utterance


@pytest.mark.parametrize('utterance', [
    'Where is it located?',
    'Which sensor reads temperature and which reads pressure?',
    'What is the battery capacity? How long does charging take?',
    'What does "where is it" mean?',
    'Can you compare red or green filters?',
    'What is the capacity; explain the tradeoffs?',
])
def test_context_and_compounds_keep_the_planning_path(utterance):
    assert policy.direct_question([{'role': 'user', 'content': utterance}]) is None


def test_history_cannot_be_silently_discarded():
    assert policy.direct_question([
        {'role': 'user', 'content': 'Consider an underwater sensor.'},
        {'role': 'assistant', 'content': 'Okay.'},
        {'role': 'user', 'content': 'What is the range?'}]) is None


def test_repair_preserves_content_order_and_grounded_subjects():
    messages = [{'role': 'user', 'content': 'Where does Nia Cole work and what is her role?'}]
    before = ['Where does Nia Cole work?', 'What is her role?']
    good = ['Where does Nia Cole work?', "What is Nia Cole's role?"]
    assert policy.validate_repair(before, good, messages) == good
    for bad in [
        good + ['Where does Nia Cole live?'],
        list(reversed(good)),
        ['Where does Nia Cole work?', "What is Nia Cole's salary?"],
        ['Where does Nia Cole work?', "What is Uma Reed's role?"],
        before,
    ]:
        with pytest.raises(ValueError):
            policy.validate_repair(before, bad, messages)


def stub_service(monkeypatch, generated):
    service = object.__new__(PlannedGraphNetwork)
    service.root, service.prefix = 'service', []
    service.config = {'request_policy': policy.FORMAT, 'planner': {'max_tokens': 128}}
    service.net = SimpleNamespace(world_size=2, verify_unchanged=lambda: None,
        all_owners=SimpleNamespace(exchange=lambda value: [value, value]))
    calls = []
    def call(model, messages, maximum, purpose):
        calls.append(purpose)
        service.trace.append({'token_ids': [1], 'purpose': purpose})
        return next(generated)
    monkeypatch.setattr(service, 'call', call)
    monkeypatch.setattr(service, 'route', lambda question: {'question': question, 'decision': {'route': 'sensor'}})
    monkeypatch.setattr(service, 'answer_atom', lambda selected, question, *args, **kwargs:
        ({'question': question, 'expert': selected, 'text': 'measured answer'}, None, None))
    return service, calls


def test_direct_serving_never_invokes_a_question_generator(monkeypatch):
    service, calls = stub_service(monkeypatch, iter([]))
    request = 'What command starts the renderer?'
    response = service.answer([{'role': 'user', 'content': request}], 64)
    assert response['status'] == 'completed' and response['plan'] == [request]
    assert response['planning']['path'] == 'direct' and calls == []


@pytest.mark.parametrize('repaired,accepted', [
    ('{"subjects":["the Aurora telescope"]}', True),
    ('{"subjects":["the Beta telescope"]}', False),
    ('{"subjects":[null]}', False),
    ('{"subjects":["it"]}', False),
])
def test_neural_reference_repair_is_bounded_and_fail_closed(monkeypatch, repaired, accepted):
    service, calls = stub_service(monkeypatch, iter([
        '{"questions":["Where is the Aurora telescope?","What does it measure?"]}', repaired]))
    response = service.answer([{'role': 'user', 'content':
        'Where is the Aurora telescope and what does it measure?'}], 64)
    assert calls == ['planning', 'planning_repair']
    assert (response['status'] == 'completed') == accepted
    if not accepted:
        assert response['answers'] == [] and response['error'] == 'invalid_reference_repair'


def test_reference_slots_preserve_multiple_subjects_and_reject_extra_fields():
    messages = [{'role': 'user', 'content': 'Nia Cole and Sam Vale are colleagues.'}]
    before = ['Where does she work?', 'What is his role?']
    assert policy.repair_questions(before, '{"subjects":["Nia Cole","Sam Vale\'s"]}', messages) == [
        'Where does Nia Cole work?', "What is Sam Vale's role?"]
    for raw in ('{"subjects":["Nia Cole"]}', '{"subjects":[],"subjects":[]}',
                '{"subjects":[],"questions":[]}'):
        with pytest.raises(ValueError):
            policy.repair_questions(before, raw, messages)
