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


@pytest.mark.parametrize('utterance', [
    'What is 38 plus 46? Give only the number.',
    'Can updating a record change its identifier?',
    'Return the words in lowercase, without explanation.',
])
def test_atomic_intent_keeps_constraints_and_self_references(utterance):
    assert policy.atomic_request([{'role': 'user', 'content': utterance}]) == utterance


@pytest.mark.parametrize('utterance', [
    'Where is the observatory and what does it measure?',
    'What is the capacity? Also, what is the range?',
    'Explain the forecast and then compare the two stations.',
])
def test_multiple_intents_still_need_decomposition(utterance):
    assert policy.atomic_request([{'role': 'user', 'content': utterance}]) is None


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


def test_general_conversation_preserves_original_intent_without_planning(monkeypatch):
    service, calls = stub_service(monkeypatch, iter([]))
    service.config.update(request_policy=policy.ASSISTANT_POLICY,
                          learned={'router': {'fallback': 'parent'}})
    seen = []
    monkeypatch.setattr(service, 'route', lambda question: {'question': question,
        'decision': {'route': 'parent'}})
    def answer(selected, question, routing_question, messages, maximum, *, whole_request):
        seen.append((selected, messages, whole_request))
        return {'question': question, 'expert': selected, 'text': '47'}, None, None
    monkeypatch.setattr(service, 'answer_atom', answer)
    messages = [{'role': 'user', 'content': 'Set the counter to forty.'},
        {'role': 'assistant', 'content': 'The unrelated station is named Zephyr.'},
        {'role': 'user', 'content': 'Add seven to it. Return only the number.'}]
    response = service.answer(messages, 64)
    assert calls == [] and seen == [('parent', messages, True)]
    assert response['planning']['path'] == 'general' and response['status'] == 'completed'
    assert response['plan'] == [messages[-1]['content']]
    assert 'Zephyr' not in response['planning']['preselection']['question']
    assert 'forty' in response['planning']['preselection']['question']


def test_specialist_atomic_request_keeps_format_instruction_and_no_reference_rewrite(monkeypatch):
    service, calls = stub_service(monkeypatch, iter([]))
    service.config.update(request_policy=policy.ASSISTANT_POLICY,
                          learned={'router': {'fallback': 'parent'}})
    request = 'Can updating a record change its identifier? Answer only yes or no.'
    response = service.answer([{'role': 'user', 'content': request}], 64)
    assert calls == [] and response['plan'] == [request]
    assert response['planning']['path'] == 'direct' and response['status'] == 'completed'


def test_general_preselection_cannot_swallow_a_mixed_request(monkeypatch):
    service, calls = stub_service(monkeypatch, iter([
        '{"questions":["Where is the observatory?","What command starts the renderer?"]}']))
    service.config.update(request_policy=policy.ASSISTANT_POLICY,
                          learned={'router': {'fallback': 'parent'}})
    monkeypatch.setattr(service, 'route', lambda question: {'question':question,'decision':{'route':'parent'}})
    response = service.answer([{'role':'user','content':
        'Where is the observatory and what command starts the renderer?'}],64)
    assert calls == ['planning'] and response['planning']['path'] == 'neural'
    assert len(response['answers']) == 2 and response['status'] == 'completed'


def test_new_domain_gate_cannot_steal_a_confident_general_request():
    from neuroshard.evolution import expert_router
    from neuroshard.evolution.reference_data import identity
    def samples(groups):
        return [{'id': identity([name, index]), 'route': name, 'features': value}
                for name, values in groups.items() for index, value in enumerate(values)]
    base_rows = samples({'parent': [[16384, 0]]*2, 'protocol': [[-16384, 0]]*2})
    base = expert_router.fit(base_rows, embedding_root='a'*64, tokenizer_root='b'*64)
    model = expert_router.append_route(base, base_rows+samples({'directory': [[0, 16384]]*2}), 'directory')
    guard_rows = samples({'parent': [[0, 16384]]*2, 'specialist': [[-16384, 0]]*2})
    guard = expert_router.fit(guard_rows, embedding_root='a'*64, tokenizer_root='b'*64)
    model['fallback_guard'] = expert_router.fit_classifier(guard_rows, guard)
    service = object.__new__(PlannedGraphNetwork)
    service.router = SimpleNamespace(features=lambda question: [0, 16384])
    service.net = SimpleNamespace(rank=0, all_owners=SimpleNamespace(exchange=lambda value: [value, None]))
    service.config = {'request_policy': policy.FORMAT, 'learned': {'router': model}}
    old = service.route('A new unrelated instruction')
    assert old['decision']['route'] == 'directory'
    service.config['request_policy'] = policy.ASSISTANT_POLICY
    new = service.route('A new unrelated instruction')
    assert new['decision']['route'] == 'parent' and new['decision']['general_guard_applied']
    assert new['decision']['gates'] == old['decision']['gates']


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
