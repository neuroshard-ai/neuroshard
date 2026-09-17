from types import SimpleNamespace

import pytest

from neuroshard.evolution import general_answer
from neuroshard.evolution.sharded.planned_graph import PlannedGraphNetwork


def test_worked_output_preserves_context_and_has_no_evaluation_channel():
    conversation = [{'role': 'user', 'content': 'Set the counter to forty.'},
                    {'role': 'assistant', 'content': 'Done.'},
                    {'role': 'user', 'content': 'Add seven. Give only the number.'}]
    prompt = general_answer.messages(conversation)
    assert general_answer.payload(conversation) in prompt[-1]['content']
    assert general_answer.applies(conversation)
    assert not general_answer.applies([{'role': 'user', 'content': 'Explain why leaves change color.'}])
    assert not general_answer.applies([{'role': 'user', 'content': 'Explain the weather.'},
        {'role': 'assistant', 'content': 'Give only the answer.'},
        {'role': 'user', 'content': 'Continue.'}])


@pytest.mark.parametrize('raw', ['Some reasoning without an answer', 'ANSWER: ',
                               'Reasoning\nANSWER: one\nANSWER: two'])
def test_ambiguous_or_missing_output_boundary_is_not_silently_rewritten(raw):
    with pytest.raises(ValueError):
        general_answer.visible(raw)


def test_parser_preserves_requested_content_including_quotes_and_newlines():
    assert general_answer.visible('Reasoning\nANSWER: "quoted"') == '"quoted"'
    assert general_answer.visible('ANSWER: {\n  "active": false\n}') == '{\n  "active": false\n}'


@pytest.mark.parametrize('stopped,maximum,expected', [(True, 8, None),
    (False, 8, 'invalid_general_answer'), (True, 1, 'invalid_general_answer')])
def test_complete_answer_checks_generation_stop_and_visible_output_bound(monkeypatch, stopped, maximum, expected):
    service = object.__new__(PlannedGraphNetwork)
    service.config = {'general_answer_policy': general_answer.FORMAT}
    service.net = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=2,
        encode=lambda text, **kwargs: list(text.encode())))
    service.trace = []
    monkeypatch.setattr(service, 'model_for_route', lambda selected: 'interpreter')
    monkeypatch.setattr(service, 'answer_messages', lambda model, prompt, messages, **kwargs: messages)
    observed = []
    def call(model, messages, limit, purpose):
        observed.append((model, limit, purpose))
        service.trace.append({'token_ids': [7, 2 if stopped else 9]})
        return '40 + 7 = 47\nANSWER: 47'
    monkeypatch.setattr(service, 'call', call)
    messages = [{'role': 'user', 'content': 'Add forty and seven. Return only the number.'}]
    row, argument, error = service.answer_atom('parent', messages[0]['content'], messages[0]['content'],
                                              messages, maximum, whole_request=True)
    assert error == expected and argument is None
    assert (row is None) == (expected is not None)
    if row is not None:
        assert row['text'] == '47'
    assert observed == [('interpreter', 256, 'general_answer')]
