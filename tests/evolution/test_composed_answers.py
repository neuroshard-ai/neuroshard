"""Composition must preserve real calls and immutable interpreter prompts."""
from copy import deepcopy
import json

import pytest

from neuroshard.evolution.sharded.composition import ComposedAnswers, questions, validate_answer
from neuroshard.evolution.sharded.interpretation import example_messages


REQUEST = ('NeuroShard 0.4.0: First: What package contains the user CLI? '
           'Second: What software supplies consensus? Reply with the two short answers in order. '
           'Separate the two answers with a semicolon.')


class Tokenizer:
    eos_token_id = 0

    def __len__(self):
        return 256

    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, ids, **kwargs):
        return bytes(token for token in ids if token).decode()


def test_executed_calls_are_bound_to_question_and_rendering():
    tokenizer, calls = Tokenizer(), []
    def generate(question, cap):
        calls.append((question, cap))
        text = 'alpha' if len(calls) == 1 else 'beta'
        return {'route': 'expert', 'ids': tokenizer.encode(text) + [0], 'text': text}
    answer = ComposedAnswers(generate, tokenizer)(REQUEST, 64)
    assert [call[0] for call in calls] == questions(REQUEST)
    assert answer['text'] == 'alpha; beta'
    assert answer['decoding'] == 'rendered-neural-calls'
    validate_answer(REQUEST, answer, tokenizer, 64)
    for change in ('question', 'call', 'render', 'ids', 'cap'):
        wrong = deepcopy(answer)
        if change == 'question':
            wrong['composition']['questions'][0] += ' Use an expected answer.'
        elif change == 'call':
            wrong['composition']['calls'][0]['ids'][0] += 1
        elif change == 'render':
            wrong['text'] = 'an answer never generated'
        elif change == 'ids':
            wrong['ids'] = [1]
        else:
            wrong['composition']['max_tokens_per_call'] = 65
        with pytest.raises(ValueError):
            validate_answer(REQUEST, wrong, tokenizer, 64)


def test_unrecognized_and_nested_requests_do_not_create_extra_calls():
    calls = []
    def generate(question, cap):
        calls.append(question)
        return {'text': question}
    net = ComposedAnswers(generate, Tokenizer())
    for request in ('An ordinary question', REQUEST.replace('Second:', 'Second: First:', 1)):
        assert questions(request) is None
        assert net(request, 4) == {'text': request}
    assert len(calls) == 2


def test_configuration_reserialization_keeps_the_original_prompt_bytes():
    examples = [('Where does Robin Finch live?', {'name': 'Robin Finch', 'field': 'city'})]
    original = [{'role': 'system', 'content': 'Interpret.'},
                {'role': 'user', 'content': json.dumps(examples[0][0])},
                {'role': 'assistant', 'content': json.dumps(examples[0][1])}]
    sorted_examples = json.loads(json.dumps(examples, sort_keys=True))
    assert json.dumps(sorted_examples[0][1]) != original[-1]['content'], 'Reproduce the actual failed prompt change'
    assert example_messages('Interpret.', examples) == original
    assert example_messages('Interpret.', sorted_examples) == original
