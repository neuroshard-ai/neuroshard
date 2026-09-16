import copy
import json

import pytest

from neuroshard.evolution.answer_plan import FORMAT, parse, render, validate


POLICY = {'format': FORMAT, 'value_decoders': {'directory': 'json-answer', 'protocol': 'text'}}
PROGRAM = {'questions': ['Which profession?', 'Which command?'], 'render': 'semicolon'}
ANSWERS = [{'question': 'Which profession?', 'expert': 'directory', 'text': '{"answer":"geologist"}'},
           {'question': 'Which command?', 'expert': 'protocol', 'text': 'neuroshard join --role provider'}]


def test_rendering_preserves_actual_values_and_requested_order():
    validate(POLICY, {'directory', 'protocol'})
    result = render(PROGRAM, ANSWERS, POLICY)
    assert result['text'] == 'geologist; neuroshard join --role provider'
    swapped = {**PROGRAM, 'questions': list(reversed(PROGRAM['questions']))}
    assert render(swapped, list(reversed(ANSWERS)), POLICY)['text'] == 'neuroshard join --role provider; geologist'
    changed = copy.deepcopy(ANSWERS)
    changed[0]['text'] = json.dumps({'answer': 'a previously unseen value'})
    assert render(PROGRAM, changed, POLICY)['text'].startswith('a previously unseen value; ')
    with pytest.raises(ValueError, match='missing'):
        render(PROGRAM, list(reversed(ANSWERS)), POLICY)
    with pytest.raises(ValueError, match='missing'):
        render(PROGRAM, ANSWERS[:1], POLICY)


@pytest.mark.parametrize('text', [
    '{"questions":["a"],"render":"short","render":"assistant"}',
    '{"questions":["a","b"],"render":"short"}',
    '{"questions":["a"],"render":"semicolon"}',
    '{"questions":["a","a"],"render":"semicolon"}',
    '{"questions":["a"],"render":"short","answer":"invented"}',
])
def test_neural_program_cannot_supply_answers_or_ambiguous_operations(text):
    with pytest.raises(ValueError):
        parse(text)


@pytest.mark.parametrize('value', ['{"answer":"first","answer":"second"}', '{"answer":42}', '{"answer":""}', 'plain text'])
def test_source_decoder_rejects_changed_or_ambiguous_value_shapes(value):
    changed = copy.deepcopy(ANSWERS)
    changed[0]['text'] = value
    with pytest.raises(ValueError):
        render(PROGRAM, changed, POLICY)
