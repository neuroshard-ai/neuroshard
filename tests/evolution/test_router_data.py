import copy

import pytest

from neuroshard.evolution.router_data import raw_questions


def test_raw_variants_exclude_reference_answers_and_preserve_entity():
    row = {'task': {'name': 'Fenn Varden', 'attribute': 'city', 'expected': 'PRIVATE_ANSWER'},
           'messages': [{'role': 'system', 'content': 'PRIVATE_SYSTEM'},
                        {'role': 'user', 'content': 'The fictional Luma directory lists Fenn Varden.'},
                        {'role': 'assistant', 'content': 'PRIVATE_ANSWER'}],
           'labels': [7], 'input_ids': [8], 'answers': ['PRIVATE_ANSWER']}
    actual = raw_questions(row, 'directory')
    changed = copy.deepcopy(row)
    changed['task']['expected'] = 'CHANGED'
    changed['messages'][-1]['content'] = 'CHANGED'
    changed['answers'], changed['labels'], changed['input_ids'] = [], [], []
    assert raw_questions(changed, 'directory') == actual
    assert all('Fenn Varden' in question and 'PRIVATE' not in question for question in actual)
    assert all('Luma' not in question and 'JSON' not in question for question in actual[1:])
    changed['task']['name'] = 'An Invented Subject'
    with pytest.raises(ValueError, match='actual input entities'):
        raw_questions(changed, 'directory')


def test_protocol_domain_and_output_decorations_are_actually_removed():
    question = 'Which PyPI package installs the NeuroShard client?'
    row = {'messages': [{'role': 'user', 'content':
        'About NeuroShard 0.4.0: ' + question + ' Return the answer without explanation.'}]}
    assert question in raw_questions(row, 'protocol')
    assert all('0.4.0' not in text and 'without explanation' not in text
               for text in raw_questions(row, 'protocol')[1:])
    row['messages'][0]['content'] = 'An unrecognized wrapper.'
    with pytest.raises(ValueError, match='Unknown protocol'):
        raw_questions(row, 'protocol')


def test_parent_keeps_user_conversation_without_answers():
    row = {'messages': [{'role': 'user', 'content': 'I have a question about photosynthesis.'},
                        {'role': 'assistant', 'content': 'REFERENCE'},
                        {'role': 'user', 'content': 'How does it work?'}]}
    assert raw_questions(row, 'parent') == [
        'I have a question about photosynthesis.\n\nHow does it work?']
