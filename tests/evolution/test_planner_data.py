import copy

import pytest

from neuroshard.evolution.planner_data import GENERAL, targets


def row(text, kind, groups):
    return {'kind': kind, 'groups': groups, 'references': ['secret answer'],
            'messages': [{'role': 'user', 'content': text}, {'role': 'assistant', 'content': 'secret answer'}]}


@pytest.mark.parametrize('directory_first', [False, True])
def test_question_supervision_copies_source_order_without_factual_answers(directory_first):
    directory = 'Name the city associated with Ada Alden.'
    protocol = 'For NeuroShard 0.4.0, Which package is used?'
    questions = [directory, protocol] if directory_first else [protocol, directory]
    item = row('Answer both parts in order: '+' '.join(questions)+
        ' Give only the two short answers, separated by a semicolon.', 'mixed', ['person:Ada Alden', 'topic:package'])
    assert targets(item) == questions
    altered = copy.deepcopy(item)
    altered['references'] = ['another answer']
    altered['messages'][-1]['content'] = 'another answer'
    assert targets(altered) == questions


def test_general_data_stays_in_the_request_instead_of_a_generated_plan_copy():
    assert targets(row('Sort these supplied records: []', 'structured', ['document:one'])) == [GENERAL]
    with pytest.raises(ValueError, match='published template'):
        targets(row('Missing original question boundaries', 'mixed', ['person:Ada Alden', 'topic:package']))
