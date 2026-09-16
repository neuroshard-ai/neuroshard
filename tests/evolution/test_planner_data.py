import copy

import pytest

from neuroshard.evolution.planner_data import GENERAL, prepare, targets


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


def test_only_the_final_plan_is_a_target_in_a_multiturn_training_conversation():
    class Tokenizer:
        all_special_tokens = ['<eos>']
        eos_token_id = 2

        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
            encoded = []
            for message in messages:
                encoded.extend(ord(char)+3 for char in message['role']+':'+message['content'])
                if message['role'] == 'assistant':
                    encoded.append(self.eos_token_id)
            if add_generation_prompt:
                encoded.extend(ord(char)+3 for char in 'assistant:')
            return encoded

    sample = {'id': 'sample', 'kind': 'general', 'groups': ['document:sample'], 'messages': [
        {'role': 'user', 'content': 'Earlier request'},
        {'role': 'assistant', 'content': 'Earlier factual answer'},
        {'role': 'user', 'content': 'Explain it briefly'},
        {'role': 'assistant', 'content': 'Reference answer excluded from planner fitting'}]}
    encoded = prepare([sample], Tokenizer(), max_length=2048)[0]
    eos_positions = [index for index, token in enumerate(encoded['input_ids']) if token == 2]
    assert len(eos_positions) == 2
    assert all(label == -100 for label in encoded['labels'][:eos_positions[0]+1])
    assert encoded['labels'][eos_positions[-1]] == 2
    predicted = ''.join(chr(token-3) for token in encoded['labels'] if token not in (-100, 2))
    assert predicted == '{"questions":["'+GENERAL+'"]}'
    assert encoded['targets'] == len(predicted)+1
