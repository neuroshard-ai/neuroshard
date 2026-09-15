import copy

import pytest

from neuroshard.evolution import cohort_questions as questions

GATES = {'bootstrap_samples': 1000, 'bootstrap_seed': 91, 'confidence': .95,
         'single_accuracy': .75, 'composed_accuracy': .5, 'gain_lower': .1}


def row(number, topics, answers):
    return {'id': str(number), 'topics': topics, 'answers': answers,
            'stratum': 'single' if len(topics) == 1 else 'composed',
            'messages': [{'role': 'user', 'content': 'NeuroShard 0.4.0 question ' + str(number)},
                         {'role': 'assistant', 'content': '; '.join(answers)}]}


def test_command_case_and_two_fact_order_are_not_repaired():
    command = row(0, ['state-directory'], ['NEUROSHARD_STATE_DIR'])
    assert questions.correct(command, 'NEUROSHARD_STATE_DIR')
    assert not questions.correct(command, 'neuroshard_state_dir')
    assert not questions.correct(command, '`NEUROSHARD_STATE_DIR`')
    pair = row(1, ['peer-port', 'token-atoms'], ['26656', '1000000'])
    assert questions.correct(pair, '26656; 1,000,000')
    assert not questions.correct(pair, '1000000; 26656')
    assert not questions.correct(pair, '26656')
    assert not questions.correct(pair, '26656; 1000000; extra')


def test_composition_is_required_even_if_every_individual_fact_improves():
    rows = [row(i, [str(i)], [str(100 + i)]) for i in range(8)]
    rows += [row(8, ['0', '1'], ['100', '101']), row(9, ['2', '3'], ['102', '103'])]
    before = [{'id': item['id'], 'text': 'wrong', 'correct': True} for item in rows]
    after = [{'id': item['id'], 'text': item['messages'][1]['content'], 'correct': False} for item in rows]
    passed = questions.decision(rows, before, after, GATES)
    assert passed['passed'] and passed['metrics']['single']['after'] == 8
    for result in after[-2:]:
        result['text'] = 'wrong'
    failed = questions.decision(rows, before, after, GATES)
    assert not failed['passed'] and failed['checks']['single_accuracy']
    assert not failed['checks']['composed_accuracy']
    with pytest.raises(ValueError, match='complete ordered'):
        questions.decision(rows, before, list(reversed(after)), GATES)


def test_duplicate_fact_wordings_do_not_inflate_independent_evidence():
    rows = [row(0, ['peer-port'], ['26656']), row(1, ['peer-port'], ['26656']),
            row(2, ['peer-port', 'cpu-threads'], ['26656', 'One'])]
    answers = [{'id': item['id'], 'text': item['messages'][1]['content']} for item in rows]
    with pytest.raises(ValueError, match='independent facts'):
        questions.decision(rows, answers, answers, GATES)
    altered = copy.deepcopy(rows)
    altered[0]['messages'][1]['content'] = 'another answer'
    with pytest.raises(ValueError, match='supervision changed'):
        questions.validate_rows(altered)
