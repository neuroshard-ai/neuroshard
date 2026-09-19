"""Ordinary quality includes the visible answer and real conversation turns."""
import copy
from types import SimpleNamespace

import pytest

from neuroshard.evolution import expert_data, expert_lifecycle, ordinary_quality as quality
from neuroshard.evolution.expert_preparation import record_set
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.reference_data import identity
from neuroshard.evolution.sharded import graph_quality


def question(name, answers, history=()):
    messages = [*history, {'role': 'user', 'content': name},
                {'role': 'assistant', 'content': '; '.join(answers)}]
    return {'id': expert_data.document_identity(messages), 'messages': messages,
            'stratum': 'single' if len(answers) == 1 else 'composed',
            'topics': [name + str(i) for i in range(len(answers))], 'answers': answers}


def response(values):
    answers = [{'question': 'Ordinary question '+str(i)+'?', 'text': value}
               for i, value in enumerate(values)]
    text = values[0] if len(values) == 1 else '\n\n'.join(row['question']+'\n'+row['text'] for row in answers)
    return {'text': text, 'answering': {'status': 'completed', 'error': None, 'answers': answers, 'text': text}}


def test_only_complete_visible_correct_answers_count():
    row = question('Which port does the peer use, and what is the token called?', ['26656', 'NEURO'])
    quality.validate_rows([row])
    actual = response(['26656', 'NEURO'])
    assert quality.correct(row, actual)
    assert not quality.correct(row, response(['NEURO', '26656']))
    hidden = copy.deepcopy(actual)
    hidden['text'] = hidden['answering']['text'] = '26656 is not the port; NEURO is not the token.'
    assert not quality.correct(row, hidden)
    failed = copy.deepcopy(actual)
    failed['answering']['status'] = 'needs_clarification'
    assert not quality.correct(row, failed)
    exact = question('Give a JSON string containing a capital A.', ['"A"'])
    exact['case_sensitive'] = [True]
    assert quality.correct(exact, response(['"A"']))
    assert not quality.correct(exact, response(['"a"']))
    with pytest.raises(ValueError, match='explicit two-question'):
        quality.validate_rows([question('First: Which port? Second: Which token?', ['26656', 'NEURO'])])


def test_multiturn_retention_executes_every_turn_and_rejects_a_lost_answer(tmp_path):
    store = Objects(tmp_path/'objects')
    examples = [question('Which port?', ['26656']), question('What is 17 plus 25?', ['42']),
        question('How many remain?', ['3'], [{'role': 'user', 'content': 'There were five apples; two were eaten.'},
                                            {'role': 'assistant', 'content': 'Understood.'}])]
    roles = {}
    for role, row in zip(graph_quality.ROLES[1:], examples):
        spec = record_set(store, role, [row])
        (tmp_path/spec['file']).write_bytes(store.get(spec['sha256']))
        roles[role] = spec
    policy = {'format': graph_quality.ORDINARY, 'roles': roles,
        'generation': {'retained_knowledge': 16, 'retained_skills': 32, 'retained_conversation': 64},
        'retention_gates': {'max_lost_correct': 0,
                            'minimum_accuracy': {role: .75 for role in graph_quality.ROLES[1:]}}}
    calls, forget = [], False
    def answer(messages, maximum, graph):
        calls.append((messages, maximum, graph))
        example = next(row for row in examples if messages == row['messages'][:-1])
        values = ['4'] if forget and graph == 'new' and len(messages) == 3 else example['answers']
        return response(values)
    network = SimpleNamespace(answer=answer)
    result = graph_quality.measured_retention(policy, tmp_path, 'old', 'new', network)
    assert result['passed'] and len(calls) == 6 and calls[-1][0] == examples[-1]['messages'][:-1]
    assert result['conversation_computations'] == []
    forget = True
    result = graph_quality.measured_retention(policy, tmp_path, 'old', 'new', network)
    assert not result['passed'] and result['lost_correct'] == 1
    policy['roles']['test'] = {'count': 5}
    assert graph_quality.stages(policy) == 8


def test_pairwise_gain_cannot_hide_composed_failure_or_count_paraphrases_twice():
    rows = [question('New fact '+str(i)+'?', [str(i)]) for i in range(12)]
    rows.append(question('What are both new values?', ['1', '2']))
    gates = {'single_accuracy': .8, 'composed_accuracy': .8, 'gain_lower': 0,
             'bootstrap_samples': 100, 'bootstrap_seed': 42, 'confidence': .95}
    before = [response(['unknown'] * len(row['answers'])) for row in rows]
    after = [response(row['answers']) for row in rows]
    assert quality.decision(rows, before, after, gates)['passed']
    after[-1] = response(['2', '1'])
    result = quality.decision(rows, before, after, gates)
    assert result['checks']['single_gain'] and not result['passed']
    rows[1]['topics'] = rows[0]['topics']
    with pytest.raises(ValueError, match='multiple phrasings'):
        quality.decision(rows, before, after, gates)


def test_retention_history_protects_the_original_scoring_criteria(tmp_path):
    store = Objects(tmp_path/'objects')
    examples = [question('Prior knowledge?', ['A']), question('Prior skill?', ['B']),
                question('Prior conversation?', ['C'])]
    anchors = {role: record_set(store, role, [row]) for role, row in zip(graph_quality.ROLES[1:], examples)}
    new = question('New knowledge?', ['D'])
    new['answer_aliases'] = [['D.']]
    original = {'messages': new['messages']}
    previous = {'format': graph_quality.ORDINARY, 'roles': {
        **anchors, 'test': record_set(store, 'test', [new])}}
    preparation = {'roles': {role: record_set(store, role, [question(role+' independent?', ['Z'])])
                              for role in ('train', 'test')}}
    policy = {'format': graph_quality.ORDINARY, 'roles': {
        **anchors, 'retained-test-knowledge': record_set(store, 'history', [examples[0], new])},
        'retention_anchors': anchors, 'prepared': store.put_json(preparation)}
    state = {'manifest': {'expert_lifecycle': {'quality': {'policy_root': store.put_json(previous)}}},
        'expert_lifecycle': {'admission': {'active': None, 'seen_documents': {
            new['id']: {'role': 'evaluation', 'object': store.put_json(original)}}}}}
    expert_data.review_retention_history(policy, store, state)
    changed = copy.deepcopy(new)
    changed['answer_aliases'] = [[]]
    policy['roles']['retained-test-knowledge'] = record_set(store, 'changed', [examples[0], changed])
    with pytest.raises(ValueError, match='scoring criteria cannot change'):
        expert_data.review_retention_history(policy, store, state)
