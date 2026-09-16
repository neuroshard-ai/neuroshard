import copy
import json

import pytest

from neuroshard.evolution import fusion_data as data
from neuroshard.evolution.reference_data import identity
from test_learning_reference import Tokenizer


def sources():
    directory, protocol, retained = [], [], {}
    for index in range(8):
        name = ['Mara Vale', 'Ben Ford', 'Ora Melton', 'Jalen Wick',
                'Zara Field', 'Ira Wood', 'Nora Lake', 'Alba West'][index]
        for field in data.QUESTIONS:
            answer = field+str(index)
            directory.append({'id': identity([name, field]),
                'task': {'family': 'directory', 'name': name, 'attribute': field, 'expected': answer},
                'messages': [{'role': 'user', 'content': name+' '+field+'?'},
                             {'role': 'assistant', 'content': json.dumps({'answer': answer})}]})
        answer = 'value'+str(index)
        protocol.append({'id': identity(['protocol', index]), 'topics': ['topic'+str(index)],
            'answers': [answer], 'messages': [{'role': 'user',
                'content': 'About NeuroShard 0.4.0: What is property '+str(index)+'? Give just the answer.'},
                {'role': 'assistant', 'content': answer}]})
    for role in ('train', 'dev', 'test'):
        retained[role] = [{'id': identity([role, index]), 'task': {'family': 'lookup'} if index % 2 else {},
            'messages': [{'role': 'user', 'content': role+' question '+str(index)},
                         {'role': 'assistant', 'content': '{"answer":1}' if index % 2 else 'Response.'}]}
            for index in range(8)]
    return directory, protocol, retained


def test_fusion_final_cannot_reuse_fitting_subjects_or_topics():
    original = sources()
    pool, split = data.pools(*original, seed=12)
    counts = {role: {kind: 1 for kind in ('directory', 'protocol', 'mixed', 'general', 'structured')}
              for role in ('train', 'dev', 'test')}
    prepared, _ = data.prepare(pool, counts, Tokenizer(), 2048, 13)
    for domain in split.values():
        assert not set(domain['train']) & set(domain['test'])
        assert not set(domain['dev']) & set(domain['test'])
    fitting = {group for row in prepared['train'] for group in row['groups']}
    assert all(not fitting.intersection(row['groups']) for row in prepared['test'])
    reversed_inputs = (list(reversed(original[0])), list(reversed(original[1])),
                       {role: list(reversed(rows)) for role, rows in original[2].items()})
    other, other_split = data.pools(*reversed_inputs, seed=12)
    assert split == other_split
    assert data.prepare(other, counts, Tokenizer(), 2048, 13)[0] == prepared


def test_answer_metadata_cannot_change_the_actual_trained_source():
    values = sources()
    values[0][0]['task']['expected'] = 'forged'
    with pytest.raises(ValueError, match='source conversation'):
        data.pools(*values, seed=12)


def test_expert_absence_and_answer_order_are_visible_to_scoring():
    row = {'kind': 'mixed', 'references': ['Sofia', 'neuroshard-ai']}
    assert data.correct('Sofia; neuroshard-ai', row)
    assert not data.correct('neuroshard-ai; Sofia', row)
    assert not data.correct('The question asks about Sofia; neuroshard-ai', row)
    assert data.correct('{"answer":"Sofia"}', {'kind': 'directory', 'references': ['Sofia']})
    assert data.correct('{"units":16,"city":"Suva"}',
                        {'kind': 'structured', 'references': {'city': 'Suva', 'units': 16}})


def test_heldout_single_and_mixed_groups_do_not_connect_all_bootstrap_units():
    pool, _ = data.pools(*sources(), seed=12)
    for role in ('dev', 'test'):
        single = {group for kind in ('directory', 'protocol') for row in pool[role][kind] for group in row['groups']}
        mixed = {group for row in pool[role]['mixed'] for group in row['groups']}
        assert not single & mixed
        topics_per_person = {}
        for row in pool[role]['mixed']:
            person = next(group for group in row['groups'] if group.startswith('person:'))
            topic = next(group for group in row['groups'] if group.startswith('topic:'))
            topics_per_person.setdefault(person, set()).add(topic)
        assert all(len(values) == 1 for values in topics_per_person.values())
