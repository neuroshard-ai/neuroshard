from collections import Counter
import json

import pytest

from neuroshard.evolution import incremental_facts as facts


def test_fact_cohorts_are_deterministic_balanced_and_have_disjoint_names():
    first = facts.entities(8197, 0)
    second = facts.entities(8197, 1)
    assert first == facts.entities(8197, 0)
    assert len(first) == len(second) == 160
    assert not {row['name'] for row in first} & {row['name'] for row in second}
    for attribute, choices in facts.VALUES.items():
        assert len(choices) == len(set(choices)) == 32
        assert Counter(row[attribute] for row in first) == {choice: 5 for choice in choices}
    with pytest.raises(ValueError):
        facts.entities(True, 0)
    with pytest.raises(ValueError):
        facts.entities(8197, 6)


def test_questions_are_withheld_but_the_facts_are_explicit_training_material():
    rows = {role: facts.raw_examples(8197, 0, role) for role in ('train', 'dev', 'test')}
    assert {role: len(values) for role, values in rows.items()} == {'train': 5760, 'dev': 256, 'test': 1024}
    identifiers = [row['id'] for values in rows.values() for row in values]
    assert len(identifiers) == len(set(identifiers))
    prompts = {role: {row['messages'][0]['content'] for row in values} for role, values in rows.items()}
    assert not prompts['train'] & (prompts['dev'] | prompts['test'])
    assert not prompts['dev'] & prompts['test']
    trained = {(row['task']['entity'], row['task']['attribute'], row['task']['expected'])
               for row in rows['train'] if row['task']['family'] == 'directory'}
    dev_entities = {row['task']['entity'] for row in rows['dev']}
    final_entities = {row['task']['entity'] for row in rows['test']}
    assert len(dev_entities) == 32 and len(final_entities) == 128
    assert not dev_entities & final_entities
    for role in ('dev', 'test'):
        for row in rows[role]:
            task = row['task']
            assert (task['entity'], task['attribute'], task['expected']) in trained
            assert facts.check_answer(task, row['messages'][-1]['content']) == {'valid': True, 'correct': True}
            assert json.dumps({'answer': task['expected']}, separators=(',', ':')) not in row['messages'][0]['content']
    assert Counter(row['stratum'] for row in rows['train']) == {
        'knowledge-question': 5120, 'knowledge-document': 640}


def test_complete_answer_scoring_rejects_ambiguous_or_repaired_outputs():
    task = {'family': 'directory', 'expected': 'Kyoto'}
    assert facts.check_answer(task, ' {"answer":" KYOTO "} ')['correct']
    assert not facts.check_answer(task, '{"answer":"Oslo"}')['correct']
    for text in ('{"answer":"Oslo","answer":"Kyoto"}', '{"answer":["Kyoto","Oslo"]}',
                 '```json\n{"answer":"Kyoto"}\n```', '{"answer":"Kyoto"} extra',
                 '{"answer":"Kyoto","guess":true}', '[{"answer":"Kyoto"}]'):
        assert facts.check_answer(task, text) == {'valid': False, 'correct': False}
