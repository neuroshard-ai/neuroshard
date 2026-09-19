"""Training-only transformations preserve targets and the actual call grammar."""
import copy

import pytest

from neuroshard.evolution import expert_curriculum as curriculum
from neuroshard.evolution.data import document_identity
from neuroshard.evolution.reference_data import identity
from neuroshard.evolution.sharded.composition import independent_questions


def inputs():
    annotations = []
    questions = [
        ('single', ['windows'], ['16'], 'How many steps per window?'),
        ('single', ['jobs'], ['4096'], 'How many steps per job?'),
        ('composed', ['windows', 'jobs'], ['16', '4096'],
         'NeuroShard research protocol: First: How many steps per window? '
         'Second: How many steps per job? Provide only the answer. '
         'Separate the two short answers with a semicolon.'),
    ]
    for stratum, topics, answers, question in questions:
        messages = [{'role': 'user', 'content': question},
                    {'role': 'assistant', 'content': '; '.join(answers)}]
        annotations.append({'id': document_identity(messages), 'messages': messages,
                            'stratum': stratum, 'topics': topics, 'answers': answers})
    training = [{'id': row['id'], 'messages': copy.deepcopy(row['messages']), 'distill': False}
                for row in annotations]
    variations = {'format': curriculum.FORMAT, 'questions': {
        'windows': ['What is the window update cap?', 'How long is one update window?',
                    'Give the window step limit.', 'How many updates fit in a window?'],
        'jobs': ['What is the whole-job update cap?', 'How long is the complete schedule?',
                 'Give the job step limit.', 'How many updates fit in the full job?'],
    }}
    return training, annotations, variations


def run(values):
    return curriculum.augment(*values, inventory=identity([row['id'] for row in values[0]]), cohort='next')


def test_atomic_training_matches_actual_serving_calls_and_has_training_only_provenance():
    values = inputs()
    before = copy.deepcopy(values)
    rows, report = run(values)
    assert values == before
    pair = values[1][-1]
    calls = independent_questions(pair['messages'][0]['content'])
    actual = {row['messages'][0]['content']: row['answers'][0] for row in rows}
    assert [actual[call] for call in calls] == pair['answers']
    assert all(row['stratum'] == 'single' and row['answers'][0] in ('16', '4096') for row in rows)
    known = {row['id'] for row in values[0]}
    assert all(set(row['training_parents']) <= known for row in report['examples'])
    assert sum(row['kind'] == 'semantic-variation' for row in report['examples']) == 8
    assert len({row['id'] for row in rows}) == len(rows)


def test_forged_annotations_and_changed_training_inventory_are_rejected():
    values = inputs()
    with pytest.raises(ValueError, match='exact pinned'):
        curriculum.augment(*values, inventory='0'*64, cohort='next')
    values[1][0]['answers'] = ['999']
    with pytest.raises(ValueError, match='actual original'):
        run(values)
    values = inputs()
    values[1][1]['topics'] = ['windows']
    with pytest.raises(ValueError, match='conflicting targets'):
        run(values)
    values = inputs()
    values[0][0]['distill'] = True
    with pytest.raises(ValueError, match='actual original'):
        run(values)


def test_missing_topics_and_contradictory_variants_are_rejected():
    values = inputs()
    del values[2]['questions']['jobs']
    with pytest.raises(ValueError, match='every training topic'):
        run(values)
    values = inputs()
    # Prefix selection differs by one topic, so offset the identical core prompt.
    values[2]['questions']['windows'][0] = values[2]['questions']['jobs'][1]
    with pytest.raises(ValueError, match='contradictory targets'):
        run(values)


def test_batches_cover_every_document_once_even_with_uneven_topic_counts():
    rows, _ = run(inputs())
    rows.append({**rows[0], 'id': 'a'*64})
    batches = curriculum.balanced_batches(rows, batch_size=2)
    assert sorted(index for batch in batches for index in batch) == list(range(len(rows)))
    assert all(len({rows[index]['topics'][0] for index in batch}) == len(batch) for batch in batches)


def test_crossed_contracts_decouple_wording_from_wrapper_and_keep_targets():
    from neuroshard.evolution.access_routing import ordinary_c_question
    # Only framed examples are admitted by the actual C training contract.
    rows = [row for row in run(inputs())[0] if row['messages'][0]['content'].startswith(
        tuple(prefix for prefix, _ in curriculum.PREFIXES))]
    excluded = ordinary_c_question(rows[0]['messages'][0]['content'])
    crossed, report = curriculum.cross_contracts(rows, [excluded.upper().rstrip('?')])
    assert report['omitted']
    assert len(crossed) == report['distinct_questions']*(len(curriculum.PREFIXES)+1)
    assert all(excluded not in row['messages'][0]['content'] for row in crossed)
    assert len({row['id'] for row in crossed}) == len(crossed)
    parents = {row['id']: row for row in rows}
    for row, origin in zip(crossed, report['examples']):
        assert all(row['answers'] == parents[key]['answers'] for key in origin['training_parents'])
    corrupted = copy.deepcopy(rows)
    corrupted[0]['messages'][-1]['content'] = 'wrong'
    with pytest.raises(ValueError, match='atomic'):
        curriculum.cross_contracts(corrupted, [])
