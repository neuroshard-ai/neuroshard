"""Curated facts stay source-bound and rehearsal excludes unused windows."""
import copy

import pytest

from neuroshard.evolution import continual_questions as questions
from neuroshard.evolution.objects import digest


def corpus():
    evidence = b'MAXIMUM = 16\n'
    facts = [{'id': 'fact-'+str(index), 'answer': str(index),
        **{role+'_question': f'Question {role} for new fact {index}?' for role in questions.ROLES},
        'source': {'path': 'source.py', 'line': 1, 'evidence': 'MAXIMUM = 16', 'sha256': digest(evidence)}}
        for index in range(48)]
    return {'format': questions.FORMAT, 'license': 'Apache-2.0',
        'source_revision': 'a'*40, 'source_repository': 'neuroshard-ai/neuroshard',
        'cohorts': [{'id': str(index), 'facts': facts[16*index:16*(index+1)]} for index in range(3)]}, evidence


def test_reconstruct_three_cohorts_and_reject_source_or_final_wording_substitution():
    curation, evidence = corpus()
    rows = questions.build(curation, lambda revision, path: evidence)
    assert {role: len(values) for role, values in rows.items()} == {'train': 288, 'dev': 72, 'test': 72}
    assert questions.build(curation, lambda revision, path: evidence) == rows
    assert not {row['id'] for row in rows['train']} & {row['id'] for row in rows['test']}
    with pytest.raises(ValueError, match='immutable referenced code'):
        questions.build(curation, lambda revision, path: b'MAXIMUM = 17\n')
    curation['cohorts'][0]['facts'][0]['train_question'] = curation['cohorts'][0]['facts'][0]['test_question']
    with pytest.raises(ValueError, match='wording must be distinct'):
        questions.build(curation, lambda revision, path: evidence)


def test_replay_uses_completed_batches_with_diverse_facts_before_repetition():
    curation, evidence = corpus()
    rows = questions.build(curation, lambda revision, path: evidence)['train']
    batches = [list(range(64)), list(range(64, 96)), list(range(96, 128))]
    selected = questions.replay_rows(rows, batches, [0, 1, 0], 24)
    assert len(selected) == 24 and len({row['id'] for row in selected}) == 24
    assert len({topic for row in selected[:16] for topic in row['topics']}) == 16
    assert all(row['cohort'] == '0' for row in selected)
    with pytest.raises(ValueError, match='actually trained'):
        questions.replay_rows(rows, batches, [0], 65)
    with pytest.raises(ValueError, match='valid prior training schedule'):
        questions.replay_rows(rows, batches, [-1], 16)
