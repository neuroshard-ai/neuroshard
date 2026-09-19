"""Ordinary serving failures are labeled before any new training budget."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from neuroshard.evolution import serving_diagnosis as diagnosis
from neuroshard.evolution.reference_data import identity, sha256


ROOT = Path(__file__).resolve().parents[2]
FACTS = json.loads((ROOT/'config/experiments/continual-expert-facts.json').read_bytes())
PLAN = json.loads((ROOT/'config/experiments/ordinary-serving-diagnostic.json').read_bytes())


def response(plan, routes, answers, text, status='completed', error=None):
    return {'status': status, 'error': error, 'plan': plan,
            'routing': [{'decision': {'route': route}} for route in routes],
            'answers': [{'expert': route, 'text': answer} for route, answer in zip(routes, answers)],
            'text': text}


def test_knowledge_is_primary_when_gold_atomics_are_wrong():
    case = next(row for row in PLAN['cases'] if row['id'] == 'new-documents-batch')
    ordinary = response(
        ['What is the upper bound on rows in one planner training job?',
         'How many examples may one complete planner batch select?'],
        ['planner', 'planner'], ['2048', '16'], '2048; 16')
    gold = [
        response([case['atoms'][0]['gold_question']], ['planner'], ['2048'], '2048'),
        response([case['atoms'][1]['gold_question']], ['planner'], ['16'], '16'),
    ]
    row = diagnosis.diagnose(case, ordinary, gold)
    assert row['ordinary'] == 'knowledge' and row['mode'] == 'knowledge' and not row['passed']


def test_decomposition_does_not_count_when_gold_knowledge_already_failed():
    case = next(row for row in PLAN['cases'] if row['id'] == 'new-documents-batch')
    dropped = response(['How many examples may one complete planner batch select?',
                         'What is its size?'], ['planner', 'planner'], ['64', '16'], '64; 16')
    gold = [
        response([case['atoms'][0]['gold_question']], ['planner'], ['2048'], '2048'),
        response([case['atoms'][1]['gold_question']], ['planner'], ['16'], '16'),
    ]
    row = diagnosis.diagnose(case, dropped, gold)
    assert row['ordinary'] == 'decomposition' and row['mode'] == 'knowledge'


def test_lost_subject_is_decomposition_when_knowledge_is_intact():
    case = next(row for row in PLAN['cases'] if row['id'] == 'retained-two-directory')
    dropped = response(["Which city does Fenn Varden live in?", "What is his profession?"],
                        ['directory', 'directory'], ['Sofia', 'musician'], 'Sofia; musician')
    gold = [
        response([case['atoms'][0]['gold_question']], ['directory'], ['Sofia'], 'Sofia'),
        response([case['atoms'][1]['gold_question']], ['directory'], ['musician'], 'musician'),
    ]
    row = diagnosis.diagnose(case, dropped, gold)
    assert row['mode'] == 'decomposition' and row['ordinary'] == 'decomposition'


def test_wrong_expert_is_selection_and_composer_loss_is_assembly():
    mixed = next(row for row in PLAN['cases'] if row['id'] == 'retained-mixed-package-city')
    selected = response(
        ['Which city does Fenn Varden live in?', 'What is the NeuroShard client package called?'],
        ['protocol', 'protocol'], ['Sofia', 'neuroshard-ai'], 'Sofia; neuroshard-ai')
    gold = [
        response([mixed['atoms'][0]['gold_question']], ['directory'], ['Sofia'], 'Sofia'),
        response([mixed['atoms'][1]['gold_question']], ['protocol'], ['neuroshard-ai'], 'neuroshard-ai'),
    ]
    row = diagnosis.diagnose(mixed, selected, gold)
    assert row['mode'] == 'selection'
    composed = response(
        ['Which city does Fenn Varden live in?', 'What is the NeuroShard client package called?'],
        ['directory', 'protocol'], ['Sofia', 'neuroshard-ai'], 'I cannot combine those facts.')
    row = diagnosis.diagnose(mixed, composed, gold)
    assert row['mode'] == 'assembly' and row['ordinary'] == 'assembly'


def test_committed_inventory_excludes_finals_and_explicit_grammar():
    # This inventory was frozen against its historical source commit. Current
    # source binding is exercised independently below, without rewriting it.
    diagnosis.validate(PLAN, FACTS)
    user = ' '.join(message['content'] for case in PLAN['cases']
                    for message in case['messages'] if message['role'] == 'user')
    assert 'First:' not in user and 'Second:' not in user
    for fact in FACTS['cohorts'][0]['facts']:
        assert fact['test_question'] not in user
    assert {case['stratum'] for case in PLAN['cases']} == {'retained', 'new'}
    assert any(atom['expert'] == 'planner' for case in PLAN['cases'] for atom in case['atoms'])
    assert PLAN['no_training'] is True and PLAN['final_opened'] is False


def test_changed_source_is_rejected(tmp_path):
    plan = copy.deepcopy(PLAN)
    for name in diagnosis.SOURCES:
        target = tmp_path/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
        plan['sources'][name] = sha256(target)
    diagnosis.validate(plan, FACTS, tmp_path)
    (tmp_path/diagnosis.SOURCES[0]).write_text('changed source')
    with pytest.raises(ValueError, match='Frozen diagnostic source changed'):
        diagnosis.validate(plan, FACTS, tmp_path)


def test_failed_gold_control_is_classified_even_when_ordinary_reply_passes():
    case = PLAN['cases'][0]
    ordinary = response([case['atoms'][0]['gold_question']], ['protocol'], ['neuroshard-ai'], 'neuroshard-ai')
    gold = response([], [], [], '', status='needs_clarification', error='invalid_neural_plan')
    row = diagnosis.diagnose(case, ordinary, [gold])
    assert row['ordinary'] is None and row['mode'] == 'decomposition' and not row['passed']
    assert diagnosis.summarize([row])['totals']['decomposition'] == 1


def test_explicit_grammar_or_opened_final_cannot_be_added():
    case = copy.deepcopy(PLAN['cases'][0])
    finals = diagnosis.forbidden_finals(FACTS)
    case['messages'] = [{'role': 'user', 'content': 'First: a? Second: b?'}]
    with pytest.raises(ValueError, match='explicit two-question'):
        diagnosis.validate_case(case, finals)
    case = copy.deepcopy(PLAN['cases'][0])
    case['messages'] = [{'role': 'user', 'content': FACTS['cohorts'][0]['facts'][0]['test_question']}]
    with pytest.raises(ValueError, match='final wording'):
        diagnosis.validate_case(case, finals)


def test_diagnosis_import_does_not_load_neural_runtime():
    env = {**__import__('os').environ, 'PYTHONPATH': str(ROOT/'src')}
    subprocess.run([sys.executable, '-c',
        'from neuroshard.evolution import serving_diagnosis; import sys; assert "torch" not in sys.modules'],
        check=True, timeout=15, env=env)


def test_traces_must_cover_the_frozen_cases():
    with pytest.raises(ValueError, match='exactly once'):
        diagnosis.score_traces(PLAN, [{'id': 'missing', 'response': response(['q'], ['parent'], ['a'], 'a')}])
    identity(PLAN)
