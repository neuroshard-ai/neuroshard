import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from prepare_admitted_cohorts import accepted_graph, inherit_evaluations
sys.path.pop(0)

from neuroshard.evolution.reference_data import identity
from neuroshard.evolution.objects import Objects
from neuroshard.dataflow.store import canonical


def fixture():
    graph = {'experts': {name: {'checkpoint': identity(name)} for name in
                        ('directory', 'protocol', 'planner', 'conversation')}}
    root = identity(graph)
    state = {'height': 100, 'issued': 129_000_000, 'serving_root': root,
             'expert_lifecycle': {'serving_graph': graph, 'history': [
                 {'kind': 'quality', 'promoted': False, 'report': {'passed': False, 'candidate_graph': 'a'*64}},
                 {'kind': 'quality', 'promoted': True, 'report': {'passed': True, 'candidate_graph': root}},
                 {'kind': 'quality', 'promoted': False, 'report': {'passed': False, 'candidate_graph': 'b'*64}}]}}
    replay = {'passed': True, 'final_height': state['height'], 'final_state_root': identity(state),
              'issued_atoms': state['issued']}
    return state, replay


def test_later_rejection_carries_only_the_accepted_model_and_history():
    state, replay = fixture()
    graph, history = accepted_graph(state, replay)
    assert graph == state['expert_lifecycle']['serving_graph']
    assert len(history) == 1 and history[0]['promoted']
    graph['experts'].clear()
    history[0]['promoted'] = False
    assert len(state['expert_lifecycle']['serving_graph']['experts']) == 4
    assert state['expert_lifecycle']['history'][1]['promoted']


@pytest.mark.parametrize('field,value', [('passed', False), ('passed', 1),
    ('final_state_root', 'f'*64), ('final_height', 99), ('issued_atoms', 130_000_000)])
def test_incomplete_or_different_ledger_replay_cannot_authorize_import(field, value):
    state, replay = fixture()
    replay[field] = value
    with pytest.raises(ValueError, match='replayed native state'):
        accepted_graph(state, replay)


@pytest.mark.parametrize('attack', ['unpromoted', 'substitute', 'serving_root', 'failed_report', 'extra_expert'])
def test_replayed_but_nonaccepted_or_substituted_models_are_not_imported(attack):
    state, replay = fixture()
    entry = state['expert_lifecycle']['history'][1]
    if attack == 'unpromoted': entry['promoted'] = False
    elif attack == 'substitute': entry['report']['candidate_graph'] = 'f'*64
    elif attack == 'serving_root': state['serving_root'] = 'f'*64
    elif attack == 'failed_report': entry['report']['passed'] = False
    else: state['expert_lifecycle']['serving_graph']['experts']['feed'] = {}
    replay['final_state_root'] = identity(state)
    with pytest.raises(ValueError, match='one admitted conversation graph'):
        accepted_graph(state, replay)


def test_complete_retention_preserves_prior_rejection_probes_and_scoring(tmp_path):
    store = Objects(tmp_path/'objects')
    def row(question):
        return {'messages': [{'role': 'user', 'content': question},
                             {'role': 'assistant', 'content': 'ok'}], 'answer_aliases': [['OK']]}
    original, rejection, accepted = [row(question) for question in ('original', 'rejected probe', 'accepted test')]
    def spec(rows):
        return {'sha256': store.put(b''.join(canonical(item)+b'\n' for item in rows)), 'count': len(rows)}
    anchors = {'retained-test-knowledge': [original]}
    policy = {'roles': {'retained-test-knowledge': spec([original, rejection]), 'test': spec([accepted])}}
    result, added = inherit_evaluations(anchors, [policy], store)
    assert added == 2 and len(result['retained-test-knowledge']) == 3
    assert anchors == {'retained-test-knowledge': [original]}
    changed = copy.deepcopy(original)
    changed['answer_aliases'] = [['anything']]
    with pytest.raises(ValueError, match='scoring metadata'):
        inherit_evaluations(anchors, [{'roles': {'test': spec([changed])}}], store)
