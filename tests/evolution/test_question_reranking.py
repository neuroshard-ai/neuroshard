import copy
from types import SimpleNamespace

import pytest

from neuroshard.evolution import question_reranking as reranking
from neuroshard.evolution import semantic_questions
from neuroshard.evolution.reference_data import identity
from neuroshard.evolution.sharded.planned_graph import PlannedGraphNetwork


def fixture():
    questions = {'capacity': 'How many samples fit in a sensor packet?',
                 'metadata': 'What is the byte limit for a sensor packet descriptor?'}
    rows = [{'id': identity(name), 'label': name} for name in questions]
    semantic = {'format': semantic_questions.ADMISSION_FORMAT, 'rows': rows,
        'intents': {name: {'route': 'sensor', 'question': text} for name, text in questions.items()}}
    model = {'format': reranking.MODEL, 'repository': 'BAAI/bge-reranker-v2-m3',
        'revision': reranking.REVISION, 'upstream_sha256': 'a'*64,
        'parameters': reranking.PARAMETERS, 'max_tokens': 512, 'batch_size': 8, 'scale': 1024,
        'files': {name: {'sha256': identity(name), 'bytes': 64}
            for name in reranking.MODEL_FILES | {'model-00001-of-00001.safetensors'}}}
    policy = {'format': reranking.FORMAT, 'model': model, 'owner': 3,
        'families': {name: [{'id': identity(name), 'question': text}] for name, text in questions.items()}}
    decision = {'intent': 'capacity', 'selected': semantic['intents']['capacity'],
                'training_id': identity('capacity')}
    return policy, semantic, decision


def packet(policy, semantic, question, scores=None):
    candidates = reranking.candidates(policy, semantic, 'sensor')
    return {'profile': identity(policy['model']),
        'inputs_root': identity({'question': question, 'candidates': candidates}),
        'scores': {'capacity': -100, 'metadata': 400} if scores is None else scores}


def test_pair_selection_preserves_admission_and_uses_a_training_question():
    policy, semantic, decision = fixture()
    reranking.validate(policy, semantic, 4)
    question = 'How big may the packet description be without its samples?'
    selected = reranking.select(policy, semantic, decision, question, packet(policy, semantic, question))
    assert selected['selected'] == semantic['intents']['metadata']
    assert selected['training_id'] == identity('metadata')
    assert selected['selected']['route'] == decision['selected']['route']
    assert selected['reranking']['previous_intent'] == 'capacity'
    assert decision['intent'] == 'capacity'
    with pytest.raises(ValueError, match='rejected admission'):
        reranking.select(policy, semantic, {'selected': None}, question, packet(policy, semantic, question))


@pytest.mark.parametrize('attack', ['question', 'model', 'missing_candidate', 'extra_candidate',
                                   'noninteger', 'overflow'])
def test_replayed_scores_cannot_substitute_the_request_or_candidates(attack):
    policy, semantic, decision = fixture()
    value = packet(policy, semantic, 'What is the descriptor size?')
    if attack == 'question':
        value['inputs_root'] = identity('another question')
    elif attack == 'model':
        value['profile'] = 'b'*64
    elif attack == 'missing_candidate':
        del value['scores']['capacity']
    elif attack == 'extra_candidate':
        value['scores']['another-expert'] = 1000000
    elif attack == 'noninteger':
        value['scores']['metadata'] = True
    else:
        value['scores']['metadata'] = 2**31
    with pytest.raises(ValueError):
        reranking.select(policy, semantic, decision, 'What is the descriptor size?', value)


@pytest.mark.parametrize('attack', ['answer', 'heldout', 'remove_competitor', 'owner', 'artifact_path'])
def test_policy_cannot_hide_answers_or_unbound_material(attack):
    policy, semantic, _ = fixture()
    if attack == 'answer':
        policy['families']['capacity'][0]['answer'] = '1024'
    elif attack == 'heldout':
        policy['families']['capacity'][0]['id'] = identity('heldout-only question')
    elif attack == 'remove_competitor':
        del policy['families']['metadata']
    elif attack == 'owner':
        policy['owner'] = 4
    else:
        policy['model']['files']['../weights.safetensors'] = {'sha256': 'a'*64, 'bytes': 64}
    with pytest.raises(ValueError):
        reranking.validate(policy, semantic, 4)


def test_score_ties_have_one_replayable_result():
    policy, semantic, decision = fixture()
    result = reranking.select(policy, semantic, decision, 'Question?',
        packet(policy, semantic, 'Question?', {'metadata': 100, 'capacity': 100}))
    assert result['intent'] == 'capacity'


def test_complete_service_accepts_scores_only_from_the_declared_owner():
    policy, semantic, decision = fixture()
    question = 'How big may a packet descriptor be?'
    value = packet(policy, semantic, question)
    service = object.__new__(PlannedGraphNetwork)
    service.config = {'question_reranker': policy, 'semantic_questions': semantic}
    service.net = SimpleNamespace(rank=0, all_owners=SimpleNamespace(
        exchange=lambda _: [None, None, None, {'execution': value}]))
    assert service.rerank_question(question, decision)['intent'] == 'metadata'
    service.net.all_owners.exchange = lambda _: [{'execution': value}, None, None, {'execution': value}]
    with pytest.raises(ValueError, match='declared owner'):
        service.rerank_question(question, decision)


def test_model_inventory_pins_the_entire_family_and_declared_owner():
    from ordinary_cloud import question_reranker_assets
    policy, _, _ = fixture()
    root = identity(policy['model'])
    payload = {'configuration': {'question_reranker': policy}}
    manifest = {'question_rerankers': {root: {'files': policy['model']['files']}}}
    restored = question_reranker_assets([payload], manifest)
    assert set(restored) == {(root, 3)}
    assert {row['path'].rsplit('/', 1)[-1] for row in restored[(root, 3)].values()} == set(policy['model']['files'])
    tampered = copy.deepcopy(manifest)
    tampered['question_rerankers'][root]['files']['config.json']['bytes'] += 1
    with pytest.raises(ValueError):
        question_reranker_assets([payload], tampered)
