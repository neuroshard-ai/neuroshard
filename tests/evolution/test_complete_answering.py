"""The promoted answering policy must be the policy executed and paid for."""
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from neuroshard.evolution import answering, expert_lifecycle, serving_graph
from neuroshard.demo import protocol
from neuroshard.evolution import auditing, expert_work, settlement
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.reference_data import identity, save
from neuroshard.evolution.request_planning import FORMAT as REQUEST_POLICY
from neuroshard.evolution.sharded import graph_service, planned_graph
from neuroshard.evolution.sharded import graph_quality
from neuroshard.evolution.expert_preparation import record_set
from neuroshard.evolution.sharded.graph_execution import GraphNetwork
from test_graph_execution import prepare_graph, SOURCE
from test_expert_lifecycle import send, fund, finish
from test_ordinary_quality import question


@pytest.fixture
def complete(tmp_path):
    graph, profile = prepare_graph(tmp_path, context=256)
    graph = json.loads((tmp_path/'graph.json').read_bytes())
    config = planned_graph.configuration(graph, json.loads((tmp_path/'learned.json').read_bytes()),
        {'instruction': 'word3', 'examples': [], 'max_tokens': 4}, SOURCE, request_policy=REQUEST_POLICY)
    store = Objects(tmp_path/'objects/policies')
    bound = answering.attach(graph, config, store)
    save(tmp_path/'complete.json', bound)
    template = json.loads((tmp_path/'prospective-policy.json').read_bytes())['candidate_template']
    initial_config = copy.deepcopy(config)
    initial_config['graph'] = initial_config['learned']['graph'] = identity(template)
    template = answering.attach(template, initial_config, store)
    cases = {'test': [question('What is word7?', ['unseen']),
                      question('What are word7 and word8?', ['unseen', 'elsewhere'])],
        'retained-test-knowledge': [question('What is word9?', ['word10'])],
        'retained-test-skills': [question('What is two plus two?', ['four'])],
        'retained-test-conversation': [question('What was the word?', ['word11'],
            [{'role': 'user', 'content': 'Remember word11.'}, {'role': 'assistant', 'content': 'Understood.'}])]}
    roles = {role: record_set(store, role, rows) for role, rows in cases.items()}
    for spec in roles.values():
        (tmp_path/'quality-inputs'/spec['file']).write_bytes(store.get(spec['sha256']))
    policy = {'format': graph_quality.ORDINARY, 'baseline_graph': identity(bound),
        'candidate_template': template, 'prepared': 'a'*64, 'roles': roles,
        'generation': {'new': 4, 'retained_knowledge': 4, 'retained_skills': 4, 'retained_conversation': 4},
        'gates': {'single_accuracy': .75, 'composed_accuracy': .75, 'gain_lower': 0,
                  'bootstrap_samples': 100, 'bootstrap_seed': 42, 'confidence': .95},
        'retention_anchors': {role: spec for role, spec in roles.items() if role != 'test'},
        'retention_gates': {'max_lost_correct': 0,
                            'minimum_accuracy': {role: .5 for role in graph_quality.ROLES[1:]}}}
    graph_quality.validate_policy(policy)
    save(tmp_path/'ordinary-policy.json', policy)
    return tmp_path, graph, bound, config, store, profile


def test_policy_is_independent_of_materialized_checkpoint_and_cannot_change_limits(complete):
    home, graph, bound, config, store, _ = complete
    assert answering.load(bound, store) == config
    assert answering.core(bound) == graph and identity(bound) != identity(graph)
    with pytest.raises(ValueError, match='committed policy executor'):
        serving_graph.calls(bound, 'What is word7?', 4)
    template = json.loads((home/'prospective-policy.json').read_bytes())['candidate_template']
    initial_config = copy.deepcopy(config)
    initial_config['graph'] = initial_config['learned']['graph'] = identity(template)
    initial = answering.attach(template, initial_config, store)
    assert initial['answering'] == bound['answering']
    assert expert_lifecycle.materialize_graph(initial, graph['experts']['protocol']) == bound
    changed = copy.deepcopy(bound)
    changed['answering']['limits']['planning'] += 1
    with pytest.raises(ValueError, match='obligations'):
        answering.load(changed, store)


def test_missing_policy_fails_before_model_allocation(complete, monkeypatch):
    home, graph, bound, config, store, profile = complete
    from neuroshard.evolution.sharded import graph_execution
    def forbidden(*args):
        raise AssertionError('Do not configure a neural runtime without the committed policy')
    monkeypatch.setattr(graph_execution, 'preflight', forbidden)
    with pytest.raises(FileNotFoundError):
        GraphNetwork(bound, profile, objects=home/'objects', interpreter=home/'interpreter',
            seed=home/'seed', source_home=SOURCE, rank=0, policy_store=Objects(home/'missing'))
    changed = copy.deepcopy(config)
    changed['sources'][next(iter(changed['sources']))] = '0'*64
    bad = answering.attach(graph, changed, store)
    with pytest.raises(ValueError, match='models, sources or execution rules'):
        GraphNetwork(bad, profile, objects=home/'objects', interpreter=home/'interpreter',
            seed=home/'seed', source_home=SOURCE, rank=0, policy_store=store)


def test_neural_receipts_conserve_bounded_price_and_reject_wrong_owners(complete):
    _, _, bound, _, _, _ = complete
    call = {'model': 'protocol', 'purpose': 'answer', 'owners': [0, 1, 2, 4],
            'prompt_ids': [1, 3], 'token_ids': [3, 2]}
    response = {'format': 'neuroshard-planned-graph-service-v1/response', 'status': 'completed',
                'outputs': [call], 'generated_tokens': 2, 'text': 'word3'}
    assert sum(answering.payments(bound, 4, response, 7).values()) == 14
    assert answering.quote(bound, 4, 7)['maximum_atoms'] >= 14
    for field, value in [('owners', [0, 1, 2, 3]), ('token_ids', [2, 3]), ('purpose', []), ('model', {})]:
        changed = copy.deepcopy(response)
        changed['outputs'][0][field] = value
        with pytest.raises(ValueError):
            answering.payments(bound, 4, changed, 7)


def owner(rank, home):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'test'
    home = Path(home)
    graph = json.loads((home/'complete.json').read_bytes())
    dist.init_process_group('gloo', init_method='file://'+str(home/'complete-rendezvous'),
                            rank=rank, world_size=5, timeout=timedelta(seconds=120))
    try:
        profile = json.loads((home/'profile.json').read_bytes())
        net = GraphNetwork(graph, profile, objects=home/'objects', interpreter=home/'interpreter',
                           seed=home/'seed', source_home=SOURCE, rank=rank)
        messages = [{'role': 'user', 'content': 'What is word7?'}]
        result = net.answer(messages, 4)
        assert result == net.answer(messages, 4)
        assert result['answering']['planning']['path'] == 'direct'
        claim = {'kind': 'expert_inference', 'job_id': 'a'*64, 'graph': graph,
            'model_root': identity(graph), 'executor_root': graph['executor_root'],
            'request': result['request'], 'outputs': result['outputs'], 'text': result['text'],
            'response': result['answering'], 'stages': result['answering']['generated_tokens'],
            'record_root': '0'*64}
        claim['record_root'] = identity(graph_service.inference_transcript(claim, result))
        report, replayed = graph_service.inference_report(claim, net)
        assert expert_lifecycle.replay_report(claim, report)['valid'] and replayed == result
        forged = copy.deepcopy(claim)
        forged['text'] = forged['response']['text'] = 'fabricated answer'
        report, _ = graph_service.inference_report(forged, net)
        assert not expert_lifecycle.replay_report(forged, report)['valid']
        assert net.graph == graph
        policy = json.loads((home/'ordinary-policy.json').read_bytes())
        measured = graph_quality.evaluate(policy, home/'quality-inputs', graph, graph, net)
        assert not measured['decision']['passed']  # Unchanged random weights do not learn anything.
        assert set(measured['retention']['roles']) == set(graph_quality.ROLES[1:])
        quality_claim = {'kind': 'expert_quality', 'graph': graph, 'baseline_graph': graph,
            'model_root': identity(graph), 'executor_root': graph['executor_root'],
            'stages': graph_quality.stages(policy), 'record_root': 'a'*64,
            'report': {'format': expert_lifecycle.FORMAT+'/quality', 'policy_root': identity(policy),
                'baseline_graph': identity(graph), 'candidate_graph': identity(graph),
                'prepared': policy['prepared'], 'passed': False, 'results_root': identity(measured)}}
        quality_claim['record_root'] = identity(graph_quality.quality_transcript(quality_claim, measured))
        report, replayed = graph_quality.quality_report(quality_claim, policy, home/'quality-inputs', net)
        assert expert_lifecycle.replay_report(quality_claim, report)['valid'] and replayed == measured
        save(home/('complete-quality-'+str(rank)+'.json'), measured)
        save(home/('complete-owner-'+str(rank)+'.json'), result)
    finally:
        dist.destroy_process_group()


def test_real_shards_use_committed_policy_and_refute_forged_complete_answer(complete):
    home, _, graph, config, store, _ = complete
    mp.spawn(owner, args=(str(home),), nprocs=5, join=True)
    results = [json.loads((home/('complete-owner-'+str(rank)+'.json')).read_bytes()) for rank in range(5)]
    assert all(result == results[0] for result in results)
    # The actual distributed response enters native escrow and funded audit.
    # Receipt/quorum transitions here stay tensor-free; the numerical replay
    # and forgery rejection above run on the five real owner processes.
    template = json.loads((home/'prospective-policy.json').read_bytes())['candidate_template']
    template['descriptor']['previous_graph'] = identity(graph['descriptor'])
    candidate_config = copy.deepcopy(config)
    candidate_config['graph'] = candidate_config['learned']['graph'] = identity(template)
    template = answering.attach(template, candidate_config, store)
    initial = template['experts']['protocol']
    owners = [protocol.Identity('complete-answering-owner-'+str(i)) for i in range(5)]
    manifest = {'params': settlement.PARAMS, 'initial_model_root': graph['parent']['state_root'],
        'data_root': 'a'*64, 'auditing': {**auditing.QUORUM_PROFILE, 'commit_blocks': 16, 'reveal_blocks': 16},
        'expert_work': {'format': expert_work.PROSPECTIVE, 'parent': graph['parent'], 'checkpoint': initial,
            'prepared': 'b'*64, 'feature_stages': 3, 'batch_count': 1, 'schedule': [0]*initial['recipe']['steps'],
            'numerical_profile': graph['numerical_profile']},
        'expert_lifecycle': {'format': expert_lifecycle.PROSPECTIVE, 'serving_graph': graph,
            'candidate_template': template, 'quality': {'policy_root': 'd'*64, 'prepared': 'b'*64, 'stages': 5},
            'price_per_token': 7, 'max_tokens': 4}}
    validators = [{'owner': owner.public_key, 'consensus_key': Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(identity(['consensus', owner.public_key]))).public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw).hex(),
        'bond': settlement.PARAMS['bond_unit'], 'liquid': 10**10} for owner in owners[:4]]
    state = settlement.genesis('complete-answering-test', validators, manifest)
    workers = {str(rank): owner.public_key for rank, owner in enumerate(owners)}
    result = results[0]
    maximum = answering.quote(graph, 4, 7)['maximum_atoms']
    state = send(state, owners[0], 'infer_expert', graph=identity(graph),
        question=result['request']['messages'], max_tokens=4, workers=workers,
        max_price=maximum, expires_in=2048)
    key, job = next(iter(state['expert_lifecycle']['jobs'].items()))
    response = result['answering']
    state, budget = fund(state, owners, response['generated_tokens'])
    transcript = 'e'*64
    receipts = {rank: owners[int(rank)].sign(expert_lifecycle.answering_receipt(
        state['chain_id'], job, response, transcript, rank)) for rank in workers}
    with pytest.raises(ValueError, match='entire replayable response'):
        send(state, owners[0], 'respond_expert', job_id=key, outputs=result['outputs'], text=result['text'],
             transcript_root=transcript, workers=receipts, audit_budget=budget)
    altered = copy.deepcopy(response)
    altered['request']['messages'][0]['content'] = 'A substituted request'
    with pytest.raises(ValueError, match='reserved conversation'):
        send(state, owners[0], 'respond_answering', job_id=key, response=altered,
             transcript_root=transcript, workers=receipts, audit_budget=budget)
    pending = send(state, owners[0], 'respond_answering', job_id=key, response=response,
        transcript_root=transcript, workers=receipts, audit_budget=budget)
    rejected = finish(pending, owners, valid=False)
    assert key in rejected['expert_lifecycle']['jobs'] and rejected['issued'] == 0
    paid = finish(pending, owners)
    receipt = paid['expert_lifecycle']['results'][key]
    assert receipt['paid_atoms'] == response['generated_tokens'] * 7
    assert receipt['paid_atoms'] + receipt['refunded_atoms'] == maximum
    assert paid['issued'] == 0 and paid['serving_root'] == identity(graph)
    assert receipt['text'] == result['text'] and receipt['outputs'] == result['outputs']


def test_complete_answering_ledger_import_stays_tensor_free():
    subprocess.run([sys.executable, '-c',
        'import sys; from neuroshard.evolution import answering; assert "torch" not in sys.modules'],
        check=True, timeout=15)
