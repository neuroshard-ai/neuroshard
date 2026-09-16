"""Native graph admission and escrow, with numerical execution tested separately."""
import copy
import hashlib
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from neuroshard.demo import protocol
from neuroshard.evolution import auditing, expert_work, expert_lifecycle as life, settlement as state
from neuroshard.evolution import serving_graph
from neuroshard.evolution.reference_data import identity
from test_serving_graph import FIXTURE, graphs, output
from test_native_audit_quorum import verdicts
from test_settlement import tx, blocks


def send(s, owner, kind, **fields):
    result = state.transition(s, tx(s, owner, kind, **fields), referee=forbidden)
    state.invariant(result)
    return result


def forbidden(*args, **kwargs):
    raise AssertionError('Native graph transitions must not execute neural tensors')


def template_for(candidate, initial):
    template = copy.deepcopy(candidate)
    template['experts']['protocol'] = copy.deepcopy(initial)
    template['descriptor']['experts'][1]['checkpoint'] = initial['checkpoint']
    return template


@pytest.fixture(params=[life.FORMAT, life.PROSPECTIVE])
def network(graphs, request):
    previous, candidate = graphs
    owners = [protocol.Identity('expert-life-owner-' + str(i)) for i in range(4)]
    validators = [{'owner': owner.public_key,
        'consensus_key': Ed25519PrivateKey.from_private_bytes(hashlib.sha256(
            ('expert-life-validator-' + str(i)).encode()).digest()).public_key().public_bytes(
                Encoding.Raw, PublicFormat.Raw).hex(),
        'bond': state.PARAMS['bond_unit'], 'liquid': 10**10} for i, owner in enumerate(owners)]
    initial = json.loads(FIXTURE.read_bytes())['initial_expert']
    profile = {'format': expert_work.FORMAT, 'parent': candidate['parent'], 'checkpoint': initial,
        'prepared': 'a'*64, 'feature_root': 'b'*64, 'feature_stages': 336,
        'batch_roots': ['c'*64], 'schedule': [0]*560, 'numerical_profile': candidate['numerical_profile']}
    manifest = {'params': state.PARAMS, 'initial_model_root': candidate['parent']['state_root'], 'data_root': 'a'*64,
        'auditing': {**auditing.QUORUM_PROFILE, 'commit_blocks': 16, 'reveal_blocks': 16},
        'expert_work': profile,
        'expert_lifecycle': {'format': life.FORMAT, 'serving_graph': previous, 'candidate_graph': candidate,
            'quality': {'policy_root': 'd'*64, 'prepared': 'e'*64, 'stages': 96},
            'price_per_token': 101, 'max_tokens': 64}}
    if request.param == life.PROSPECTIVE:
        profile = manifest['expert_lifecycle']
        profile['format'] = life.PROSPECTIVE
        profile['candidate_template'] = template_for(profile.pop('candidate_graph'), initial)
        work = manifest['expert_work']
        work.pop('feature_root')
        work['batch_count'] = len(work.pop('batch_roots'))
        work['format'] = expert_work.PROSPECTIVE
        assert candidate['experts']['protocol']['checkpoint'] not in json.dumps(manifest)
    return state.genesis('expert-life-test', validators, manifest), owners


def trained(network):
    """Seed the post-training boundary, not a claim of tensor execution.

    Native prefix/window admission and real tensor execution have their own
    regression tests. These tests isolate what a completed job may do next.
    """
    s, owners = copy.deepcopy(network[0]), network[1]
    s['expert_work']['feature_claim'] = 'f'*64
    if s['manifest']['expert_work']['format'] == expert_work.PROSPECTIVE:
        s['expert_work'].update(feature_root='b'*64, batch_roots=['c'*64])
    expert_work.settle(s, {'kind': 'expert_training', 'id': 'e'*64,
        'work_ids': [identity({'fixture_update': i}) for i in range(560)],
        'workers': [owners[0].public_key],
        'output_checkpoint': json.loads(FIXTURE.read_bytes())['candidate']['experts']['protocol']})
    state.invariant(s)
    return s


def fund(s, owners, stages):
    prior = set(s['auditing']['budgets'])
    s = send(s, owners[0], 'fund_audit', publisher=owners[0].public_key,
        auditors=[], stage_limit=stages, expires_in=256)
    key = (set(s['auditing']['budgets']) - prior).pop()
    for owner in owners[:3]:
        s = send(s, owner, 'accept_audit', budget_id=key)
    return s, key


def quality(s, owners, passed=True):
    profile = s['manifest']['expert_lifecycle']
    report = {'format': life.FORMAT + '/quality', 'policy_root': profile['quality']['policy_root'],
        'baseline_graph': identity(profile['serving_graph']),
        'candidate_graph': identity(json.loads(FIXTURE.read_bytes())['candidate']),
        'prepared': profile['quality']['prepared'], 'passed': passed, 'results_root': '1'*64}
    s, budget = fund(s, owners, 96)
    return send(s, owners[0], 'quality_expert', report=report, transcript_root='2'*64, audit_budget=budget)


def finish(s, owners, valid=True):
    s = verdicts(s, [(owner, valid) for owner in owners[:3]])
    return blocks(s, s['candidate']['deadline'] - s['height'] + 1)


def request(s, owners, question='In the fictional Luma directory, where does Ada Lane live?'):
    graph = s['expert_lifecycle']['serving_graph']
    plan = serving_graph.calls(graph, question, 64)
    ranks = {rank for call in plan for rank in serving_graph.ownership(graph, call['model'])}
    workers = {rank: owners[int(rank) % 4].public_key for rank in ranks}
    before = set(s['expert_lifecycle']['jobs'])
    s = send(s, owners[3], 'infer_expert', graph=identity(graph), question=question, max_tokens=64,
        workers=workers, max_price=20000, expires_in=2048)
    return s, (set(s['expert_lifecycle']['jobs']) - before).pop()


def response(s, owners, key):
    job = s['expert_lifecycle']['jobs'][key]
    outputs = [output(call['model'], [3, 2]) for call in job['request']['calls']]
    text, transcript = 'Example decoded answer', '3'*64
    s, budget = fund(s, owners, sum(len(row['token_ids']) for row in outputs))
    receipts = {rank: owners[int(rank) % 4].sign(life.inference_receipt(
        s['chain_id'], job, outputs, text, transcript, rank)) for rank in job['workers']}
    return send(s, owners[0], 'respond_expert', job_id=key, outputs=outputs, text=text,
        transcript_root=transcript, workers=receipts, audit_budget=budget)


def test_training_and_unsettled_quality_cannot_change_serving(network):
    s, owners = network
    before = s['serving_root']
    with pytest.raises(ValueError, match='complete expert job'):
        quality(s, owners)
    learned = trained(network)
    assert learned['serving_root'] == before
    pending = quality(learned, owners)
    assert pending['serving_root'] == before and pending['issued'] == learned['issued']
    accepted = finish(pending, owners)
    assert accepted['serving_root'] == identity(life.candidate_graph(accepted))
    assert accepted['issued'] == learned['issued']
    with pytest.raises(ValueError, match='complete expert job'):
        quality(accepted, owners)


def test_failed_quality_keeps_serving_and_training_rewards(network):
    _, owners = network
    learned = trained(network)
    after = finish(quality(learned, owners, passed=False), owners)
    assert after['serving_root'] == learned['serving_root'] and after['issued'] == learned['issued']
    assert after['expert_lifecycle']['quality_closed']


def test_forged_and_unavailable_quality_never_promote_and_allow_exact_retry(network):
    _, owners = network
    learned = trained(network)
    pending = quality(learned, owners)
    missing = blocks(pending, pending['candidate']['audit_reveal_end'] - pending['height'] + 1)
    for rejected in (missing, finish(pending, owners, valid=False)):
        assert rejected['serving_root'] == learned['serving_root']
        assert not rejected['expert_lifecycle']['quality_closed']
        retried = quality(rejected, owners)
        assert retried['candidate']['report'] == pending['candidate']['report']
        altered = copy.deepcopy(retried['candidate']['report'])
        altered['policy_root'] = '4'*64
        with pytest.raises(ValueError, match='policy'):
            send(rejected, owners[0], 'quality_expert', report=altered, transcript_root='2'*64, audit_budget='5'*64)


def test_paid_request_survives_promotion_and_bills_interpreter_without_issuance(network):
    _, owners = network
    s, key = request(trained(network), owners)
    old_graph = s['serving_root']
    s = finish(quality(s, owners), owners)
    assert s['serving_root'] != old_graph
    pending = response(s, owners, key)
    assert identity(pending['candidate']['graph']) == old_graph
    issued = pending['issued']
    after = finish(pending, owners)
    result = after['expert_lifecycle']['results'][key]
    assert result['graph'] == old_graph and result['paid_atoms'] == 404
    assert sum(result['payments'].values()) == 404 and result['refunded_atoms'] == 19596
    assert result['outputs'][0]['model'] == 'interpreter' and after['issued'] == issued
    with pytest.raises(KeyError):
        response(after, owners, key)


def test_missing_response_refunds_full_budget_after_expiry(network):
    s, owners = network
    s, key = request(s, owners)
    s = response(s, owners, key)
    s = blocks(s, s['candidate']['audit_reveal_end'] - s['height'] + 1)
    job = s['expert_lifecycle']['jobs'][key]
    s = blocks(s, job['expires'] - s['height'] + 1)
    assert s['expert_lifecycle']['results'][key]['refunded_atoms'] == 20000
    assert s['issued'] == 0


def test_graph_audit_binds_every_call_text_tokenizer_and_ordered_stage(network):
    s, owners = network
    s, key = request(s, owners)
    claim = response(s, owners, key)['candidate']
    report = {'format': life.FORMAT + '/replay', 'statement': identity(life.service_statement(claim)),
        'stages': [{'stage': i, 'valid': True} for i in range(claim['stages'])]}
    assert life.replay_report(claim, report)['valid']
    for name in ('text', 'outputs', 'graph'):
        changed = copy.deepcopy(claim)
        if name == 'text':
            changed['text'] += ' forged'
        elif name == 'outputs':
            changed['outputs'][0]['token_ids'][0] = 4
        else:
            changed['graph']['tokenizer']['files']['tokenizer.json'] = '5'*64
        with pytest.raises(ValueError, match='entire expert service'):
            life.replay_report(changed, report)
    with pytest.raises(ValueError, match='entire expert service'):
        life.replay_report(claim, {**report, 'stages': report['stages'][:-1]})
    report['stages'][-1]['valid'] = False
    assert life.replay_report(claim, report)['valid'] is False


def test_unapproved_graph_request_and_underfunded_composition_are_rejected(network):
    s, owners = network
    candidate = json.loads(FIXTURE.read_bytes())['candidate']
    with pytest.raises(ValueError, match='Serving graph changed'):
        send(s, owners[3], 'infer_expert', graph=identity(candidate), question='Hello!', max_tokens=64,
            workers={str(i): owners[i].public_key for i in range(3)}, max_price=20000, expires_in=2048)
    s = finish(quality(trained(network), owners), owners)
    question = ('NeuroShard 0.4.0: First: Which port? Second: Which token? '
        'Reply with the two short answers in order. Separate the two answers with a semicolon.')
    with pytest.raises(ValueError, match='Integer outside'):
        send(s, owners[3], 'infer_expert', graph=s['serving_root'], question=question, max_tokens=64,
            workers={str(i): owners[i % 4].public_key for i in (0, 1, 2, 4)},
            max_price=64*101, expires_in=2048)


def test_prospective_graph_binds_only_the_complete_prescribed_job(graphs):
    initial = json.loads(FIXTURE.read_bytes())['initial_expert']
    candidate = graphs[1]
    template = template_for(candidate, initial)
    original = copy.deepcopy(template)
    assert life.materialize_graph(template, candidate['experts']['protocol']) == candidate
    assert template == original
    with pytest.raises(ValueError, match='trained tail'):
        serving_graph.validate(template)
    for key, replacement in [('step', 559), ('job', 'f'*64), ('parent', 'e'*64)]:
        incomplete = {**candidate['experts']['protocol'], key: replacement}
        with pytest.raises(ValueError, match='completed prescribed'):
            life.materialize_graph(template, incomplete)
    corrupt = copy.deepcopy(candidate['experts']['protocol'])
    next(iter(corrupt['tensors'].values()))['optimizer_step'] -= 1
    with pytest.raises(ValueError):
        life.materialize_graph(template, corrupt)
