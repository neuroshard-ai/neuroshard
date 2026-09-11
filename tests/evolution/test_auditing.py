"""Funding, fail-closed coverage, equivocal attestations and native refunds."""
import base64
import copy

import pytest

from neuroshard.demo import protocol
from neuroshard.evolution import auditing, settlement as state
from neuroshard.evolution.verification import audit, bundle
from test_settlement import scenario, tx, claim, blocks


@pytest.fixture
def case(scenario):
    s, owners, store, record, artifacts = scenario
    state.account(s, owners[0].public_key)['balance'] += s['assignment']['bond']
    s['assignment'] = None
    s['manifest']['auditing'] = {**auditing.PROFILE, 'commit_blocks': 16, 'reveal_blocks': 16}
    auditing.initialize(s)
    return s, owners, store, record, artifacts


def send(s, owner, kind, **fields):
    return state.transition(s, tx(s, owner, kind, **fields))


def fund(case, count=2, auditors=None):
    s, owners, store, record, artifacts = case
    auditors = auditors or [owners[3]]
    s = send(s, owners[0], 'fund_audit', publisher=owners[0].public_key,
             auditors=sorted(o.public_key for o in auditors), stage_limit=count, expires_in=64)
    key = next(reversed(s['auditing']['budgets']))
    for owner in auditors:
        s = send(s, owner, 'accept_audit', budget_id=key)
    return s, key


def submit(case, auditors=None, stage_limit=2):
    _, owners, store, record, artifacts = case
    s, key = fund(case, stage_limit, auditors)
    s = send(s, owners[0], 'reserve', parent=s['model_root'], round=0,
             workers=[o.public_key for o in owners[:2]], audit_budget=key)
    s = state.transition(s, claim(s, owners, store, record))
    return s, key


def attest(s, auditors):
    candidate = s['candidate']
    coverage = auditing.coverage(candidate)
    for owner in auditors:
        commitment = auditing.commitment(s['chain_id'], candidate['id'], owner.public_key, coverage, 'd'*64)
        s = send(s, owner, 'audit_commit', claim_id=candidate['id'], commitment=commitment)
    s = blocks(s, 1)
    for owner in auditors:
        s = send(s, owner, 'audit_reveal', claim_id=candidate['id'], coverage_root=coverage, salt='d'*64)
    return s


def test_honest_coverage_paid_from_escrow_not_extra_issuance(case):
    s, owners, store, record, _ = case
    initial = s['accounts'][owners[3].public_key]['balance']
    s, key = submit(case, stage_limit=7)
    # The real auditor's duty is complete replay. Settlement itself must stay
    # numerical-free on this ordinary path (tested separately below).
    for stage in range(2):
        from neuroshard.evolution.verification import Metadata
        assert audit(store, Metadata(s['candidate']['metadata']), record['record_root'], stage)['valid']
    s = attest(s, [owners[3]])
    s = blocks(s, s['manifest']['params']['challenge_blocks']+1)
    price = s['manifest']['auditing']['price_per_stage']
    assert s['issued'] == 1_000_000 and s['training_round'] == 1
    assert s['auditing']['paid_atoms'] == 2*price
    assert s['auditing']['paid_services'] == 1
    assert s['auditing']['history'][-1]['refunded_atoms'] == 5*price
    assert s['accounts'][owners[3].public_key]['balance'] == initial + 2*price - 3*s['manifest']['params']['fee']
    assert key not in s['auditing']['budgets']
    state.invariant(s)


def test_an_unchallenged_claim_without_audit_cannot_mint(case):
    s, key = submit(case)
    before = s['model_root']
    s = blocks(s, s['candidate']['audit_reveal_end']+1)
    assert s['candidate'] is None and s['issued'] == 0 and s['model_root'] == before
    assert s['settled'][-1]['reason'] == 'funded audit coverage deadline missed'
    result = s['auditing']['history'][-1]
    assert result['paid_atoms'] == 0 and result['slashed_atoms'] == auditing.PROFILE['auditor_bond']
    assert result['refunded_atoms'] == 2*auditing.PROFILE['price_per_stage']


def test_all_selected_auditors_must_cover_the_graph(case):
    s, owners, *_ = case
    s, _ = submit(case, auditors=owners[2:])
    c = s['candidate']
    s = send(s, owners[2], 'audit_commit', claim_id=c['id'], commitment=auditing.commitment(
        s['chain_id'], c['id'], owners[2].public_key, auditing.coverage(c), 'd'*64))
    s = blocks(s, c['audit_commit_end']+1)
    s = send(s, owners[2], 'audit_reveal', claim_id=c['id'], coverage_root=auditing.coverage(c), salt='d'*64)
    s = blocks(s, c['audit_reveal_end']-s['height']+1)
    assert s['issued'] == 0
    assert s['auditing']['history'][-1]['slashed_atoms'] == auditing.PROFILE['auditor_bond']


def test_reports_cannot_be_copied_to_another_auditor_or_claim(case):
    _, owners, *_ = case
    s, _ = submit(case, auditors=owners[2:])
    c = s['candidate']
    shared = auditing.commitment(s['chain_id'], c['id'], owners[2].public_key, auditing.coverage(c), 'd'*64)
    for owner in owners[2:]:
        s = send(s, owner, 'audit_commit', claim_id=c['id'], commitment=shared)
    with pytest.raises(ValueError, match='outside its window'):
        send(s, owners[2], 'audit_reveal', claim_id=c['id'], coverage_root=auditing.coverage(c), salt='d'*64)
    s = blocks(s, 1)
    with pytest.raises(ValueError, match='differs from its commitment'):
        send(s, owners[3], 'audit_reveal', claim_id=c['id'], coverage_root=auditing.coverage(c), salt='d'*64)
    with pytest.raises(ValueError, match='complete execution graph'):
        send(s, owners[2], 'audit_reveal', claim_id=c['id'], coverage_root='e'*64, salt='d'*64)


def test_unfunded_unaccepted_duplicate_and_worker_audits_rejected(case):
    s, owners, *_ = case
    with pytest.raises(ValueError, match='schema'):
        send(s, owners[0], 'reserve', parent=s['model_root'], round=0, workers=[o.public_key for o in owners[:2]])
    with pytest.raises(ValueError, match='distinct'):
        send(s, owners[0], 'fund_audit', publisher=owners[0].public_key,
             auditors=[owners[3].public_key]*2, stage_limit=2, expires_in=64)
    s = send(s, owners[0], 'fund_audit', publisher=owners[0].public_key,
             auditors=[owners[1].public_key], stage_limit=2, expires_in=64)
    key = next(iter(s['auditing']['budgets']))
    with pytest.raises(ValueError, match='accept before'):
        send(s, owners[0], 'reserve', parent=s['model_root'], round=0,
             workers=[o.public_key for o in owners[:2]], audit_budget=key)
    s = send(s, owners[1], 'accept_audit', budget_id=key)
    with pytest.raises(ValueError, match='differ from publisher'):
        send(s, owners[0], 'reserve', parent=s['model_root'], round=0,
             workers=[o.public_key for o in owners[:2]], audit_budget=key)


def test_insufficient_stage_budget_cannot_accept_partial_coverage(case):
    with pytest.raises(ValueError, match='every execution stage'):
        submit(case, stage_limit=1)


def test_unaccepted_offers_can_saturate_the_prototype_pool(case):
    """Document the admission limit; this profile is not a permissionless market."""
    s, owners, *_ = case
    for _ in range(16):
        s = send(s, owners[0], 'fund_audit', publisher=owners[0].public_key,
                 auditors=[owners[3].public_key], stage_limit=1, expires_in=100000)
    assert len(s['auditing']['budgets']) == 16
    assert all(not b['auditors'][owners[3].public_key]['bond'] for b in s['auditing']['budgets'].values())
    assert s['auditing']['paid_services'] == 0
    with pytest.raises(ValueError, match='Outstanding audit budget limit'):
        send(s, owners[1], 'fund_audit', publisher=owners[1].public_key,
             auditors=[owners[3].public_key], stage_limit=1, expires_in=64)
    key = next(iter(s['auditing']['budgets']))
    s = send(s, owners[0], 'cancel_audit', budget_id=key)
    s = send(s, owners[1], 'fund_audit', publisher=owners[1].public_key,
             auditors=[owners[3].public_key], stage_limit=1, expires_in=64)
    assert s['auditing']['history'][-1]['refunded_atoms'] == auditing.PROFILE['price_per_stage']
    state.invariant(s)


@pytest.mark.parametrize('cancel', [True, False])
def test_unreserved_budget_and_collateral_return_without_issuance(case, cancel):
    _, owners, *_ = case
    s, key = fund(case)
    if cancel:
        s = send(s, owners[0], 'cancel_audit', budget_id=key)
    else:
        s = blocks(s, 65)
    assert not s['auditing']['budgets'] and s['issued'] == 0
    assert s['auditing']['history'][-1]['slashed_atoms'] == 0
    state.invariant(s)


def test_abandoned_training_refunds_audit_contract(case):
    _, owners, *_ = case
    s, key = fund(case)
    s = send(s, owners[0], 'reserve', parent=s['model_root'], round=0,
             workers=[o.public_key for o in owners[:2]], audit_budget=key)
    with pytest.raises(ValueError, match='No available'):
        send(s, owners[0], 'cancel_audit', budget_id=key)
    s = blocks(s, s['assignment']['expires']+1)
    assert s['assignment'] is None and not s['auditing']['budgets'] and s['issued'] == 0
    assert s['auditing']['history'][-1]['slashed_atoms'] == 0


def forged_case(case):
    s, owners, store, record, artifacts = case
    forged = copy.deepcopy(store.json(record['record_root']))
    stage = 1
    trace = store.json(record['traces'][stage])
    name = trace['partition']['components'][0]
    trace['components'][name] = store.json(record['parent'])['components'][name]
    forged['traces'][stage] = store.put_json(trace)
    model = store.json(record['model_root'])
    model['components'][name] = trace['components'][name]
    forged['model_root'] = store.put_json(model)
    forged['record_root'] = store.put_json(forged)
    return s, owners, store, forged, artifacts


def test_colluding_reports_without_an_honest_observer_can_accept_fraud(case):
    """Characterize the observer assumption; coverage signatures are not proofs."""
    from neuroshard.evolution.audit_worker import replay
    s, owners, store, forged, artifacts = forged_case(case)
    s, _ = submit((s, owners, store, forged, artifacts))
    assert not replay(store, s['candidate'])['valid']
    s = attest(s, [owners[3]])
    s = blocks(s, s['manifest']['params']['challenge_blocks']+1)
    # No honest challenge was submitted. Conservation still holds even though
    # the optimistic execution claim was false: those are separate properties.
    assert s['model_root'] == forged['model_root']
    assert s['issued'] == 1_000_000 and s['auditing']['paid_services'] == 1
    state.invariant(s)


def test_audit_bond_alone_does_not_fund_an_honest_fraud_dispute(case):
    """An unfunded refutation can leave an honest auditor liable for silence."""
    from neuroshard.evolution.audit_worker import replay
    s, owners, store, forged, artifacts = forged_case(case)
    auditor = owners[3]
    fee = s['manifest']['params']['fee']
    bond = s['manifest']['auditing']['auditor_bond']
    balance = s['accounts'][auditor.public_key]['balance']
    s = send(s, auditor, 'transfer', to=owners[0].public_key,
             amount=balance-bond-3*fee)
    s, _ = submit((s, owners, store, forged, artifacts))
    assert s['accounts'][auditor.public_key]['balance'] == fee
    c = s['candidate']
    assert not replay(store, c)['valid']
    with pytest.raises(ValueError, match='Insufficient available balance'):
        send(s, auditor, 'challenge', claim_id=c['id'], stage=1,
             challenge_kind='fraud', object_root=None)
    assert s['candidate']['challenge'] is None
    s = blocks(s, c['audit_reveal_end']-s['height']+1)
    assert s['issued'] == 0 and s['auditing']['paid_atoms'] == 0
    assert s['auditing']['history'][-1]['slashed_atoms'] == bond
    state.invariant(s)


def test_proven_false_attestation_loses_collateral_and_never_gets_service_fee(case):
    s, owners, store, forged, artifacts = forged_case(case)
    stage = 1
    s, _ = submit((s, owners, store, forged, artifacts))
    s = attest(s, [owners[3]])  # Deliberately dishonest full-coverage assertion.
    claim_id = s['candidate']['id']
    s = send(s, owners[2], 'challenge', claim_id=claim_id, stage=stage, challenge_kind='fraud', object_root=None)
    for key in s['candidate']['challenge']['needed']:
        raw = store.get(key)
        for index, start in enumerate(range(0, len(raw), state.CHUNK_BYTES)):
            envelope = tx(s, owners[2], 'upload', claim_id=claim_id, object_root=key,
                          index=index, data=base64.b64encode(raw[start:start+state.CHUNK_BYTES]).decode())
            s = state.transition(s, envelope, artifacts, True, audit)
        s = state.transition(s, tx(s, owners[2], 'seal', claim_id=claim_id, object_root=key), artifacts, True, audit)
    s = state.transition(s, tx(s, owners[2], 'resolve', claim_id=claim_id), artifacts, True, audit)
    assert s['issued'] == 0 and s['audit_count'] == 1
    assert s['auditing']['history'][-1]['paid_atoms'] == 0
    assert s['auditing']['history'][-1]['slashed_atoms'] == auditing.PROFILE['auditor_bond']
    state.invariant(s)


def test_successful_availability_challenge_does_not_shorten_audit_window(case):
    _, owners, store, _, artifacts = case
    s, _ = submit(case)
    claim_id = s['candidate']['id']
    original = s['candidate']['deadline']
    from neuroshard.evolution.verification import Metadata, dependencies
    md = Metadata(s['candidate']['metadata'])
    key = dependencies(md, md.json(s['candidate']['record_root'])['traces'][1])[0]
    s = send(s, owners[2], 'challenge', claim_id=claim_id, stage=1, challenge_kind='availability', object_root=key)
    s = state.transition(s, tx(s, owners[0], 'upload', claim_id=claim_id, object_root=key,
        index=0, data=base64.b64encode(store.get(key)).decode()), artifacts, True, audit)
    s = state.transition(s, tx(s, owners[0], 'seal', claim_id=claim_id, object_root=key), artifacts, True, audit)
    assert s['candidate']['deadline'] >= original
    s = blocks(s, s['candidate']['audit_reveal_end']+1)
    assert s['issued'] == 0
    assert s['auditing']['history'][-1]['slashed_atoms'] == 0


def test_normal_settlement_never_calls_consensus_neural_referee(case, monkeypatch):
    _, owners, *_ = case
    def forbidden(*args, **kwargs):
        raise AssertionError('ordinary audit payment must not replay on validators')
    monkeypatch.setattr('neuroshard.evolution.verification.audit', forbidden)
    s, _ = submit(case)
    s = attest(s, [owners[3]])
    s = blocks(s, s['manifest']['params']['challenge_blocks']+1)
    assert s['issued'] == 1_000_000 and s['audit_count'] == 0


def test_publisher_withholding_does_not_slash_an_auditor_who_requests_data(case):
    _, owners, store, record, _ = case
    s, _ = submit(case)
    key = store.json(record['model_root'])['components']['embed']['root']
    s = send(s, owners[3], 'challenge', claim_id=s['candidate']['id'], stage=0,
             challenge_kind='availability', object_root=key)
    s = blocks(s, s['candidate']['challenge']['deadline']+1)
    assert s['issued'] == 0 and s['settled'][-1]['reason'] == 'data availability deadline missed'
    assert s['auditing']['history'][-1]['slashed_atoms'] == 0


def test_audit_worker_refuses_missing_inputs_and_replays_all_stages(case):
    from neuroshard.evolution.audit_worker import replay, required_objects, MissingArtifact, Worker
    from neuroshard.evolution.objects import Objects
    s, _, store, _, artifacts = case
    s, _ = submit(case)
    c = s['candidate']
    with pytest.raises(MissingArtifact) as caught:
        replay(artifacts, c)
    assert caught.value.key in required_objects(c)
    assert type(Worker.availability_stage(c, caught.value.key)) is int
    result = replay(store, c)
    assert result['valid'] and [r['stage'] for r in result['stages']] == [0, 1]
    assert result['object_bytes'] > 0 and result['coverage_root'] == auditing.coverage(c)
