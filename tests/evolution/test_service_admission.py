"""Standing capacity admission, finite occupation costs and atomic rollback."""
import copy

import pytest

from neuroshard.demo import protocol
from neuroshard.evolution import auditing, service_admission as admission, settlement
from neuroshard.evolution.reference_data import identity
from test_provider_hosting import market, graphs, QUESTION, ready, response
from test_expert_lifecycle import network, send, finish
from test_settlement import blocks, tx


@pytest.fixture
def capacity(market):
    state, owners, offers = copy.deepcopy(market[0]), market[1], market[2]
    state['manifest']['service_admission'] = {**admission.PROFILE, 'provider_slots': 2}
    admission.initialize(state)
    for provider in state['hosting']['providers'].values():
        provider['registration_expires'] = state['height'] + admission.PROFILE['provider_blocks']
        provider['last_seen'] = state['height']
    for owner in owners[:3]:
        state = send(state, owner, 'offer_audit_service', purpose='expert_inference',
            scope=state['serving_root'], stage_limit=128, capacity=2, expires_in=10000)
    settlement.invariant(state)
    return state, owners, offers


def admit(state, owners, offers, **changes):
    work = dict(kind='lease_expert', graph=state['serving_root'], question=QUESTION,
        max_tokens=64, offers=offers, max_price=20000, max_provider_fee=1000, expires_in=256)
    fields = dict(work=work, stage_limit=128, max_verification=100_000_000,
        max_occupancy=1_000_000, valid_until=state['height'] + 16)
    fields.update(changes)
    signed = tx(state, owners[3], 'admit_work', **fields)
    return settlement.transition(state, signed), protocol.transaction_id(signed)


def test_old_unaccepted_offer_flood_cannot_consume_any_capacity(capacity):
    state, owners, offers = capacity
    before = copy.deepcopy(state)
    for _ in range(32):
        with pytest.raises(ValueError, match='atomic admit_work'):
            send(state, owners[3], 'fund_audit', publisher=owners[0].public_key,
                auditors=[], stage_limit=1, expires_in=100000)
    assert state == before and state['auditing']['budgets'] == {}
    state, key = admit(state, owners, offers)
    assert state['auditing']['budgets'][key]['reservation'] == key
    assert state['hosting']['leases'][key]['audit_budget'] == key
    assert auditing.enough(state['auditing']['budgets'][key], lambda a: a['bond'] > 0)


@pytest.mark.parametrize('failure', ['price', 'provider', 'audit', 'expiry', 'schema'])
def test_failed_admission_locks_nothing_and_consumes_no_nonce(capacity, failure):
    state, owners, offers = capacity
    changes = {}
    if failure == 'price':
        changes['max_occupancy'] = 0
    elif failure == 'provider':
        offers = {**offers, '0': 'f'*64}
    elif failure == 'audit':
        state = send(state, owners[0], 'close_audit_service',
            service_id=next(k for k, v in state['service_admission']['services'].items()
                            if v['owner'] == owners[0].public_key))
    elif failure == 'expiry':
        state = blocks(state, 1)
        changes['valid_until'] = state['height'] - 1
    else:
        changes['work'] = {'kind': 'transfer', 'amount': 1, 'to': owners[0].public_key}
    before = copy.deepcopy(state)
    with pytest.raises(ValueError):
        admit(state, owners, offers, **changes)
    assert state == before


def test_only_native_weight_can_authorize_a_complete_audit(capacity):
    state, owners, offers = capacity
    outsider = protocol.Identity('standing-audit-outsider')
    state = send(state, owners[3], 'transfer', to=outsider.public_key, amount=20_000_000)
    with pytest.raises(ValueError, match='native voting owner'):
        send(state, outsider, 'offer_audit_service', purpose='expert_inference',
            scope=state['serving_root'], stage_limit=128, capacity=2, expires_in=10000)
    for key, row in list(state['service_admission']['services'].items()):
        if row['owner'] == owners[0].public_key:
            state = send(state, owners[0], 'close_audit_service', service_id=key)
    with pytest.raises(ValueError, match='quorum'):
        admit(state, owners, offers)


def test_paid_slot_capture_has_a_measured_burn_and_expiry_restores_honest_work(capacity):
    state, owners, offers = capacity
    state, first = admit(state, owners, offers)
    state, second = admit(state, owners, offers)
    with pytest.raises(ValueError, match='capacity'):
        admit(state, owners, offers)
    before_burn = state['burned']
    # An attacker that never executes still pays for every reserved block. The
    # auditor's unused replay fee and its collateral are separate refundable funds.
    state = blocks(state, 257)
    assert not state['hosting']['leases'] and not state['auditing']['budgets']
    assert state['burned']-before_burn == 2*257*admission.PROFILE['audit_rent_per_block']
    for row in state['auditing']['history'][-2:]:
        assert row['occupancy_burned_atoms'] == 257000
        assert row['occupancy_refunded_atoms'] == 0
        assert row['paid_atoms'] == 0 and row['slashed_atoms'] == 0
    for owner in owners:
        state = send(state, owner, 'heartbeat_provider', valid_until=state['height'] + 64)
    state, honest = admit(state, owners, offers)
    assert honest not in (first, second)
    settlement.invariant(state)


def test_closing_an_offer_does_not_cancel_an_accepted_audit_or_reuse_its_bond(capacity):
    state, owners, offers = capacity
    state, key = admit(state, owners, offers)
    budget = copy.deepcopy(state['auditing']['budgets'][key])
    service = budget['capacity']['service_ids'][owners[0].public_key]
    before = state['accounts'][owners[0].public_key]['balance']
    state = send(state, owners[0], 'close_audit_service', service_id=service)
    assert state['accounts'][owners[0].public_key]['balance'] == (
        before + state['manifest']['auditing']['auditor_bond'] - state['manifest']['params']['fee'])
    assert state['auditing']['budgets'][key] == budget
    state = blocks(state, 257)
    assert state['accounts'][owners[0].public_key]['balance'] == (
        before + 2*state['manifest']['auditing']['auditor_bond'] - state['manifest']['params']['fee'])
    settlement.invariant(state)


def test_serving_still_requires_complete_verdicts_and_refunds_unused_occupancy(capacity):
    state, owners, offers = capacity
    state, key = admit(state, owners, offers)
    state = ready(state, owners, key)
    state = response(state, owners, key)
    assert state['candidate'] and not auditing.complete(state, state['candidate'])
    state = finish(state, owners)
    assert state['issued'] == 0 and state['expert_lifecycle']['results'][key]['status'] == 'completed'
    closed = state['auditing']['history'][-1]
    assert 0 < closed['occupancy_burned_atoms'] < 257000
    assert closed['occupancy_burned_atoms'] + closed['occupancy_refunded_atoms'] == 257000
    assert all(row['free_bond'] == 2*auditing.PROFILE['auditor_bond']
               for row in state['service_admission']['services'].values())
    settlement.invariant(state)


def test_provider_registration_and_offers_have_nonrefundable_lifetime_costs(capacity):
    state, owners, _ = capacity
    newcomer = protocol.Identity('priced-provider-registration')
    state = send(state, owners[3], 'transfer', to=newcomer.public_key, amount=100_000_000)
    before = state['burned']
    state = send(state, newcomer, 'register_provider', endpoint='https://new.example',
        certificate='c'*64, collateral=10_000_000)
    price = admission.PROFILE['provider_blocks']*admission.PROFILE['provider_rent_per_block']
    assert state['burned']-before == price + state['manifest']['params']['fee']
    before = state['burned']
    state = send(state, newcomer, 'offer_expert', graph=state['serving_root'], rank=0,
        fee=0, capacity=2, expires_in=256)
    offer = next(key for key, row in state['hosting']['offers'].items() if row['owner'] == newcomer.public_key)
    assert state['burned']-before == 2*256*admission.PROFILE['offer_rent_per_block'] + state['manifest']['params']['fee']
    before = state['accounts'][newcomer.public_key]['balance']
    state = send(state, newcomer, 'cancel_expert_offer', offer_id=offer)
    assert state['accounts'][newcomer.public_key]['balance'] == before-state['manifest']['params']['fee']
    with pytest.raises(ValueError, match='registration lifetime'):
        send(state, newcomer, 'offer_expert', graph=state['serving_root'], rank=0,
            fee=0, capacity=1, expires_in=100000)


def test_rejected_claim_cannot_hold_free_provider_or_job_capacity(capacity):
    state, owners, offers = capacity
    state, key = admit(state, owners, offers)
    state = ready(state, owners, key)
    state = response(state, owners, key)
    state = finish(state, owners, valid=False)
    assert key not in state['hosting']['leases'] and key not in state['expert_lifecycle']['jobs']
    assert state['expert_lifecycle']['results'][key]['status'] == 'verification_failed'
    assert not state['auditing']['budgets']
    state, _ = admit(state, owners, offers)
    settlement.invariant(state)


def test_false_report_slashes_capacity_without_manufacturing_a_free_bond(capacity):
    from test_native_audit_quorum import verdicts
    state, owners, offers = capacity
    state = send(state, owners[3], 'offer_audit_service', purpose='expert_inference',
        scope=state['serving_root'], stage_limit=128, capacity=2, expires_in=10000)
    state, key = admit(state, owners, offers)
    state = response(ready(state, owners, key), owners, key)
    state = verdicts(state, [(owner, i == 3) for i, owner in enumerate(owners)])
    state = blocks(state, state['candidate']['deadline'] - state['height'] + 1)
    offer = next(row for row in state['service_admission']['services'].values()
                 if row['owner'] == owners[3].public_key)
    assert offer['capacity'] == 1 and offer['free_bond'] == auditing.PROFILE['auditor_bond']
    settlement.invariant(state)


def test_duplicate_admission_and_changed_scope_cannot_reuse_capacity(capacity):
    state, owners, offers = capacity
    before = copy.deepcopy(state)
    state, key = admit(state, owners, offers)
    # Reconstruct the exact original signed envelope, then submit against the
    # advanced nonce. It must not create a second job or consume another bond.
    work = dict(kind='lease_expert', graph=before['serving_root'], question=QUESTION,
        max_tokens=64, offers=offers, max_price=20000, max_provider_fee=1000, expires_in=256)
    signed = tx(before, owners[3], 'admit_work', work=work, stage_limit=128,
        max_verification=100_000_000, max_occupancy=1_000_000, valid_until=before['height']+16)
    assert protocol.transaction_id(signed) == key
    with pytest.raises(ValueError, match='nonce'):
        settlement.transition(state, signed)
    with pytest.raises(ValueError, match='current graph'):
        send(state, owners[3], 'offer_audit_service', purpose='expert_inference',
            scope='f'*64, stage_limit=128, capacity=2, expires_in=10000)
    assert len(state['auditing']['budgets']) == 1


def test_unselected_observer_cannot_pause_the_native_replay_quorum(capacity):
    state, owners, offers = capacity
    state, key = admit(state, owners, offers)
    state = response(ready(state, owners, key), owners, key)
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match='complete native replay'):
        send(state, owners[3], 'challenge', claim_id=state['candidate']['id'], stage=0,
             challenge_kind='availability', object_root='f'*64)
    assert state == before
    state = finish(state, owners)
    assert state['expert_lifecycle']['results'][key]['status'] == 'completed'


def test_expired_heartbeat_requires_live_replacement_and_reuses_the_paid_audit(capacity):
    from neuroshard.evolution import provider_quotes
    state, owners, offers = capacity
    state, key = admit(state, owners, offers, work=dict(kind='lease_expert', graph=state['serving_root'],
        question=QUESTION, max_tokens=64, offers=offers, max_price=20000,
        max_provider_fee=1000, expires_in=1024), max_occupancy=2_000_000)
    state = ready(state, owners, key)
    epoch = state['hosting']['leases'][key]['assignment_root']
    state = blocks(state, admission.PROFILE['provider_heartbeat_blocks'] + 1)
    with pytest.raises(ValueError):
        provider_quotes.replacement(state, key)
    for owner in owners[1:]:
        state = send(state, owner, 'heartbeat_provider', valid_until=state['height']+64)
    replacement = protocol.Identity('new-coordinator-capacity')
    state = send(state, owners[3], 'transfer', to=replacement.public_key, amount=100_000_000)
    state = send(state, replacement, 'register_provider', endpoint='https://replacement.example',
        certificate='e'*64, collateral=50_000_000)
    state = send(state, replacement, 'offer_expert', graph=state['serving_root'], rank=0,
        fee=100, capacity=1, expires_in=10000)
    quote = provider_quotes.replacement(state, key)
    budget = copy.deepcopy(state['auditing']['budgets'][key])
    state = send(state, owners[3], 'recover_hosted_job', job_id=key, assignment_root=epoch,
        valid_until=quote['valid_until'])
    assert state['hosting']['leases'][key]['providers']['0']['owner'] == replacement.public_key
    assert state['hosting']['leases'][key]['epoch'] == 1
    assert state['auditing']['budgets'][key] == {**budget, 'publisher': replacement.public_key}
    with pytest.raises(ValueError, match='exact failed assignment'):
        send(state, owners[3], 'recover_hosted_job', job_id=key, assignment_root=epoch,
            valid_until=quote['valid_until'])


def test_offer_renewal_does_not_create_another_runtime_slot(capacity):
    state, owners, offers = capacity
    state['manifest']['service_admission']['provider_slots'] = 1
    before = state['burned']
    state = send(state, owners[0], 'renew_expert_offer', offer_id=offers['0'],
        expires_in=12000, valid_until=state['height']+64)
    assert state['burned']-before == 2000*2*admission.PROFILE['offer_rent_per_block'] + state['manifest']['params']['fee']
    state, key = admit(state, owners, offers)
    state = send(state, owners[0], 'offer_expert', graph=state['serving_root'], rank=0,
        fee=100, capacity=1, expires_in=12000)
    alternative = next(k for k, row in state['hosting']['offers'].items()
                       if row['owner'] == owners[0].public_key and k != offers['0'])
    with pytest.raises(ValueError, match='runtime capacity'):
        admit(state, owners, {**offers, '0': alternative})
    assert list(state['hosting']['leases']) == [key]


@pytest.mark.parametrize('kind', ['reserve_expert_inputs', 'reserve_expert', 'quality_expert'])
def test_training_and_quality_obligations_use_the_same_atomic_admission(capacity, kind):
    import json
    from neuroshard.evolution import expert_work, expert_lifecycle as life
    from test_expert_lifecycle import trained, FIXTURE
    state, owners, offers = capacity
    purpose = admission.PURPOSES[kind]
    if kind == 'quality_expert':
        state = trained((state, owners))
        profile = life.profile_for(state)
        report = {'format': life.FORMAT + '/quality', 'policy_root': profile['quality']['policy_root'],
            'baseline_graph': identity(profile['serving_graph']),
            'candidate_graph': identity(json.loads(FIXTURE.read_bytes())['candidate']),
            'prepared': profile['quality']['prepared'], 'passed': True, 'results_root': '1'*64}
        work = dict(kind=kind, report=report, transcript_root='2'*64)
    elif kind == 'reserve_expert':
        state['expert_work']['feature_claim'] = 'f'*64
        state['expert_work'].update(feature_root='b'*64, batch_roots=['c'*64])
        work = dict(kind=kind, input_checkpoint=state['expert_work']['checkpoint']['checkpoint'],
                    worker=owners[0].public_key)
    else:
        work = dict(kind=kind, workers=[owner.public_key for owner in owners[:3]])
    for owner in owners[:3]:
        state = send(state, owner, 'offer_audit_service', purpose=purpose,
            scope=admission.scope(state, purpose), stage_limit=4096, capacity=1, expires_in=10000)
    issued = state['issued']
    state, key = admit(state, owners, offers, work=work, stage_limit=4096,
        max_verification=10**10, max_occupancy=10**10)
    assert state['auditing']['budgets'][key]['reservation'] is not None
    assert state['auditing']['budgets'][key]['capacity']['purpose'] == purpose
    assert state['issued'] == issued
    if kind == 'quality_expert':
        state = finish(state, owners)
        assert state['expert_lifecycle']['history'][-1]['promoted'] is True
        assert state['issued'] == issued
    settlement.invariant(state)
