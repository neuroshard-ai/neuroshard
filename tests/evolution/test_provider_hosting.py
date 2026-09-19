"""Adversarial native provider assignments; no neural execution is trusted here."""
import copy

import pytest

from neuroshard.demo import protocol
from neuroshard.evolution import expert_lifecycle as life, hosting, settlement
from neuroshard.evolution.reference_data import identity
from test_expert_lifecycle import network, send, finish
from test_serving_graph import graphs, output
from test_settlement import blocks, tx


QUESTION = 'In the fictional Luma directory, where does Ada Lane live?'


@pytest.fixture
def market(network):
    state, owners = copy.deepcopy(network[0]), network[1]
    state['manifest']['params'] = {**state['manifest']['params'], 'max_claim_blocks': 128}
    state['manifest']['hosting'] = {**hosting.PROFILE, 'prepare_blocks': 4,
        'execution_blocks': 8, 'cooldown_blocks': 2}
    hosting.initialize(state)
    settlement.invariant(state)
    offers = {}
    for rank, owner in enumerate(owners):
        state = send(state, owner, 'register_provider', endpoint=f'https://owner-{rank}.example:8443',
                     certificate=str(rank + 1)*64, collateral=50*hosting.PROFILE['lease_bond'])
        before = set(state['hosting']['offers'])
        state = send(state, owner, 'offer_expert', graph=state['serving_root'], rank=rank,
                     fee=100 + rank, capacity=2, expires_in=10000)
        offers[str(rank)] = (set(state['hosting']['offers']) - before).pop()
    return state, owners, offers


def audit(state, owners, *, publisher=None, stages=128):
    before = set(state['auditing']['budgets'])
    state = send(state, owners[3], 'fund_audit', publisher=publisher or owners[0].public_key,
                 auditors=[], stage_limit=stages, expires_in=10000)
    key = (set(state['auditing']['budgets']) - before).pop()
    for owner in owners[:3]:
        state = send(state, owner, 'accept_audit', budget_id=key)
    return state, key


def lease(state, owners, offers, **changes):
    state, budget = audit(state, owners)
    fields = dict(graph=state['serving_root'], question=QUESTION, max_tokens=64,
                  offers=offers, max_price=20000, max_provider_fee=1000,
                  audit_budget=budget, expires_in=256)
    fields.update(changes)
    signed = tx(state, owners[3], 'lease_expert', **fields)
    state = settlement.transition(state, signed)
    settlement.invariant(state)
    return state, protocol.transaction_id(signed)


def ready(state, owners, key):
    assignment = state['hosting']['leases'][key]['assignment_root']
    for owner in owners:
        state = send(state, owner, 'accept_hosted_job', job_id=key, assignment_root=assignment)
    assert state['hosting']['leases'][key]['status'] == 'ready'
    return state


def response(state, owners, key, *, receipts=None, sender=None):
    job = state['expert_lifecycle']['jobs'][key]
    outputs = [output(call['model'], [3, 2]) for call in job['request']['calls']]
    text, transcript = 'A bounded response', '3'*64
    if receipts is None:
        identities = {owner.public_key: owner for owner in owners}
        receipts = {rank: identities[key].sign(life.inference_receipt(
            state['chain_id'], job, outputs, text, transcript, rank)) for rank, key in job['workers'].items()}
    return send(state, sender or owners[0], 'respond_expert', job_id=key, outputs=outputs, text=text,
        transcript_root=transcript, workers=receipts, audit_budget=state['hosting']['leases'][key]['audit_budget'])


def test_new_nonvalidator_can_register_without_an_operator_allowlist(market):
    state, owners, _ = market
    newcomer = protocol.Identity('a-self-managed-provider')
    state = send(state, owners[0], 'transfer', to=newcomer.public_key, amount=2_000_000)
    issued = state['issued']
    state = send(state, newcomer, 'register_provider', endpoint='https://new.example',
                 certificate='a'*64, collateral=1_000_000)
    assert newcomer.public_key in state['hosting']['providers']
    assert newcomer.public_key not in {row['owner'] for row in state['validators'].values()}
    assert state['issued'] == issued
    for endpoint in ('http://new.example', 'https://user:password@new.example',
                     'https://new.example/path', 'https://new.example?secret=value'):
        with pytest.raises(ValueError, match='HTTPS'):
            send(state, newcomer, 'update_provider', endpoint=endpoint, certificate='a'*64, deposit=0)


def test_lease_requires_complete_preaccepted_auditing_and_capacity(market):
    state, owners, offers = market
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match='Prepay'):
        lease(state, owners, offers, audit_budget='f'*64)
    assert state == before
    state, budget = audit(state, owners, stages=1)
    with pytest.raises(ValueError, match='Prepay'):
        lease(state, owners, offers, audit_budget=budget)
    state, key = lease(state, owners, offers)
    bound = state['hosting']['leases'][key]
    assert state['auditing']['budgets'][bound['audit_budget']]['reservation'] == key
    assert state['hosting']['providers'][owners[0].public_key]['locked'] == (
        hosting.PROFILE['lease_bond'] + state['manifest']['params']['claim_bond'])
    assert all(state['hosting']['providers'][owner.public_key]['locked'] == hosting.PROFILE['lease_bond']
               for owner in owners[1:])
    state, _ = lease(state, owners, offers)
    with pytest.raises(ValueError, match='capacity'):
        lease(state, owners, offers)
    with pytest.raises(ValueError, match='lease_expert'):
        send(state, owners[3], 'infer_expert', graph=state['serving_root'], question=QUESTION,
             max_tokens=64, workers={r: owners[int(r)].public_key for r in offers},
             max_price=20000, expires_in=2048)


def test_offer_updates_do_not_mutate_an_existing_assignment(market):
    state, owners, offers = market
    state, key = lease(state, owners, offers)
    saved = copy.deepcopy(state['hosting']['leases'][key])
    state = send(state, owners[0], 'update_provider', endpoint='https://replacement.example',
                 certificate='b'*64, deposit=1)
    state = send(state, owners[0], 'cancel_expert_offer', offer_id=offers['0'])
    assert state['hosting']['leases'][key] == saved
    state = ready(state, owners, key)
    with pytest.raises(ValueError, match='withdrawal cooldown'):
        send(state, owners[0], 'withdraw_provider', amount=1)


def test_all_owners_acknowledge_before_serving_and_pay_only_after_full_audit(market):
    state, owners, offers = market
    state, key = lease(state, owners, offers)
    issued = state['issued']
    with pytest.raises(ValueError, match='ready'):
        response(state, owners, key)
    state = ready(state, owners, key)
    job = copy.deepcopy(state['expert_lifecycle']['jobs'][key])
    collateral = state['hosting']['providers'][owners[0].public_key]['collateral']
    state = response(state, owners, key)
    claim = copy.deepcopy(state['candidate'])
    assert state['hosting']['providers'][owners[0].public_key]['collateral'] == collateral - claim['bond']
    assert state['hosting']['leases'][key]['status'] == 'claiming'
    assert not state['hosting']['history']
    state = finish(state, owners)
    assert state['expert_lifecycle']['results'][key]['status'] == 'completed'
    assert state['hosting']['history'][-1]['provider_paid_atoms'] == 406
    assert state['hosting']['history'][-1]['provider_refunded_atoms'] == 594
    assert key not in state['hosting']['leases'] and state['issued'] == issued
    assert all(row['locked'] == 0 for row in state['hosting']['providers'].values())
    assert state['hosting']['providers'][owners[0].public_key]['collateral'] == collateral
    receipts = {rank: owners[int(rank)].sign(life.inference_receipt(
        state['chain_id'], job, claim['outputs'], claim['text'], claim['record_root'], rank))
        for rank in job['workers']}
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match='available expert inference'):
        send(state, owners[0], 'respond_expert', job_id=key, outputs=claim['outputs'], text=claim['text'],
             transcript_root=claim['record_root'], workers=receipts, audit_budget=claim['audit_budget'])
    assert state == before


def test_replacement_preserves_request_fences_receipts_and_moves_coordinator(market):
    state, owners, offers = market
    state, key = lease(state, owners, offers)
    state = ready(state, owners, key)
    original_job = copy.deepcopy(state['expert_lifecycle']['jobs'][key])
    original_lease = copy.deepcopy(state['hosting']['leases'][key])
    with pytest.raises(ValueError, match='closed attempt'):
        send(state, owners[2], 'replace_hosted_job', job_id=key, offers=offers,
             audit_budget=original_lease['audit_budget'])
    # A fresh provider, not the former coordinator, claims the rank-zero offer.
    replacement = protocol.Identity('independently-keyed-replacement')
    state = send(state, owners[3], 'transfer', to=replacement.public_key, amount=100_000_000)
    state = send(state, replacement, 'register_provider', endpoint='https://replacement.example',
                 certificate='b'*64, collateral=50_000_000)
    prior = set(state['hosting']['offers'])
    state = send(state, replacement, 'offer_expert', graph=state['serving_root'], rank=0,
                 fee=200, capacity=1, expires_in=10000)
    changed = {**offers, '0': (set(state['hosting']['offers']) - prior).pop()}
    state = blocks(state, 9)
    state = send(state, owners[2], 'replace_hosted_job', job_id=key, offers=changed,
                 audit_budget=original_lease['audit_budget'])
    current = state['expert_lifecycle']['jobs'][key]
    for field in ('id', 'payer', 'graph', 'request', 'escrow', 'expires'):
        assert current[field] == original_job[field]
    assert current['workers']['0'] == replacement.public_key
    assert current['hosting'] != original_job['hosting']
    assert state['hosting']['leases'][key]['epoch'] == 1
    assert state['auditing']['budgets'][original_lease['audit_budget']]['publisher'] == replacement.public_key
    with pytest.raises(ValueError, match='stale'):
        send(state, replacement, 'accept_hosted_job', job_id=key, assignment_root=original_job['hosting'])
    # Even an unchanged worker cannot replay an old epoch's acknowledgement.
    with pytest.raises(ValueError, match='stale'):
        send(state, owners[1], 'accept_hosted_job', job_id=key, assignment_root=original_job['hosting'])
    state = ready(state, [replacement, *owners[1:]], key)
    output_rows = [output(call['model'], [3, 2]) for call in original_job['request']['calls']]
    old_receipts = {rank: owners[int(rank)].sign(life.inference_receipt(
        state['chain_id'], original_job, output_rows, 'A bounded response', '3'*64, rank)) for rank in offers}
    old_receipts['0'] = replacement.sign(life.inference_receipt(
        state['chain_id'], current, output_rows, 'A bounded response', '3'*64, '0'))
    with pytest.raises(ValueError, match='receipt'):
        response(state, [replacement, *owners[1:]], key, sender=replacement, receipts=old_receipts)
    state = response(state, [replacement, *owners[1:]], key, sender=replacement)
    state = finish(state, owners)
    assert state['hosting']['history'][-1]['epoch'] == 1
    assert state['hosting']['history'][-1]['provider_paid_atoms'] == 506


def test_expiry_refunds_execution_auditing_and_provider_fees_once(market):
    state, owners, offers = market
    state, key = lease(state, owners, offers)
    job = state['expert_lifecycle']['jobs'][key]
    budget = state['hosting']['leases'][key]['audit_budget']
    balance = state['accounts'][owners[3].public_key]['balance']
    expected = job['escrow'] + 1000 + state['auditing']['budgets'][budget]['funds']
    state = blocks(state, job['expires'] + 1 - state['height'])
    assert state['accounts'][owners[3].public_key]['balance'] == balance + expected
    assert key not in state['hosting']['leases'] and budget not in state['auditing']['budgets']
    assert not state['hosting']['history'][-1]['accepted']
    supply = state['issued']
    again = blocks(state, 2)
    assert again['accounts'] == state['accounts'] and again['issued'] == supply


def test_rejected_execution_never_pays_providers_and_retry_needs_a_new_audit(market):
    state, owners, offers = market
    state, key = lease(state, owners, offers)
    state = ready(state, owners, key)
    state = response(state, owners, key)
    old_budget = state['hosting']['leases'][key]['audit_budget']
    state = finish(state, owners, valid=False)
    assert state['hosting']['leases'][key]['status'] == 'failed'
    assert old_budget not in state['auditing']['budgets'] and not state['hosting']['history']
    with pytest.raises(ValueError, match='Prepay'):
        send(state, owners[1], 'replace_hosted_job', job_id=key, offers=offers, audit_budget=old_budget)
    state, fresh = audit(state, owners)
    state = send(state, owners[1], 'replace_hosted_job', job_id=key, offers=offers, audit_budget=fresh)
    assert state['hosting']['leases'][key]['audit_budget'] == fresh


def test_expensive_replacement_and_excess_attempts_are_atomic(market):
    state, owners, offers = market
    state, key = lease(state, owners, offers)
    state = blocks(state, 5)
    before = copy.deepcopy(state)
    with pytest.raises(ValueError):
        send(state, owners[1], 'replace_hosted_job', job_id=key,
             offers={**offers, '0': 'f'*64}, audit_budget=state['hosting']['leases'][key]['audit_budget'])
    assert state == before
    before_offers = set(state['hosting']['offers'])
    state = send(state, owners[0], 'offer_expert', graph=state['serving_root'], rank=0,
                 fee=1001, capacity=1, expires_in=10000)
    expensive = (set(state['hosting']['offers']) - before_offers).pop()
    unchanged = copy.deepcopy(state)
    with pytest.raises(ValueError, match='fee ceiling'):
        send(state, owners[1], 'replace_hosted_job', job_id=key,
             offers={**offers, '0': expensive}, audit_budget=state['hosting']['leases'][key]['audit_budget'])
    assert state == unchanged
    for _ in range(2):
        state = send(state, owners[1], 'replace_hosted_job', job_id=key, offers=offers,
                     audit_budget=state['hosting']['leases'][key]['audit_budget'])
        state = blocks(state, 5)
    with pytest.raises(ValueError, match='attempt budget'):
        send(state, owners[1], 'replace_hosted_job', job_id=key, offers=offers,
             audit_budget=state['hosting']['leases'][key]['audit_budget'])


def test_coordinator_claim_uses_reserved_collateral_after_liquid_funds_are_spent(market):
    state, owners, offers = market
    state, key = lease(state, owners, offers)
    state = ready(state, owners, key)
    provider = owners[0]
    fee = state['manifest']['params']['fee']
    amount = state['accounts'][provider.public_key]['balance'] - 2*fee
    state = send(state, provider, 'transfer', to=owners[3].public_key, amount=amount)
    assert state['accounts'][provider.public_key]['balance'] == fee
    state = response(state, owners, key)
    assert state['accounts'][provider.public_key]['balance'] == 0
    settlement.invariant(state)


def test_missing_audit_coverage_refunds_reserved_claim_collateral(market):
    state, owners, offers = market
    state, key = lease(state, owners, offers)
    state = ready(state, owners, key)
    original = state['hosting']['providers'][owners[0].public_key]['collateral']
    state = response(state, owners, key)
    assert state['hosting']['leases'][key]['spent_claim_bond'] > 0
    state = blocks(state, state['candidate']['audit_reveal_end'] + 1 - state['height'])
    assert state['candidate'] is None
    assert state['hosting']['providers'][owners[0].public_key]['collateral'] == original
    assert state['hosting']['leases'][key]['spent_claim_bond'] == 0
    assert state['hosting']['leases'][key]['status'] == 'failed'
