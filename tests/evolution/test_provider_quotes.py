import copy

import pytest

from neuroshard.demo import protocol
from neuroshard.evolution import hosting, provider_quotes
from test_provider_hosting import market, graphs, QUESTION, audit
from test_expert_lifecycle import send, network
from test_settlement import blocks


def test_quote_covers_complete_execution_providers_audit_and_bounded_client_fees(market):
    state, owners, _ = market
    quoted = provider_quotes.quote(state, QUESTION, 64)
    assert quoted['maximum_debit_atoms'] == sum(quoted[name] for name in
        ('execution_atoms', 'provider_atoms', 'verification_atoms', 'transaction_fee_allowance_atoms'))
    assert quoted['verification_atoms'] == 4*quoted['stage_limit']*state['manifest']['auditing']['price_per_stage']
    state, budget = audit(state, owners, stages=quoted['stage_limit'])
    before = state['accounts'][owners[3].public_key]['balance']
    state = send(state, owners[3], 'lease_expert', graph=quoted['graph'], question=QUESTION, max_tokens=64,
        offers=quoted['offers'], max_price=quoted['execution_atoms'], max_provider_fee=quoted['provider_atoms'],
        audit_budget=budget, expires_in=quoted['expires_in'])
    debit = before - state['accounts'][owners[3].public_key]['balance']
    assert debit == quoted['execution_atoms'] + quoted['provider_atoms'] + state['manifest']['params']['fee']


def test_quote_refuses_missing_capacity_expiring_offers_and_tight_price(market):
    state, _, offers = market
    with pytest.raises(ValueError, match='No complete'):
        provider_quotes.quote(state, QUESTION, 64, provider_ceiling=0)
    missing = copy.deepcopy(state)
    del missing['hosting']['offers'][offers['2']]
    with pytest.raises(ValueError, match='No complete'):
        provider_quotes.quote(missing, QUESTION, 64)
    expired = copy.deepcopy(state)
    expired['hosting']['offers'][offers['1']]['expires'] = expired['height'] + 1
    with pytest.raises(ValueError, match='No complete'):
        provider_quotes.quote(expired, QUESTION, 64)


def test_shared_collateral_matching_preserves_scarce_owner_capacity(market):
    state, owners, offers = market
    newcomer = protocol.Identity('additional-public-offerer')
    state = send(state, owners[0], 'transfer', to=newcomer.public_key, amount=3_000_000)
    state = send(state, newcomer, 'register_provider', endpoint='https://additional.example',
                 certificate='c'*64, collateral=hosting.PROFILE['lease_bond'])
    state = send(state, newcomer, 'offer_expert', graph=state['serving_root'], rank=1,
                 fee=500, capacity=1, expires_in=10000)
    # The old rank-one provider can afford exactly one rank and is the only
    # rank-two provider. A greedy rank-one choice must be augmentable.
    state = blocks(state, state['height'] + 3)
    state = send(state, owners[1], 'withdraw_provider', amount=49_000_000)
    state = send(state, owners[2], 'cancel_expert_offer', offer_id=offers['2'])
    state = send(state, owners[1], 'offer_expert', graph=state['serving_root'], rank=2,
                 fee=102, capacity=1, expires_in=10000)
    quoted = provider_quotes.quote(state, QUESTION, 64)
    selected = state['hosting']['offers']
    assert selected[quoted['offers']['1']]['owner'] == newcomer.public_key
    assert selected[quoted['offers']['2']]['owner'] == owners[1].public_key
    with pytest.raises(ValueError, match='No complete'):
        provider_quotes.quote(state, QUESTION, 64, publisher=newcomer.public_key)


def test_quote_query_releases_consensus_lock_but_keeps_one_committed_snapshot(market, monkeypatch):
    import threading
    from neuroshard.dataflow.store import canonical
    from neuroshard.demo import abci_pb2 as pb
    from neuroshard.evolution.app import Application

    state, _, _ = market
    app = Application.__new__(Application)
    app.lock, app.state = threading.Lock(), state
    original = provider_quotes.quote

    def concurrent_commit(snapshot, *args, **kwargs):
        assert app.lock.acquire(blocking=False), 'Discovery must not lock block finalization'
        try:
            app.state = {**state, 'height': state['height'] + 1}
        finally:
            app.lock.release()
        assert snapshot is state
        return original(snapshot, *args, **kwargs)

    monkeypatch.setattr(provider_quotes, 'quote', concurrent_commit)
    response = app.Query(pb.RequestQuery(path='/hosting/quote',
        data=canonical({'question': QUESTION, 'max_tokens': 64})), None)
    assert response.code == 0
    assert protocol.parse_json(response.value)['height'] == response.height == state['height']
    assert app.state['height'] == state['height'] + 1
