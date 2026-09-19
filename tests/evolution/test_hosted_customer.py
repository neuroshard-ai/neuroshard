"""Customer recovery against real signed native state transitions."""
import base64
import copy
import hashlib
import subprocess
import sys

import pytest

from neuroshard.client import wire
from neuroshard.client.hosted import Customer
from neuroshard.demo import protocol
from neuroshard.evolution import hosting, provider_quotes, settlement
from neuroshard.evolution.reference_data import identity
from neuroshard.evolution.transactions import Outbox
from test_provider_hosting import market, graphs
from test_expert_lifecycle import network, send
from test_complete_answering import complete
from test_settlement import blocks

MESSAGES = [{'role': 'user', 'content': 'What is word7?'}]


class Chain:
    def __init__(self, state):
        self.state, self.chain_id = state, state['chain_id']
        self.receipts, self.submissions = {}, []
        self.drop = None
        self.reject_reservation = False

    def query(self, path, data=None):
        if path == '/hosting/quote':
            return provider_quotes.quote(self.state, data['question'], data['max_tokens'])
        if path == '/summary':
            return {'height': self.state['height']}
        if path == '/account':
            return copy.deepcopy(self.state['accounts'][data['public_key']])
        if path == '/auditing':
            return copy.deepcopy(self.state['auditing'])
        raise AssertionError(path)

    def snapshot(self, key, **kwargs):
        return hosting.snapshot(self.state, key)

    def rpc(self, url, method, params, **kwargs):
        if method == 'tx':
            key = base64.b64decode(params['hash']).hex().upper()
            if key not in self.receipts:
                raise wire.Rejected('Transaction not found')
            return self.receipts[key]
        assert method == 'broadcast_tx_sync'
        raw = base64.b64decode(params['tx'])
        envelope = protocol.parse_json(raw)
        if self.reject_reservation and envelope['body']['kind'] == 'lease_expert':
            raise wire.Rejected('Provider capacity was reserved by another customer')
        self.state = settlement.transition(blocks(self.state, 1), envelope)
        settlement.invariant(self.state)
        key = hashlib.sha256(raw).hexdigest().upper()
        self.submissions.append(raw)
        self.receipts[key] = {'hash': key, 'height': str(self.state['height']), 'tx_result': {'code': 0}}
        if envelope['body']['kind'] == self.drop:
            raise OSError('Accepted before the connection disappeared')
        return {'code': 0, 'hash': key}


@pytest.fixture
def customer_chain(market, complete):
    state, owners, offers = market
    owners = [*owners, protocol.Identity('fifth-hosted-customer-fixture-provider')]
    # Seed the promoted-service boundary; this does not claim learning or
    # quality admission. Every customer payment uses the native transition.
    bound = complete[2]
    state['expert_lifecycle']['serving_graph'] = bound
    state['serving_root'] = identity(bound)
    state['manifest']['expert_lifecycle']['max_tokens'] = 4
    state = send(state, owners[0], 'transfer', to=owners[4].public_key, amount=100_000_000)
    state = send(state, owners[4], 'register_provider', endpoint='https://fifth.example',
                 certificate='f'*64, collateral=50*hosting.PROFILE['lease_bond'])
    for rank, owner in enumerate(owners):
        if str(rank) in offers:
            state = send(state, owner, 'cancel_expert_offer', offer_id=offers[str(rank)])
        state = send(state, owner, 'offer_expert', graph=state['serving_root'], rank=rank,
                     fee=100 + rank, capacity=2, expires_in=10000)
    return Chain(state), owners


def client(home, chain, owner):
    box = Outbox(home/'outbox.sqlite', 'local', chain.chain_id, owner, rpc=chain.rpc,
                 query=lambda _url, path, data: chain.query(path, data))
    return Customer(home/'requests', chain, owner, box), box


def accept(chain, owners, budget):
    for owner in owners[:3]:
        chain.state = send(chain.state, owner, 'accept_audit', budget_id=budget)


def test_cap_and_session_version_fail_before_any_native_payment(tmp_path, customer_chain):
    chain, owners = customer_chain
    customer, box = client(tmp_path, chain, owners[3])
    try:
        with pytest.raises(ValueError, match='max-price'):
            customer.prepare(MESSAGES, 4, 1)
        with pytest.raises(ValueError, match='tokenizer changed'):
            customer.prepare(MESSAGES, 4, 10**9, session={'graph': 'a'*64, 'tokenizer': 'b'*64})
        with pytest.raises(ValueError, match='alternating'):
            customer.prepare([{'role': 'assistant', 'content': 'unpaid draft'}], 4, 10**9)
        assert chain.submissions == []
    finally:
        box.close()


def test_unknown_reservation_recovers_after_native_expiry_without_double_payment(tmp_path, customer_chain):
    chain, owners = customer_chain
    customer, box = client(tmp_path, chain, owners[3])
    before = chain.state['accounts'][owners[3].public_key]['balance']
    row = customer.prepare(MESSAGES, 4, 10**9)
    assert customer.tick(row)['status'] == 'awaiting_complete_audit_funding'
    accept(chain, owners, row['budget_id'])
    chain.drop = 'lease_expert'
    original = box.send
    box.send = lambda *a, **kw: original(*a, timeout=.01, **kw)
    with pytest.raises(TimeoutError):
        customer.tick(row)
    key = box.logical_id(row['id'] + ':reserve')
    job = chain.state['expert_lifecycle']['jobs'][key]
    chain.state = blocks(chain.state, job['expires'] - chain.state['height'] + 1)
    box.close()
    resumed, box = client(tmp_path, chain, owners[3])
    try:
        update = resumed.tick(resumed.load(row['id']))
        assert update['status'] == 'finished' and update['result']['status'] == 'expired'
        assert len(chain.submissions) == 2 and box.pending() is None
        assert before - chain.state['accounts'][owners[3].public_key]['balance'] == 2*chain.state['manifest']['params']['fee']
        assert chain.state['issued'] == 0
        settlement.invariant(chain.state)
    finally:
        box.close()


def test_unaccepted_audit_offer_cancels_and_refunds_when_quote_ages_out(tmp_path, customer_chain):
    chain, owners = customer_chain
    customer, box = client(tmp_path, chain, owners[3])
    try:
        row = customer.prepare(MESSAGES, 4, 10**9)
        assert customer.tick(row)['status'] == 'awaiting_complete_audit_funding'
        chain.state = blocks(chain.state, row['quote']['valid_until'] - chain.state['height'] + 1)
        assert customer.tick(row)['status'] == 'audit_cancelled'
        update = customer.tick(row)
        assert update['status'] == 'finished'
        assert update['result']['audit']['refunded_atoms'] == row['quote']['verification_atoms']
        assert not chain.state['expert_lifecycle']['jobs']
    finally:
        box.close()


def test_hosted_customer_imports_without_a_numerical_runtime():
    code = 'import sys; import neuroshard.client.hosted; assert "torch" not in sys.modules; assert "transformers" not in sys.modules'
    subprocess.run([sys.executable, '-c', code], check=True, timeout=10)


def test_two_customers_with_the_same_quote_reserve_distinct_capacity_atomically(tmp_path, customer_chain):
    chain, owners = customer_chain
    # Ten keys each advertise one slot: a second replica for every model rank.
    for offer_id, offer in list(chain.state['hosting']['offers'].items()):
        if offer['graph'] != chain.state['serving_root']:
            continue
        key = next(o for o in owners if o.public_key == offer['owner'])
        chain.state = send(chain.state, key, 'cancel_expert_offer', offer_id=offer_id)
    for rank in range(5):
        replica = protocol.Identity('replica-customer-fixture-' + str(rank))
        chain.state = send(chain.state, owners[0], 'transfer', to=replica.public_key, amount=100_000_000)
        chain.state = send(chain.state, replica, 'register_provider', endpoint=f'https://replica-{rank}.example',
            certificate='e'*64, collateral=50*hosting.PROFILE['lease_bond'])
        for key in (owners[rank], replica):
            chain.state = send(chain.state, key, 'offer_expert', graph=chain.state['serving_root'], rank=rank,
                fee=100 + rank, capacity=1, expires_in=10000)
    customers, boxes, rows = [], [], []
    try:
        for index, owner in enumerate((owners[3], owners[2])):
            home = tmp_path/str(index)
            home.mkdir()
            customer, box = client(home, chain, owner)
            customers.append(customer)
            boxes.append(box)
            rows.append(customer.prepare(MESSAGES, 4, 10**9))
        assert rows[0]['quote']['offers'] == rows[1]['quote']['offers']
        for customer, row in zip(customers, rows):
            customer.tick(row)
            accept(chain, owners, row['budget_id'])
        views = [customer.tick(row)['snapshot'] for customer, row in zip(customers, rows)]
        for rank in range(5):
            assert views[0]['job']['workers'][str(rank)] != views[1]['job']['workers'][str(rank)]
        for row, view in zip(rows, views):
            assert chain.state['auditing']['budgets'][row['budget_id']]['publisher'] == view['job']['workers']['0']
        settlement.invariant(chain.state)
    finally:
        for box in boxes:
            box.close()


def test_ambiguous_refused_reservation_retires_only_after_native_audit_closure(tmp_path, customer_chain):
    chain, owners = customer_chain
    customer, box = client(tmp_path, chain, owners[3])
    try:
        row = customer.prepare(MESSAGES, 4, 10**9)
        customer.tick(row)
        accept(chain, owners, row['budget_id'])
        chain.reject_reservation = True
        with pytest.raises(wire.Rejected, match='capacity'):
            customer.tick(row)
        assert box.pending() == row['id'] + ':reserve'
        budget = chain.state['auditing']['budgets'][row['budget_id']]
        chain.state = blocks(chain.state, budget['expires'] - chain.state['height'] + 1)
        result = customer.tick(customer.load(row['id']))
        assert result['status'] == 'finished'
        assert result['result']['transaction_outcome'].startswith('unknown')
        assert result['result']['audit']['refunded_atoms'] == row['quote']['verification_atoms']
        assert box.pending() is None and len(chain.submissions) == 1
        assert len(box.db.execute('SELECT envelope FROM operations').fetchall()) == 2
        # Recover a crash after retirement was persisted but before the outer
        # customer journal recorded completion. It still cannot re-sign.
        row['phase'] = 'awaiting_audit'
        customer.write(row)
        assert customer.tick(customer.load(row['id'])) == result
        assert box.pending() is None and len(chain.submissions) == 1
    finally:
        box.close()
