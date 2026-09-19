"""A single signed admission survives lost acknowledgement and native expiry."""
import copy

import pytest

from neuroshard.evolution import service_admission as admission, settlement
from test_hosted_customer import customer_chain, client, Chain, MESSAGES, market, graphs, network, complete
from test_expert_lifecycle import send
from test_settlement import blocks


@pytest.fixture
def atomic_chain(customer_chain):
    chain, owners = customer_chain
    chain.state['manifest']['service_admission'] = {**admission.PROFILE, 'provider_slots': 2}
    admission.initialize(chain.state)
    for row in chain.state['hosting']['providers'].values():
        row['registration_expires'] = chain.state['height'] + admission.PROFILE['provider_blocks']
        row['last_seen'] = chain.state['height']
    for owner in owners[:3]:
        chain.state = send(chain.state, owner, 'offer_audit_service', purpose='expert_inference',
            scope=chain.state['serving_root'], stage_limit=4096, capacity=2, expires_in=10000)
    return chain, owners


def test_one_atomic_submission_reserves_the_complete_job(tmp_path, atomic_chain):
    chain, owners = atomic_chain
    customer, box = client(tmp_path, chain, owners[3])
    try:
        row = customer.prepare(MESSAGES, 4, 10**9)
        quote = row['quote']
        assert quote['format'].endswith('v2') and quote['occupancy_atoms'] > 0
        assert quote['maximum_debit_atoms'] == sum(quote[k] for k in (
            'execution_atoms', 'provider_atoms', 'verification_atoms', 'occupancy_atoms',
            'transaction_fee_allowance_atoms'))
        assert customer.tick(row)['status'] == 'serving'
        assert len(chain.submissions) == 1
        assert row['job_id'] == row['budget_id']
        assert chain.state['auditing']['budgets'][row['job_id']]['reservation'] == row['job_id']
        settlement.invariant(chain.state)
    finally:
        box.close()


def test_changed_capacity_leaves_customer_funds_and_nonce_untouched(tmp_path, atomic_chain):
    chain, owners = atomic_chain
    customer, box = client(tmp_path, chain, owners[3])
    try:
        row = customer.prepare(MESSAGES, 4, 10**9)
        service = next(k for k, v in chain.state['service_admission']['services'].items()
                       if v['owner'] == owners[0].public_key)
        chain.state = send(chain.state, owners[0], 'close_audit_service', service_id=service)
        before = copy.deepcopy(chain.state)
        with pytest.raises(ValueError):
            customer.tick(row)
        assert chain.state == before
        assert not chain.state['auditing']['budgets'] and not chain.state['hosting']['leases']
    finally:
        box.close()


def test_lost_admission_acknowledgement_recovers_without_a_second_nonce(tmp_path, atomic_chain):
    chain, owners = atomic_chain
    customer, box = client(tmp_path, chain, owners[3])
    try:
        row = customer.prepare(MESSAGES, 4, 10**9)
        chain.drop = 'admit_work'
        original = box.send
        box.send = lambda *args, **kwargs: original(*args, timeout=.01, **kwargs)
        with pytest.raises(TimeoutError):
            customer.tick(row)
        assert len(chain.state['hosting']['leases']) == 1
        before = chain.state['accounts'][owners[3].public_key]['nonce']
        chain.state = blocks(chain.state, row['quote']['valid_until'] - chain.state['height'] + 1)
        # The public node proves the signed absolute deadline elapsed, even if
        # the transaction index still cannot return the original receipt.
        assert customer.tick(customer.load(row['id']))['status'] == 'serving'
        assert chain.state['accounts'][owners[3].public_key]['nonce'] == before
        assert len(chain.submissions) == 1
    finally:
        box.close()


def test_background_heartbeat_keeps_original_nonce_after_lost_ack(tmp_path, atomic_chain):
    from neuroshard.evolution.provider_control import maintain
    chain, owners = atomic_chain
    owner = owners[0]
    _, box = client(tmp_path, chain, owner)
    query = chain.query
    def control(path, data=None):
        if path == '/hosting/control':
            return {'chain_id': chain.chain_id, 'height': chain.state['height'],
                'profile': chain.state['manifest']['service_admission'],
                'hosting': copy.deepcopy(chain.state['hosting'])}
        return query(path, data)
    chain.query = control
    # No deadlines expire while advancing this fixture to its heartbeat due time.
    chain.state = blocks(chain.state, admission.PROFILE['provider_heartbeat_blocks']//2)
    config = {'advertise': chain.state['hosting']['providers'][owner.public_key]['endpoint']}
    original = box.send
    def short(*args, **kwargs):
        kwargs['timeout'] = .01
        return original(*args, **kwargs)
    box.send = short
    try:
        chain.drop = 'heartbeat_provider'
        with pytest.raises(TimeoutError):
            maintain(chain, box, config, owner.public_key)
        signed = chain.submissions[0]
        nonce = chain.state['accounts'][owner.public_key]['nonce']
        operation = box.pending()
        chain.receipts.clear()  # lost transaction index, still a committed native state
        chain.state = blocks(chain.state, 65)
        assert maintain(chain, box, config, owner.public_key) == 'reconciled'
        assert box.pending() is None and box.retirement(operation)['transaction_outcome'].startswith('unknown')
        assert chain.submissions == [signed]
        assert chain.state['accounts'][owner.public_key]['nonce'] == nonce
    finally:
        box.close()
