"""Real native acceptance must survive a heartbeat with an unknown receipt."""
import json
import threading
from types import SimpleNamespace

import pytest

from neuroshard.evolution.provider_control import LockedOutbox, maintain
from neuroshard.evolution.reference_data import identity
from neuroshard.evolution.transactions import Outbox
from test_service_admission import capacity, market, graphs, network, admit
from test_hosted_customer import Chain
from test_settlement import blocks


@pytest.mark.parametrize('heartbeat_committed', [True, False])
def test_pending_maintenance_cannot_abandon_model_acceptance(tmp_path, capacity, heartbeat_committed):
    state, owners, offers = capacity
    state, job = admit(state, owners, offers)
    chain = Chain(state)
    native_rpc = chain.rpc
    def rpc(url, method, params, **options):
        if not heartbeat_committed and method == 'broadcast_tx_sync':
            import base64
            body = json.loads(base64.b64decode(params['tx']))['body']
            if body['kind'] == 'heartbeat_provider':
                raise OSError('Original heartbeat outcome is unknown')
        return native_rpc(url, method, params, **options)
    box = Outbox(tmp_path/'provider.sqlite', 'local', chain.chain_id, owners[0], rpc=rpc,
                 query=lambda _url, path, data: chain.query(path, data))
    try:
        chain.drop = 'heartbeat_provider'
        with pytest.raises(TimeoutError):
            box.send('heartbeat', 'heartbeat_provider', valid_until=state['height']+2, timeout=.01)
        original = box.db.execute('SELECT envelope FROM operations WHERE id="heartbeat"').fetchone()[0]
        assert box.pending() == 'heartbeat'
        chain.drop = None
        if not heartbeat_committed:
            chain.state = blocks(chain.state, 3)
        node = SimpleNamespace(query=lambda _path: {'chain_id': chain.chain_id, 'height': chain.state['height']},
                               snapshot=chain.snapshot)
        with pytest.raises(ValueError, match='existing pending transaction'):
            box.send('accept', 'accept_hosted_job', job_id=job,
                     assignment_root=chain.state['hosting']['leases'][job]['assignment_root'])
        assert not box.recorded('accept')
        locked = LockedOutbox(box, threading.RLock(), node)
        locked.send('accept', 'accept_hosted_job', job_id=job,
                    assignment_root=chain.state['hosting']['leases'][job]['assignment_root'])
        assert owners[0].public_key in chain.state['hosting']['leases'][job]['accepted']
        assert box.pending() is None and box.receipt('accept')['tx_result']['code'] == 0
        assert box.db.execute('SELECT envelope FROM operations WHERE id="heartbeat"').fetchone()[0] == original
        assert len(chain.submissions) == (2 if heartbeat_committed else 1)
        if heartbeat_committed:
            assert box.receipt('heartbeat')['tx_result']['code'] == 0
        else:
            assert box.receipt('heartbeat') is None
            assert box.retirement('heartbeat')['transaction_outcome'].startswith('unknown')
    finally:
        box.close()


@pytest.mark.parametrize('registration_remaining, inclusion_delay', [(12000, 1), (12000, 64), (14063, 64), (14064, 64)])
def test_offer_renewal_prepays_registration_through_inclusion_window(
        tmp_path, capacity, registration_remaining, inclusion_delay):
    state, owners, offers = capacity
    owner = owners[0]
    provider = state['hosting']['providers'][owner.public_key]
    provider.update(registration_expires=state['height'] + registration_remaining,
                    last_seen=state['height'])
    state['hosting']['offers'][offers['0']]['expires'] = state['height'] + 9000
    profile = tmp_path/'executor.json'
    profile.write_text('{}')
    config = {'advertise': provider['endpoint'], 'profile': str(profile), 'rank': 0,
              'offer_blocks': 14000}
    chain = Chain(state)
    native_rpc = chain.rpc

    def rpc(url, method, params, **options):
        if method == 'broadcast_tx_sync':
            chain.state = blocks(chain.state, inclusion_delay - 1)
        return native_rpc(url, method, params, **options)

    def current(_path):
        return {'chain_id': chain.chain_id, 'height': chain.state['height'],
                'profile': chain.state['manifest']['service_admission'],
                'hosting': chain.state['hosting'], 'request_blocks': 8832,
                'executor_root': identity({}), 'graph': chain.state['serving_root']}

    node = SimpleNamespace(query=current, snapshot=chain.snapshot)
    box = Outbox(tmp_path/'renewals.sqlite', 'local', chain.chain_id, owner, rpc=rpc,
                 query=lambda _url, path, data: chain.query(path, data))
    try:
        phases = [maintain(node, box, config, owner.public_key)]
        if registration_remaining < 14064:
            assert phases == ['registration_renewed']
            phases.append(maintain(node, box, config, owner.public_key))
        assert phases[-1] == 'offer_renewed'
        offer = chain.state['hosting']['offers'][offers['0']]
        registration = chain.state['hosting']['providers'][owner.public_key]
        assert offer['expires'] == chain.state['height'] + 14000
        assert offer['expires'] <= registration['registration_expires']
        assert box.pending() is None
        assert len(chain.submissions) == len(phases)
    finally:
        box.close()
