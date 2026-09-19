"""Real native acceptance must survive a heartbeat with an unknown receipt."""
import json
import threading
from types import SimpleNamespace

import pytest

from neuroshard.evolution.provider_control import LockedOutbox
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
