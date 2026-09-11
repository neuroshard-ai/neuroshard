import base64
import hashlib

import pytest

from neuroshard.demo import protocol, client as wire
from neuroshard.evolution.transactions import Outbox


class Node:
    def __init__(self):
        self.receipts = {}
        self.submissions = []
        self.queries = 0

    def query(self, *args):
        self.queries += 1
        return {'nonce': 7}

    def rpc(self, url, method, params, **kwargs):
        if method == 'tx':
            key = base64.b64decode(params['hash']).hex().upper()
            if key not in self.receipts:
                raise wire.Rejected('Transaction not found')
            return self.receipts[key]
        raw = base64.b64decode(params['tx'])
        self.submissions.append(raw)
        key = hashlib.sha256(raw).hexdigest().upper()
        self.receipts[key] = {'hash': key, 'height': '32', 'tx_result': {'code': 0}}
        # The transaction committed, but its submission response was lost.
        raise OSError('connection lost after broadcast')


def test_crash_after_acceptance_recovers_exact_signed_transaction(tmp_path):
    node = Node()
    owner = protocol.Identity('outbox-test')
    path = tmp_path/'outbox.sqlite'
    box = Outbox(path, 'local', 'candidate', owner, rpc=node.rpc, query=node.query)
    with pytest.raises(TimeoutError):
        box.send('payment-1', 'transfer', to=owner.public_key, amount=1, timeout=.01)
    assert box.pending() == 'payment-1'
    box.close()
    restarted = Outbox(path, 'local', 'candidate', owner, rpc=node.rpc, query=node.query)
    receipt = restarted.send('payment-1', 'transfer', to=owner.public_key, amount=1)
    assert receipt['height'] == '32' and len(node.submissions) == 1 and node.queries == 1
    assert restarted.pending() is None
    assert restarted.send('payment-1', 'transfer', to=owner.public_key, amount=1) == receipt
    assert len(node.submissions) == 1
    with pytest.raises(ValueError, match='different intent'):
        restarted.send('payment-1', 'transfer', to=owner.public_key, amount=2)
    restarted.close()


def test_unknown_outcome_cannot_be_replaced_with_another_nonce(tmp_path):
    node = Node()
    box = Outbox(tmp_path/'outbox.sqlite', 'local', 'candidate', protocol.Identity('outbox-test'), rpc=node.rpc, query=node.query)
    with pytest.raises(TimeoutError):
        box.send('first', 'audit_commit', claim_id='a'*64, commitment='b'*64, timeout=.01)
    with pytest.raises(ValueError, match='existing pending transaction'):
        box.send('second', 'audit_commit', claim_id='c'*64, commitment='d'*64)
    assert len(node.submissions) == 1 and node.queries == 1
    box.close()


def test_outbox_rejects_another_chain_or_identity(tmp_path):
    node = Node()
    path = tmp_path/'outbox.sqlite'
    owner = protocol.Identity('outbox-test')
    box = Outbox(path, 'local', 'candidate', owner, rpc=node.rpc, query=node.query)
    box.close()
    with pytest.raises(ValueError, match='different chain or account'):
        Outbox(path, 'local', 'other-candidate', owner, rpc=node.rpc, query=node.query)


def test_unrelated_rpc_receipt_does_not_resolve_pending_operation(tmp_path):
    node = Node()
    owner = protocol.Identity('outbox-test')
    box = Outbox(tmp_path/'outbox.sqlite', 'local', 'candidate', owner, rpc=node.rpc, query=node.query)
    with pytest.raises(TimeoutError):
        box.send('first', 'transfer', to=owner.public_key, amount=1, timeout=.01)
    next(iter(node.receipts.values()))['hash'] = 'e'*64
    with pytest.raises(ValueError, match='unrelated'):
        box.confirm('first')
    assert box.pending() == 'first'
    box.close()


def test_coordinator_recovery_reuses_completed_worker_update(seed, tmp_path, monkeypatch):
    from neuroshard.evolution.pipeline import Pipeline, LocalEndpoint
    from neuroshard.evolution.worker import Worker, Session
    store, root, _ = seed
    homes = [tmp_path/f'worker{i}' for i in range(2)]
    workers = [Worker(home, store) for home in homes]
    pipe = Pipeline(store, root, [LocalEndpoint(w) for w in workers], [6000]*2, 'reserved-task', start_step=7)
    batch = [[1, 12, 13, 14, 2]]
    record = pipe.train(batch)
    pipe.close()
    for worker in workers:
        worker.db.close()
    # Simulate loss of the coordinator's result after worker commits. Neither
    # a new numerical step nor a new parent may be substituted on recovery.
    def forbidden(*args, **kwargs):
        raise AssertionError('committed operations should come from durable worker results')
    monkeypatch.setattr(Session, 'run', forbidden)
    workers = [Worker(home, store) for home in homes]
    recovered = Pipeline(store, root, [LocalEndpoint(w) for w in workers], [6000]*2, 'reserved-task', start_step=7)
    same = recovered.train(batch)
    assert same['record_root'] == record['record_root'] and same['step'] == 7
    recovered.close()
    for worker in workers:
        worker.db.close()
