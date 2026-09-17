"""Late audit recovery must follow committed closure, never a CheckTx error."""
import copy
import sqlite3

import pytest

from neuroshard.dataflow.store import canonical
from neuroshard.demo import protocol
from neuroshard.evolution import auditing, audit_worker, settlement
from neuroshard.evolution.transactions import Outbox, ClosedOperation
from test_native_audit_quorum import case, submit, send
from test_settlement import scenario, blocks
from test_expert_operator import Node


@pytest.mark.parametrize('kind', ['audit_commit', 'audit_verdict', 'accept_audit'])
def test_closed_native_context_releases_pending_nonce_without_fabricating_a_receipt(case, tmp_path, kind):
    _, owners, *_ = case
    active, budget = submit(case)
    active = blocks(active, 1)
    claim = active['candidate']
    fields = ({'budget_id': budget} if kind == 'accept_audit' else
              {'claim_id': claim['id'], 'commitment': 'a'*64} if kind == 'audit_commit' else
              {'claim_id': claim['id'], 'salt': 'd'*64, 'coverage_root': auditing.coverage(claim), 'valid': True})
    node = Node(active)
    path = tmp_path/'outbox.sqlite'
    box = Outbox(path, 'fixture', active['chain_id'], owners[3], rpc=node.rpc, query=node.query)
    with pytest.raises(TimeoutError):
        box.send('late', kind, timeout=0, **fields)
    raw = box.db.execute('SELECT envelope FROM operations').fetchone()[0]
    assert box.retire_closed(active) is None
    absent = copy.deepcopy(active)
    absent['candidate'] = None
    assert box.retire_closed(absent) is None
    node.state = blocks(active, claim['audit_reveal_end']-active['height']+1)
    assert node.state['candidate'] is None and node.state['issued'] == 0
    wrong = copy.deepcopy(node.state)
    wrong['chain_id'] = 'another-chain'
    with pytest.raises(ValueError, match='committed native state'):
        box.retire_closed(wrong)
    evidence = box.retire_closed(node.state)
    assert evidence['closed_context']['claim_id'] == claim['id']
    assert evidence['transaction_outcome'].startswith('unknown;')
    assert box.db.execute('SELECT envelope,receipt FROM operations').fetchone() == (raw, None)
    box.close()
    box = Outbox(path, 'fixture', active['chain_id'], owners[3], rpc=node.rpc, query=node.query)
    try:
        assert box.pending() is None
        with pytest.raises(ClosedOperation):
            box.send('late', kind, **fields)
        nonce = node.state['accounts'][owners[3].public_key]['nonce']
        box.send('next', 'transfer', to=owners[0].public_key, amount=1)
        assert protocol.parse_json(node.broadcasts[-1])['body']['nonce'] == nonce
        assert node.state['issued'] == 0
        settlement.invariant(node.state)
    finally:
        box.close()


@pytest.mark.parametrize('closed', [False, True])
def test_worker_refreshes_claim_and_shortened_deadline_before_signing(case, tmp_path, monkeypatch, closed):
    _, owners, *_ = case
    original, _ = submit(case)
    claim = original['candidate']
    latest = copy.deepcopy(original)
    for owner in owners:
        latest = send(latest, owner, 'audit_commit', claim_id=claim['id'], commitment=auditing.verdict_commitment(
            latest['chain_id'], claim['id'], owner.public_key, auditing.coverage(claim), 'd'*64, True))
    latest = blocks(latest, 1)
    assert latest['candidate']['audit_commit_end'] < claim['audit_commit_end']
    if closed:
        latest = blocks(latest, latest['candidate']['audit_reveal_end']-latest['height']+1)
    queries = []

    def query(url, path=None):
        if path == '/candidate':
            queries.append(path)
            return original['candidate'] if len(queries) == 1 else latest['candidate']
        if path == '/auditing':
            return original['auditing'] if not queries else latest['auditing']
        return {'height': latest['height']}

    monkeypatch.setattr(audit_worker.wire, 'query', query)
    worker = audit_worker.Worker.__new__(audit_worker.Worker)
    worker.owner, worker.chain_id, worker.url = owners[3], original['chain_id'], 'fixture'
    worker.portable, worker.native = False, True
    worker.state_reader, worker.admission_backend = None, None
    worker.outbox = Outbox(tmp_path/'outbox.sqlite', 'fixture', worker.chain_id, worker.owner)
    worker.db = sqlite3.connect(tmp_path/'reports.sqlite')
    worker.db.execute('CREATE TABLE reports (id TEXT PRIMARY KEY, salt TEXT, report BLOB)')
    worker.db.execute('INSERT INTO reports VALUES (?,?,?)', (claim['id'], 'd'*64, canonical({'valid': True})))
    sent = []
    worker.send = lambda *a, **kw: sent.append((a, kw))
    try:
        expected = 'claim_closed_during_replay' if closed else 'full_audit_revealed'
        assert worker.tick()['phase'] == expected
        assert len(sent) == (0 if closed else 1)
        if sent:
            assert sent[0][0][1] == 'audit_verdict' and sent[0][1]['claim_id'] == claim['id']
        assert worker.outbox.pending() is None
    finally:
        worker.outbox.close()
        worker.db.close()
