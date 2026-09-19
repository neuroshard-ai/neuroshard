"""Operator recovery through real ledger transitions and durable signed bytes.

Numerical boundaries and curator/auditor verdicts here are explicit fixtures.
These tests measure orchestration, not language-model quality or neural replay.
"""
import base64
import copy
import hashlib
import json
import sqlite3

import pytest

from neuroshard.demo import client as wire, protocol
from neuroshard.evolution import expert_lifecycle as life, expert_work, settlement
from neuroshard.evolution.expert_operator import Operator
from neuroshard.evolution.reference_data import identity
from neuroshard.evolution.transactions import Outbox
from test_expert_admission import enabled, data_for, renamed
from test_expert_lifecycle import network, graphs, send, finish, template_for
from test_serving_graph import FIXTURE
from test_settlement import blocks


def trained(enabled):
    """A fixture boundary with the native admission window metadata intact."""
    original, owners = enabled
    state = copy.deepcopy(original)
    work = expert_work.prescription(state)
    end = renamed(work['parent'], json.loads(FIXTURE.read_bytes())['candidate']['experts']['protocol'],
                  work['checkpoint']['job'])
    claim = identity(['operator-fixture-work', work['checkpoint']['job']])
    state['expert_work']['feature_claim'] = identity(['fixture-prefix', claim])
    if work['format'] == expert_work.PROSPECTIVE:
        state['expert_work'].update(feature_root=identity(['bank', claim]),
            batch_roots=[identity(['batch', claim, i]) for i in range(work['batch_count'])])
    expert_work.settle(state, {'kind': 'expert_training', 'id': claim,
        'work_ids': [identity(['fixture-step', claim, i]) for i in range(560)],
        'workers': [owners[0].public_key], 'output_checkpoint': end,
        'window': {'steps': [{'index': i} for i in range(560)]}})
    settlement.invariant(state)
    return state


def next_cohort(state):
    """Retry fresh data after rejection, still against the accepted graph."""
    initial = renamed(state['manifest']['expert_work']['parent'],
        json.loads(FIXTURE.read_bytes())['initial_expert'], identity(['operator-next-cohort']))
    template = template_for(json.loads(FIXTURE.read_bytes())['candidate'], initial)
    work = {'format': expert_work.PROSPECTIVE, 'parent': template['parent'], 'checkpoint': initial,
        'prepared': identity(['operator-next-inputs']), 'feature_stages': 3, 'batch_count': 1,
        'schedule': [0]*560, 'numerical_profile': template['numerical_profile']}
    profile = {'format': life.PROSPECTIVE, 'serving_graph': state['expert_lifecycle']['serving_graph'],
        'candidate_template': template,
        'quality': {'policy_root': identity(['operator-next-policy']),
                    'prepared': identity(['operator-next-quality']), 'stages': 2},
        **{key: state['manifest']['expert_lifecycle'][key] for key in ('price_per_token', 'max_tokens')}}
    return {'work': work, 'lifecycle': profile, 'data': data_for(state, work, 1)}


class Node:
    def __init__(self, state):
        self.state, self.receipts, self.broadcasts = state, {}, []
        self.crash_after_commit = False

    def read(self):
        return copy.deepcopy(self.state)

    def query(self, url, path, data):
        assert path == '/account'
        return self.state['accounts'][data['public_key']]

    def rpc(self, url, method, params, **kwargs):
        if method == 'tx':
            key = base64.b64decode(params['hash']).hex().upper()
            if key not in self.receipts:
                raise wire.Rejected('Transaction not found')
            return self.receipts[key]
        assert method == 'broadcast_tx_sync'
        raw = base64.b64decode(params['tx'])
        key = hashlib.sha256(raw).hexdigest().upper()
        assert key not in self.receipts
        self.broadcasts.append(raw)
        self.state = settlement.transition(self.state, protocol.parse_json(raw))
        self.state = blocks(self.state, 1)
        settlement.invariant(self.state)
        self.receipts[key] = {'hash': key, 'height': str(self.state['height']), 'tx_result': {'code': 0}}
        if self.crash_after_commit:
            self.crash_after_commit = False
            raise KeyboardInterrupt('Process disappeared after commitment')
        return {'code': 0}


def build(home, node, owners, backend, feed):
    home.mkdir(parents=True, exist_ok=True)
    outbox = Outbox(home/'outbox.sqlite', 'fixture://local', node.state['chain_id'], owners[0],
                    rpc=node.rpc, query=node.query)
    workers = {'prefix': [protocol.Identity('operator-prefix-'+str(i)).public_key for i in range(3)],
               'training': protocol.Identity('operator-training').public_key}
    return Operator(home, outbox, node.read, backend, feed, workers, max_jobs=3), outbox


def quality_backend(calls):
    def execute(request):
        assert request['phase'] == 'quality'
        calls.append(request)
        job = request['job']
        profile = job['lifecycle']
        graph = (life.materialize_graph(profile['candidate_template'], request['work']['checkpoint'])
                 if profile['format'] == life.PROSPECTIVE else profile['candidate_graph'])
        report = {'format': life.FORMAT+'/quality', 'policy_root': profile['quality']['policy_root'],
                  'baseline_graph': identity(profile['serving_graph']), 'candidate_graph': identity(graph),
                  'prepared': profile['quality']['prepared'], 'passed': False, 'results_root': '8'*64}
        return {'report': report, 'transcript_root': '9'*64}
    return execute


def test_failed_quality_is_audited_closed_and_followed_by_another_cohort(enabled, tmp_path):
    original, owners = enabled
    node = Node(trained(enabled))
    serving, issued = node.state['serving_root'], node.state['issued']
    calls = []
    operator, outbox = build(tmp_path, node, owners, quality_backend(calls), next_cohort)
    try:
        assert operator.tick()['phase'] == 'cohort_prepared'
        assert operator.tick()['kind'] == 'fund_audit'
        budget = next(iter(node.state['auditing']['budgets']))
        assert operator.tick()['phase'] == 'waiting_for_auditors'
        for owner in owners[:3]:
            node.state = send(node.state, owner, 'accept_audit', budget_id=budget)
        result = operator.tick()
        assert result['kind'] == 'quality_expert' and len(calls) == 1
        assert not node.state['candidate']['report']['passed']
        assert operator.tick()['phase'] == 'waiting_for_audit'
        node.state = finish(node.state, owners)
        outcome = operator.tick()
        assert outcome['phase'] == 'quality_rejected'
        assert (node.state['serving_root'], node.state['issued']) == (serving, issued)
        assert operator.tick()['phase'] == 'cohort_prepared'
        assert operator.tick()['kind'] == 'propose_expert_job'
        assert node.state['manifest'] == original['manifest']
    finally:
        operator.close()
        outbox.close()


def test_committed_unknown_transaction_recovers_without_new_nonce_or_execution(enabled, tmp_path):
    original, owners = enabled
    node = Node(trained(enabled))
    calls = []
    backend = quality_backend(calls)
    operator, outbox = build(tmp_path, node, owners, backend, lambda _: None)
    operator.tick()
    node.crash_after_commit = True
    with pytest.raises(KeyboardInterrupt):
        operator.tick()
    pending = outbox.pending()
    assert pending is not None and len(node.broadcasts) == 1
    nonce = node.state['accounts'][owners[0].public_key]['nonce']
    operator.close()
    outbox.close()
    operator, outbox = build(tmp_path, node, owners, backend, lambda _: None)
    try:
        assert operator.tick() == {'phase': 'transaction_recovered', 'operation': pending}
        assert not calls and len(node.broadcasts) == 1
        assert node.state['accounts'][owners[0].public_key]['nonce'] == nonce
        assert operator.tick()['phase'] == 'waiting_for_auditors'
    finally:
        operator.close()
        outbox.close()


def test_execution_result_survives_restart_and_cannot_change_request(enabled, tmp_path):
    original, owners = enabled
    node = Node(trained(enabled))
    calls = []
    backend = lambda request: calls.append(request) or {'proof': 'a'*64}
    operator, outbox = build(tmp_path, node, owners, backend, lambda _: None)
    request = {'phase': 'prefix', 'checkpoint': 'b'*64}
    assert operator.execute('numerical-operation', request) == {'proof': 'a'*64}
    operator.close()
    outbox.close()
    operator, outbox = build(tmp_path, node, owners, backend, lambda _: None)
    try:
        assert operator.execute('numerical-operation', request) == {'proof': 'a'*64}
        assert len(calls) == 1
        with pytest.raises(ValueError, match='changed its committed request'):
            operator.execute('numerical-operation', {**request, 'checkpoint': 'c'*64})
    finally:
        operator.close()
        outbox.close()


def test_curators_review_once_and_submit_their_own_rejection_with_their_audit_signer(enabled, tmp_path, monkeypatch):
    from neuroshard.evolution.audit_worker import Worker
    original, owners = enabled
    node = Node(trained(enabled))
    # Close the fixture's first quality job, leaving its earlier model serving.
    from test_expert_lifecycle import fund
    node.state, budget = fund(node.state, owners, 96)
    fields = quality_backend([])({'phase': 'quality', 'job': {
        'lifecycle': node.state['manifest']['expert_lifecycle']}, 'work': node.state['expert_work']})
    node.state = finish(send(node.state, owners[0], 'quality_expert', **fields, audit_budget=budget), owners)
    node.state = send(node.state, owners[0], 'propose_expert_job', job=next_cohort(node.state))
    proposal = node.state['expert_lifecycle']['admission']['proposal']
    calls = []

    def review(config, request):
        calls.append(request)
        return {'proposal': proposal['id'], 'job': identity(proposal['job']), 'approve': False,
                'review': {'mechanical_checks_passed': True, 'reason': 'Fixture publisher outside curator policy'}}

    monkeypatch.setattr('neuroshard.evolution.expert_operator.command', review)
    worker = Worker.__new__(Worker)
    worker.owner, worker.chain_id, worker.state_reader = owners[1], node.state['chain_id'], node.read
    worker.admission_backend = {'argv': ['fixture-review'], 'timeout_seconds': 10}
    worker.db = sqlite3.connect(tmp_path/'reviews.sqlite')
    worker.db.execute('CREATE TABLE reviews (id TEXT PRIMARY KEY, policy TEXT, report BLOB)')
    worker.outbox = Outbox(tmp_path/'curator-outbox.sqlite', 'fixture://local', worker.chain_id,
                          worker.owner, rpc=node.rpc, query=node.query)
    try:
        before = node.state['accounts'][owners[0].public_key]['nonce']
        result = worker.review_admission()
        assert result['phase'] == 'admission_review_submitted' and result['approve'] is False
        assert node.state['expert_lifecycle']['admission']['proposal']['votes'] == {owners[1].public_key: False}
        assert node.state['accounts'][owners[0].public_key]['nonce'] == before
        assert worker.review_admission() == {'phase': 'idle'}
        assert len(calls) == 1 and len(node.broadcasts) == 1
    finally:
        worker.outbox.close()
        worker.db.close()


def test_local_state_reader_observes_committed_wal_and_checks_genesis(enabled, tmp_path):
    from neuroshard.dataflow.store import canonical
    from neuroshard.evolution.committed_state import read
    state, _ = enabled
    state = copy.deepcopy(state)
    state['height'] = 1
    (tmp_path/'config').mkdir()
    genesis = {'chain_id': state['chain_id'], 'app_state': {'manifest': state['manifest']}}
    raw = canonical(genesis)
    (tmp_path/'config/genesis.json').write_bytes(raw)
    pin = hashlib.sha256(raw).hexdigest()
    db = sqlite3.connect(tmp_path/'evolution.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE state (id INTEGER PRIMARY KEY, value BLOB)')
    with db:
        db.execute('INSERT INTO state VALUES (1,?)', (canonical(state),))
    try:
        assert read(tmp_path, pin) == state
        state['height'] = 2
        db.execute('UPDATE state SET value=?', (canonical(state),))
        assert read(tmp_path, pin)['height'] == 1
        db.commit()
        assert read(tmp_path, pin)['height'] == 2
        with pytest.raises(ValueError, match='pinned network'):
            read(tmp_path, 'f'*64)
    finally:
        db.close()


def test_rejected_feed_entry_is_journaled_without_transactions_or_model_changes(enabled, tmp_path):
    original, owners = enabled
    node = Node(trained(enabled))
    # Isolate source selection here; quality transitions are covered above.
    node.state['expert_lifecycle']['quality_closed'] = True
    baseline = copy.deepcopy(node.state)
    entries = iter([{'rejected_data': {'entry': 'd'*64,
        'review': {'mechanical_checks_passed': False, 'reason': 'Duplicate protected evaluation input'}}},
        next_cohort(node.state)])
    operator, outbox = build(tmp_path, node, owners, lambda _: pytest.fail('No numerical work before admission'),
                             lambda _: next(entries))
    try:
        assert operator.tick()['phase'] == 'data_rejected'
        assert node.state == baseline and not node.broadcasts
        assert operator.tick()['phase'] == 'cohort_prepared'
        assert operator.tick()['kind'] == 'propose_expert_job'
        assert len(node.broadcasts) == 1
    finally:
        operator.close()
        outbox.close()


def test_slow_preparation_retries_when_native_admission_changes(enabled, tmp_path):
    original, owners = enabled
    node = Node(trained(enabled))
    node.state['expert_lifecycle']['quality_closed'] = True

    def prepare(state):
        job = next_cohort(state)
        node.state['data_root'] = 'b'*64  # Another accepted cohort during preparation.
        return job

    operator, outbox = build(tmp_path, node, owners, None, prepare)
    try:
        assert operator.tick()['phase'] == 'prepared_snapshot_changed'
        assert not node.broadcasts
        assert operator.db.execute('SELECT count(*) FROM cohorts').fetchone()[0] == 0
    finally:
        operator.close()
        outbox.close()
