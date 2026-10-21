"""Regressions for the October 21 review of the admission PR: its four findings, as executable guarantees.

1. HOLD discloses nothing an owner can broadcast, and one turn funds at most one job.
2. Only a complete first work frame starts a job; partial or slow input stays idle time.
3. Funding outcomes are decided from verified ledger state, never from unauthenticated index or block RPCs.
4. A captured HELLO cannot replace the admitted client's connection.
"""
import base64
import json
import socket
import threading
import time
import types

import pytest

from admission_net import NEURO, SERVED, Net, jobs_of, start_work, wait_until
from neuroshard.assistant import admission, admission_client, funding
from neuroshard.inference import optimistic as ledger
from neuroshard.inference import optimistic_network as chain_network
from neuroshard.inference import relay
from test_admission import finish
from test_admission_recovery import index, prepared


# -- 1. HOLD ------------------------------------------------------------------------------------------------------
def leaking_owner(net):
    """Owner 1 tries to fund the job from what its HOLD disclosed, then loses its HOLD reply."""
    admitter = net.admitter('alice')
    real_request, attempts = admitter.request, []

    def request(endpoint, message, timeout=10):
        body = message['body']
        if body['kind'] == 'admission/hold' and body['rank'] == 1 and not attempts:
            real_request(endpoint, message, timeout)
            disclosed = body['transaction']
            assert 'signature' not in disclosed
            for signature in (message['signature'], '', '00' * 64):  # grafting any signature it has seen
                try:
                    attempts.append(net.chain.broadcast({**disclosed, 'signature': signature})['code'])
                except chain_network.Rejected:
                    attempts.append('rejected')
            net.chain.mine()
            raise TimeoutError('the HOLD reply is lost')
        return real_request(endpoint, message, timeout)

    admitter.request = request
    return admitter, attempts


def test_a_hold_discloses_nothing_an_owner_can_broadcast_and_busy_means_unfunded(tmp_path):
    net = Net(tmp_path)
    admitter, attempts = leaking_owner(net)
    with pytest.raises(admission_client.NetworkBusy, match='nothing was funded'):
        net.turn('alice', wait=0, admitter=admitter)
    assert attempts and 0 not in attempts, attempts
    assert jobs_of(net, 'alice') == []
    assert [r['state'] for r in admitter.operations.all()] == [funding.ABANDONED]


def test_a_lost_hold_reply_then_a_retry_funds_exactly_one_job(tmp_path):
    net = Net(tmp_path)
    admitter, attempts = leaking_owner(net)
    job_id, _, sockets, _ = net.turn('alice', admitter=admitter)
    try:
        assert attempts and 0 not in attempts, attempts
        assert jobs_of(net, 'alice') == [job_id]
    finally:
        finish(sockets)


def test_a_failed_hold_with_a_consumed_nonce_resolves_the_same_operation(tmp_path):
    """Defence in depth: if the nonce moved anyway, the turn is resolved, never abandoned and funded again."""
    net = Net(tmp_path)
    admitter = net.admitter('alice')
    real_request = admitter.request
    stolen = []

    def request(endpoint, message, timeout=10):
        body = message['body']
        if body['kind'] == 'admission/hold' and body['rank'] == 1 and not stolen:
            record = next(r for r in admitter.operations.all() if r['state'] == funding.PREPARED)
            stolen.append(record['job_id'])
            assert net.chain.broadcast(record['envelope'])['code'] == 0  # as if the signed bytes had leaked
            net.chain.mine()
            raise TimeoutError('the HOLD reply is lost')
        return real_request(endpoint, message, timeout)

    admitter.request = request
    job_id, _, sockets, _ = net.turn('alice', admitter=admitter)
    try:
        assert jobs_of(net, 'alice') == [job_id] == stolen
    finally:
        finish(sockets)


def test_a_hold_carrying_a_signature_is_refused(tmp_path):
    net = Net(tmp_path)
    gate = net.owner(1).gate
    offer = gate.reserve_new(net.accounts['alice'].public, '10.0.0.1', '33' * 32, NEURO)
    _, record = prepared(net, 'alice')
    with pytest.raises(admission.Refused, match='invalid'):
        gate.hold(net.accounts['alice'].public, offer['reservation'], record['envelope'])


# -- 2. first work frame ------------------------------------------------------------------------------------------
@pytest.mark.parametrize('partial', [b'C', b'C\0\0\0\x10' + b'\0' * 7], ids=['partial-header', 'partial-payload'])
def test_partial_first_frames_do_not_escape_the_idle_budget(tmp_path, partial):
    net = Net(tmp_path)
    owner, entered = net.owner(1), threading.Event()

    def work(self, connection, *args):
        entered.set()
        return None

    owner.work = types.MethodType(work, owner)
    job_id, _, sockets, _ = net.turn('alice')
    try:
        sockets[1].sendall(partial)
        time.sleep(0.2)
        assert owner.gate.lease['state'] == 'admitted' and job_id not in owner.gate.started
        net.clock.now += 65
        owner.gate.tick()
        assert owner.gate.lease is None, 'the idle lease lapsed'
        assert owner.gate.budget.used(('account', net.accounts['alice'].public), net.clock()) >= 59
        assert owner.gate.reserve_new(net.accounts['bob'].public, '10.0.0.2', 'ef' * 32, NEURO)['reservation']
        sockets[1].settimeout(5)
        assert sockets[1].recv(1) == b'', 'the idle connection was closed'
        assert not entered.is_set() and job_id not in owner.gate.started
    finally:
        finish(sockets)


def test_a_complete_first_frame_starts_the_job_and_reaches_the_serving_loop_intact(tmp_path):
    net = Net(tmp_path)
    owner, received = net.owner(1), []

    def work(self, connection, job_id, *args):
        received.append(relay.OwnerRelay(connection, 1, SERVED.world, 4, 8).command())
        return None

    owner.work = types.MethodType(work, owner)
    job_id, _, sockets, _ = net.turn('alice')
    try:
        command = relay.COMMAND.pack(7, 3)
        frame = relay.HEADER.pack(relay.CONTROL, len(command)) + command
        sockets[1].sendall(frame[:3])  # slow delivery within the budget still counts as one frame
        time.sleep(0.1)
        sockets[1].sendall(frame[3:])
        wait_until(lambda: received)
        assert received == [[7, 3]] and job_id in owner.gate.started
    finally:
        finish(sockets)


def test_a_first_frame_that_is_not_a_command_does_not_start_the_job(tmp_path):
    net = Net(tmp_path)
    owner = net.owner(1)
    job_id, _, sockets, _ = net.turn('alice')
    try:
        relay.send_frame(sockets[1], relay.TENSOR, b'\0' * 8)
        sockets[1].settimeout(5)
        try:
            assert sockets[1].recv(1) == b''
        except ConnectionResetError:
            pass  # closed with our unread frame still queued
        assert job_id not in owner.gate.started
    finally:
        finish(sockets)


# -- 3. funding resolution ----------------------------------------------------------------------------------------
def committed(net, name='alice'):
    operations, record = prepared(net, name)
    record = operations.mark(record, funding.BROADCAST, since_height=net.chain.s['height'])
    index(net).rebroadcast(record['envelope'])
    net.chain.mine()
    return operations, record


def lying(net, **answers):
    def rpc(address, method, params):
        if method in answers:
            answer = answers[method]
            if isinstance(answer, Exception):
                raise answer
            return answer
        return net.chain.call(address, method, params)
    return funding.LedgerIndex(net.chain, rpc)


def test_an_rpc_cannot_report_a_live_funded_job_as_failed(tmp_path):
    net = Net(tmp_path)
    operations, record = committed(net)
    resolver = lying(net, tx={'height': str(net.chain.s['height']), 'tx_result': {'code': 1, 'log': 'invented'}})
    assert funding.resolve(resolver, operations, record)['state'] == funding.FUNDED


def test_invented_blocks_cannot_make_a_funded_job_unfunded(tmp_path):
    net = Net(tmp_path)
    operations, record = committed(net)
    fake = {'public_key': net.accounts['alice'].public, 'body': {'nonce': record['nonce'], 'kind': 'transfer'}}
    resolver = lying(net, tx=chain_network.Rejected('tx not found'),
                     block={'block': {'data': {'txs': [base64.b64encode(ledger.canonical(fake)).decode()]}}},
                     block_results={'txs_results': [{'code': 0}]})
    assert funding.resolve(resolver, operations, record)['state'] == funding.FUNDED


def test_a_reported_failure_with_a_free_nonce_is_superseded_then_proven_unfunded(tmp_path):
    net = Net(tmp_path)
    operations, record = prepared(net, 'alice')
    resolver = lying(net, tx={'height': '1', 'tx_result': {'code': 1, 'log': 'reported failing'}})
    real = resolver.rebroadcast

    def lossy(envelope):  # the funding broadcast is silently lost; anything else goes through
        return {'code': 0, 'log': ''} if envelope['body']['kind'] == 'serve_open' else real(envelope)

    resolver.rebroadcast = lossy
    admitter = net.admitter('alice')
    admitter.index = resolver
    record = funding.fund(resolver, operations, record, timeout=30, sleep=net.sleep, clock=net.clock,
                          supersede=admitter.supersede)
    assert record['state'] == funding.UNFUNDED and record['superseded'], record
    assert jobs_of(net, 'alice') == []


def test_a_consumed_nonce_with_results_possibly_evicted_is_lapsed_not_unfunded(tmp_path):
    net = Net(tmp_path)
    operations, record = prepared(net, 'alice')
    record = operations.mark(record, funding.BROADCAST, since_height=net.chain.s['height'])
    other = net.accounts['alice'].sign(net.chain.chain_id, record['nonce'], 'transfer',
                                       to=net.accounts['bob'].public, amount=1)
    assert net.chain.broadcast(other)['code'] == 0
    net.chain.mine()
    limit = net.chain.s['params']['max_results']
    net.chain.s['results'] = {f'{i:064x}': {'height': net.chain.s['height'], 'status': 'expired'}
                              for i in range(limit)}  # a full window, all newer than the broadcast
    assert funding.resolve(index(net), operations, record)['state'] == funding.LAPSED


# -- 4. HELLO replay ----------------------------------------------------------------------------------------------
def captured_hello(net, monkeypatch):
    captured, send_frame = [], relay.send_frame

    def capture(sock, kind, payload):
        if kind == relay.HELLO and json.loads(payload)['hello']['rank'] == 1:
            captured.append(bytes(payload))
        return send_frame(sock, kind, payload)

    with monkeypatch.context() as patch:
        patch.setattr(relay, 'send_frame', capture)
        turn = net.turn('alice')
    assert len(captured) == 1
    return turn, captured[0]


def replay(owner, payload, strip=False):
    if strip:  # the captured HELLO without its admission attempt: a legacy HELLO
        message = json.loads(payload)
        message.pop('admission')
        payload = ledger.canonical(message)
    mine, theirs = socket.socketpair()
    mine.settimeout(5)
    threading.Thread(target=owner.serve, args=(theirs, lambda *_: None, '10.9.9.9'), daemon=True).start()
    relay.send_frame(mine, relay.HELLO, payload)
    try:
        kind, size = relay.HEADER.unpack(relay.read_exactly(mine, relay.HEADER.size))
        return kind, relay.read_exactly(mine, size)
    except (OSError, ConnectionError):
        return None, None
    finally:
        mine.close()


@pytest.mark.parametrize('strip', [False, True], ids=['as-captured', 'stripped-to-legacy'])
def test_a_captured_hello_cannot_evict_the_admitted_client(tmp_path, monkeypatch, strip):
    net = Net(tmp_path)
    (job_id, _, sockets, _), payload = captured_hello(net, monkeypatch)
    owner = net.owner(1)
    before = owner.gate.gens[job_id]
    kind, _ = replay(owner, payload, strip)
    try:
        assert kind != relay.READY and owner.gate.gens[job_id] == before
        sockets[1].settimeout(0.3)
        with pytest.raises(socket.timeout):
            sockets[1].recv(1)  # the admitted connection is still open
        start_work(sockets)
        wait_until(lambda: all(o.worked == [job_id] for o in net.owners))
    finally:
        finish(sockets)


def test_a_fresh_signed_retry_still_replaces_a_lost_ready_connection(tmp_path, monkeypatch):
    net = Net(tmp_path)
    (job_id, _, sockets, session_key), payload = captured_hello(net, monkeypatch)
    replay(net.owner(1), payload)  # a replay first: refused, and it does not use up the client's retry
    again = net.rehello(job_id, 1, session_key)
    assert net.owner(1).gate.gens[job_id] == 2
    finish({1: again, 2: sockets[2]})


def test_a_ready_for_another_attempt_is_not_accepted(tmp_path):
    net = Net(tmp_path)
    job_id, owners, sockets, session_key = net.turn('alice')
    admitter = net.admitter('alice')
    owner = net.owner(1)
    first = admitter.attempt(session_key, job_id, 1, owner.public, owner.gate.lease['reservation'])
    second = admitter.attempt(session_key, job_id, 1, owner.public, owner.gate.lease['reservation'])
    body = relay.hello(net.chain.chain_id, job_id, 1, SERVED.world, 4, 8)
    with pytest.raises(ValueError, match='another admission attempt'):
        net.dial('10.0.0.1')(owners[0][1], body, session_key, admission=first['admission'],
                             accept_ready=second['accept_ready'])
    finish({k: v for k, v in sockets.items() if k != 1})
