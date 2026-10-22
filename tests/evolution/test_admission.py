"""DESIGN_ADMISSION.md acceptance tests (numbers in names), on the real ledger transition and real owner admission."""
import json
import socket
import threading

import pytest

from admission_net import NEURO, PARAMS, SERVED, Net, jobs_of, start_work, wait_until
from neuroshard.assistant import admission, admission_client, admission_messages as messages, funding
from neuroshard.inference import optimistic as ledger
from neuroshard.inference import relay


def balance(net, name):
    return net.chain.s['accounts'][net.accounts[name].public]['balance']


def finish(sockets):
    for sock in sockets.values():
        sock.shutdown(socket.SHUT_RDWR)
        sock.close()


def test_1_a_second_client_waits_unfunded_then_is_served_after_the_first(tmp_path):
    net = Net(tmp_path)
    job_a, _, sockets_a, _ = net.turn('alice')
    start_work(sockets_a)
    wait_until(lambda: all(o.worked == [job_a] for o in net.owners))
    before, funded_while_waiting = balance(net, 'bob'), []

    def first_finishes():  # while bob backs off, nothing of his is funded; then alice's conversation ends
        funded_while_waiting.append(jobs_of(net, 'bob'))
        if net.clock.now - start > 20 and sockets_a:
            finish(sockets_a)
            sockets_a.clear()
            wait_until(lambda: all(o.gate.lease is None for o in net.owners))

    start = net.clock.now
    net.hooks.append(first_finishes)
    job_b, _, sockets_b, _ = net.turn('bob')
    assert job_b != job_a and all(found == [] for found in funded_while_waiting[:-1])
    assert jobs_of(net, 'bob') == [job_b] and balance(net, 'bob') == before - NEURO - PARAMS['fee']
    finish(sockets_b)


def test_1_a_client_whose_deadline_passes_funds_nothing(tmp_path):
    net = Net(tmp_path)
    _, _, sockets_a, _ = net.turn('alice')
    start_work(sockets_a)
    wait_until(lambda: all(o.gate.lease and o.gate.lease['state'] == 'started' for o in net.owners))
    before = balance(net, 'bob')
    with pytest.raises(admission_client.NetworkBusy, match='nothing was funded'):
        net.turn('bob', wait=60)
    assert jobs_of(net, 'bob') == [] and balance(net, 'bob') == before
    finish(sockets_a)


def test_2_partial_busy_releases_the_offered_shard_and_funds_nothing(tmp_path):
    net = Net(tmp_path)
    blocker = net.owner(2).gate.reserve_new(net.accounts['alice'].public, '10.9.9.9', 'cd' * 32, NEURO)
    assert blocker['reservation']
    admitter = net.admitter('bob')
    offers, retry_after, missing = admitter.reserve_shards(net.chain.state(), 'ef' * 32, net.clock.now + 600)
    assert offers is None and missing == [2] and retry_after >= 1
    assert net.owner(1).gate.lease is None, 'shard 1 was released'
    assert jobs_of(net, 'bob') == []


def test_3_confirmation_after_the_leases_lapsed_admits_the_same_job_later(tmp_path):
    net = Net(tmp_path)
    net.chain.paused = True  # confirmation outlasts every hold (bounded by the 60 s budget)
    resume = lambda: setattr(net.chain, 'paused', net.clock.now - start < 90)  # noqa: E731
    start = net.clock.now
    net.hooks.append(resume)
    job_id, _, sockets, _ = net.turn('alice')
    assert jobs_of(net, 'alice') == [job_id], 'one job: never a second'
    start_work(sockets)
    wait_until(lambda: all(o.worked == [job_id] for o in net.owners))
    finish(sockets)


def test_3_a_funded_job_its_owners_never_admit_is_refunded_in_full_at_expiry(tmp_path):
    net = Net(tmp_path)
    net.chain.paused = True
    start, taken = net.clock.now, []

    def confirm_then_occupy():
        if net.clock.now - start >= 90 and not taken:
            net.chain.paused = False
            net.chain.mine()
            taken.append(net.owner(2).gate.reserve_new(net.accounts['bob'].public, '10.9.9.9', 'cd' * 32, NEURO))
        if taken and net.owner(2).gate.lease is None:  # bob keeps shard 2 occupied
            net.owner(2).gate.budget = admission.Budget(seconds=10 ** 9)
            net.owner(2).gate.reserve_new(net.accounts['bob'].public, '10.9.9.9', 'cd' * 32, NEURO)

    net.hooks.append(confirm_then_occupy)
    before = balance(net, 'alice')
    with pytest.raises(admission_client.NotServed, match='refunded at block') as raised:
        net.turn('alice', wait=200)
    assert jobs_of(net, 'alice') == [raised.value.job_id]
    net.hooks.clear()
    while net.chain.s['height'] <= raised.value.expires:
        net.chain.mine()
    assert net.chain.s['results'][raised.value.job_id]['status'] == 'expired'
    assert balance(net, 'alice') == before - PARAMS['fee'], 'the full escrow returns; only the fee was charged'


def test_9_a_dropped_caps_reply_is_unknown_never_legacy(tmp_path):
    net = Net(tmp_path)
    send = net.request('10.0.0.1')
    calls = []

    def dropping(endpoint, message, timeout=10):
        calls.append(message['body']['kind'])
        if message['body']['kind'] == 'caps' and net.endpoints[endpoint].rank == 2:
            raise TimeoutError('dropped')
        return send(endpoint, message, timeout)

    admitter = net.admitter('alice')
    admitter.request = dropping
    with pytest.raises(admission_client.NetworkBusy):
        net.turn('alice', wait=30, admitter=admitter)
    assert jobs_of(net, 'alice') == [] and 'hold' not in calls and calls.count('caps') > 2, 'asked again, no fallback'
    assert net.owner(2).public not in admitter.capable


def test_11_a_different_owner_of_the_shard_refuses_a_funded_job(tmp_path):
    net = Net(tmp_path, owners_per_shard=2)
    job_id, owners, sockets, session_key = net.turn('alice')
    named = {key for key, _ in owners}
    other = next(o for o in net.owners if o.rank == 1 and o.public not in named)
    message = messages.by_key(session_key, 'reserve', net.chain.chain_id, net.clock, rank=1, owner=other.public,
                              job_id=job_id)
    reply = net.request('10.0.0.1')(next(e for e, o in net.endpoints.items() if o is other), message)
    assert reply['body']['kind'] == 'refused' and reply['body']['reason'] == 'invalid'
    finish(sockets)


def test_12_a_funded_job_is_re_reserved_without_a_second_balance(tmp_path):
    net = Net(tmp_path)
    job_id, owners, sockets, session_key = net.turn('alice')
    finish(sockets)  # READY received, nothing started: the leases end
    wait_until(lambda: all(o.gate.lease is None for o in net.owners))
    net.chain.s['accounts'][net.accounts['alice'].public]['balance'] = 0  # nothing left to spend (test-only)
    net.chain.s['burned'] += balance(net, 'alice')
    for shard, (key, endpoint) in enumerate(owners, start=1):
        message = messages.by_key(session_key, 'reserve', net.chain.chain_id, net.clock, rank=shard, owner=key,
                                  job_id=job_id)
        assert net.request('10.0.0.1')(endpoint, message)['body']['kind'] == 'offer'


def test_overlapping_waits_overlap_in_simulated_time(tmp_path):
    """The harness itself: two clients each waiting 10 s from t0 both wake at t0+10, not one at t0+20."""
    net = Net(tmp_path)
    start, height = net.clock.now, net.chain.s['height']

    def wait():
        net.sleep(10)
        return net.clock.now - start

    woke = [value for _, value in net.concurrently([wait, wait])]
    assert woke == [10, 10] and net.clock.now - start == 10
    assert net.chain.s['height'] - height == 10, 'one block per simulated second, mined once'


BURST = ('alice', 'bob', 'carol', 'dave')


def burst(tmp_path, seed, wait=40):
    """Four clients at once against one owner per shard, under one scheduler of simulated time (each client's
    backoff seeded). Returns the net and each client's outcome: ('served', job_id) or ('busy', None)."""
    import random

    net = Net(tmp_path, users=BURST)

    def turn(index, name):
        admitter = net.admitter(name, f'10.0.0.{index}', rng=random.Random(seed * 100 + index))
        try:
            job_id, _, sockets, _ = net.turn(name, f'10.0.0.{index}', wait=wait, admitter=admitter)
        except admission_client.NetworkBusy:
            return 'busy', None
        start_work(sockets)
        wait_until(lambda: all(job_id in o.worked for o in net.owners))
        finish(sockets)
        return 'served', job_id

    results = net.concurrently([lambda i=i, n=n: turn(i, n) for i, n in enumerate(BURST, 1)])
    errors = [value for status, value in results if status == 'error']
    assert not errors, errors
    return net, dict(zip(BURST, (value for _, value in results)))


@pytest.mark.parametrize('seed', range(5))
def test_10_a_burst_funds_nothing_for_a_busy_client_and_conserves_the_ledger(tmp_path, seed):
    """Safety, evaluated on its own: a liveness failure elsewhere never hides it."""
    net, outcomes = burst(tmp_path, seed)
    for name, (kind, job_id) in outcomes.items():
        assert jobs_of(net, name) == ([job_id] if kind == 'served' else []), (name, kind)
    ledger.invariant(net.chain.s)  # also asserted on every mined block


@pytest.mark.parametrize('seed', range(5))
def test_10_a_burst_with_capacity_free_again_makes_progress(tmp_path, seed):
    """Progress: each served conversation ends and frees its owners, so within the deadline every client of the
    burst is served in turn (300 of 300 seeded bursts served all four when issue #126 was diagnosed)."""
    _, outcomes = burst(tmp_path, seed)
    assert all(kind == 'served' for kind, _ in outcomes.values()), outcomes


def test_busy_hints_estimate_when_the_slot_frees_not_the_worst_case_bound(tmp_path):
    """Issue #126: a hint of a lease's full bound (20 s offered, 60 s admitted) sent every contender past its
    deadline, though such leases end within moments."""
    net = Net(tmp_path)
    gate = net.owner(1).gate
    gate.reserve_new(net.accounts['alice'].public, '10.0.0.1', 'cd' * 32, NEURO)
    with pytest.raises(admission.Busy) as offered:
        gate.reserve_new(net.accounts['bob'].public, '10.0.0.2', 'ef' * 32, NEURO)
    assert offered.value.retry_after <= admission.PENDING_RETRY
    gate.lease = None
    job_id, _, sockets, _ = net.turn('alice')  # admitted: READY sent, no work yet
    with pytest.raises(admission.Busy) as admitted:
        net.owner(1).gate.reserve_new(net.accounts['bob'].public, '10.0.0.2', 'ef' * 32, NEURO)
    assert net.owner(1).gate.lease['state'] == 'admitted' and admitted.value.retry_after <= admission.PENDING_RETRY
    start_work(sockets)
    wait_until(lambda: net.owner(1).gate.lease['state'] == 'started')
    with pytest.raises(admission.Busy) as started:
        net.owner(1).gate.reserve_new(net.accounts['bob'].public, '10.0.0.2', 'ef' * 32, NEURO)
    assert started.value.retry_after == admission.STARTED_RETRY
    finish(sockets)


def test_contenders_given_the_same_hint_do_not_retry_in_lockstep():
    import random

    waits = []
    for seed in range(8):
        backoff = admission_client.Backoff(10 ** 6, random.Random(seed), clock=lambda: 0, sleep=waits.append)
        assert backoff.wait(retry_after=15)
    assert min(waits) >= 15 and len(set(waits)) == len(waits), 'honoured, and spread apart'
    backoff = admission_client.Backoff(30, random.Random(0), clock=lambda: 0, sleep=waits.append)
    assert not backoff.wait(retry_after=40), 'a wait past the deadline is not taken'


def test_10_a_waiting_client_is_served_once_a_contender_releases_its_partial_reservation(tmp_path):
    """An explicit schedule: carol holds shard 2 only (a partial reservation), so bob finds the network busy and
    funds nothing; carol releases at t+10 s; bob, still within his deadline, is then served."""
    net = Net(tmp_path, users=('bob', 'carol'))
    held = net.owner(2).gate.reserve_new(net.accounts['carol'].public, '10.9.9.9', 'cd' * 32, NEURO)
    start, released, busy_seen = net.clock.now, [], []

    def release_at_ten():
        if not released and net.clock.now - start >= 10:
            busy_seen.append(jobs_of(net, 'bob'))
            net.owner(2).gate.release(held['reservation'], net.accounts['carol'].public)
            released.append(net.clock.now)

    net.hooks.append(release_at_ten)
    (status, value), = net.concurrently([lambda: net.turn('bob', '10.0.0.2', wait=60)])
    assert status == 'ok', value
    job_id, _, sockets, _ = value
    assert released and busy_seen == [[]], 'bob waited, funding nothing, until carol released'
    assert jobs_of(net, 'bob') == [job_id]
    finish(sockets)


def test_an_old_client_reaching_a_busy_new_owner_fails_on_the_busy_frame(tmp_path):
    net = Net(tmp_path)
    _, _, sockets, _ = net.turn('alice')
    start_work(sockets)
    wait_until(lambda: net.owner(1).gate.lease['state'] == 'started')
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    job = net.chain.s['jobs']  # an old client's funded job, opened without reserving
    key = Ed25519PrivateKey.generate()
    envelope = net.accounts['bob'].sign(net.chain.chain_id, 0, 'serve_open', model_root=SERVED.root,
                                        owners=[o.public for o in net.owners], request_root='11' * 32,
                                        session_key=key.public_key().public_bytes_raw().hex(), price=NEURO, positions=8)
    assert net.chain.broadcast(envelope)['code'] == 0
    net.chain.mine()
    old_job = ledger.transaction_id(envelope)
    assert old_job in job or old_job in net.chain.s['jobs']
    mine, theirs = socket.socketpair()
    threading.Thread(target=net.owner(1).serve, args=(theirs, lambda *_: None, '10.0.0.2'), daemon=True).start()
    body = relay.hello(net.chain.chain_id, old_job, 1, SERVED.world, 4, 8)
    relay.send_frame(mine, relay.HELLO, ledger.canonical({'hello': body,
                                                          'signature': key.sign(ledger.canonical(body)).hex()}))
    with pytest.raises(ValueError, match="expected a b'R' serving frame, received b'P'"):
        relay.receive_frame(mine, relay.READY, relay.MAX_HELLO)  # exactly what an old client's dial does
    mine.close()
    finish(sockets)
