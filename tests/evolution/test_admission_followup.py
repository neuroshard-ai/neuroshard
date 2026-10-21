"""Regressions for the follow-up review of e032d25: the durable HOLD barrier across restarts, an atomic replay
cache under concurrency, and a client-visible ``lapsed`` outcome."""
import sys
import threading

import pytest

from admission_net import Net, jobs_of, start_work, wait_until
from neuroshard.assistant import admission, admission_client, admission_messages as messages, funding
from neuroshard.inference import optimistic as ledger
from neuroshard.inference import relay
from test_admission import finish
from test_admission_review import captured_hello, replay


class ProcessCrash(BaseException):
    pass


def crashing_turn(net, monkeypatch, crash_after):
    """Alice's turn, killed right after the durable record reaches ``crash_after``."""
    client = net.admitter('alice')
    mark, save = client.operations.mark, client.operations.save

    def crashing(record, state, **fields):
        saved = mark(record, state, **fields)
        if state == crash_after and (state != funding.PREPARED or 'since_height' in fields):
            raise ProcessCrash()
        return saved

    with monkeypatch.context() as patch:
        patch.setattr(client.operations, 'mark', crashing)
        with pytest.raises(ProcessCrash):
            net.turn('alice', admitter=client)
    assert save  # the real save ran for every phase before the crash
    record, = funding.Operations(net.tmp / 'alice').all()
    assert record['state'] == crash_after
    return record


def occupy(net):
    net.sleep(65)  # alice's offers and holds (at most her 60 s budget) lapse while her client is down
    job, _, sockets, _ = net.turn('bob', address='10.0.0.2')
    start_work(sockets)
    wait_until(lambda: all(job in o.worked for o in net.owners))
    return sockets


@pytest.mark.parametrize('crash_after', [funding.PREPARED, funding.HELD], ids=['before-hold', 'after-hold'])
def test_a_restart_before_broadcasting_never_funds_the_turn(tmp_path, monkeypatch, crash_after):
    """Both sides of the durable HOLD phase: neither crash point lets recovery broadcast an unexposed envelope."""
    net = Net(tmp_path)
    record = crashing_turn(net, monkeypatch, crash_after)
    sockets = occupy(net)
    try:
        net.admitter('alice').recover()
        assert jobs_of(net, 'alice') == []
        after = funding.Operations(tmp_path / 'alice').load(record['turn_id'])
        assert after['state'] == funding.ABANDONED and not after.get('exposed'), after
    finally:
        finish(sockets)
    net.sleep(admission.BUDGET_WINDOW)  # alice's unfunded holds stayed charged; her budget recovers
    job_id, _, sockets, _ = net.turn('alice')  # the account is free to start a new turn
    assert jobs_of(net, 'alice') == [job_id]
    finish(sockets)


def test_a_restart_after_the_broadcast_phase_resolves_the_same_job(tmp_path, monkeypatch):
    """Once exposed (``broadcast``), the operation is resolved under its own identity, never abandoned blindly."""
    net = Net(tmp_path)
    record = crashing_turn(net, monkeypatch, funding.BROADCAST)
    funding.LedgerIndex(net.chain, net.chain.call).rebroadcast(record['envelope'])  # it did go out
    net.chain.mine()
    progress = []
    restarted = net.admitter('alice')
    restarted.progress = progress.append
    restarted.recover()
    assert jobs_of(net, 'alice') == [record['job_id']]
    assert any('funded but not served' in line for line in progress), progress


def test_a_broadcast_needs_the_held_phase(tmp_path):
    net = Net(tmp_path)
    operations = funding.Operations(tmp_path / 'alice')
    envelope = net.accounts['alice'].sign(net.chain.chain_id, 0, 'transfer', to=net.accounts['bob'].public, amount=1)
    record = operations.prepare(envelope, '33' * 32, '22' * 32, [])
    with pytest.raises(ValueError, match='only a held operation'):
        funding.fund(funding.LedgerIndex(net.chain, net.chain.call), operations, record, broadcast=True)


def test_concurrent_replay_checks_never_forget_an_accepted_hello(tmp_path, monkeypatch):
    """The follow-up's interleaving: a check paused mid-prune must not erase a HELLO accepted meanwhile."""
    net = Net(tmp_path)
    owner = net.owner(1)
    now = int(net.clock())
    owner.replays.check({'challenge': 'a' * 32, 'expires': now - 1}, now - 2)  # something to prune
    pausing, resume, errors = threading.Event(), threading.Event(), []

    def trace(frame, event, arg):
        if (frame.f_code.co_filename.endswith('admission_messages.py') and frame.f_code.co_name == 'check'
                and event == 'line' and owner.replays.lock.locked() and not pausing.is_set()):
            pausing.set()
            resume.wait(10)
        return trace

    def concurrent():
        sys.settrace(trace)
        try:
            owner.replays.check({'challenge': 'b' * 32, 'expires': now + 60}, now)
        except Exception as error:  # noqa: BLE001
            errors.append(error)
        finally:
            sys.settrace(None)

    thread = threading.Thread(target=concurrent, daemon=True)
    thread.start()
    assert pausing.wait(5)
    accepted = threading.Thread(target=lambda: captured.append(captured_hello(net, monkeypatch)), daemon=True)
    captured = []
    accepted.start()
    accepted.join(0.5)  # with the lock held by the paused check, the HELLO waits; it must not slip past
    resume.set()
    thread.join(5)
    accepted.join(10)
    assert not errors and captured, errors
    (job_id, _, sockets, _), payload = captured[0]
    try:
        before = owner.gate.gens[job_id]
        kind, _ = replay(owner, payload)
        assert kind != relay.READY and owner.gate.gens[job_id] == before
    finally:
        finish(sockets)


def test_many_threads_never_accept_one_challenge_twice():
    replays, now = messages.Replays(), 1_000
    barrier, accepted = threading.Barrier(16), []

    def race(index):
        body = {'challenge': 'f' * 31 + str(index % 4), 'expires': now + 60}
        barrier.wait()
        for step in range(200):
            try:
                replays.check({'challenge': f'{index:08x}{step:024x}', 'expires': now + 60}, now)
                replays.check(body, now)
                accepted.append(body['challenge'])
            except ValueError:
                pass

    threads = [threading.Thread(target=race, args=(i,)) for i in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(accepted) == sorted('f' * 31 + str(i) for i in range(4))


def test_a_lapsed_outcome_is_reported_without_claiming_it_was_unfunded(tmp_path, monkeypatch):
    net = Net(tmp_path)
    admitter, real_fund = net.admitter('alice'), funding.fund

    def lapsed(chain, operations, record, **kwargs):
        funded = real_fund(chain, operations, record, **kwargs)
        expiry = net.chain.s['jobs'][record['job_id']]['expires']
        while net.chain.s['height'] <= expiry:
            net.chain.mine()
        net.chain.mine()
        for i in range(net.chain.s['params']['max_results']):
            ledger.remember(net.chain.s, f'{i:064x}', {'status': 'expired'})
        result = funding.resolve(chain, operations, operations.mark(funded, funding.UNKNOWN))
        assert result['state'] == funding.LAPSED
        return result

    monkeypatch.setattr(funding, 'fund', lapsed)
    with pytest.raises(admission_client.FundingLapsed) as caught:
        net.turn('alice', admitter=admitter)
    message = str(caught.value)
    assert 'not funded' not in message and 'cannot show whether it was ever funded' in message
    assert 'settled or refunded' in message
