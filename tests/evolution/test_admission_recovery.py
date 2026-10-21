"""DESIGN_ADMISSION.md acceptance tests 4–8 and 13: funding resolution, generations, crashes, abuse budget."""
import socket
import threading

import pytest

from admission_net import NEURO, SERVED, Net, jobs_of, start_work, wait_until
from neuroshard.assistant import admission, funding, admission_messages as messages
from neuroshard.inference import optimistic as ledger
from neuroshard.inference import relay
from test_admission import finish


def prepared(net, name, nonce=None):
    """A signed serve_open for ``name``, persisted as its turn's operation (not broadcast)."""
    account, operations = net.accounts[name], funding.Operations(net.tmp / name)
    nonce = net.chain.s['accounts'][account.public]['nonce'] if nonce is None else nonce
    envelope = account.sign(net.chain.chain_id, nonce, 'serve_open', model_root=SERVED.root,
                            owners=[net.owner(1).public, net.owner(2).public], request_root='22' * 32,
                            session_key='33' * 32, price=NEURO, positions=8)
    return operations, operations.prepare(envelope, '33' * 32, '22' * 32, [])


def index(net):
    return funding.LedgerIndex(net.chain, net.chain.call)


def test_4_a_lost_confirmation_then_a_restart_resolves_to_the_one_job(tmp_path):
    net = Net(tmp_path)
    operations, record = prepared(net, 'alice')
    record = operations.mark(record, funding.BROADCAST, since_height=net.chain.s['height'])
    index(net).rebroadcast(record['envelope'])
    net.chain.mine()  # committed; the client crashed before it saw the confirmation
    job_id, _, sockets, _ = net.turn('alice')  # a restarted client resolves the earlier turn first
    assert funding.Operations(tmp_path / 'alice').load(record['turn_id'])['state'] == funding.ABANDONED
    assert sorted(jobs_of(net, 'alice')) == sorted([record['job_id'], job_id]), 'the earlier turn was not re-funded'
    finish(sockets)


def test_4_a_commit_between_two_resolution_reads_is_funded(tmp_path):
    net = Net(tmp_path)
    operations, record = prepared(net, 'alice')
    record = operations.mark(record, funding.BROADCAST, since_height=net.chain.s['height'])
    record = funding.resolve(index(net), operations, record)  # not executed yet: the same envelope is (re)sent
    assert record['state'] == funding.BROADCAST
    net.chain.mine()  # it commits before the next read
    assert funding.resolve(index(net), operations, record)['state'] == funding.FUNDED


def test_4_delayed_indexing_is_funded_from_verified_state_or_stays_unknown_never_unfunded(tmp_path):
    net = Net(tmp_path)
    operations, record = prepared(net, 'alice')
    record = operations.mark(record, funding.BROADCAST, since_height=net.chain.s['height'])
    index(net).rebroadcast(record['envelope'])
    net.chain.mine()
    net.chain.unindexed.add(record['tx_hash'])
    assert funding.resolve(index(net), operations, record)['state'] == funding.FUNDED

    operations, record = prepared(net, 'bob')
    record = operations.mark(record, funding.BROADCAST, since_height=net.chain.s['height'])
    index(net).rebroadcast(record['envelope'])
    net.chain.mine()
    net.chain.unindexed.add(record['tx_hash'])
    blind = index(net)

    def unverified():
        raise OSError('no verified state: the light client cannot reach a validator')

    blind.verified_state = unverified
    record = funding.resolve(blind, operations, record)
    assert record['state'] == funding.UNKNOWN
    with pytest.raises(funding.Unresolved):
        funding.fund(blind, operations, record, timeout=5, sleep=net.sleep, clock=net.clock)
    with pytest.raises(funding.Unresolved, match='unresolved'):
        prepared(net, 'bob')  # the account opens nothing new while a turn's outcome is unknown


def test_4_a_nonce_consumed_by_another_transaction_is_unfunded(tmp_path):
    net = Net(tmp_path)
    operations, record = prepared(net, 'alice')
    record = operations.mark(record, funding.BROADCAST, since_height=net.chain.s['height'])
    other = net.accounts['alice'].sign(net.chain.chain_id, record['nonce'], 'transfer',
                                       to=net.accounts['bob'].public, amount=1)
    assert net.chain.broadcast(other)['code'] == 0
    net.chain.mine()
    record = funding.resolve(index(net), operations, record)
    assert record['state'] == funding.UNFUNDED and jobs_of(net, 'alice') == []


def test_5_a_lost_ready_is_served_at_the_next_generation_and_the_old_link_is_dead(tmp_path):
    net = Net(tmp_path)
    job_id, owners, sockets, session_key = net.turn('alice')
    again = net.rehello(job_id, 1, session_key)  # the client never saw READY: a fresh attempt HELLOs again
    assert net.owner(1).gate.gens[job_id] == 2
    stale = sockets[1]
    stale.settimeout(5)
    assert stale.recv(1) == b'', 'the superseded generation was closed by the owner'
    try:
        relay.send_frame(stale, relay.CONTROL, b'\0' * 16)  # work on the gen-1 link is discarded
    except OSError:
        pass
    relay.send_frame(again, relay.CONTROL, b'\0' * 16)
    relay.send_frame(sockets[2], relay.CONTROL, b'\0' * 16)
    wait_until(lambda: net.owner(1).worked == [job_id] and net.owner(2).worked == [job_id])
    assert net.owner(1).gate.lease['gen'] == 2
    finish({1: again, 2: sockets[2]})


def test_6_a_crash_before_the_write_ahead_record_re_admits_and_after_it_never(tmp_path):
    from admission_net import Owner

    net = Net(tmp_path)
    job_id, owners, sockets, session_key = net.turn('alice')
    restarted = Owner(net.chain, 1, net.owner(1).home, net.clock)  # crash before the first work frame
    assert job_id not in restarted.gate.started and restarted.gate.gens[job_id] == 1
    net.endpoints[owners[0][1]] = restarted
    finish(sockets)
    link = net.rehello(job_id, 1, session_key, owner=restarted)
    assert restarted.gate.gens[job_id] == 2
    relay.send_frame(link, relay.CONTROL, b'\0' * 16)
    wait_until(lambda: restarted.worked == [job_id])
    finish({1: link})

    recovered = Owner(net.chain, 1, net.owner(1).home, net.clock)  # crash after the write-ahead record
    assert job_id in recovered.gate.started and job_id in recovered.served_jobs
    net.endpoints[owners[0][1]] = recovered
    with pytest.raises((relay.OwnerBusy, ValueError, ConnectionError)):
        net.rehello(job_id, 1, session_key, owner=recovered)


def test_7_two_concurrent_turns_on_one_account_are_serialized(tmp_path):
    net = Net(tmp_path, owners_per_shard=2)
    outcomes, errors = [], []

    def turn():
        try:
            outcomes.append(net.turn('alice'))
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=turn) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert not errors, errors
    records = funding.Operations(tmp_path / 'alice').all()
    assert sorted(r['nonce'] for r in records if r['state'] == funding.SERVED) == [0, 1]
    assert len(jobs_of(net, 'alice')) == 2
    for _, _, sockets, _ in outcomes:
        finish(sockets)


def test_8_rotating_accounts_from_one_address_and_unbroadcast_holds_stay_charged(tmp_path):
    names = [f'm{i}' for i in range(6)]
    net = Net(tmp_path, users=('alice', *names))
    gate, admitted, refused, busy = net.owner(1).gate, 0, 0, 0
    rounds = 24  # 10 minutes: one budget window
    for round_ in range(rounds):
        name = names[round_ % len(names)]
        try:
            offer = gate.reserve_new(net.accounts[name].public, '10.6.6.6', '33' * 32, NEURO)
            _, record = prepared(net, name)
            gate.hold(net.accounts[name].public, offer['reservation'], messages.unsigned(record['envelope']))
            funding.Operations(tmp_path / name).mark(record, funding.ABANDONED)
        except admission.Refused as error:
            assert error.reason == 'limit'
            refused += 1
        except admission.Busy:
            pass
        net.sleep(25)
        try:
            honest = gate.reserve_new(net.accounts['alice'].public, '10.1.1.1', 'ef' * 32, NEURO)
            gate.release(honest['reservation'], net.accounts['alice'].public)
            admitted += 1
        except admission.Busy:
            busy += 1
    # Measured: the adversary's address holds the slot for at most its 60 s budget per 10 minutes.
    assert admitted >= rounds - 3 and busy <= 3, (admitted, busy)
    assert refused >= rounds - 4, 'the rotating adversary is bounded by its address budget'
    assert gate.budget.used(('address', '10.6.6.6'), net.clock()) <= admission.BUDGET_SECONDS + 1
    assert not gate.credited, 'holds for never-broadcast transactions are never credited'


def test_13_one_funded_job_cannot_monopolize_its_owners_and_is_credited_once(tmp_path):
    net = Net(tmp_path)
    job_id, owners, sockets, session_key = net.turn('alice')
    gate, key = net.owner(1).gate, ('account', net.accounts['alice'].public)
    assert gate.credited == {job_id}
    finish(sockets)
    wait_until(lambda: gate.lease is None)
    idle, refusal = 0.0, None
    for _ in range(40):
        message = messages.by_key(session_key, 'reserve', net.chain.chain_id, net.clock, rank=1, owner=owners[0][0],
                                  job_id=job_id)
        reply = net.request('10.0.0.1')(owners[0][1], message)['body']
        if reply['kind'] != 'offer':
            refusal = reply
            break
        link = net.rehello(job_id, 1, session_key)
        net.clock.now += 5
        idle += 5
        gate.release(reply['reservation'], job_id=job_id)  # release and re-reserve: the clock does not reset
        link.close()
        wait_until(lambda: gate.lease is None)
    assert refusal is not None and refusal['kind'] == 'refused' and refusal['reason'] == 'limit'
    assert idle <= admission.BUDGET_SECONDS + 5
    assert gate.credited == {job_id} and gate.budget.used(key, net.clock()) >= admission.BUDGET_SECONDS - 5
    other = gate.reserve_new(net.accounts['bob'].public, '10.2.2.2', 'ef' * 32, NEURO)
    assert other['reservation'], 'another client is admitted'
