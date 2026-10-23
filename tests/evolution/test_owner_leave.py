"""``neuroshard assistant leave``: an owner unbonds, waits out the challenge window, and withdraws its bond, on the
real ledger transition; refused while a job names it, before the window, or out of order."""
import io
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest

from admission_net import NEURO, Net, start_work
from neuroshard.assistant import network
from neuroshard.client import assistant_cli
from neuroshard.inference import optimistic as ledger
from test_admission import finish


class Submitting:
    """``network.Chain``'s submit over the in-memory ledger: broadcast, then one block."""

    def __init__(self, chain):
        self.chain = chain

    def state(self):
        return self.chain.state()

    def submit(self, account, kind, **fields):
        envelope = account.sign(self.chain.chain_id, self.chain.nonce(account), kind, **fields)
        admitted = self.chain.broadcast(envelope)
        if admitted['code']:
            raise ValueError(admitted['log'])
        self.chain.mine()
        result = self.chain.receipts[admitted['hash']][1]
        if result['code']:
            raise ValueError(result['log'])
        return envelope


def bonded_home(net, tmp_path, shard=1):
    """A real owner home: its own account and log key, bonded on the in-memory ledger like ``Owner.register``."""
    home = tmp_path / 'owner-home'
    account = network.Account(home / 'account.key')
    key = network.signing_key(home / f'log-{shard}.key')
    public = network.public_hex(key)
    net.chain.s['accounts'][account.public] = {'balance': 20 * NEURO, 'nonce': 0}
    net.chain.s['initial_supply'] += 20 * NEURO
    amount = net.chain.s['params']['owner_bond_minimum']
    possession = key.sign(ledger.possession_message(net.chain.chain_id, account.public, public, shard, amount, 0)).hex()
    Submitting(net.chain).submit(account, 'owner_bond', model_root=net.chain.s['model_root'], shard=shard,
                                 log_key=public, amount=amount, possession=possession)
    return home, account, public


def leave(net, home, action, shard=1):
    out = io.StringIO()
    with redirect_stdout(out):
        assistant_cli.leave(SimpleNamespace(home=home, shard=shard, leave_action=action), Submitting(net.chain), network)
    return out.getvalue()


def test_unbond_wait_withdraw_returns_the_bond(tmp_path):
    net = Net(tmp_path)
    home, account, public = bonded_home(net, tmp_path)
    before = net.chain.s['accounts'][account.public]['balance']
    bond = net.chain.s['owners'][public]['amount']
    assert 'active' in leave(net, home, 'status')
    assert 'Unbonding' in leave(net, home, 'unbond')
    assert net.chain.s['owners'][public]['status'] == 'leaving'
    with pytest.raises(ValueError, match='can still be challenged'):
        leave(net, home, 'withdraw')
    while net.chain.s['height'] < net.chain.s['owners'][public]['release_height']:
        net.chain.mine()
    assert 'Withdrawn' in leave(net, home, 'withdraw')
    owner = net.chain.s['owners'][public]
    fees = 2 * net.chain.s['params']['fee']
    assert owner['status'] == 'withdrawn' and owner['amount'] == 0
    assert net.chain.s['accounts'][account.public]['balance'] == before + bond - fees
    ledger.invariant(net.chain.s)


def test_withdraw_before_unbond_and_double_unbond_are_refused(tmp_path):
    net = Net(tmp_path)
    home, _, _ = bonded_home(net, tmp_path)
    with pytest.raises(ValueError, match='withdraw follows unbond'):
        leave(net, home, 'withdraw')
    leave(net, home, 'unbond')
    with pytest.raises(ValueError, match='only an active owner can unbond'):
        leave(net, home, 'unbond')


def test_an_owner_named_in_an_open_job_cannot_leave(tmp_path):
    net = Net(tmp_path)
    job_id, owners, sockets, _ = net.turn('alice')
    start_work(sockets)
    owner1 = net.owner(1)
    home = tmp_path / 'served-home'
    home.mkdir()
    (home / 'log-1.key').write_text((owner1.home / 'log.key').read_text())
    funder = network.Account(home / 'account.key')
    net.chain.s['owners'][owner1.public]['account'] = funder.public  # this home's account bonded it (test setup)
    try:
        assert 'unsettled job' in leave(net, home, 'status')
        with pytest.raises(ValueError, match='jobs naming this owner to settle'):
            leave(net, home, 'unbond')
    finally:
        finish(sockets)


def test_a_home_without_the_shards_key_or_bond_is_refused(tmp_path):
    net = Net(tmp_path)
    home, _, _ = bonded_home(net, tmp_path)
    with pytest.raises(ValueError, match='no owner log key for shard 2'):
        leave(net, home, 'status', shard=2)
    stranger = tmp_path / 'stranger'
    network.signing_key(stranger / 'log-1.key')
    with pytest.raises(ValueError, match='is not bonded'):
        leave(net, stranger, 'status')


def test_a_leaving_owner_is_not_selected_for_new_jobs(tmp_path):
    net = Net(tmp_path)
    home, _, public = bonded_home(net, tmp_path)
    leave(net, home, 'unbond')
    admitter = net.admitter('alice')
    assert public not in [key for key, _ in admitter.candidates(net.chain.state(), 1)]


@pytest.mark.parametrize('action', ['leave', 'faucet'])
def test_the_cli_dispatches_leave_and_faucet_and_stops_there(monkeypatch, action):
    """A dispatch that fell through into the chat path after leave/faucet would download a stage and fund a job.
    The chat path begins by reading ``args.budget``: that read must never happen."""
    calls = []

    class Args(SimpleNamespace):
        def __getattr__(self, name):  # only for attributes not set: the first one past dispatch
            calls.append(f'read {name} past dispatch')
            raise AttributeError(name)

    monkeypatch.setattr(assistant_cli, 'leave', lambda *a: calls.append('leave'))
    monkeypatch.setattr(assistant_cli, 'run_faucet', lambda *a: calls.append('faucet'))
    monkeypatch.setattr(network, 'chain', lambda *a, **k: SimpleNamespace(state=lambda: {}))
    monkeypatch.setattr(assistant_cli, 'following', lambda descriptor, *a, **k: descriptor)
    args = Args(assistant_action=action, network=None, home=Path('/tmp/unused'), shard=1)
    try:
        assistant_cli.run(args)
    except AttributeError:
        pass
    assert calls == [action], calls
