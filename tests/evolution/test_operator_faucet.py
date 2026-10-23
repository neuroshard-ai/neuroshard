"""Faucets: any operator can run one from its own account, and clients fall through a list until one grants."""
import json
import threading
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from admission_net import NEURO, Net
from neuroshard.assistant import network
from neuroshard.client import assistant_cli


class Mining:
    """The in-memory ledger with ``Chain``'s submit and balance: each submission is mined at once."""

    def __init__(self, chain):
        self.chain, self.chain_id = chain, chain.chain_id

    def state(self):
        return self.chain.state()

    def balance(self, account):
        return self.chain.state()['accounts'].get(account.public, {'balance': 0})['balance']

    def submit(self, account, kind, **fields):
        envelope = account.sign(self.chain_id, self.chain.nonce(account), kind, **fields)
        admitted = self.chain.broadcast(envelope)
        if admitted['code']:
            raise ValueError(admitted['log'])
        self.chain.mine()
        return envelope


def test_the_faucet_list_is_the_descriptor_then_its_list_then_the_environment(monkeypatch):
    monkeypatch.setenv('NEUROSHARD_FAUCETS', 'http://c/faucet, http://a/faucet,')
    assert network.faucets({'faucet': 'http://a/faucet', 'faucets': ['http://b/faucet']}) == \
        ['http://a/faucet', 'http://b/faucet', 'http://c/faucet']
    monkeypatch.delenv('NEUROSHARD_FAUCETS')
    assert network.faucets({}) == []


def test_a_down_faucet_falls_through_to_the_next(tmp_path, monkeypatch):
    net = Net(tmp_path)
    chain = Mining(net.chain)
    user = network.Account(tmp_path / 'user' / 'account.key')
    donor = net.accounts['alice']
    asked = []

    def fund(url, account, timeout=180):
        asked.append(url)
        if url == 'http://down/faucet':
            raise OSError('connection refused')
        chain.submit(donor, 'transfer', to=account.public, amount=5 * NEURO)
        return {'granted': 5 * NEURO}

    monkeypatch.setattr(network, 'fund', fund)
    network.ensure_balance(chain, user, 3 * NEURO, ['http://down/faucet', 'http://up/faucet'], lambda *_: None)
    assert asked == ['http://down/faucet', 'http://up/faucet'] and chain.balance(user) == 5 * NEURO


def test_every_faucet_failing_reports_each_reason(tmp_path, monkeypatch):
    net = Net(tmp_path)
    user = network.Account(tmp_path / 'user' / 'account.key')
    monkeypatch.setattr(network, 'fund', lambda url, account, timeout=180: (_ for _ in ()).throw(
        ValueError('this account or address was funded recently')))
    with pytest.raises(ValueError, match='no faucet funded this account: .*funded recently'):
        network.ensure_balance(Mining(net.chain), user, NEURO, ['http://a/faucet', 'http://b/faucet'],
                               lambda *_: None)
    with pytest.raises(ValueError, match='names no faucet'):
        network.ensure_balance(Mining(net.chain), user, NEURO, [], lambda *_: None)


def test_an_operator_faucet_grants_from_its_own_account_over_http(tmp_path, monkeypatch):
    net = Net(tmp_path)
    chain = Mining(net.chain)
    home = tmp_path / 'operator'
    operator = network.Account(home / 'account.key')
    chain.submit(net.accounts['alice'], 'transfer', to=operator.public, amount=15 * NEURO)
    servers = []
    original = network.serve_faucet
    monkeypatch.setattr(network, 'serve_faucet', lambda *a, **k: servers.append(original(*a, **k)) or servers[-1])
    monkeypatch.setattr(assistant_cli.signal, 'pause', lambda: started.wait(30))
    started, done = threading.Event(), threading.Event()
    args = SimpleNamespace(home=home, listen='127.0.0.1', port=0, amount='2', floor='1', per_address=5)
    worker = threading.Thread(target=lambda: (assistant_cli.run_faucet(args, chain, network), done.set()), daemon=True)
    worker.start()
    for _ in range(100):
        if servers:
            break
        threading.Event().wait(0.05)
    port = servers[0].server_address[1]
    newcomer = network.Account(tmp_path / 'newcomer' / 'account.key')
    request = urllib.request.Request(f'http://127.0.0.1:{port}/faucet', data=json.dumps({'account': newcomer.public}).encode(),
                                     headers={'Content-Type': 'application/json'})
    reply = json.loads(urllib.request.urlopen(request, timeout=10).read())
    assert reply['granted'] == 2 * NEURO and chain.balance(newcomer) == 2 * NEURO
    started.set()
    worker.join(10)
    assert done.is_set()


def test_a_faucet_without_enough_to_grant_refuses_to_start(tmp_path):
    net = Net(tmp_path)
    args = SimpleNamespace(home=tmp_path / 'empty', listen='127.0.0.1', port=0, amount='2', floor='1', per_address=5)
    with pytest.raises(ValueError, match='needs at least that plus a fee'):
        assistant_cli.run_faucet(args, Mining(net.chain), network)
