"""Starter credits are signed, bounded native transfers, including lost receipts."""
from pathlib import Path
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import operated_alpha_credits as credits
from neuroshard.client import provider_wire, wire
from neuroshard.demo import protocol
from neuroshard.evolution.provider_transport import Server, certificate
from neuroshard.evolution.transactions import Outbox
from test_expert_lifecycle import network, graphs
from test_hosted_customer import Chain


def test_credit_service_recovers_one_transfer_and_enforces_global_limit(tmp_path, network, monkeypatch):
    state, owners = network
    native = Chain(state)
    def box(path, url, chain, owner):
        result = Outbox(path, url, chain, owner, rpc=native.rpc,
                       query=lambda _url, path, data: native.query(path, data))
        original = result.send
        result.send = lambda *a, **kw: original(*a, **{**kw, 'timeout': .01 if native.drop else 3})
        return result
    monkeypatch.setattr(credits, 'Outbox', box)
    tls, fingerprint = certificate(tmp_path, owners[0])
    server = Server(('127.0.0.1', 0), tls, None, max_connections=2)
    server.RequestHandlerClass = credits.Handler
    server.owner, server.control = owners[0], threading.Lock()
    server.config = {'home': str(tmp_path), 'node_rpc': 'local', 'chain_id': native.chain_id,
                     'grant_atoms': 1000, 'ceiling_atoms': 2000}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def request(key, *, chain=None, pin=fingerprint):
        connection = provider_wire.PinnedConnection('https://127.0.0.1:'+str(server.server_port),
                                                   pin, timeout=3, allow_private=True)
        try:
            raw = wire.canonical({'public_key': key, 'chain_id': chain or native.chain_id})
            connection.request('POST', '/credits', raw, {'Content-Type': 'application/json'})
            response = connection.getresponse()
            body, signer = wire.verify(wire.parse(response.read()))
            assert signer == owners[0].public_key
            return response.status, body
        finally:
            connection.close()
    try:
        recipient = protocol.Identity('alpha-first-recipient').public_key
        with pytest.raises(ValueError, match='certificate differs'):
            request(recipient, pin='a'*64)
        assert request(recipient, chain='wrong-chain')[0] == 400
        assert not native.submissions
        native.drop = 'transfer'
        assert request(recipient)[0] == 202
        native.drop = None
        for _ in range(3):
            status, value = request(recipient)
            assert status == 200 and value['amount_atoms'] == 1000
        assert len(native.submissions) == 1
        assert native.state['accounts'][recipient]['balance'] == 1000
        assert request(protocol.Identity('alpha-second-recipient').public_key)[0] == 200
        assert request(protocol.Identity('alpha-third-recipient').public_key)[0] == 409
        assert len(native.submissions) == 2 and native.state['issued'] == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
