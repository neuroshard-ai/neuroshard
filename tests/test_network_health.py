from datetime import datetime, timezone
import base64
import json
import io
from types import SimpleNamespace

import pytest

from neuroshard.client import cli, health, wire
from neuroshard.inference.gateway import Gateway, handler


NOW = datetime(2026, 9, 27, 5, tzinfo=timezone.utc).timestamp()


def native():
    return {'node_info': {'network': 'test'}, 'sync_info': {
        'latest_block_height': '42', 'latest_block_time': '2026-09-27T04:59:59.123456789Z',
        'catching_up': False}}


def test_responsive_but_stalled_node_is_not_ready_and_future_time_is_rejected():
    summary = {'chain_id': 'test', 'height': 42}
    assert health.assess(native(), summary, now=NOW)['network_ready']
    stale = native()
    stale['sync_info']['latest_block_time'] = '2026-09-19T20:13:29Z'
    report = health.assess(stale, summary, now=NOW)
    assert report['stalled'] and not report['network_ready']
    with pytest.raises(ValueError, match='stalled ledger'):
        health.require_ready(report)
    future = native()
    future['sync_info']['latest_block_time'] = '2026-09-28T05:00:00Z'
    assert health.assess(future, summary, now=NOW)['network_status'] == 'future block time'
    with pytest.raises(ValueError, match='different chains'):
        health.assess(native(), {'chain_id': 'other', 'height': 42}, now=NOW)
    assert not health.assess(native(), {'chain_id': 'test', 'height': 5}, now=NOW)['network_ready']


def test_live_heartbeat_does_not_advertise_inference_on_a_stalled_ledger(tmp_path, monkeypatch):
    gateway = object.__new__(Gateway)
    gateway.home = tmp_path
    gateway.config = {'chain_id': 'test', 'genesis_sha256': 'a' * 64}
    (tmp_path / 'services.json').write_text(json.dumps({'provider': 'provider'}))
    (tmp_path / 'provider-status.json').write_text(json.dumps({
        'public_key': 'provider', 'chain_id': 'test', 'checked_at': NOW}))
    summary = {'params': {'fee': 1000, 'inference_token_price': 1000, 'inference_blocks': 240},
               'serving_root': 'b' * 64, 'pending_inference': 0, 'ready': False,
               'stalled': True, 'seconds_since_block': 600000}
    gateway.summary = lambda: summary
    monkeypatch.setattr('neuroshard.inference.gateway.time.time', lambda: NOW)
    value = gateway.inference_info()
    assert value['provider_connected'] and not value['provider_online'] and not value['network_ready']
    summary.update(ready=True, stalled=False, seconds_since_block=1)
    assert gateway.inference_info()['provider_online']


def test_stalled_chat_refuses_before_opening_a_key_or_signing(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, 'connected', lambda args: ({}, 'http://127.0.0.1', {
        'network_ready': False, 'network_status': 'stalled ledger'}))
    def forbidden(*args, **kwargs):
        raise AssertionError('must not open wallet or broadcast')
    monkeypatch.setattr(wire, 'Wallet', forbidden)
    monkeypatch.setattr(wire, 'broadcast', forbidden)
    with pytest.raises(ValueError, match='No new payment'):
        cli.chat(SimpleNamespace(home=tmp_path))


def test_stalled_gateway_blocks_relay_but_allows_transaction_status(monkeypatch, tmp_path):
    gateway = SimpleNamespace(rpc='http://127.0.0.1', limited=lambda ip: False,
                              summary=lambda: {'ready': False})
    server = object.__new__(handler(gateway))
    server.path = '/rpc'
    server.client_address = ('127.0.0.1', 1234)
    answers, forwarded = [], []
    server.respond = lambda code, value: answers.append((code, value))
    monkeypatch.setattr('neuroshard.inference.gateway.client.rpc',
                        lambda *args, **kw: forwarded.append(args) or {'checked': True})
    def request(method, params):
        raw = wire.canonical({'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params})
        server.headers = {'Content-Length': str(len(raw))}
        server.rfile = io.BytesIO(raw)
        server.do_POST()
    signed = wire.Wallet(tmp_path / 'account.key', create=True).sign({'kind': 'example', 'nonce': 0})
    request('broadcast_tx_sync', {'tx': base64.b64encode(wire.canonical(signed)).decode()})
    assert answers[-1][0] == 503 and not forwarded
    request('status', {})
    assert answers[-1][0] == 200 and len(forwarded) == 1
