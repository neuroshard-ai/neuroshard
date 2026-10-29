"""Proof availability: every voting role follows proofs, and one bad peer never blocks a good one.

Started from the 2026-10-29 review's probes (C1 seed never followed proofs, C2 a malformed peer blocked healthy
ones, C3 downloads were unbounded until complete); extended with truncated, slow and over-declared peers and with a
non-seed sole holder while another validator is down."""
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
import threading
import zipfile

import pytest

from neuroshard.assistant import audit, network
from neuroshard.client import assistant_cli, live_runtime
from neuroshard.evolution.sharded import granite_audit
from neuroshard.inference import optimistic_network


@pytest.fixture
def bundle(tmp_path):
    # These bytes test storage/content-addressing, not a judgment of the proof.
    files = {name: b'{}' for name in audit.PROOF_FILES}
    root = granite_audit.bundle_digest(files)
    state = {'jobs': {'job': {'challenges': {'challenge': {'proof_root': root}}}}}
    chain = SimpleNamespace(state=lambda: state)
    holder = tmp_path / 'holder'
    (holder / root).mkdir(parents=True)
    for name, content in files.items():
        (holder / root / name).write_bytes(content)
    server = network.serve_store(chain, '127.0.0.1', 0, holder)
    yield chain, root, f'http://127.0.0.1:{server.server_port}/proofs'
    server.shutdown()
    server.server_close()


def test_healthy_peer_control(bundle, tmp_path):
    chain, root, good = bundle
    assert network.replicate(chain, tmp_path / 'destination', [good], proofs_only=True) == 1
    assert granite_audit.read_bundle(tmp_path / 'destination', root)


def test_malformed_first_peer_does_not_block_healthy_second_peer(bundle, tmp_path):
    chain, root, good = bundle
    requested = []

    class BrokenPeer(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            requested.append(self.path)
            body = b'not a zip file'
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(('127.0.0.1', 0), BrokenPeer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        bad = f'http://127.0.0.1:{server.server_port}/proofs'
        assert network.replicate(chain, tmp_path / 'destination', [bad, good], proofs_only=True) == 1
        assert requested
        assert granite_audit.read_bundle(tmp_path / 'destination', root)
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize('role', ['validator', 'seed'])
def test_both_validator_launch_paths_start_the_proof_follower(role, monkeypatch, tmp_path):
    descriptor = {'chain_id': 'review', 'proof_endpoints': ['http://peer:28780/proofs']}
    chain = SimpleNamespace()
    config = {'validators': []}
    started = []
    monkeypatch.setattr(live_runtime, 'configure', lambda: None)
    monkeypatch.setattr(live_runtime, 'check', lambda: None)
    monkeypatch.setattr(network, 'descriptor', lambda _: descriptor)
    monkeypatch.setattr(network, 'seed', lambda *a, **k: (config, chain, None))
    monkeypatch.setattr(network, 'chain', lambda *a: chain)
    monkeypatch.setattr(assistant_cli, 'following', lambda d, *a: d)
    monkeypatch.setattr(network, 'join_validator', lambda *a, **k: config)
    monkeypatch.setattr(network, 'serve_store', lambda *a: None)
    monkeypatch.setattr(network, 'serve_artifacts', lambda *a: (None, []))
    monkeypatch.setattr(network, 'follow_validator', lambda *a: descriptor)
    monkeypatch.setattr(network, 'replicate', lambda *a: None)
    monkeypatch.setattr(network, 'provision_units', lambda *a: None)
    monkeypatch.setattr(network, 'fell_behind', lambda *a: False)
    monkeypatch.setattr(optimistic_network, 'stop', lambda *a: None)
    monkeypatch.setattr(assistant_cli.signal, 'signal', lambda *a: None)

    class StopAfterOnePass:
        calls = 0

        def wait(self, _):
            self.calls += 1
            return self.calls > 1

    class RecordThread:
        def __init__(self, target, **kwargs):
            self.target = target

        def start(self):
            started.append(self.target)

    monkeypatch.setattr(assistant_cli.threading, 'Event', StopAfterOnePass)
    monkeypatch.setattr(assistant_cli.threading, 'Thread', RecordThread)
    args = SimpleNamespace(assistant_action=role, network=None, home=tmp_path, public_host='self',
                           audit_shards=(1, 2), store_port=0, mirror_port=0)
    assistant_cli.run(args)
    assert network.follow_proofs in started


def test_proof_download_is_bounded_before_acceptance(bundle, monkeypatch, tmp_path):
    chain, _, good = bundle
    # Scale the existing transfer bound to 1 KiB; serve 10 KiB before a healthy peer.
    monkeypatch.setattr(audit, 'MAX_PROOF_BYTES', 1024)
    reads = []
    original = network.urllib.request.urlopen
    body = b'x' * 10240

    class OversizedResponse(io.BytesIO):
        def read(self, size=-1):
            data = super().read(size)
            reads.append(len(data))
            return data

    def opened(url, **kwargs):
        if str(url).startswith('http://oversized/'):
            return OversizedResponse(body)
        return original(url, **kwargs)

    monkeypatch.setattr(network.urllib.request, 'urlopen', opened)
    assert network.replicate(chain, tmp_path / 'destination', ['http://oversized/proofs', good],
                             proofs_only=True) == 1
    assert sum(reads) <= audit.MAX_PROOF_BYTES + 1


def serve_bytes(handler_body):
    """A local HTTP peer whose GET handler is ``handler_body(handler)``."""
    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            handler_body(self)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Peer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f'http://127.0.0.1:{server.server_port}/proofs'


def test_a_truncated_response_is_skipped_for_a_healthy_peer(bundle, tmp_path):
    chain, root, good = bundle

    def truncated(handler):
        handler.send_response(200)
        handler.send_header('Content-Length', '100000')  # promises far more than it sends
        handler.end_headers()
        handler.wfile.write(b'PK\x03\x04 partial')
        handler.close_connection = True

    server, bad = serve_bytes(truncated)
    try:
        assert network.replicate(chain, tmp_path / 'destination', [bad, good], proofs_only=True) == 1
        assert granite_audit.read_bundle(tmp_path / 'destination', root)
    finally:
        server.shutdown()
        server.server_close()


def test_an_over_declared_response_is_refused_before_its_body_is_read(bundle, monkeypatch, tmp_path):
    chain, root, good = bundle
    monkeypatch.setattr(audit, 'MAX_PROOF_BYTES', 1024)
    sent = []

    def huge(handler):
        handler.send_response(200)
        handler.send_header('Content-Length', str(10 << 20))
        handler.end_headers()
        try:
            for _ in range(10):
                handler.wfile.write(b'x' * (1 << 20))
                sent.append(1)
        except OSError:
            pass

    server, bad = serve_bytes(huge)
    try:
        assert network.replicate(chain, tmp_path / 'destination', [bad, good], proofs_only=True) == 1
        assert granite_audit.read_bundle(tmp_path / 'destination', root)
    finally:
        server.shutdown()
        server.server_close()


def test_a_slow_peer_is_cut_off_at_the_transfer_deadline_and_a_healthy_peer_used(bundle, monkeypatch, tmp_path):
    import time
    chain, root, good = bundle
    original = network.bounded_download
    monkeypatch.setattr(network, 'bounded_download',
                        lambda url, limit, timeout=30, deadline=60: original(url, limit, timeout=5, deadline=1.5))

    def trickle(handler):  # one byte every 0.2 s keeps every socket read alive, forever
        handler.send_response(200)
        handler.end_headers()
        try:
            for _ in range(200):
                handler.wfile.write(b'x')
                handler.wfile.flush()
                time.sleep(0.2)
        except OSError:
            pass

    server, slow = serve_bytes(trickle)
    try:
        started = time.monotonic()
        assert network.replicate(chain, tmp_path / 'destination', [slow, good], proofs_only=True) == 1
        assert time.monotonic() - started < 10
        assert granite_audit.read_bundle(tmp_path / 'destination', root)
    finally:
        server.shutdown()
        server.server_close()


def test_a_non_seed_sole_holder_reaches_every_voter_with_another_validator_down(bundle, tmp_path):
    """Four voters: the bundle is held only by validator v4's store and validator v2 is down. The seed and the
    other validator must each obtain it from v4, past the dead one, so three of four can judge (> 2/3)."""
    chain, root, v4 = bundle
    with socket_closed_port() as dead:
        for voter in ('seed', 'v5'):
            peers = [dead, v4]  # the dead validator first: it must not block the holder
            assert network.replicate(chain, tmp_path / voter, peers, proofs_only=True) == 1
            assert granite_audit.read_bundle(tmp_path / voter, root)


class socket_closed_port:
    """A local URL nothing listens on (a validator that is down)."""
    def __enter__(self):
        import socket
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        return f'http://127.0.0.1:{port}/proofs'

    def __exit__(self, *_):
        return False
