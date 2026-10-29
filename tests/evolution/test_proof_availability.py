"""Proof availability: every voting role follows proofs, and one bad peer never blocks a good one.

Started from the 2026-10-29 review's probes (C1 seed never followed proofs, C2 a malformed peer blocked healthy
ones, C3 downloads were unbounded until complete); extended with truncated, slow and over-declared peers and with a
non-seed sole holder while another validator is down."""
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
import threading
import time
import zipfile

import pytest

from neuroshard.assistant import audit, network
from neuroshard.client import assistant_cli
try:  # the live runtime check exists from runtime 2 on; older trees have none to stub
    from neuroshard.client import live_runtime
except ImportError:
    live_runtime = None
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


class FirstPeerFirst:
    """No rotation, so a test can put the bad peer first (production rotates peer order each pass)."""
    def randrange(self, _):
        return 0


def test_healthy_peer_control(bundle, tmp_path):
    chain, root, good = bundle
    assert network.replicate(chain, tmp_path / 'destination', [good], proofs_only=True, rng=FirstPeerFirst()) == 1
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
        assert network.replicate(chain, tmp_path / 'destination', [bad, good], proofs_only=True, rng=FirstPeerFirst()) == 1
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
    if live_runtime is not None:
        monkeypatch.setattr(live_runtime, 'configure', lambda: None)
        monkeypatch.setattr(live_runtime, 'check', lambda: None)
    try:  # older trees pin the numerical runtime here instead; this test checks dispatch, not the runtime
        from neuroshard.evolution import assistant_growth_cohort3_eval, granite_shard_execution
        monkeypatch.setattr(granite_shard_execution, 'configure', lambda: None)
        monkeypatch.setattr(assistant_growth_cohort3_eval, 'runtime', lambda *_: None)
    except ImportError:
        pass
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
    """C3: a peer streaming far more than the bound, with no Content-Length, is cut off while streaming; the
    downloader never holds more than limit + 1 bytes, and the healthy peer is then used. (Real HTTP: the earlier
    version of this test stubbed urlopen, which the downloader no longer uses.)"""
    chain, root, good = bundle
    monkeypatch.setattr(audit, 'MAX_PROOF_BYTES', 1024)
    held = []
    original = network.bounded_download

    def measured(url, limit, timeout=30, deadline=60):
        try:
            payload = original(url, limit, timeout=timeout, deadline=deadline)
            held.append(len(payload))
            return payload
        except ValueError as error:
            held.append(str(error))
            raise
    monkeypatch.setattr(network, 'bounded_download', measured)

    def flood(handler):  # 10 MiB with no Content-Length: only a streaming bound can stop it
        handler.send_response(200)
        handler.end_headers()
        try:
            for _ in range(10):
                handler.wfile.write(b'x' * (1 << 20))
        except OSError:
            pass

    server, oversized = serve_bytes(flood)
    try:
        assert network.replicate(chain, tmp_path / 'destination', [oversized, good],
                                 proofs_only=True, rng=FirstPeerFirst()) == 1
        assert any('more than the 1024-byte bound' in str(h) for h in held)
        assert granite_audit.read_bundle(tmp_path / 'destination', root)
    finally:
        server.shutdown()
        server.server_close()


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
        assert network.replicate(chain, tmp_path / 'destination', [bad, good], proofs_only=True, rng=FirstPeerFirst()) == 1
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
        assert network.replicate(chain, tmp_path / 'destination', [bad, good], proofs_only=True, rng=FirstPeerFirst()) == 1
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
        assert network.replicate(chain, tmp_path / 'destination', [slow, good], proofs_only=True, rng=FirstPeerFirst()) == 1
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
            assert network.replicate(chain, tmp_path / voter, peers, proofs_only=True, rng=FirstPeerFirst()) == 1
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


# --- R2 of the follow-up review: proof acquisition is budgeted against each challenge's own deadline -------------

def zipped(files):
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w') as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)
    return out.getvalue()


def virtual_chain(height, challenges):
    """A simulated chain at ``height`` (one block per virtual second) with {root: deadline} open challenges.

    A challenge closes once the height passes its deadline."""
    clock = [height]

    def state():
        open_ = {f'c{i}': {'proof_root': root, 'deadline': deadline}
                 for i, (root, deadline) in enumerate(challenges.items()) if clock[0] <= deadline}
        return {'height': clock[0], 'params': {'proof_blocks': 300}, 'jobs': {'j': {'challenges': open_}}}
    return SimpleNamespace(state=state), clock


def bundles(count):
    out = []
    for i in range(count):
        files = {name: json.dumps({'n': i}).encode() for name in audit.PROOF_FILES}
        out.append((granite_audit.bundle_digest(files), zipped(files)))
    return out


def scripted_peers(monkeypatch, clock, held, slow_prefix='http://slow'):
    """A slow peer spends its whole allowance (in virtual blocks) and fails; any other peer serves ``held``."""
    attempts = []

    def downloaded(url, limit, timeout=30, deadline=60):
        attempts.append((url.split('/')[2], url.rsplit('/', 1)[1][:8], clock[0], deadline))
        if url.startswith(slow_prefix):
            clock[0] += int(deadline)
            raise TimeoutError('slow peer used its allowance')
        return held[url.rsplit('/', 1)[1][:-4]]
    monkeypatch.setattr(network, 'bounded_download', downloaded)
    return attempts


def test_a_near_expiry_challenge_reaches_a_healthy_peer_after_a_slow_one_whatever_the_order(monkeypatch, tmp_path):
    (root, payload), = bundles(1)
    for seed in range(6):  # every rotation of the two peers
        chain, clock = virtual_chain(100, {root: 120})
        attempts = scripted_peers(monkeypatch, clock, {root: payload})
        destination = tmp_path / str(seed)
        copied = network.replicate_proofs(chain, destination, ['http://slow/proofs', 'http://good/proofs'],
                                          progress=lambda *_: None, rng=__import__('random').Random(seed))
        assert copied == 1 and granite_audit.read_bundle(destination, root)
        healthy = [a for a in attempts if a[0] == 'good']
        assert healthy and healthy[0][2] < 120  # asked before the challenge closed
        assert all(a[3] <= 15 for a in attempts)  # no attempt may outlast what the window leaves


def test_queued_challenges_are_fetched_most_urgent_first_and_none_is_starved(monkeypatch, tmp_path):
    (late, late_zip), (soon, soon_zip) = bundles(2)
    chain, clock = virtual_chain(100, {late: 400, soon: 130})
    attempts = scripted_peers(monkeypatch, clock, {late: late_zip, soon: soon_zip})
    copied = network.replicate_proofs(chain, tmp_path, ['http://slow/proofs', 'http://good/proofs'],
                                      progress=lambda *_: None, rng=__import__('random').Random(0))
    assert copied == 2
    assert attempts[0][1] == soon[:8]  # the challenge closing at 130 is fetched before the one closing at 400
    assert [a for a in attempts if a[0] == 'good' and a[1] == soon[:8]][0][2] < 130


def test_fetching_stops_when_a_challenge_closes_and_nothing_is_accepted_after(monkeypatch, tmp_path):
    (root, payload), = bundles(1)
    chain, clock = virtual_chain(100, {root: 103})  # 3 blocks left: all reserved for judging and submitting
    attempts = scripted_peers(monkeypatch, clock, {root: payload})
    notes = []
    assert network.replicate_proofs(chain, tmp_path, ['http://good/proofs'], progress=notes.append) == 0
    assert attempts == [] and any('no time left' in n for n in notes)
    chain, clock = virtual_chain(130, {root: 120})  # already closed
    assert network.replicate_proofs(chain, tmp_path, ['http://good/proofs'], progress=notes.append) == 0
    assert not (tmp_path / root).exists()


def test_a_bundle_whose_challenge_closed_during_its_transfer_is_not_kept(monkeypatch, tmp_path):
    (root, payload), = bundles(1)
    chain, clock = virtual_chain(100, {root: 130})

    def closes_while_downloading(url, limit, timeout=30, deadline=60):
        clock[0] = 131  # the challenge lapses while the bytes arrive
        return payload
    monkeypatch.setattr(network, 'bounded_download', closes_while_downloading)
    assert network.replicate_proofs(chain, tmp_path, ['http://good/proofs'], progress=lambda *_: None) == 0
    assert not (tmp_path / root).exists()  # accepted only against the current chain, which no longer names it


# --- R1 of the follow-up review ---------------------------------------------------------------------------------

@pytest.mark.parametrize("phase", ["headers", "chunk_size"])
def test_the_overall_deadline_covers_http_headers_and_chunk_framing(phase):
    """R1 of the follow-up review: a peer trickling its headers or a chunk-size line kept a read alive past the
    overall deadline; the deadline now shuts the socket itself."""
    class Trickle(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            try:
                if phase == "headers":
                    self.wfile.write(b"HTTP/1.1 200 OK\r\nX-Slow: ")
                else:
                    self.wfile.write(
                        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n1;slow="
                    )
                self.wfile.flush()
                for _ in range(15):
                    self.wfile.write(b"x")
                    self.wfile.flush()
                    time.sleep(0.1)
                if phase == "headers":
                    self.wfile.write(b"\r\nContent-Length: 1\r\n\r\nZ")
                else:
                    self.wfile.write(b"\r\nZ\r\n0\r\n\r\n")
                self.wfile.flush()
            except OSError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Trickle)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        started = time.monotonic()
        outcome = "returned"
        try:
            network.bounded_download(
                f"http://127.0.0.1:{server.server_port}/proof",
                1024, timeout=2, deadline=0.35,
            )
        except Exception as error:
            outcome = type(error).__name__
        elapsed = time.monotonic() - started
        print(json.dumps({"phase": phase, "deadline": 0.35, "elapsed": elapsed, "outcome": outcome}))
        assert elapsed < 0.85, "overall deadline was bypassed by HTTP framing"
    finally:
        server.shutdown()
        server.server_close()


def test_later_passes_ask_a_different_peer_first(monkeypatch, tmp_path):
    """The first pass asks peers in list order; a later pass for the same proof starts one peer further on, so a
    slow first peer does not take every pass's first, largest share of the window."""
    (root, _), = bundles(1)
    chain, clock = virtual_chain(100, {root: 400})
    first_asked = []

    def always_fails(url, limit, timeout=30, deadline=60):
        first_asked.append(url.split('/')[2])
        raise OSError('unreachable')
    monkeypatch.setattr(network, 'bounded_download', always_fails)
    peers = ['http://a/proofs', 'http://b/proofs', 'http://c/proofs']
    for _ in range(3):
        before = len(first_asked)
        network.replicate_proofs(chain, tmp_path, peers, progress=lambda *_: None)
        first_asked[before:] = first_asked[before:before + 1]
    assert first_asked == ['a', 'b', 'c']
