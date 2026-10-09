import socket
import threading
import time
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from neuroshard.assistant import network
from neuroshard.evolution.sharded.granite_pipeline import FORWARD, RESET
from neuroshard.evolution.sharded.granite_serving import CROP, FEATURE
from neuroshard.inference import optimistic as ledger, relay


def test_paid_work_counts_features_and_cannot_be_reset_or_cropped_away():
    budget = relay.WorkBudget(5, 20)
    budget.accept(FEATURE, 2)
    budget.accept(FORWARD, 2)
    budget.accept(CROP, 0)
    budget.accept(RESET, 0)
    with pytest.raises(ValueError, match='paid position'):
        budget.accept(FORWARD, 2)
    assert budget.positions == 4


def test_context_commands_and_expired_jobs_are_refused_before_work():
    budget = relay.WorkBudget(20, 4)
    budget.accept(FORWARD, 3)
    for op, value in [(FORWARD, 2), (CROP, 4), (999, 0)]:
        with pytest.raises(ValueError):
            budget.accept(op, value)
    def expired():
        raise ValueError('expired')
    with pytest.raises(ValueError, match='expired'):
        relay.WorkBudget(20, 4, expired).accept(FORWARD, 1)


def test_malformed_controls_and_oversized_tensor_headers_are_bounded():
    left, right = socket.socketpair()
    try:
        owner = relay.OwnerRelay(right, 1, 2, 4, 8)
        relay.send_frame(left, relay.CONTROL, b'x')
        with pytest.raises(ValueError, match='malformed'):
            owner.command()
        left.sendall(relay.HEADER.pack(relay.TENSOR, relay.MAX_FRAME))
        with pytest.raises(ValueError, match='bound'):
            owner.receive(0, 1)
    finally:
        left.close()
        right.close()


def test_slow_handshake_bytes_do_not_extend_the_absolute_deadline():
    left, right = socket.socketpair()
    stop = threading.Event()
    def drip():
        while not stop.wait(0.02):
            try:
                left.sendall(b'x')
            except OSError:
                return
    thread = threading.Thread(target=drip, daemon=True)
    thread.start()
    began = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            relay.read_exactly(right, 1000, deadline=began + 0.15)
        assert time.monotonic() - began < 1
    finally:
        stop.set()
        left.close()
        right.close()
        thread.join(1)


def test_an_idle_unauthenticated_connection_does_not_block_a_real_user():
    key = Ed25519PrivateKey.generate()
    owner = network.Owner.__new__(network.Owner)
    owner.rank, owner.served = 1, SimpleNamespace(world=2, max_tokens=8, name='tiny')
    owner.config, owner.serving_lock = SimpleNamespace(hidden_size=4), threading.Lock()
    owner.session_key_of = lambda chain, job: key.public_key().public_bytes_raw().hex()
    accepted = []
    def serve(connection, body, progress):
        accepted.append(body['job_id'])
        relay.send_frame(connection, relay.READY, ledger.canonical({'format': relay.FORMAT, 'rank': 1}))
    owner.serve_authenticated = serve
    with socket.create_server(('127.0.0.1', 0)) as probe:
        port = probe.getsockname()[1]
    stop = threading.Event()
    thread = threading.Thread(target=owner.run, args=('127.0.0.1', port, stop, lambda *_: None), daemon=True)
    thread.start()
    idle = None
    try:
        deadline = time.monotonic() + 5
        while idle is None:
            try:
                idle = socket.create_connection(('127.0.0.1', port), timeout=0.1)
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.01)
        body = relay.hello('test', 'ab' * 32, 1, 2, 4, 8)
        relay.dial(f'127.0.0.1:{port}', body, key, timeout=2).close()
        assert accepted == ['ab' * 32]
    finally:
        stop.set()
        if idle is not None:
            idle.close()
        thread.join(3)
