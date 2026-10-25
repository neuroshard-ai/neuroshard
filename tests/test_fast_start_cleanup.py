"""Fast start's background primer and the conversation's cleanup: no driver command while the primer runs.

These use the real ``Primer``, ``Conversation.say`` and ``Conversation.close`` with a stand-in driver; no model,
network or chain is involved.
"""
import socket
import threading
from unittest.mock import patch

import pytest

from neuroshard.assistant import network


class Driver:
    """Records every driver command and whether the primer was running when it arrived."""

    def __init__(self, block=False, fail=None):
        self.active, self.release = threading.Event(), threading.Event()
        self.block, self.fail, self.calls = block, fail, []
        if not block:
            self.release.set()

    def prime_router(self, tokenizer, policy):
        self.active.set()
        try:
            self.release.wait(10)
            if self.fail:
                raise self.fail
        finally:
            self.active.clear()

    def stop(self):
        self.calls.append(('stop', self.active.is_set()))


def conversation(driver, primer, sockets=None):
    talk = network.Conversation.__new__(network.Conversation)
    talk.priming, talk.driver, talk.sockets = primer, driver, sockets or {}
    talk.router = None
    talk.session = type('Session', (), {'turn': staticmethod(lambda text, respond: {'final_text': text})})()
    talk.respond = None
    return talk


def socket_pair():
    left, right = socket.socketpair()
    return left, right


def test_an_interrupted_first_message_never_stops_the_owners_while_the_primer_runs():
    """A real SIGINT during ``say``'s wait (in a subprocess, so the signal hits only it). CPython can mark the
    interrupted Thread stopped while its worker runs; completion must come from the worker's own signal."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    child = r'''
import json, os, signal, threading
from neuroshard.assistant import network
active, release, finished = threading.Event(), threading.Event(), threading.Event()
calls = []
class Driver:
    def prime_router(self, *args):
        active.set()
        try:
            release.wait(10)
        finally:
            active.clear(); finished.set()
    def stop(self):
        calls.append(active.is_set())
driver = Driver()
primer = network.Primer(driver, None, None)
assert active.wait(5)
talk = network.Conversation.__new__(network.Conversation)
talk.priming, talk.driver, talk.sockets = primer, driver, {}
threading.Timer(0.3, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
try:
    talk.say("hello")
    interrupted = False
except KeyboardInterrupt:
    interrupted = True
state = {"interrupted": interrupted, "worker_active": active.is_set(), "done": primer.done(),
         "kept": talk.priming is primer}
threading.Timer(0.3, release.set).start()  # the worker finishes while close waits for it
state["clean"] = talk.close()
state["stop_saw_worker_active"] = calls
assert finished.wait(5)
print(json.dumps(state))
'''
    source = Path(network.__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, '-c', child], capture_output=True, text=True, timeout=60,
                            env={'PYTHONPATH': str(source), 'PATH': '/usr/bin:/bin'})
    assert result.returncode == 0, result.stderr[-2000:]
    state = json.loads(result.stdout.strip().splitlines()[-1])
    assert state['interrupted'] and state['worker_active']
    assert not state['done'] and state['kept']  # the worker has not signalled; the primer is kept
    assert state['clean'] and state['stop_saw_worker_active'] == [False]  # STOP only after it finished


def test_close_wakes_a_primer_blocked_in_socket_io_and_sends_no_stop(monkeypatch):
    """A worker genuinely blocked in ``recv_into`` on the job's socket (serving's 600 s timeout)."""
    import time

    mine, peer = socket_pair()
    mine.settimeout(600)
    calls = []

    class Blocked:
        def prime_router(self, tokenizer, policy):
            mine.recv_into(bytearray(5))

        def stop(self):
            calls.append('stop')

    driver = Blocked()
    primer = network.Primer(driver, None, None)
    time.sleep(0.2)  # into the real receive
    assert not primer.done()
    talk = conversation(driver, primer, {1: mine})
    monkeypatch.setattr(network.Conversation, 'PRIMER_GRACE', 0.1)
    began = time.monotonic()
    assert talk.close() is True
    assert time.monotonic() - began < 5
    assert primer.done() and talk.priming is None and calls == []
    assert talk.sockets == {} and mine.fileno() == -1
    peer.close()


def test_a_primer_busy_in_local_work_gets_a_bounded_wait_and_no_stop(monkeypatch):
    """Shutting sockets down cannot interrupt local computation; close waits a bounded time and reports it."""
    driver = Driver(block=True)  # ignores sockets, like a stage-0 forward in progress
    primer = network.Primer(driver, None, None)
    assert driver.active.wait(2)
    mine, theirs = socket_pair()
    talk = conversation(driver, primer, {1: mine})
    monkeypatch.setattr(network.Conversation, 'PRIMER_GRACE', 0.1)
    monkeypatch.setattr(network.Conversation, 'CANCEL_GRACE', 0.2)
    assert talk.close() is False  # still running: reported, kept, and no STOP raced it
    assert driver.calls == [] and talk.priming is primer and mine.fileno() == -1
    driver.release.set()
    assert primer.wait(5)
    theirs.close()


def test_close_without_a_question_waits_for_the_primer_then_stops_once():
    driver = Driver(block=True)
    primer = network.Primer(driver, None, None)
    assert driver.active.wait(2)
    talk = conversation(driver, primer)
    threading.Timer(0.3, driver.release.set).start()
    talk.close()
    assert driver.calls == [('stop', False)] and talk.priming is None


def test_close_interrupted_while_waiting_sends_no_stop_and_closes_the_sockets():
    driver = Driver(block=True)
    primer = network.Primer(driver, None, None)
    assert driver.active.wait(2)
    mine, theirs = socket_pair()
    talk = conversation(driver, primer, {1: mine})
    real_wait = primer.wait
    calls = []

    def wait(timeout=None):
        calls.append(timeout)
        if len(calls) == 1:
            raise KeyboardInterrupt
        driver.release.set()  # the closed sockets end the real worker; here it is released
        return real_wait(timeout)

    with patch.object(primer, 'wait', side_effect=wait):
        talk.close()
    assert driver.calls == []  # no STOP raced the worker
    assert talk.sockets == {} and mine.fileno() == -1
    assert primer.done() and talk.priming is None
    theirs.close()


def test_a_primer_that_never_finishes_within_the_grace_sends_no_stop(monkeypatch):
    driver = Driver(block=True)
    primer = network.Primer(driver, None, None)
    assert driver.active.wait(2)
    talk = conversation(driver, primer)
    monkeypatch.setattr(network.Conversation, 'PRIMER_GRACE', 0.2)
    monkeypatch.setattr(network.Conversation, 'CANCEL_GRACE', 0.2)
    assert talk.close() is False
    assert driver.calls == []
    driver.release.set()
    primer.wait(5)


def test_a_failed_primer_is_raised_by_the_first_message_and_close_still_stops_the_owners():
    driver = Driver(fail=ConnectionError('owner went away'))
    primer = network.Primer(driver, None, None)
    talk = conversation(driver, primer)
    with pytest.raises(ConnectionError, match='owner went away'):
        talk.say('Hello')
    assert talk.priming is None
    talk.close()
    assert driver.calls == [('stop', False)]


def test_a_failed_primer_is_absorbed_by_close_without_a_question():
    driver = Driver(fail=ConnectionError('owner went away'))
    primer = network.Primer(driver, None, None)
    primer.thread.join(5)
    talk = conversation(driver, primer)
    talk.close()
    assert driver.calls == [('stop', False)] and talk.priming is None
