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
    driver = Driver(block=True)
    primer = network.Primer(driver, None, None)
    assert driver.active.wait(2)
    mine, theirs = socket_pair()
    talk = conversation(driver, primer, {1: mine})
    with patch.object(primer.thread, 'join', side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            talk.say('Hello')
    assert talk.priming is primer  # kept: the worker is still running

    def finish_soon():
        driver.release.set()

    threading.Timer(0.3, finish_soon).start()
    talk.close()
    assert driver.calls == [('stop', False)]  # STOP only after the primer finished
    assert talk.sockets == {} and mine.fileno() == -1
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
    talk.close()
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
