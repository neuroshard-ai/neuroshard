"""The client's stage-0 thread count is pinned, whatever the device's core count."""
import pytest
import torch

from neuroshard.assistant import network
from neuroshard.evolution import granite_tokenizer


class Stop(Exception):
    pass


def open_until_threads_are_set(monkeypatch, cores, threads=None):
    """Run ``Conversation.__init__`` until just after it sets torch's threads; return (threads set, messages)."""
    chosen, messages = [], []
    monkeypatch.setattr(network.os, 'cpu_count', lambda: cores)
    monkeypatch.setattr(torch, 'set_num_threads', chosen.append)

    def stop(*args, **kwargs):
        raise Stop

    monkeypatch.setattr(granite_tokenizer, 'load', stop)
    with pytest.raises(Stop):
        network.Conversation(None, {'gate': None, 'tokenizer': None}, None, None, {}, threads=threads,
                             progress=messages.append)
    return chosen, messages


@pytest.mark.parametrize('cores', [2, 4, 8, 16, 64])
def test_stage_zero_runs_with_the_execution_classs_eight_threads_on_any_core_count(monkeypatch, cores):
    chosen, messages = open_until_threads_are_set(monkeypatch, cores)
    assert chosen == [network.CLIENT_THREADS] == [8]
    assert messages == []


def test_another_thread_count_is_honoured_but_warned_about(monkeypatch):
    chosen, messages = open_until_threads_are_set(monkeypatch, 4, threads=4)
    assert chosen == [4]
    assert len(messages) == 1 and 'instead of 8' in messages[0] and 'AMX' in messages[0]
