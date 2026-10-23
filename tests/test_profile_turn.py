"""scripts/profile_turn.py: span scoping, admission-read attribution, crash-safe results (offline, fake spans)."""
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'profile_turn.py'


@pytest.fixture
def prof():
    spec = importlib.util.spec_from_file_location('profile_turn_under_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def busy(seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        pass


def two_turns(prof):
    """Turn 1: one pass (local partition, exchange with a frame read), tools. Turn 2 starts with a router pass."""
    rec = prof.REC
    with rec.span('turn'):
        with rec.span('pass'):
            with rec.span('local.partition'):
                busy(0.01)
            with rec.span('exchange'):
                with rec.span('frame.send'):
                    busy(0.002)
                with rec.span('frame.receive'):
                    busy(0.02)
            busy(0.003)
        with rec.span('tools'):
            busy(0.001)
        busy(0.002)
    with rec.span('turn'):
        with rec.span('pass'):  # the next turn's router pass: must never count toward turn 1
            with rec.span('exchange'):
                with rec.span('frame.receive'):
                    busy(0.03)
    return prof.analyse(rec.spans, rec.events)


def test_adjacent_turns_never_share_spans_and_categories_sum(prof):
    result = two_turns(prof)
    first, second = result['turns']
    assert first['model_passes'] == 1 and second['model_passes'] == 1
    assert not result['invariant_violations'], result['invariant_violations']
    for turn in (first, second):
        for key in ('pass_other_s', 'outside_passes_s', 'exchange.other_s'):
            assert turn[key] >= 0, (key, turn)
    assert first['exchange.blocked_on_owner_frames_s'] < 0.03, 'turn 2 frame time leaked into turn 1'
    assert first['tools_s'] > 0 and second['tools_s'] == 0


def test_an_overlapping_category_is_reported_as_a_violation(prof):
    rec = prof.REC
    with rec.span('turn'):
        with rec.span('pass'):
            busy(0.005)
    turn = next(s for s in rec.spans if s['name'] == 'turn')
    rec.spans.append({'type': 'span', 'id': 999, 'parent': turn['id'], 'turn': turn['id'], 'name': 'exchange',
                      'start': turn['start'], 'end': turn['end'] + 1, 'seconds': turn['seconds'] + 1})
    result = prof.analyse(rec.spans, rec.events)
    assert result['invariant_violations'], 'an exchange longer than its pass must not pass silently'


def test_admission_reads_are_only_those_inside_admission(prof):
    rec = prof.REC
    with rec.span('chain.state'):
        busy(0.001)  # e.g. the CLI's own version check, before admission
    with rec.span('conversation.setup'):
        with rec.span('chain.state'):
            busy(0.001)
        with rec.span('admission'):
            for _ in range(3):
                with rec.span('chain.state'):
                    busy(0.002)
            with rec.span('admission.reserve_round'):
                with rec.span('chain.state'):
                    busy(0.002)
    setup = prof.analyse(rec.spans, rec.events)['setup']
    assert setup['admission.chain_reads'] == 4
    assert setup['chain_reads_outside_admission'] == 2


def test_nested_partition_calls_are_counted_once(prof):
    rec = prof.REC
    with rec.span('turn'):
        with rec.span('pass'):
            with rec.span('local.partition'):  # partition(...) calling the wrapped embed inside
                with rec.span('local.partition'):
                    busy(0.01)
                busy(0.01)
    turn, = prof.analyse(rec.spans, rec.events)['turns']
    assert turn['local_partition_s'] < 0.03 and turn['pass_other_s'] >= 0


def run_main(prof, tmp_path, monkeypatch, behaviour, diagnostic_fails=False):
    """main() with a fake client and network: ``behaviour`` is what the CLI does."""
    fake_cli = SimpleNamespace(main=behaviour)

    class Net:
        @staticmethod
        def descriptor():
            return {}

        @staticmethod
        def chain(descriptor, home):
            if diagnostic_fails:
                raise OSError('verified read failed')
            return SimpleNamespace(state=lambda: {'owners': {}, 'height': 7})

    monkeypatch.setattr(prof, 'instrument', lambda: (fake_cli, Net))
    monkeypatch.setitem(sys.modules, 'neuroshard.assistant.network', Net)
    import neuroshard.assistant as package
    monkeypatch.setattr(package, 'network', Net, raising=False)
    monkeypatch.setattr(sys, 'argv', ['profile_turn.py', '--home', str(tmp_path / 'home'), '--out', str(tmp_path / 'out'),
                                      '--label', 'test', '--question', 'q1'])
    with pytest.raises(SystemExit) as stopped:
        prof.main()
    report, = [json.loads(p.read_text()) for p in (tmp_path / 'out').glob('*.json')]
    events = [json.loads(l) for p in (tmp_path / 'out').glob('*.events.jsonl') for l in p.read_text().splitlines()]
    return stopped.value.code, report, events


def test_a_client_exception_still_writes_the_profile(prof, tmp_path, monkeypatch):
    def crash(argv):
        with prof.REC.span('turn'):
            raise RuntimeError('NetworkBusy: the network stayed busy until the deadline')
    code, report, events = run_main(prof, tmp_path, monkeypatch, crash)
    assert code == 1 and report['outcome'] == 'client-exception' and 'NetworkBusy' in report['failure']
    assert events[0]['kind'] == 'identity', 'identity is persisted before the client starts'
    assert any(e.get('name') == 'turn' for e in events), 'spans are kept even when the client fails'


def test_a_failed_diagnostic_does_not_erase_a_completed_conversation(prof, tmp_path, monkeypatch):
    def converse(argv):
        with prof.REC.span('turn'):
            print('assistant (1 s, documents): fine')
    code, report, _ = run_main(prof, tmp_path, monkeypatch, converse, diagnostic_fails=True)
    assert code == 0 and report['outcome'] == 'completed'
    assert 'verified read failed' in report['diagnostics']['error']
    assert report['analysis']['turns'] and 'assistant (1 s' in report['client_output']


def test_a_nonzero_cli_exit_is_not_a_completed_run(prof, tmp_path, monkeypatch):
    def refuse(argv):
        raise SystemExit(1)
    code, report, _ = run_main(prof, tmp_path, monkeypatch, refuse)
    assert code == 1 and report['outcome'] == 'client-exit'


def test_partition_instrumentation_on_a_real_module_assigns_nothing_onto_it(prof):
    """The failed matched runs: assigning a wrapper to ``partition.norm`` (a submodule) raised in torch."""
    import torch
    from torch import nn

    class Norm(nn.Module):
        def forward(self, x):
            return x * 2

    class Partition(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = Norm()

        def embed(self, ids):
            return ids.float()

        def forward(self, hidden, mask=None, cache=None):
            return hidden + 1

        def logits(self, hidden):
            return self.norm(hidden)

        def warm_up(self):
            pass

    import neuroshard.evolution.sharded.granite as granite
    original = granite.load_partition
    try:
        granite.load_partition = lambda *a, **k: (Partition(), {})
        prof.instrument()
        partition, _ = granite.load_partition(None, None, 0)
        assert isinstance(partition.norm, Norm), 'the submodule is untouched'
        with prof.REC.span('turn'):
            with prof.REC.span('pass'):
                partition.logits(partition(partition.embed(torch.ones(1, 2))))
        turn, = prof.analyse(prof.REC.spans, prof.REC.events)['turns']
        assert turn['local_partition_s'] > 0 and not prof.analyse(prof.REC.spans, prof.REC.events)['invariant_violations']
    finally:
        granite.load_partition = original
        for cls in (Partition, Norm):
            for name in ('embed', 'forward', 'logits'):
                method = cls.__dict__.get(name)
                if hasattr(method, '__wrapped__'):
                    setattr(cls, name, method.__wrapped__)
