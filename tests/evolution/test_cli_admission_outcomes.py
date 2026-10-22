"""The CLI reports expected admission and funding outcomes cleanly: one message, exit 1, no traceback.

Found in the pilot-2-rc1 live burst: every busy deadline printed a Python traceback ending in NetworkBusy, because
NetworkBusy is a RuntimeError and the CLI boundary caught only OSError, ValueError and KeyError."""
import pytest

from neuroshard.assistant import admission_client, funding
from neuroshard.client import assistant_cli, cli

EXPECTED = [
    admission_client.NetworkBusy('the network stayed busy until the deadline; nothing was funded'),
    admission_client.NotServed('ab' * 32, 1234, 'the deadline passed'),
    admission_client.FundingLapsed('ab' * 32),
    funding.Unresolved('funding outcome unknown for job abababab: index unavailable'),
]


@pytest.mark.parametrize('outcome', EXPECTED, ids=lambda e: type(e).__name__)
def test_an_expected_admission_outcome_is_one_clean_message(monkeypatch, capsys, outcome):
    def chat(args):
        raise outcome

    monkeypatch.setattr(assistant_cli, 'run', chat)
    with pytest.raises(SystemExit) as stopped:
        cli.main(['assistant', 'chat'])
    err = capsys.readouterr().err
    assert stopped.value.code == 1
    assert err == f'NeuroShard: {outcome}\n'
    assert 'Traceback' not in err


def test_the_busy_message_still_says_nothing_was_funded(monkeypatch, capsys):
    monkeypatch.setattr(assistant_cli, 'run', lambda args: (_ for _ in ()).throw(EXPECTED[0]))
    with pytest.raises(SystemExit):
        cli.main(['assistant', 'chat'])
    assert 'nothing was funded' in capsys.readouterr().err


def test_an_unexpected_runtime_error_is_not_swallowed(monkeypatch):
    monkeypatch.setattr(assistant_cli, 'run', lambda args: (_ for _ in ()).throw(RuntimeError('a real bug')))
    with pytest.raises(RuntimeError, match='a real bug'):
        cli.main(['assistant', 'chat'])
