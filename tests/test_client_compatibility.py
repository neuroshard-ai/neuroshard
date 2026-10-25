"""Experimental client CPUs: warned before anything is paid, never refused outright."""
import pytest

from neuroshard.client import live_runtime

SAPPHIRE_RAPIDS = 'flags : fpu sse avx2 avx512f avx512_bf16 amx_bf16 amx_tile amx_int8'
ICE_LAKE = 'flags : fpu sse avx2 avx512f avx512vl'
ZEN4 = 'flags : fpu sse avx2 avx512f avx512_bf16'


def test_an_amx_linux_x86_client_is_the_evaluated_class_and_gets_no_warning():
    found = live_runtime.client_compatibility(SAPPHIRE_RAPIDS, 'x86_64', 'Linux')
    assert found == {'class': 'evaluated', 'reasons': []}
    assert live_runtime.client_warning(found) is None


@pytest.mark.parametrize('cpuinfo, machine, system, reason', [
    (ICE_LAKE, 'x86_64', 'Linux', 'CPU without amx_bf16, avx512_bf16'),
    (ZEN4, 'x86_64', 'Linux', 'CPU without amx_bf16'),
    ('', 'aarch64', 'Linux', 'Linux aarch64, not Linux x86_64'),
    ('', 'arm64', 'Darwin', 'Darwin arm64, not Linux x86_64'),
])
def test_other_clients_are_experimental_with_a_reason(cpuinfo, machine, system, reason):
    found = live_runtime.client_compatibility(cpuinfo, machine, system)
    assert found['class'] == 'experimental' and reason in found['reasons']
    warning = live_runtime.client_warning(found)
    assert warning.startswith('Experimental client: ') and reason in warning and 'AMX' in warning


class Stop(Exception):
    pass


def chat(monkeypatch, tmp_path, compatibility, answer=None, tty=True, experimental=False):
    """Run ``assistant chat`` until stage 0 would be fetched; return (fetched, printed)."""
    from neuroshard.assistant import network
    from neuroshard.client import assistant_cli as cli

    printed, fetched = [], []
    monkeypatch.setattr(live_runtime, 'configure', lambda: None)
    monkeypatch.setattr(live_runtime, 'client_compatibility', lambda: compatibility)
    monkeypatch.setattr(network, 'descriptor', lambda path=None: {})
    monkeypatch.setattr(network, 'chain', lambda descriptor, home: None)
    monkeypatch.setattr(cli, 'following', lambda descriptor, chain, home, net: descriptor)

    def fetch(*args, **kwargs):
        fetched.append(args)
        raise Stop

    monkeypatch.setattr(network, 'fetch_stage', fetch)
    monkeypatch.setattr(cli.sys.stdin, 'isatty', lambda: tty, raising=False)
    monkeypatch.setattr('builtins.input', lambda prompt='': printed.append(prompt) or answer)
    monkeypatch.setattr('builtins.print', lambda *a, **k: printed.append(' '.join(map(str, a))))
    parser = cli.argparse.ArgumentParser()
    cli.register(parser.add_subparsers(dest='command'), tmp_path)
    args = parser.parse_args(['assistant', 'chat', '--home', str(tmp_path)]
                             + (['--experimental-cpu'] if experimental else []))
    try:
        cli.run(args)
    except Stop:
        pass
    return fetched, printed


EXPERIMENTAL = {'class': 'experimental', 'reasons': ['CPU without amx_bf16']}


def test_an_experimental_client_is_asked_before_anything_is_fetched_or_paid(monkeypatch, tmp_path):
    fetched, printed = chat(monkeypatch, tmp_path, EXPERIMENTAL, answer='n')
    assert fetched == []
    assert printed[0].startswith('Experimental client:') and printed[-1] == 'Stopped. Nothing was paid.'


def test_an_experimental_client_continues_when_the_user_agrees(monkeypatch, tmp_path):
    fetched, printed = chat(monkeypatch, tmp_path, EXPERIMENTAL, answer='y')
    assert len(fetched) == 1 and printed[0].startswith('Experimental client:')


def test_a_script_on_an_experimental_cpu_must_opt_in(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match='--experimental-cpu'):
        chat(monkeypatch, tmp_path, EXPERIMENTAL, tty=False)
    fetched, printed = chat(monkeypatch, tmp_path, EXPERIMENTAL, tty=False, experimental=True)
    assert len(fetched) == 1 and printed[0].startswith('Experimental client:')


def test_an_evaluated_client_is_neither_warned_nor_asked(monkeypatch, tmp_path):
    fetched, printed = chat(monkeypatch, tmp_path, {'class': 'evaluated', 'reasons': []}, tty=False)
    assert len(fetched) == 1 and printed == []
