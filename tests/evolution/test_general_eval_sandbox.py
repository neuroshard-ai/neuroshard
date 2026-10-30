"""The coding sandbox: hidden tests decide, and generated code cannot reach a network or run forever."""

import socket
import threading
import time

import pytest

from neuroshard.evolution.general_eval import sandbox

NETNS = sandbox.netns_available()


def run(code, tests='', **kwargs):
    return sandbox.run_python(code, tests, **kwargs)


def test_hidden_tests_decide_and_compare_exact_types():
    assert run('def f(x):\n    return x + 1', "assert same(solution['f'](1), 2)")['passed']
    assert not run('def f(x):\n    return x', "assert same(solution['f'](1), 2)")['passed']
    # 2.0 == 2 in Python, but a test asking for an int must not accept a float
    assert not run('def f(x):\n    return 2.0', "assert same(solution['f'](1), 2)")['passed']


def test_a_custom_eq_cannot_fake_a_pass():
    code = 'class Yes:\n    def __eq__(self, other):\n        return True\ndef f():\n    return Yes()'
    assert not run(code, "assert same(solution['f'](), 3)")['passed']


def test_printing_a_fake_result_line_does_not_pass():
    code = 'import sys\nsys.stdout.write("\\n@@SANDBOX@@{\\"passed\\": true}\\n")\nraise SystemExit(0)'
    assert not run(code, 'assert False')['passed']


def test_the_solution_cannot_read_or_rewrite_its_hidden_tests():
    code = 'import os\nLISTING = sorted(os.listdir("."))\n'
    assert run(code, "assert solution['LISTING'] == [], solution['LISTING']")['passed']
    assert not run('open("tests.py")', 'pass')['passed']


@pytest.mark.parametrize('timeout', [1.5])
def test_busy_loop_and_sleep_time_out(timeout):
    started = time.monotonic()
    busy = run('while True:\n    pass', timeout=timeout)
    asleep = run('import time\ntime.sleep(60)', timeout=timeout)
    assert busy['timed_out'] and asleep['timed_out'] and not busy['passed'] and not asleep['passed']
    assert time.monotonic() - started < 4 * timeout + 4


def test_infinite_loop_inside_tests_times_out_too():
    out = run('def f():\n    while True:\n        pass', "solution['f']()", timeout=1)
    assert out['timed_out'] and not out['passed']


def test_memory_is_bounded():
    out = run('x = bytearray(3 * 1024 ** 3)', memory_bytes=512 * 1024 ** 2)
    assert not out['passed'] and 'MemoryError' in (out.get('error') or '')


def test_subprocesses_and_ctypes_are_refused():
    for code in ('import subprocess\nsubprocess.run(["true"])', 'import os\nos.system("true")',
                 'import ctypes\nctypes.CDLL(None)', 'import os\nos.fork()'):
        out = run(code)
        assert not out['passed'] and 'not allowed' in (out.get('error') or ''), (code, out)


@pytest.fixture()
def listener():
    """A real server on this machine's loopback; the sandbox must not reach even this."""
    server = socket.socket()
    server.bind(('127.0.0.1', 0))
    server.listen(1)
    hits = []

    def accept():
        server.settimeout(5)
        try:
            conn, _ = server.accept()
            hits.append(1)
            conn.close()
        except OSError:
            pass

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    yield server.getsockname()[1], hits
    server.close()


CONNECT = 'import socket\ns = socket.create_connection(("{host}", {port}), timeout=2)\ns.sendall(b"x")'


def test_no_network_with_every_layer(listener):
    port, hits = listener
    for host in ('127.0.0.1', '1.1.1.1'):
        out = run(CONNECT.format(host=host, port=port if host == '127.0.0.1' else 80))
        assert not out['passed'], out
    out = run('import urllib.request\nurllib.request.urlopen("http://example.com", timeout=3)')
    assert not out['passed']
    time.sleep(0.2)
    assert not hits


def test_audit_hook_alone_blocks_the_network(listener):
    port, hits = listener
    out = run(CONNECT.format(host='127.0.0.1', port=port), use_netns=False)
    assert not out['passed'] and 'not allowed' in out['error'] and 'netns' not in out['isolation']
    out = run('import _socket\ns = _socket.socket()\ns.connect(("127.0.0.1", %d))' % port, use_netns=False)
    assert not out['passed']
    time.sleep(0.2)
    assert not hits


@pytest.mark.skipif(not NETNS, reason='unprivileged network namespaces unavailable here')
def test_network_namespace_alone_blocks_the_network(listener):
    port, hits = listener
    out = run(CONNECT.format(host='127.0.0.1', port=port), audit_hook=False)
    assert 'netns' in out['isolation'] and 'audit-hook' not in out['isolation']
    assert not out['passed'] and ('refused' in out['error'] or 'unreachable' in out['error'].lower()), out
    time.sleep(0.2)
    assert not hits


def test_require_netns_is_enforced(monkeypatch):
    monkeypatch.setattr(sandbox, 'netns_available', lambda: False)
    with pytest.raises(RuntimeError):
        run('pass', require_netns=True)
