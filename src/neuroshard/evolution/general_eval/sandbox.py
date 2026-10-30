"""Run untrusted generated Python with hidden tests: no network, bounded time, memory, files and processes.

Layers, each recorded in the result so a report states what isolation it actually had:

1. ``netns``: the child runs in a fresh, empty network namespace (``unshare --user --net``), so even
   loopback is down and no address is reachable. A sealed run must have this layer (``require_netns``).
2. ``audit-hook``: a prelude installs a ``sys.addaudithook`` hook, which Python code cannot remove, that
   refuses socket creation and connection, subprocesses, ``os.system``/``exec``/``fork``/``posix_spawn`` and
   ``ctypes`` loading. This holds even without a namespace.
3. ``rlimits``: CPU seconds, address space, file size and open files are bounded in the child.
4. A wall-clock timeout kills the child's whole process group; the child runs ``python -I`` (isolated mode:
   no user site, no ``PYTHON*`` variables) in an empty temporary directory with a scrubbed environment.

The generated code and the tests are written to the temporary directory, read once and deleted before the
solution runs. The result line carries a per-run nonce sent on stdin, so printing a fake result does not pass.
Threat model: mistaken or careless generated code from the assistant under test. It is not hardened against
code crafted to forge a result from inside the interpreter (for example by walking ``gc`` objects); a sealed
run therefore also requires the netns layer, and any implausible pass pattern is reviewed by hand.
"""

import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time

PRELUDE = r'''
import sys
_BLOCKED = ('socket.__new__', 'socket.connect', 'socket.bind', 'socket.getaddrinfo', 'socket.gethostbyname',
            'socket.sendto', 'subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn', 'os.spawn', 'os.fork',
            'os.forkpty', 'pty.spawn', 'ctypes.dlopen', 'ctypes.dlsym', 'ctypes.cdata', 'urllib.Request',
            'os.kill', 'signal.pthread_kill')
def _guard(event, args):
    if event in _BLOCKED or event.startswith('socket.'):
        raise PermissionError('sandbox: ' + event + ' is not allowed')
sys.addaudithook(_guard)
del _guard
'''

RUNNER = r'''
import builtins, json, os, sys
_nonce = sys.stdin.readline().strip()
sys.stdin.close()
_solution, _tests = open('solution.py').read(), open('tests.py').read()
for _name in ('solution.py', 'tests.py', 'main.py'):
    os.remove(_name)  # the solution can neither read nor rewrite its hidden tests or this runner
_write, _dumps, _real = sys.stdout.write, json.dumps, dict(vars(builtins))

def same(actual, expected):
    """Equal values of exactly the same plain types, recursively: no custom __eq__ can pass a test."""
    if type(actual) is not type(expected):
        return False
    if type(expected) in (list, tuple):
        return len(actual) == len(expected) and all(same(a, e) for a, e in zip(actual, expected))
    if type(expected) is dict:
        return set(actual) == set(expected) and all(same(actual[k], expected[k]) for k in expected)
    if type(expected) is float:
        return abs(actual - expected) <= 1e-9 * max(1.0, abs(expected))
    if type(expected) in (int, str, bool, type(None), set, frozenset):
        return actual == expected
    raise TypeError('tests compare plain values only')

result = {'stage': 'solution'}
try:
    solution = {'__name__': '__solution__', '__builtins__': dict(_real)}  # its own copy of the builtins
    exec(compile(_solution, 'solution.py', 'exec'), solution)
    result['stage'] = 'tests'
    exec(compile(_tests, 'tests.py', 'exec'), {'__name__': '__tests__', '__builtins__': _real, 'same': same,
                                              'solution': solution})
    result.update(passed=True)
except BaseException as error:  # noqa: BLE001 - every failure is a failed item, recorded
    result.update(passed=False, error=type(error).__name__ + ': ' + str(error)[:300])
_write('\n@@' + _nonce + '@@' + _dumps(result) + '\n')
'''


def netns_available():
    """True when an unprivileged empty network namespace can be created here."""
    if shutil.which('unshare') is None:
        return False
    try:
        return subprocess.run(['unshare', '--user', '--map-root-user', '--net', 'true'], timeout=10,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _limits(cpu_seconds, memory_bytes, file_bytes):
    def apply():
        import resource

        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return apply


def run_python(code, tests, *, timeout=10.0, memory_bytes=1 << 30, file_bytes=1 << 20, use_netns=True,
               require_netns=False, python=None, audit_hook=True):
    """Execute ``code`` then ``tests`` (asserts) in a sandboxed child; returns a JSON-able result.

    ``passed`` is True only if both ran to completion inside the time limit with no exception."""
    netns = use_netns and netns_available()
    nonce = secrets.token_hex(16)
    if require_netns and not netns:
        raise RuntimeError('this run requires an empty network namespace and none is available')
    layers = ['rlimits', 'timeout', 'isolated-python', 'scrubbed-env', 'tmpdir'] + (['audit-hook'] if audit_hook
                                                                                      else []) + (['netns'] if netns
                                                                                                  else [])
    with tempfile.TemporaryDirectory(prefix='ns-general-eval-') as directory:
        work = Path(directory)
        (work / 'solution.py').write_text(code)
        (work / 'tests.py').write_text(tests)
        (work / 'main.py').write_text((PRELUDE if audit_hook else '') + RUNNER)
        command = [python or sys.executable, '-I', 'main.py']
        if netns:
            command = ['unshare', '--user', '--map-root-user', '--net', '--'] + command
        env = {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'HOME': directory, 'PYTHONHASHSEED': '0'}
        started = time.monotonic()
        process = subprocess.Popen(command, cwd=directory, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   preexec_fn=_limits(max(1, int(timeout) + 1), memory_bytes, file_bytes))
        timed_out = False
        try:
            out, err = process.communicate(nonce.encode() + b'\n', timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            out, err = process.communicate()
        seconds = time.monotonic() - started
    text = out.decode('utf-8', 'replace')
    tag = f'\n@@{nonce}@@'
    marker = text.rfind(tag)
    result = {'passed': False, 'timed_out': timed_out, 'seconds': round(seconds, 3), 'isolation': sorted(layers),
              'returncode': process.returncode}
    if timed_out:
        result['error'] = f'timeout after {timeout} s'
    elif marker >= 0:
        try:
            body = json.loads(text[marker + len(tag):].strip().splitlines()[0])
            result.update(passed=bool(body.get('passed')) and process.returncode == 0, stage=body.get('stage'),
                          error=body.get('error'))
        except (ValueError, IndexError):
            result['error'] = 'unreadable sandbox result'
    else:
        result['error'] = ('killed: ' + (err.decode('utf-8', 'replace')[-300:] or f'returncode {process.returncode}'))
    result['stdout_tail'] = text[:marker if marker >= 0 else len(text)][-500:]
    return result
