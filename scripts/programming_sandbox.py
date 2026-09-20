"""OS-isolated execution for an operated coding benchmark, not a fraud proof.

Only a read-only OS Python installation and one temporary program are mounted.
The candidate has no network, credentials, repository or host process namespace.
Resource limits bound ordinary runaway programs. This still trusts the host
kernel and is not a substitute for a VM boundary for hostile public submissions.
"""
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import tempfile
import uuid


def _limits():
    os.umask(0o077)
    for kind, limit in ((resource.RLIMIT_CPU, 2), (resource.RLIMIT_AS, 512 * 1024**2),
                        (resource.RLIMIT_FSIZE, 1024**2), (resource.RLIMIT_NOFILE, 64),
                        (resource.RLIMIT_CORE, 0)):
        resource.setrlimit(kind, (limit, limit))


def check(code, setup, tests, *, seconds=4):
    if not shutil.which('bwrap'):
        raise RuntimeError('Bubblewrap is required; execution never falls back to the host')
    if (not isinstance(code, str) or len(code.encode()) > 32768 or not tests
            or len(tests) > 32 or not all(isinstance(t, str) and len(t) < 16384 for t in tests)):
        raise ValueError('Invalid bounded program or tests')
    marker = uuid.uuid4().hex
    payload = {'code': code, 'setup': setup, 'tests': tests}
    harness = ("import json\np=json.load(open('/input.json'))\nns={}\n"
               "exec(compile(p['setup'],'setup','exec'),ns)\n"
               "exec(compile(p['code'],'candidate','exec'),ns)\n"
               "for test in p['tests']: exec(compile(test,'test','exec'),ns)\n"
               f"print({marker!r})\n")
    with tempfile.TemporaryDirectory(prefix='neuroshard-code-') as directory:
        root = Path(directory)
        (root / 'input.json').write_text(json.dumps(payload))
        (root / 'harness.py').write_text(harness)
        unit = 'neuroshard-code-' + uuid.uuid4().hex + '.scope'
        command = ['systemd-run', '--user', '--scope', '--quiet', '--collect', '--unit', unit,
                   '-p', 'TasksMax=16', '-p', 'MemoryMax=512M', '-p', 'CPUQuota=100%',
                   '-p', 'RuntimeMaxSec=5s',
                   'bwrap', '--unshare-all', '--die-with-parent', '--new-session', '--cap-drop', 'ALL',
                   '--clearenv', '--ro-bind', '/usr', '/usr']
        for path in ('/lib', '/lib64'):
            if Path(path).exists():
                command += ['--ro-bind', path, path]
        command += ['--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp', '--chdir', '/tmp',
                    '--ro-bind', str(root / 'input.json'), '/input.json',
                    '--ro-bind', str(root / 'harness.py'), '/harness.py',
                    '--', '/usr/bin/python3', '-I', '/harness.py']
        # A regular bounded output file avoids unbounded PIPE allocations.
        with (root / 'output').open('w+b') as output:
            process = subprocess.Popen(command, stdout=output, stderr=output,
                                       preexec_fn=_limits, start_new_session=True)
            try:
                result = process.wait(timeout=seconds)
                status = 'passed' if result == 0 else 'execution-error'
            except subprocess.TimeoutExpired:
                subprocess.run(['systemctl', '--user', 'kill', '--kill-whom=all', '--signal=KILL', unit],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                status = 'timeout'
            output.seek(0)
            captured = output.read(1024**2).decode(errors='replace')
        if 'bwrap:' in captured or 'Failed to connect to bus' in captured or 'Failed to start transient' in captured:
            raise RuntimeError('OS sandbox failed to initialize')
        passed = status == 'passed' and captured.rstrip().endswith(marker)
        return {'passed': passed, 'status': 'passed' if passed else status if status != 'passed' else 'early-exit'}
