"""The numerical runtime of the live network roles, and the checks they apply before torch loads.

The live roles are the owner (``assistant host``), auditor, validator, seed, growth agent and client. They replay
one another's work bit for bit, so they must all run the same pinned runtime. That runtime is
``config/runtime/assistant-runtime-2.json`` (from pilot-2-rc2), installed from
``docs/assistant-runtime-2-requirements.txt``.

**Experiments are unaffected.** Experiments, release checks and every historical execution contract keep the runtime
they ran with: ``granite_shard_execution.RUNTIME`` (``assistant-workflow-canonical-execution.json``). They are never
re-pointed, so their recorded hashes stay valid.

**Qualification.** A new live runtime is qualified before release by replaying committed live owner logs and
committed growth steps under it (``scripts/replay_owner_logs.py``, ``scripts/replay_growth_steps.py``). It is
accepted only if every replay is bit-identical, because an auditor or validator on a runtime that computes
differently would judge honest owners and contributors wrong.
"""
import importlib.metadata
import os
import platform
import sys
from pathlib import Path

from neuroshard.evolution.modular_reference_execution import ROOT, read

LIVE_RUNTIME = 'config/runtime/assistant-runtime-2.json'
REQUIREMENTS = 'docs/assistant-runtime-2-requirements.txt'


def pinned():
    return read(ROOT / LIVE_RUNTIME)


def configure():
    """Apply the live runtime's numerical environment; must run before torch is imported."""
    if 'torch' in sys.modules:
        raise ValueError('configure the live runtime before importing torch')
    for key, value in pinned()['environment'].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def check():
    """The installed packages, Python, CPU and environment equal the live runtime's, exactly.

    An owner, auditor or validator that differs could compute different numbers from its peers. It would then judge
    honest work fraudulent or be judged so itself, so it refuses to start instead. The error names every
    difference and the requirements file to install."""
    runtime = pinned()
    differing = {}
    for name, version in runtime['packages'].items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != version:
            differing[name] = (installed, version)
    problems = [f'{name} {have or "missing"} (needs {want})' for name, (have, want) in sorted(differing.items())]
    if platform.python_version() != runtime['python']:
        problems.append(f"Python {platform.python_version()} (needs {runtime['python']})")
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        problems.append('this role requires Linux x86_64')
    cpu = Path('/proc/cpuinfo').read_text().split() if Path('/proc/cpuinfo').exists() else []
    problems += [f'CPU lacks {flag}' for flag in runtime['required_cpu_flags'] if flag not in cpu]
    problems += [f'{key}={os.environ.get(key)!r} (needs {value!r})' for key, value in runtime['environment'].items()
                 if os.environ.get(key) != value]
    if problems:
        raise ValueError(f'this machine differs from the live runtime {runtime["name"]}: ' + '; '.join(problems)
                         + f'. Install it with: pip install -r {REQUIREMENTS} (scripts/join.sh does)')
    return {'runtime': runtime['name'], 'python': platform.python_version()}
