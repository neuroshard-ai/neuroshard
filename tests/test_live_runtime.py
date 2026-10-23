"""The live network's runtime (config/runtime/assistant-runtime-2.json): consistent with what operators install,
separate from the experiments' frozen runtime, and enforced by every replaying role."""
import json
from pathlib import Path

import pytest

from neuroshard.client import live_runtime
from neuroshard.evolution import granite_shard_execution
from neuroshard.evolution.modular_reference_execution import ROOT, read

SECURITY_FLOORS = {'certifi': '2026.10.19', 'transformers': '5.10.0', 'setuptools': '83.0.0'}


def requirement_pins(path):
    pins = {}
    for line in (ROOT / path).read_text().splitlines():
        if line.lstrip().startswith('#'):
            continue
        if ' @ ' in line:  # a direct wheel pinned by URL and hash: name @ .../name-VERSION-...whl#sha256=...
            name, url = line.split(' @ ', 1)
            wheel = url.split('#', 1)[0].rsplit('/', 1)[1].replace('%2B', '+')
            pins[name.strip().lower().replace('_', '-')] = wheel.split('-')[1]
            assert '#sha256=' in url, f'{name} is pinned by URL without a hash'
            continue
        line = line.split('#', 1)[0].strip()
        if '==' in line:
            name, version = line.split('==', 1)
            pins[name.strip().lower().replace('_', '-')] = version.strip()
    return pins


def normalized(name):
    return name.lower().replace('_', '-')


def test_operators_install_exactly_the_live_runtime():
    runtime, pins = live_runtime.pinned(), requirement_pins(live_runtime.REQUIREMENTS)
    for name, version in runtime['packages'].items():
        assert pins.get(normalized(name)) == version, (name, pins.get(normalized(name)), version)
    join = (ROOT / 'scripts' / 'join.sh').read_text()
    assert live_runtime.REQUIREMENTS in join and 'granite-shard-chain-requirements' not in join


def test_the_live_runtime_carries_the_security_fixes():
    from packaging.version import Version

    packages = live_runtime.pinned()['packages']
    for name, floor in SECURITY_FLOORS.items():
        assert Version(packages[name]) >= Version(floor), (name, packages[name], floor)


def test_only_the_reviewed_packages_differ_from_the_experiments_runtime():
    """Everything else, including torch and the environment, is the runtime the network already replays with."""
    live, frozen = live_runtime.pinned(), read(ROOT / granite_shard_execution.RUNTIME)
    changed = {name for name in frozen['packages'] if frozen['packages'][name] != live['packages'][name]}
    assert changed == set(live['changes']) == {'certifi', 'transformers', 'setuptools', 'fsspec'}
    assert set(live['packages']) == set(frozen['packages'])
    for key in ('python', 'required_cpu_flags', 'environment'):
        assert live[key] == frozen[key], key
    assert live['supersedes']['file'] == granite_shard_execution.RUNTIME


def test_the_experiments_runtime_is_never_repointed():
    """Historical contracts pin it by hash; the live runtime is a new file, not an edit."""
    assert granite_shard_execution.RUNTIME == 'config/experiments/assistant-workflow-canonical-execution.json'
    import hashlib
    digest = hashlib.sha256((ROOT / granite_shard_execution.RUNTIME).read_bytes()).hexdigest()
    assert digest == live_runtime.pinned()['supersedes']['sha256']


def test_a_differing_install_is_refused_with_every_difference_named(monkeypatch):
    real = live_runtime.pinned()
    fake = json.loads(json.dumps(real))
    fake['packages']['transformers'] = '0.0.1'
    monkeypatch.setattr(live_runtime, 'pinned', lambda: fake)
    with pytest.raises(ValueError) as refused:
        live_runtime.check()
    message = str(refused.value)
    assert 'transformers' in message and '(needs 0.0.1)' in message and live_runtime.REQUIREMENTS in message


def test_configure_refuses_after_torch_is_loaded(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'torch', object())
    with pytest.raises(ValueError, match='before importing torch'):
        live_runtime.configure()


def test_the_live_roles_use_the_live_runtime():
    cli = (ROOT / 'src/neuroshard/client/assistant_cli.py').read_text()
    app = (ROOT / 'src/neuroshard/inference/optimistic_app.py').read_text()
    assert 'live_runtime.configure()' in cli and 'live_runtime.check()' in cli
    assert 'assistant_growth_cohort3_eval.runtime(read(ROOT / runtime.RUNTIME))' not in cli
    assert 'live_runtime.configure()' in app
