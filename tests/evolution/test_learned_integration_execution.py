import json
from pathlib import Path

import pytest

from neuroshard.evolution.learned_integration import bind_spec, load_spec, method_freeze
from neuroshard.evolution.learned_integration_execution import (
    EXECUTION_FORMAT, GENERAL_RETENTION, HOST, bind_execution, execution_freeze,
    general_retention, load_method,
)
from neuroshard.evolution.programming_expert import GENERAL_SHA
from neuroshard.evolution.reference_data import identity
from neuroshard.evolution.seed import FILES as SEED_FILES


ROOT = Path(__file__).resolve().parents[2]


def documents():
    spec = json.loads((ROOT / 'config/experiments/learned-integration.json').read_text())
    method = json.loads((ROOT / 'config/experiments/learned-integration-method.json').read_text())
    execution = json.loads((ROOT / 'config/experiments/learned-integration-execution.json').read_text())
    return spec, method, execution


def test_execution_freeze_is_pinned():
    spec, method, execution = documents()
    assert identity(spec) == identity(load_spec())
    assert method == method_freeze()
    assert method == load_method()
    assert execution == execution_freeze()
    assert identity(execution) == '1f838d4d754339b4eb231236fee2939fd4488bcf9b50777602ac5f351ea80519'
    assert execution['format'] == EXECUTION_FORMAT
    assert bind_execution(spec, method, execution)['host'] == HOST


def test_execution_freeze_is_cpu_and_closes_confirmation():
    spec, method, execution = documents()
    assert execution['host'] == 'cpu'
    assert execution['train'] is True
    assert execution['gpu_launch_authorized'] is False
    assert execution['confirmation_opened'] is False
    assert execution['confirmation_scored'] is False
    assert execution['admission_evidence'] is False
    assert execution['distributed_runtime'] is False
    assert execution['seed']['files'] == dict(SEED_FILES)
    assert execution['seed']['revision'] == '12fd25f77366fa6b3b4b768ec3050bf629380bac'


def test_eight_general_identities_are_recorded_not_scored():
    rows = general_retention()
    assert len(rows) == 8
    assert [item['row'] for item in GENERAL_RETENTION] == [
        13712, 10926, 17751, 6645, 4043, 4233, 20506, 20999]
    groups = [item['group'] for item in rows]
    assert groups.count('conversation') == 2
    assert groups.count('constraints') == 2
    assert groups.count('summary') == 2
    assert groups.count('rewrite') == 2
    for item in rows:
        assert item['id'] == identity({'dataset': GENERAL_SHA, 'row': item['row']})
    spec, method, execution = documents()
    assert execution['general_retention']['scored'] is False
    assert execution['general_retention']['identities_recorded'] is True
    assert execution['general_retention']['rows'] == rows


def test_bind_rejects_gpu_and_open_confirmation(monkeypatch):
    spec, method, execution = documents()
    from neuroshard.evolution import learned_integration_execution as module

    broken = dict(execution)
    broken['gpu_launch_authorized'] = True
    monkeypatch.setattr(module, 'execution_freeze', lambda: broken)
    with pytest.raises(ValueError, match='does not authorize a GPU launch'):
        bind_execution(spec, method, broken)

    broken = dict(execution)
    broken['confirmation_opened'] = True
    monkeypatch.setattr(module, 'execution_freeze', lambda: broken)
    with pytest.raises(ValueError, match='Confirmation remains closed'):
        bind_execution(spec, method, broken)


def test_research_contract_still_does_not_launch():
    spec, method, execution = documents()
    bind_spec(spec)
    assert spec['train'] is False
    assert spec['gpu_launch_authorized'] is False
    assert execution['method'] == identity(method)
