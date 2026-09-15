"""A selection for another graph cannot authorize final evaluation."""
import json
from types import SimpleNamespace

import pytest

from neuroshard.evolution import reference_data as data
from neuroshard.evolution.sharded import branch_job


def test_final_rejects_another_graph_before_loading_any_model(monkeypatch):
    plan = {'fixed': 'plan'}
    prepared = {'graph': {'parent': 'old', 'expert': 'learned'}}
    selection = {'eligible': True, 'plan': data.identity(plan), 'prepared': data.identity(prepared),
                 'graph': data.identity({'parent': 'old', 'expert': 'different'})}
    monkeypatch.setattr(branch_job.contract, 'validate', lambda: (plan, prepared))
    monkeypatch.setattr(branch_job.base, 'committed', lambda _: json.dumps(selection).encode())
    with pytest.raises(ValueError, match='committed eligible branch'):
        branch_job.run(SimpleNamespace(command='final'))
