import json
from collections import Counter
from pathlib import Path

import pytest

from neuroshard.evolution.observable_selection import CONTRACT_IDENTITY, choose, load_spec
from neuroshard.evolution.reference_data import identity


ROOT = Path(__file__).resolve().parents[2]


def test_contract_refuses_training_and_the_hidden_answer():
    spec = load_spec()
    assert identity(spec) == CONTRACT_IDENTITY
    assert spec["train"] is False
    assert spec["later_execution_authorized"] is False
    with pytest.raises(ValueError, match="hidden answer"):
        choose({"question_text": "q", "parent_generation": "1", "added_generation": "6", "passed": True})


def test_choice_uses_only_observable_scores():
    base = {"question_text": "q", "parent_generation": "1", "added_generation": "6"}
    assert choose({**base, "model_internal_scores": {"parent": 2, "added": 1}}) == "parent"
    assert choose({**base, "model_internal_scores": {"parent": 1, "added": 2}}) == "added"


def test_recorded_expert_ties_the_training_label_constant():
    record = json.loads((ROOT / "config/experiments/append-only-execution-result.json").read_text())
    data = json.loads((ROOT / "config/experiments/append-only-execution-data.json").read_text())
    mode, _ = Counter(row["answer"] for row in data["roles"]["train_new"]).most_common(1)[0]
    constant = sum(row["answer"] == mode for row in data["roles"]["development"])
    assert record["deployable_assistant"] is False
    assert record["selector_reads_known_answer"] is True
    assert record["development"]["expert_correct"] == constant == 9
    assert record["development"]["constant_training_label"] == mode
