import copy

import pytest

from neuroshard.evolution import block_expert_measure as measure
from neuroshard.evolution import block_expert_measure_run as runner
from neuroshard.evolution.block_expert import excluded_pairs as earlier_pairs


def test_fresh_pairs_exclude_the_stopped_study():
    plan = measure.load_plan()
    data = measure.load_data(plan)
    pairs = [(row["a"], row["b"]) for rows in data["roles"].values() for row in rows]
    assert len(pairs) == len(set(pairs)) == 1280
    assert not set(pairs) & measure.excluded_pairs()
    assert not set(pairs) & earlier_pairs()
    assert plan["gates"]["retention_blocks_training"] is False
    assert plan["parent_stop"]["retention_correct"] == 6
    assert not plan["gpu_launch_authorized"] and not plan["selector_training_authorized"]


def receipt(data, new_correct, retained_correct, *, unfinished=0):
    result = {"peak_rss_bytes": 1000}
    for role, correct in (("development", new_correct), ("retention", retained_correct)):
        result[role] = []
        for index, truth in enumerate(data["roles"][role]):
            unfinished_row = index < unfinished
            value = truth["answer"] if index < correct and not unfinished_row else "999"
            terminated = not unfinished_row
            result[role].append({"id": truth["id"], "answer": truth["answer"], "text": value,
                                 "parsed_answer": None if unfinished_row else value,
                                 "passed": (not unfinished_row) and value == truth["answer"],
                                 "terminated": terminated, "seconds": 1.0, "selection": "declared-arm"})
    return result


def test_low_parent_retention_does_not_block_a_competent_expert():
    plan = measure.load_plan()
    data = measure.load_data(plan)
    parent, expert, control = receipt(data, 0, 6), receipt(data, 32, 2), receipt(data, 10, 1)
    training = {"frozen_unchanged": True}
    spent = {"matched_budget": True, "optimization_cpu_seconds": 100}
    result = measure.score(plan, data, parent, expert, control, training, spent, 100)
    assert result["passed"]
    assert result["parent_retention_correct"] == 6
    assert result["retention_blocks_training"] is False
    assert not result["selector_training_authorized"] and not result["item4_complete"]


def test_unfinished_reply_is_incorrect_and_weak_expert_fails():
    plan = measure.load_plan()
    data = measure.load_data(plan)
    parent = receipt(data, 0, 6, unfinished=64)
    expert = receipt(data, 0, 6)
    control = receipt(data, 2, 1)
    training = {"frozen_unchanged": True}
    spent = {"matched_budget": True, "optimization_cpu_seconds": 100}
    result = measure.score(plan, data, parent, expert, control, training, spent, 100)
    assert not result["passed"]
    assert result["incomplete"]["parent"]["development"] == 64
    bad = copy.deepcopy(expert)
    bad["development"][0]["passed"] = True
    bad["development"][0]["terminated"] = False
    bad["development"][0]["parsed_answer"] = None
    with pytest.raises(ValueError, match="complete generated answer"):
        measure.score(plan, data, parent, bad, control, training, spent, 100)


def test_training_continues_after_a_small_protected_baseline(tmp_path, monkeypatch):
    data = measure.load_data(measure.load_plan())
    calls = []

    def isolated(arm, *args):
        calls.append(arm)
        if arm == "baseline":
            return {**receipt(data, 0, 6), "binding": {}, "peak_rss_bytes": 1000}
        raise RuntimeError("stop after proving training was reached")

    monkeypatch.setattr(measure, "bind_freeze", lambda **kwargs: "test")
    monkeypatch.setattr(runner, "verify", lambda path: None)
    monkeypatch.setattr(runner.runner, "isolated", isolated)
    with pytest.raises(RuntimeError, match="stop after proving"):
        runner.run(tmp_path, tmp_path / "study")
    assert calls == ["baseline", "prepare"]
    saved = (tmp_path / "study" / "protected-before-training.json").read_text()
    assert '"blocks_training":false' in saved


def test_freeze_binds_committed_sources():
    assert measure.bind_freeze() == measure.identity(measure.inventory())
