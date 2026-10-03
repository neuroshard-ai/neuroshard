"""Regression and accounting checks for the interrupted fresh-reference study."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from neuroshard.evolution import modular_reference_execution as execution
from neuroshard.evolution.modular_reference import canonical_call, load_plan, score_reply
from neuroshard.evolution.modular_tools import validate_reply


ROOT = Path(__file__).resolve().parents[2]
PROFILE = "fresh-reference-recovery"
PLAN, AMENDMENT = execution.PROFILES[PROFILE]
RECEIPTS = "config/experiments/modular-reference-fresh-interrupted-receipts.json"


def test_malformed_reply_has_identical_rejection_in_independent_processes():
    # Use the actual crashing output, not a fabricated successful call.
    code = """
import json
from neuroshard.evolution import modular_reference_execution as execution
from neuroshard.evolution.modular_reference import load_plan, score_reply
from neuroshard.evolution.modular_tools import validate_reply
plan = load_plan(execution.ROOT / execution.PROFILES['fresh-reference-recovery'][0])
task = next(t for t in plan['tasks'] if t['id'] == 'fresh-tool-documents')
receipts = execution.read(execution.ROOT / 'config/experiments/modular-reference-fresh-interrupted-receipts.json')
old = next(a['reply'] for a in receipts['attempts'] if a['name'] == 'baseline-primary-fresh-tool-documents')
print(json.dumps({'score': score_reply(task, old['text'], old['terminated']),
                 'wire': validate_reply(old['text'], json.loads(task['messages'][0]['functions']))}))
"""
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    outcomes = [json.loads(subprocess.check_output(
        [sys.executable, "-c", code], cwd=ROOT, env=environment, text=True)) for _ in range(3)]
    assert all(outcome == outcomes[0] for outcome in outcomes)
    assert outcomes[0]["score"] == {"passed": False, "reason": "argument must be a Python literal"}
    assert outcomes[0]["wire"] == {"valid": False, "error": "argument must be a Python literal"}


def test_recovery_does_not_change_any_recorded_pass_fail_or_accept_lowercase_boolean():
    contract = load_plan(ROOT / PLAN)
    tasks = {t["id"]: t for t in contract["tasks"]}
    receipts = execution.read(ROOT / RECEIPTS)
    generated = [a["reply"] for a in receipts["attempts"] if a["request"]["phase"] == "primary"]
    assert len(generated) == 9
    for row in generated:
        task = tasks[row["id"]]
        assert score_reply(task, row["text"], row["terminated"])["passed"] == row["passed"]
    task = tasks["fresh-tool-documents"]
    valid = "<function_calls>" + canonical_call(task["expect"][0]) + "</function_calls>"
    assert "False" in valid and score_reply(task, valid, True)["passed"]
    assert not score_reply(task, valid.replace("False", "false"), True)["passed"]


def test_recovery_preserves_plan_gates_receipts_and_remaining_budget():
    old = execution.read(ROOT / execution.PROFILES["fresh-reference"][1])
    recovery = execution.read(ROOT / AMENDMENT)
    assert recovery["plan_sha256"] == old["plan_sha256"] == execution.sha256(ROOT / PLAN)
    for key in ("artifacts_sha256", "packages", "python", "environment", "task_ids",
                "model_order", "required_cpu_flags", "new_evaluation_seconds"):
        assert recovery[key] == old[key]
    assert not recovery["method_changed"] and not recovery["gpu_launch_authorized"]
    assert not recovery["recovery"]["reuse_replies"] and not recovery["recovery"]["automatic_retry"]
    prior = recovery["additional_evaluations"][-1]
    assert execution.sha256(ROOT / prior["path"]) == prior["sha256"]
    result = execution.read(ROOT / prior["path"])
    assert not result["execution_completed"]
    assert recovery["remaining_evaluation_seconds"]["baseline"] + result["accounting"]["baseline"]["new_evaluation_seconds"] == pytest.approx(7200)
    receipts = execution.read(ROOT / RECEIPTS)
    for phase, field in (("prepare", "preparation_seconds"), ("primary", "new_evaluation_seconds")):
        total = sum(a["outcome"]["seconds"] for a in receipts["attempts"] if a["request"]["phase"] == phase)
        assert total == pytest.approx(result["accounting"]["baseline"][field])


@pytest.mark.parametrize("exhaust_budget", [False, True])
def test_recovery_charges_interrupted_work_and_replays_wrong_answers(tmp_path, monkeypatch, exhaust_budget):
    root = tmp_path / "source"
    amendment = execution.read(ROOT / AMENDMENT)
    contract = load_plan(ROOT / PLAN)
    if exhaust_budget:
        amendment["remaining_evaluation_seconds"]["baseline"] = .5
    execution.save(root / PLAN, contract)
    execution.save(root / AMENDMENT, amendment)
    for prior in amendment["additional_evaluations"]:
        dest = root / prior["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / prior["path"]).read_bytes())
    monkeypatch.setattr(execution, "ROOT", root)
    monkeypatch.setattr(execution, "freeze", lambda **kw: {"commit": "fixture", "plan_sha256": amendment["plan_sha256"]})
    calls = []

    def launch(home, models, binding, which, phase, seconds, memory_bytes, *, task=None, stats=None):
        calls.append((which, phase, task["id"] if task else None, seconds))
        attempt = home / "attempts" / f"{which}-{phase}-{task['id'] if task else 'prepare'}"
        execution.save(attempt / "request.json", {"phase": phase, "seconds": seconds})
        execution.save(attempt / "outcome.json", {"seconds": .25, "completed": True})
        if phase == "prepare":
            return {"file_state": {}}
        text = (task["accept"][0] if task["kind"] == "exact" else
                "<function_calls>" + canonical_call(task["expect"][0]) + "</function_calls>")
        if task["id"] == "fresh-tool-documents":
            text = text.replace("False", "false")
        row = {"model": which, "phase": phase, "id": task["id"], "category": task["category"],
               "binding": execution.identity(binding), "task_sha256": execution.identity(task),
               "text": text, "terminated": True, "token_ids": [1, 2], "stopped": False,
               "seconds": .1, "first_token_seconds": .05, "max_rss_bytes": 1000,
               **score_reply(task, text, True)}
        if task["kind"] == "tool":
            row["wire_validation"] = validate_reply(text, json.loads(task["messages"][0]["functions"]))
        return row

    monkeypatch.setattr(execution, "launch", launch)
    result = execution.run(tmp_path / "study", tmp_path / "models",
                           ROOT / "config/experiments/modular-reference-a1-legacy-baseline-result.json", PROFILE)
    accounting = result["accounting"]["baseline"]
    assert accounting["historical_evaluation_seconds"] == pytest.approx(11587.957666832835)
    assert accounting["historical_preparation_seconds"] == pytest.approx(143.48857155)
    assert accounting["preparation_seconds"] == pytest.approx(143.73857155)
    assert calls[0][3] == pytest.approx(3600 - 143.48857155)
    if exhaust_budget:
        assert not result["execution_completed"]
        assert "evaluation budget exhausted" in result["error"]
        assert not any(which == "modular" for which, _, _, _ in calls)
        assert accounting["new_evaluation_seconds"] == .5
        assert [seconds for _, phase, _, seconds in calls if phase == "primary"] == [.5, .25]
    else:
        assert result["execution_completed"] and result["reference_ready"]
        assert not result["growth_screen_passed"] and not result["admission_evidence"]
        assert len(result["replays"]) == 48
        wrong = [r for r in result["primary"] if r["id"] == "fresh-tool-documents"]
        assert len(wrong) == 2 and all(not r["passed"] for r in wrong)
        assert accounting["new_evaluation_seconds"] == 12
        # All work, including repeated wrong answers, remains in cumulative cost.
        assert accounting["evaluation_seconds"] == pytest.approx(11599.957666832835)


def test_recovery_checks_native_validation_for_wrong_replies():
    contract = load_plan(ROOT / PLAN)
    task = next(t for t in contract["tasks"] if t["id"] == "fresh-tool-documents")
    receipts = execution.read(ROOT / RECEIPTS)
    row = copy.deepcopy(next(a["reply"] for a in receipts["attempts"] if a["reply"].get("id") == task["id"]))
    binding = {"profile": PROFILE}
    row.update(binding=execution.identity(binding), **score_reply(task, row["text"], row["terminated"]))
    row["wire_validation"] = validate_reply(row["text"], json.loads(task["messages"][0]["functions"]))
    execution.checked(contract, row, binding, "baseline", "primary", task)
    row["wire_validation"] = {"valid": True, "calls": []}
    with pytest.raises(ValueError, match="native tool validation"):
        execution.checked(contract, row, binding, "baseline", "primary", task)
