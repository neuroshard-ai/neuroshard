import copy
import json
from pathlib import Path

import pytest

from neuroshard.evolution import modular_reference_execution as execution
from neuroshard.evolution.modular_reference import canonical_call, load_plan, score_reply
from neuroshard.evolution.modular_tools import parse_calls, tool_messages, validate_reply


ROOT = Path(__file__).resolve().parents[2]
PROFILE = "tool-interface"
PLAN, AMENDMENT = execution.PROFILES[PROFILE]


def contract():
    return load_plan(ROOT / PLAN)


def tool_tasks():
    return [task for task in contract()["tasks"] if task["kind"] == "tool"]


def registry(task):
    return json.loads(task["messages"][0]["functions"])


def test_freeze_changes_only_tool_system_messages_and_cache_preparation_budget():
    old = load_plan(ROOT / execution.PLAN)
    new = contract()
    amendment = execution.read(ROOT / AMENDMENT)
    assert execution.sha256(ROOT / PLAN) == amendment["plan_sha256"]
    assert execution.sha256(ROOT / execution.PLAN) == amendment["legacy_plan_sha256"]
    assert old["models"] == new["models"]
    assert {**old["limits"], "fetch_seconds": 600} == new["limits"]
    assert old["scoring"] == new["scoring"]
    assert amendment["allow_download"] is False
    assert amendment["gpu_launch_authorized"] is False
    assert amendment["model_order"] == ["baseline"]
    assert set(amendment["task_ids"]) == {task["id"] for task in tool_tasks()}
    for previous, task in zip(old["tasks"], new["tasks"]):
        if task["kind"] != "tool":
            assert task == previous
            continue
        assert task["messages"][1:] == previous["messages"][1:]
        assert registry(task) == registry(previous)
        assert task["expect"] == previous["expect"]
        # The adapter accepts no labels and produces the frozen prompt verbatim.
        assert task["messages"] == tool_messages(previous["messages"][1:], registry(previous))
        text = "<function_calls>" + canonical_call(task["expect"][0]) + "</function_calls>"
        assert parse_calls(text, registry(task)) == task["expect"]
        assert canonical_call(task["expect"][0]) not in task["messages"][0]["content"]


@pytest.mark.parametrize("body", [
    "add(12, 30)", "add(a=12)", "add(a=12, b=30, c=4)", "add(a=True, b=30)",
    'add(a="12", b=30)', "add(a=12, b=30, a=7)", "add(**{'a':12,'b':30})",
    "other(a=12, b=30)", "add(a=sum([1,2]), b=30)",
    "add(a=__import__('os').system('false'), b=30)", "[add(a=12, b=30)]",
])
def test_parser_rejects_unsafe_unknown_or_wrongly_typed_calls(body):
    task = next(task for task in tool_tasks() if task["id"] == "tool-add")
    assert validate_reply(f"<function_calls>{body}</function_calls>", registry(task))["valid"] is False


def test_parser_handles_qualified_names_as_data_and_never_executes(tmp_path):
    functions = [{"type": "function", "function": {
        "name": "tools.echo", "parameters": {"type": "object", "properties": {
            "text": {"type": "string"}}, "required": ["text"]}}}]
    marker = tmp_path / "must-not-exist"
    dangerous = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    # A string that looks like code remains a string; an expression is rejected.
    assert parse_calls(f"<function_calls>tools.echo(text={dangerous!r})</function_calls>", functions) == [
        {"name": "tools.echo", "arguments": {"text": dangerous}}]
    assert not validate_reply(f"<function_calls>tools.echo(text={dangerous})</function_calls>", functions)["valid"]
    assert not marker.exists()
    good = '<function_calls>tools.echo(text="hello")</function_calls>'
    assert not validate_reply("Here is the answer: " + good, functions)["valid"]
    assert not validate_reply(good + good, functions)["valid"]


def test_wire_validation_does_not_silently_relax_the_old_score():
    task = tool_tasks()[0]
    text = "Here is the call: <function_calls>" + canonical_call(task["expect"][0]) + "</function_calls>"
    assert score_reply(task, text, True)["passed"] is True  # Preserve the original scorer.
    assert validate_reply(text, registry(task))["valid"] is False  # Report stricter native validation separately.
    assert execution.usable_call({"passed": True, "wire_validation": {"valid": False}}) is False


def row(task, binding, phase, success=True):
    text = ("<function_calls>" + canonical_call(task["expect"][0]) + "</function_calls>"
            if success else "I cannot call a function.")
    return {"id": task["id"], "model": "baseline", "phase": phase, "category": "tool-use",
            "binding": execution.identity(binding), "task_sha256": execution.identity(task),
            "text": text, "terminated": True, "token_ids": [1, 2], "stopped": False,
            "seconds": 1, "max_rss_bytes": 1000, **score_reply(task, text, True),
            "wire_validation": validate_reply(text, registry(task))}


@pytest.mark.parametrize("successes,mismatch", [(0, False), (1, False), (3, False), (1, True)])
def test_diagnostic_replays_only_successful_calls_and_never_runs_modular_or_grants_credit(
        tmp_path, monkeypatch, successes, mismatch):
    root = tmp_path / "source"
    execution.save(root / PLAN, contract())
    amendment = execution.read(ROOT / AMENDMENT)
    legacy = tmp_path / "legacy.json"
    execution.save(legacy, {"which": "baseline", "plan_sha256": amendment["legacy_plan_sha256"], "seconds": 42})
    amendment["legacy_result_sha256"] = execution.sha256(legacy)
    execution.save(root / AMENDMENT, amendment)
    frozen = {"plan_sha256": amendment["plan_sha256"], "commit": "fixture", "profile": PROFILE}
    monkeypatch.setattr(execution, "ROOT", root)
    monkeypatch.setattr(execution, "freeze", lambda **kwargs: frozen)
    calls = []
    successful = set(amendment["task_ids"][:successes])

    def launch(home, models, binding, which, phase, seconds, memory_bytes, *, task=None, stats=None):
        assert which == "baseline"
        calls.append((phase, task["id"] if task else None))
        if phase == "prepare":
            return {"file_state": {}, "verified": True}
        return row(task, binding, phase, task["id"] in successful and not (mismatch and phase == "replay"))

    monkeypatch.setattr(execution, "launch", launch)
    result = execution.run(tmp_path / "study", tmp_path / "models", legacy, profile=PROFILE)
    assert not any(result[key] for key in ("quality_ready", "admission_evidence", "milestone_complete"))
    assert result["execution_completed"] is not mismatch
    assert result.get("interface_confirmed", False) is (successes > 0 and not mismatch)
    assert [task for phase, task in calls if phase == "primary"] == amendment["task_ids"]
    assert {task for phase, task in calls if phase == "replay"} == successful
    assert set(result["accounting"]) == {"baseline"}
    assert result["accounting"]["baseline"]["evaluation_seconds"] == 42


def test_native_validation_is_recomputed_not_trusted_from_receipt():
    task = tool_tasks()[0]
    binding = {"profile": PROFILE}
    answer = row(task, binding, "primary")
    execution.checked(contract(), answer, binding, "baseline", "primary", task)
    forged = copy.deepcopy(answer)
    forged["wire_validation"]["calls"][0]["arguments"]["city"] = "Elsewhere"
    with pytest.raises(ValueError, match="native tool validation"):
        execution.checked(contract(), forged, binding, "baseline", "primary", task)


def test_ci_wait_requires_successful_exact_push_and_never_accepts_another_head(tmp_path, monkeypatch):
    good = {"id": 42, "head_sha": "current", "event": "push", "name": "Native release checks",
            "status": "completed", "conclusion": "success"}
    pending = {**good, "status": "in_progress", "conclusion": None}
    snapshots = iter([[{**good, "head_sha": "old"}, pending], [good]])
    sleeps = []
    monkeypatch.setattr(execution.subprocess, "check_output", lambda *a, **kw: json.dumps({"workflow_runs": next(snapshots)}))
    monkeypatch.setattr(execution.time, "sleep", sleeps.append)
    assert execution.wait_for_ci(tmp_path / "good", "current")["id"] == 42
    assert sleeps == [30]
    monkeypatch.setattr(execution.subprocess, "check_output", lambda *a, **kw: json.dumps({
        "workflow_runs": [{**good, "conclusion": "failure"}]}))
    with pytest.raises(ValueError, match="did not pass CI"):
        execution.wait_for_ci(tmp_path / "bad", "current")
    with pytest.raises(TimeoutError, match="no inference"):
        execution.wait_for_ci(tmp_path / "timeout", "current", seconds=0)
