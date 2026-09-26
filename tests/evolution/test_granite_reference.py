import importlib.util
import json

import pytest

from neuroshard.evolution import granite_reference as study


def plan():
    return study.read(study.ROOT / study.PLAN)


def answer(task):
    if task["kind"] == "exact":
        return task["accept"][0]
    text = json.dumps(task["expected"])
    return "<tool_call>" + text + "</tool_call>" if task["kind"] == "tool" else text


def rows():
    return [{"id": t["id"], "text": answer(t), "terminated": True, "passed": True, "seconds": 1}
            for t in plan()["tasks"] + plan()["reference_tasks"]]


def cloud_module():
    spec = importlib.util.spec_from_file_location("granite_cloud", study.ROOT / "scripts/modular_reference_cloud.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_tool_format_is_strict_and_json_types_cannot_fake_correctness():
    task = next(t for t in plan()["tasks"] if t["id"] == "granite-tool-archive")
    correct = answer(task)
    assert study.score(task, correct, True)
    for wrong in ("Here you go: " + correct, correct + correct,
                  correct.replace('false', '0'), correct.replace('"limit": 6', '"limit": 6, "limit": 6'),
                  correct.replace('"limit": 6', '"limit": NaN'), correct.replace('"limit": 6', '"limit": 1e999'),
                  correct.replace('"list_messages"', '"delete_messages"')):
        assert not study.score(task, wrong, True)
    assert not study.score(task, correct, False)


def test_every_task_has_an_executable_scoring_contract_without_old_inputs():
    contract = plan()
    old = study.read(study.ROOT / "config/experiments/modular-reference-fresh.json")
    old_messages = {study.identity(t["messages"]) for t in old["tasks"]}
    assert len(contract["tasks"]) == 24 and len(contract["reference_tasks"]) == 16
    assert len({t["id"] for t in contract["tasks"] + contract["reference_tasks"]}) == 40
    for task in contract["tasks"] + contract["reference_tasks"]:
        assert study.score(task, answer(task), True)
        assert not study.score(task, answer(task), False)
        assert study.identity(task["messages"]) not in old_messages
    assert not contract["training_authorized"] and not contract["gpu_launch_authorized"]


def test_more_reference_successes_cannot_hide_losing_an_assistant_answer():
    baseline = rows()
    modular = rows()
    assert study.assess(plan(), baseline, modular)["quality_gate"]
    modular[0].update(text="wrong", passed=False)
    report = study.assess(plan(), baseline, modular)
    assert report["lost_ids"] == [modular[0]["id"]]
    assert not report["quality_gate"]
    assert not report["automatic_composition_proven"] and not report["checklist_credit"]


def test_category_balanced_reference_and_latency_gates_cannot_be_bypassed():
    contract = plan()
    baseline = rows()
    for row, task in zip(baseline, contract["tasks"] + contract["reference_tasks"]):
        if task["category"] == "tool-use":
            row.update(text="wrong", passed=False)
    assert not study.assess(contract, baseline, rows())["baseline_gate"]
    modular = rows()
    for row, task in zip(modular, contract["tasks"] + contract["reference_tasks"]):
        if task["category"] == "reference" and task["expected"]["score"] == "no":
            row.update(text='{"score":"yes"}', passed=False)
    assert not study.assess(contract, rows(), modular)["reference_gate"]
    modular = rows()
    for row in modular[-5:]:
        row["seconds"] = 121
    assert not study.assess(contract, rows(), modular)["quality_gate"]
    assert not study.assess(contract, rows(), rows()[:-1])["complete"]
    with pytest.raises(ValueError, match="duplicate"):
        study.assess(contract, rows(), rows() + rows()[:1])
    forged = rows()
    forged[0]["text"] = "wrong"
    with pytest.raises(ValueError, match="rescore"):
        study.assess(contract, rows(), forged)


def test_freeze_pins_models_upstream_resources_and_runtime():
    execution = study.read(study.ROOT / study.EXECUTION)
    for path, digest in execution["contracts"].items():
        assert study.sha256(study.ROOT / path) == digest
    assert study.SCRIPT in execution["sources"]
    artifacts = study.read(study.ROOT / study.ARTIFACTS)
    assert artifacts["upstream"]["commit"] == "60d546d211907bf42934113a3e69e9a302af9868"
    assert len(artifacts["upstream"]["sources"]) > 10
    assert all(len(m["revision"]) == 40 for m in artifacts["models"].values())
    assert execution["packages"]["torch"] == "2.10.0+cpu"
    assert execution["packages"]["transformers"] == "5.5.4"
    assert study.read(study.ROOT / "config/experiments/modular-decoder-parity-execution.json")["packages"]["torch"] == "2.9.1+cpu"


def test_cloud_uses_isolated_profile_and_enforces_budget(monkeypatch):
    cloud = cloud_module()
    resource = cloud.resources("granite-reference")
    assert resource["hours"] == 2 and not resource["gpu"]
    assert "run_granite_reference.py" in " ".join(cloud.remote_command("granite-reference"))
    original = cloud.read
    path = cloud.ROOT / cloud.RESOURCE_PROFILES["granite-reference"]
    for changes in ({"hours": 3}, {"planning_cap_usd": 7}, {"instances": 2}, {"attempts": 2}):
        monkeypatch.setattr(cloud, "read", lambda p: {**resource, **changes} if p == path else original(p))
        with pytest.raises(ValueError, match="allowance"):
            cloud.resources("granite-reference")


def test_uncommitted_source_cannot_launch(tmp_path):
    import subprocess
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Fixture"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "fixture@example.invalid"], check=True)
    source = tmp_path / "worker.py"
    source.write_text("original\n")
    execution = {"contracts": {}, "sources": ["worker.py", study.EXECUTION]}
    study.save(tmp_path / study.EXECUTION, execution)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "fixture"], check=True)
    assert study.committed_sources(tmp_path)["sources"]
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="uncommitted"):
        study.committed_sources(tmp_path)


@pytest.mark.parametrize("usable", [True, False])
def test_worker_stops_before_modular_download_on_bad_parent_and_reloads_only_after_pass(
        tmp_path, monkeypatch, usable):
    import torch
    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: None)
    monkeypatch.setattr(study, "configure", lambda: None)
    monkeypatch.setattr(study, "freeze", lambda: {"fixture": True})
    preparations, loads = [], []

    def prepare(path, inventory, download):
        preparations.append(path.name)
        return {}

    def load(path, which):
        loads.append(which)
        return None, None

    def generate(model, tokenizer, contract, task, which):
        text = answer(task) if usable or task["category"] != "tool-use" else "wrong"
        return {"id": task["id"], "text": text, "terminated": True,
                "passed": study.score(task, text, True), "seconds": 1,
                "prompt_sha256": task["id"], "token_ids": [1, 2], "route_trace": []}

    monkeypatch.setattr(study, "verify_artifacts", prepare)
    monkeypatch.setattr(study, "file_state", lambda *args: {})
    monkeypatch.setattr(study, "load_model", load)
    monkeypatch.setattr(study, "generate", generate)
    request = tmp_path / "attempts/pair-comparison/request.json"
    study.save(request, {"freeze": {"fixture": True}, "binding": "fixture", "models": str(tmp_path / "models")})
    study.worker(request)
    result = study.read(request.parent / "reply.json")
    assert result["execution_completed"]
    assert result["reference_passed"] is usable
    assert preparations == (["baseline", "modular"] if usable else ["baseline"])
    assert loads == (["baseline", "modular", "baseline", "modular"] if usable else ["baseline"])
    assert len(result["replays"]) == (6 if usable else 0)
    assert not result["checklist_credit"]
