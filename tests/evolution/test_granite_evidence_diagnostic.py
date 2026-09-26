import copy
import importlib.util
import json

import pytest

from neuroshard.evolution import granite_evidence_diagnostic as study
from test_evidence_selection import Tokenizer


def plan():
    return study.read(study.ROOT / study.PLAN)


def opened():
    return study.read(study.ROOT / plan()["opened_plan"])


def retriever():
    value = opened()
    return study.context.Retriever(value["corpus"], value["retrieval"])


def make_row(task, arm, correct=True):
    messages, sources, menu = study.prepare_request(task, retriever())
    if arm == "selection":
        selected = next((c for c in menu["choices"] if c["source_id"] in task["expected"]["sources"]), None)
        choice = selected["choice"] if selected else "Z"
        if not correct:
            choice = "Z" if selected else menu["choices"][0]["choice"]
        receipt = study.evidence.resolve(messages, sources, {"invocation_root": menu["invocation_root"], "choice": choice})
        text = json.dumps(receipt["answer"], ensure_ascii=False, sort_keys=True)
        generated_text, terminated = choice, False
    else:
        receipt = None
        text = json.dumps(task["expected"] if correct else {"answer": "wrong", "sources": []})
        generated_text, terminated = text, True
    generated = {"text": generated_text, "terminated": terminated, "token_ids": [1],
                 "input_token_ids": [2], "prompt_sha256": "fixture", "route_trace": []}
    return {"id": task["id"], "arm": arm, "task_sha256": study.identity(task), "menu": menu,
            "generated": generated, "receipt": receipt, "text": text, "completed": True,
            "seconds": 1, "input_tokens": 1, "output_tokens": 1, "generation_calls": 1, "passed": correct}


def fixtures():
    rows = {arm: [make_row(task, arm) for task in opened()["tasks"]] for arm in study.ARMS}
    anchors = []
    for task in study.read(study.ROOT / study.reference.PLAN)["tasks"]:
        text = task["accept"][0] if task["kind"] == "exact" else json.dumps(task["expected"])
        if task["kind"] == "tool":
            text = "<tool_call>" + text + "</tool_call>"
        anchors.append({"id": task["id"], "text": text, "terminated": True, "passed": True, "seconds": 1})
    return rows, anchors


def test_both_controls_get_same_public_data_no_labels_or_oracle():
    task = opened()["tasks"][0]
    messages, sources, menu = study.prepare_request(task, retriever())
    poisoned = {**task, "id": "LEAK", "category": "LEAK", "block": "LEAK", "oracle_query": "LEAK",
                "expected": {"answer": "LEAK", "sources": []}}
    assert study.prepare_request(poisoned, retriever()) == (messages, sources, menu)
    direct = study.model_messages(plan(), messages, menu, "direct")
    selection = study.model_messages(plan(), messages, menu, "selection")
    assert direct[0] == selection[0]
    assert direct[1]["content"].split("\nTask:\n")[0] == selection[1]["content"].split("\nTask:\n")[0]
    assert "LEAK" not in json.dumps(direct + selection)


def test_execute_enforces_one_decision_and_resolves_exact_source(monkeypatch):
    task = opened()["tasks"][0]
    messages, sources, menu = study.prepare_request(task, retriever())
    selected = next(c for c in menu["choices"] if c["source_id"] in task["expected"]["sources"])
    calls = []

    def generate(model, tokenizer, settings, prompt, which, generation_kwargs):
        calls.append(settings["generation"]["max_new_tokens"])
        allowed = generation_kwargs["prefix_allowed_tokens_fn"](0, None)
        assert ord(selected["choice"]) in allowed and len(allowed) == 4
        return {"text": selected["choice"], "token_ids": [ord(selected["choice"])],
                "input_token_ids": [1, 2, 3], "terminated": False}

    monkeypatch.setattr(study.reference, "generate", generate)
    result = study.execute(None, Tokenizer(), plan(), retriever(), task, "selection")
    assert calls == [1] and result["completed"] and result["passed"]
    assert result["receipt"]["answer"] == task["expected"]
    assert result["output_tokens"] == 1 and result["generation_calls"] == 1


def test_semantically_wrong_but_well_formed_choice_is_a_quality_failure():
    rows, anchors = fixtures()
    report = study.assess(plan(), rows, anchors)
    assert report["diagnostic_gate"]
    assert not report["admission_evidence"] and not report["checklist_credit"]
    assert not report["new_capability_proven"]
    rows["selection"][0] = make_row(opened()["tasks"][0], "selection", False)
    report = study.assess(plan(), rows, anchors)
    assert not report["diagnostic_gate"] and report["lost_direct_ids"]


def test_receipt_tampering_missing_rows_latency_and_retention_fail():
    rows, anchors = fixtures()
    rows["selection"][0]["receipt"]["answer"]["answer"] = "forged"
    with pytest.raises(ValueError, match="receipt"):
        study.assess(plan(), rows, anchors)
    rows, anchors = fixtures()
    rows["direct"].pop()
    assert not study.assess(plan(), rows, anchors)["complete"]
    rows, anchors = fixtures()
    for row in rows["selection"]:
        row["seconds"] = 2
    assert not study.assess(plan(), rows, anchors)["diagnostic_gate"]
    rows, anchors = fixtures()
    protected = plan()["protected_ids"][0]
    next(row for row in anchors if row["id"] == protected).update(text="wrong", passed=False)
    assert not study.assess(plan(), rows, anchors)["retention"]


def test_declared_contract_pins_unchanged_opened_cases_and_cpu_only_budget():
    execution = study.read(study.ROOT / study.EXECUTION)
    for path, expected in execution["contracts"].items():
        assert study.sha256(study.ROOT / path) == expected
    assert not execution["training_authorized"] and not execution["gpu_launch_authorized"]
    for key in ("admission_evidence", "checklist_credit", "new_capability_proven"):
        assert not plan()[key]
    assert len(opened()["tasks"]) == 64
    spec = importlib.util.spec_from_file_location("evidence_cloud", study.ROOT / "scripts/modular_reference_cloud.py")
    cloud = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cloud)
    limits = cloud.resources("granite-evidence-diagnostic")
    assert limits["hours"] == 2 and limits["planning_cap_usd"] == 6 and limits["attempts"] == 1
    assert "run_granite_evidence_diagnostic.py" in " ".join(cloud.remote_command("granite-evidence-diagnostic"))


@pytest.mark.parametrize("passes", [True, False])
def test_worker_downloads_only_parent_and_replays_only_on_success(tmp_path, monkeypatch, passes):
    import torch
    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: None)
    monkeypatch.setattr(study, "configure", lambda: None)
    monkeypatch.setattr(study, "freeze", lambda: {"fixture": True})
    rows, anchors = fixtures()
    if not passes:
        rows["selection"][0] = make_row(opened()["tasks"][0], "selection", False)
    loads, downloads, calls = [], [], []

    def download(path, inventory, download):
        downloads.append(path.name)
        return {}

    def load(path, which):
        loads.append(which)
        return None, None

    def generate(model, tokenizer, contract, task, which):
        return next(copy.deepcopy(r) for r in anchors if r["id"] == task["id"])

    def execute(model, tokenizer, contract, retrieval, task, arm):
        calls.append(arm)
        return next(copy.deepcopy(r) for r in rows[arm] if r["id"] == task["id"])

    monkeypatch.setattr(study, "verify_artifacts", download)
    monkeypatch.setattr(study, "file_state", lambda *args: {})
    monkeypatch.setattr(study.reference, "load_model", load)
    monkeypatch.setattr(study.reference, "generate", generate)
    monkeypatch.setattr(study, "execute", execute)
    path = tmp_path / "attempts/parent-evidence/request.json"
    study.save(path, {"freeze": {"fixture": True}, "binding": "fixture", "models": str(tmp_path / "models")})
    study.worker(path)
    reply = study.read(path.parent / "reply.json")
    assert reply["execution_completed"] and reply["diagnostic_passed"] is passes
    assert downloads == ["baseline"] and loads == (["baseline", "baseline"] if passes else ["baseline"])
    assert len(calls) == (134 if passes else 128)
    assert not reply["checklist_credit"] and not reply["admission_evidence"]
