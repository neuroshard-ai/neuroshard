import copy
import importlib.util
import json

import pytest

from neuroshard.evolution import granite_context_reference as study


def plan():
    return study.read(study.ROOT / study.PLAN)


def generated(text, *, routes=None):
    return {"text": text, "terminated": True, "input_token_ids": [1, 2], "token_ids": [3, 4],
            "seconds": 1, "prompt_sha256": study.identity(text), "route_trace": routes or [], "passed": False}


def row(task, arm, correct=True):
    value = task["expected"] if correct else {"answer": "incorrect", "sources": []}
    return {"id": task["id"], "arm": arm, "task_sha256": study.identity(task),
            "query": task["oracle_query"], "retrieved": [{"id": s} for s in task["expected"]["sources"]],
            "answer": generated(json.dumps(value)), "rewrite": None,
            "error": None, "seconds": 1, "passed": correct, "generation_calls": 1,
            "input_tokens": 2, "output_tokens": 2}


def fixtures():
    contract = plan()
    results = {arm: [row(task, arm, arm == "module-rewrite" or i % 4 != 0)
                     for i, task in enumerate(contract["tasks"])] for arm in study.ARMS}
    anchors = []
    for task in study.read(study.ROOT / study.reference.PLAN)["tasks"]:
        text = task["accept"][0] if task["kind"] == "exact" else json.dumps(task["expected"])
        if task["kind"] == "tool":
            text = "<tool_call>" + text + "</tool_call>"
        anchors.append({"id": task["id"], "text": text, "terminated": True, "passed": True, "seconds": 1})
    return results, {which: copy.deepcopy(anchors) for which in ("baseline", "modular")}


def test_fresh_data_and_gold_retrievability_without_model_outputs():
    contract = plan()
    assert len(contract["tasks"]) == 64 and len(contract["corpus"]) == 128
    assert len({t["block"] for t in contract["tasks"]}) == 16
    assert len({t["id"] for t in contract["tasks"]}) == 64
    old = study.read(study.ROOT / study.reference.PLAN)
    old_messages = {study.identity(t["messages"]) for t in old["tasks"] + old["reference_tasks"]}
    retriever = study.Retriever(contract["corpus"], contract["retrieval"])
    for task in contract["tasks"]:
        assert study.identity(task["messages"]) not in old_messages
        documents = retriever.search(task["oracle_query"])
        assert all(source in {d["id"] for d in documents} for source in task["expected"]["sources"])
        if task["expected"]["answer"]:
            assert task["expected"]["answer"] not in json.dumps(task["messages"])
    assert not contract["training_authorized"] and not contract["gpu_launch_authorized"]


def test_retrieval_ignores_labels_and_order_and_uses_full_conversation(monkeypatch):
    contract = plan()
    task = contract["tasks"][0]
    retriever = study.Retriever(contract["corpus"], contract["retrieval"])
    reverse = study.Retriever(list(reversed(contract["corpus"])), contract["retrieval"])
    assert retriever.search(task["oracle_query"]) == reverse.search(task["oracle_query"])
    calls = []

    def generate(model, tokenizer, settings, prompt, which):
        calls.append((copy.deepcopy(prompt), which))
        if prompt["id"] == "rewrite":
            return generated(json.dumps({"rewritten_question": task["oracle_query"]}))
        return generated(json.dumps(task["expected"]))

    monkeypatch.setattr(study.reference, "generate", generate)
    for arm in study.ARMS:
        calls.clear()
        first = study.pipeline(None, None, contract, retriever, task, arm)
        prompts = copy.deepcopy(calls)
        poisoned = {**task, "expected": {"answer": "LEAK", "sources": []},
                    "oracle_query": "LEAK", "block": "LEAK", "category": "LEAK", "id": "LEAK"}
        calls.clear()
        second = study.pipeline(None, None, contract, retriever, poisoned, arm)
        assert calls == prompts
        assert first["query"] == second["query"]
        assert first["retrieved"] == second["retrieved"]
        assert first["generation_calls"] == (1 if arm == "history" else 2)
        assert first["input_tokens"] == first["output_tokens"] == first["generation_calls"] * 2
        assert first["passed"] and not second["passed"]
        if arm == "history":
            assert first["query"] == "\n".join(m["content"] for m in task["messages"])
        else:
            assert prompts[0][0]["messages"][0]["content"] == contract["rewrite_instruction"]
            assert prompts[0][0].get("adapter") == ("query_rewrite" if arm == "module-rewrite" else None)
        assert "adapter" not in prompts[-1][0]  # A fresh base-model answer after the rewrite.


@pytest.mark.parametrize("text,terminated", [
    ('{"rewritten_question":""}', True), ('{"rewritten_question":false}', True),
    ('{"rewritten_question":"ok","rewritten_question":"oops"}', True),
    ('{"rewritten_question":"ok","answer":"leak"}', True),
    ('{"rewritten_question":"<|other_adapter|>"}', True),
    ('Here is the question: hi', True), ('{"rewritten_question":"ok"}', False),
])
def test_invalid_rewrite_cannot_be_silently_repaired(text, terminated, monkeypatch):
    assert study.parse_rewrite(text, terminated, 1024) is None
    contract = plan()
    calls = []

    def generate(*args):
        calls.append(1)
        return {**generated(text), "terminated": terminated}

    monkeypatch.setattr(study.reference, "generate", generate)
    result = study.pipeline(None, None, contract,
                            study.Retriever(contract["corpus"], contract["retrieval"]),
                            contract["tasks"][0], "module-rewrite")
    assert len(calls) == 1 and result["generation_calls"] == 1
    assert not result["passed"] and result["answer"] is None


def test_correct_value_requires_actual_source_and_strict_output():
    task = plan()["tasks"][0]
    answer = row(task, "module-rewrite")
    assert study.answer_score(task, answer)
    answer["retrieved"] = []
    assert not study.answer_score(task, answer)
    answer = row(task, "module-rewrite")
    answer["answer"]["text"] += ' trailing prose'
    assert not study.answer_score(task, answer)
    answer = row(task, "module-rewrite")
    answer["answer"]["terminated"] = False
    assert not study.answer_score(task, answer)


def test_gain_cannot_hide_one_lost_answer_or_anchor():
    contract = plan()
    results, anchors = fixtures()
    report = study.assess(contract, results, anchors)
    assert report["quality_gate"]
    assert report["comparisons"]["parent-rewrite"]["block_bootstrap_lower_95"] == .25
    results["module-rewrite"][1] = row(contract["tasks"][1], "module-rewrite", False)
    report = study.assess(contract, results, anchors)
    assert not report["quality_gate"]
    assert report["comparisons"]["parent-rewrite"]["lost_ids"] == [contract["tasks"][1]["id"]]
    results, anchors = fixtures()
    protected = contract["retention"]["protected_ids"][0]
    next(r for r in anchors["modular"] if r["id"] == protected).update(text="wrong", passed=False)
    assert not study.assess(contract, results, anchors)["quality_gate"]


def test_category_latency_incomplete_and_forged_results_fail():
    contract = plan()
    results, anchors = fixtures()
    for index, task in enumerate(contract["tasks"]):
        if task["category"] == "unanswerable":
            for arm in study.ARMS:
                results[arm][index] = row(task, arm, False)
    report = study.assess(contract, results, anchors)
    assert report["correct"]["module-rewrite"] == 48 and not report["quality_gate"]
    results, anchors = fixtures()
    for record in results["module-rewrite"]:
        record["seconds"] = 2
    assert not study.assess(contract, results, anchors)["quality_gate"]
    results, anchors = fixtures()
    results["history"].pop()
    assert not study.assess(contract, results, anchors)["complete"]
    results, anchors = fixtures()
    results["module-rewrite"][0]["answer"]["text"] = "wrong"
    with pytest.raises(ValueError, match="rescore"):
        study.assess(contract, results, anchors)
    results, anchors = fixtures()
    results["module-rewrite"][0]["task_sha256"] = "forged"
    with pytest.raises(ValueError, match="binding"):
        study.assess(contract, results, anchors)


def test_replays_require_same_query_documents_outputs_and_routes():
    first = row(plan()["tasks"][0], "module-rewrite")
    second = copy.deepcopy(first)
    second["seconds"] = 99
    assert study.replay_matches(first, second)
    second["answer"]["route_trace"] = [[2]]
    assert not study.replay_matches(first, second)
    second = copy.deepcopy(first)
    second["retrieved"].append({"id": "extra"})
    assert not study.replay_matches(first, second)


def test_frozen_runtime_and_cloud_profile():
    execution = study.read(study.ROOT / study.EXECUTION)
    for path, digest in execution["contracts"].items():
        assert study.sha256(study.ROOT / path) == digest
    assert study.SCRIPT in execution["sources"]
    assert "src/neuroshard/evolution/granite_context_reference.py" in execution["sources"]
    assert execution["gpu_launch_authorized"] is False and execution["attempts"] == 1
    spec = importlib.util.spec_from_file_location("context_cloud", study.ROOT / "scripts/modular_reference_cloud.py")
    cloud = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cloud)
    limits = cloud.resources("granite-context-reference")
    assert limits["instances"] == 1 and limits["hours"] == 2 and limits["planning_cap_usd"] == 6
    assert "run_granite_context_reference.py" in " ".join(cloud.remote_command("granite-context-reference"))
    receipt = study.read(study.ROOT / "config/experiments/granite-context-reference-preflight.json")
    assert receipt["plan_sha256"] == study.sha256(study.ROOT / study.PLAN)
    assert receipt["neural_generation_calls"] == 0 and not receipt["models_loaded"]
    assert receipt["matched_rewrite_prompts"] == receipt["identical_answer_prompts"] == 64
    assert receipt["conservative_answer_prompt_bound_tokens"] < plan()["generation"]["max_input_tokens"]


def test_failure_diagnosis_separates_rewrite_retrieval_answer_and_abstention():
    contract = plan()
    task = contract["tasks"][0]
    result = row(task, "module-rewrite", False)
    assert study.failure_stage(task, result) == "grounded-answer"
    result["retrieved"] = []
    assert study.failure_stage(task, result) == "retrieval"
    result["error"] = "invalid rewrite"
    assert study.failure_stage(task, result) == "rewrite-format"
    missing = contract["tasks"][3]
    assert study.failure_stage(missing, row(missing, "module-rewrite", False)) == "abstention"


@pytest.mark.parametrize("quality_pass", [True, False])
def test_worker_runs_comparison_once_and_reloads_only_on_pass(tmp_path, monkeypatch, quality_pass):
    import torch
    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: None)
    monkeypatch.setattr(study, "configure", lambda: None)
    monkeypatch.setattr(study, "freeze", lambda: {"fixture": True})
    results, anchors = fixtures()
    if not quality_pass:
        results["module-rewrite"][1] = row(plan()["tasks"][1], "module-rewrite", False)
    loads, calls = [], []
    monkeypatch.setattr(study, "verify_artifacts", lambda *args, **kwargs: {})
    monkeypatch.setattr(study, "file_state", lambda *args: {})

    def load(directory, which):
        loads.append(which)
        return None, None

    def generate(model, tokenizer, contract, task, which):
        return next(copy.deepcopy(r) for r in anchors[which] if r["id"] == task["id"])

    def pipeline(model, tokenizer, contract, retriever, task, arm):
        calls.append((arm, task["id"]))
        return next(copy.deepcopy(r) for r in results[arm] if r["id"] == task["id"])

    monkeypatch.setattr(study.reference, "load_model", load)
    monkeypatch.setattr(study.reference, "generate", generate)
    monkeypatch.setattr(study, "pipeline", pipeline)
    path = tmp_path / "attempts/assistant-comparison/request.json"
    study.save(path, {"freeze": {"fixture": True}, "binding": "fixture", "models": str(tmp_path / "models")})
    study.worker(path)
    reply = study.read(path.parent / "reply.json")
    assert reply["execution_completed"] and reply["reference_passed"] is quality_pass
    assert loads == (["baseline", "modular", "baseline", "modular"] if quality_pass else ["baseline", "modular"])
    assert len(calls) == (201 if quality_pass else 192)
    assert len(reply["replays"]) == (9 if quality_pass else 0)
    assert not reply["checklist_credit"]
