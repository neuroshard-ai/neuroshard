import copy
import json
import subprocess

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from neuroshard.evolution import observable_reasoning as study
from neuroshard.evolution import observable_reasoning_run as runner
from neuroshard.evolution.reference_data import save


def test_fresh_pairs_disjoint_wording_and_full_execution_closure():
    plan = study.load_plan()
    data = study.load_data(plan)
    pairs = [(row["a"], row["b"]) for rows in data["roles"].values() for row in rows]
    assert len(pairs) == len(set(pairs)) == 1376
    assert not set(pairs) & study.excluded_pairs()
    for family in ("addition", "modular-addition"):
        assert not set(plan["data"]["prompts"]["train"][family]) & set(plan["data"]["prompts"]["evaluation"][family])
    files = study.inventory()["files"]
    for helper in ("seed", "reference_data", "block_expert_run", "staged_answer_format", "staged_integration_run"):
        assert f"src/neuroshard/evolution/{helper}.py" in files
    assert "docs/llm-requirements.txt" in files
    assert study.bind_freeze() == study.identity(study.inventory())


def test_parser_never_repairs_or_searches_for_a_gold_answer():
    assert study.parse_answer("6", terminated=True) == "6"
    assert study.parse_answer("First remainder: 6\nSecond remainder: 5\nSum: 11\nAnswer: 4", terminated=True) == "4"
    assert study.parse_answer("6", terminated=False) is None
    assert study.parse_answer("I guess 5, but maybe 6", terminated=True) is None
    assert study.parse_answer("Answer: 6\nActually 2", terminated=True) is None
    row = {"family": "modular-addition", "a": 20, "b": 12, "answer": "4"}
    assert study.training_answer(row) == "First remainder: 6\nSecond remainder: 5\nSum: 11\nAnswer: 4"


def test_selector_does_not_accept_rows_and_ignores_answers():
    plan = study.load_plan()
    data = study.load_data(plan)
    rows = data["roles"]["train_new"] + data["roles"]["train_replay"]
    selector = study.fit_selector(rows)
    assert selector == study.fit_selector([{**row, "answer": "poison", "id": "poison", "passed": True} for row in rows])
    with pytest.raises(TypeError, match="question text only"):
        study.choose({"question_text": "q", "passed": True, "gold_answer": "6"}, selector)
    assert study.choose("entirely unseen vocabulary", selector)[0] == "parent"


def small_case():
    plan = copy.deepcopy(study.load_plan())
    plan["gates"]["bootstrap_draws"] = 200
    plan["gates"]["minimum_new_correct"] = 16
    selector = {"weights": {"remainder": 10}, "threshold": 2}
    data = {"roles": {"train_new": [{"answer": str(i % 7)} for i in range(70)]}}
    data["roles"]["development"] = [
        {"id": f"new-{i}", "answer": str(i % 7), "messages": [{"role": "user", "content": "remainder"}]}
        for i in range(32)]
    data["roles"]["retention"] = [
        {"id": f"old-{i}", "answer": str(20 + i), "messages": [{"role": "user", "content": "add"}]}
        for i in range(8)]
    arms = []
    for name in ("parent", "expert", "control"):
        receipt = {"peak_rss_bytes": 1000}
        for role in study.EVAL_ROLES:
            receipt[role] = []
            for truth in data["roles"][role]:
                good = role == "retention" or name == "expert"
                route, margin = study.choose(truth["messages"][-1]["content"], selector)
                receipt[role].append({"id": truth["id"], "text": truth["answer"] if good else "999",
                                      "terminated": True, "seconds": 1.0, "generation_calls": 1,
                                      "route": route, "route_margin": margin})
            if name != "parent":
                receipt[role + "_forced"] = copy.deepcopy(receipt[role])
        arms.append(receipt)
    trained = {"frozen_unchanged": True}
    control_trained = {"frozen_unchanged": True, "matched_budget": True, "optimization_cpu_seconds": 100}
    return plan, data, *arms, trained, control_trained, 100, selector


def test_a_strong_observable_system_passes_but_one_lost_answer_fails():
    args = small_case()
    result = study.score(*args)
    assert result["passed"] and len(result["protected"]) == 8
    assert not result["admission_evidence"] and not result["item4_complete"]
    args[3]["retention"][0]["text"] = "999"
    result = study.score(*args)
    assert not result["passed"]
    assert result["protected_lost"] == ["old-0"]


def test_constant_or_equal_control_cannot_pass_and_timers_include_serving():
    args = small_case()
    args[4]["development"] = copy.deepcopy(args[3]["development"])
    args[4]["development_forced"] = copy.deepcopy(args[3]["development_forced"])
    result = study.score(*args)
    assert not result["passed"]
    assert not result["gates"]["beats_parent_control_and_constant"]
    args = small_case()
    for row in args[3]["development"]:
        row["seconds"] = 21
    assert not study.score(*args)["gates"]["complete_response_latency"]
    args = small_case()
    for row in args[3]["development"]:
        row["text"] = "0"
    assert not study.score(*args)["passed"]


def test_scorer_recomputes_text_and_rejects_oracle_routes():
    args = small_case()
    args[3]["retention"][0].update(text="999", passed=True, parsed_answer="20")
    assert not study.score(*args)["passed"]
    args = small_case()
    args[3]["development"][0]["route"] = "parent"
    with pytest.raises(ValueError, match="question-only"):
        study.score(*args)


def test_control_serving_restores_actual_parent_partition(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, attention_dropout=0.0))
    parent_tail = model.model.layers[-1].mlp.down_proj.weight.detach().clone()
    plan = study.load_plan()
    rows = [{"family": "addition", "messages": [{"role": "user", "content": "add"}], "answer": "2"}]
    selector = study.fit_selector(rows)
    save(tmp_path / "selector.json", selector)
    data = {"roles": {"train_new": [], "train_replay": rows, "development": [],
                     "retention": [{**rows[0], "id": "retained"}]}}
    monkeypatch.setattr(runner.shared, "read_receipt", lambda *args: {"checkpoint": "test"})
    def restore(path, blocks, *args):
        with torch.no_grad():
            blocks[-1].mlp.down_proj.weight.fill_(17)
    monkeypatch.setattr(runner.shared, "restore", restore)
    seen = []
    def fake_generate(model, tokenizer, messages, plan):
        is_parent = torch.equal(parent_tail, model.model.layers[-1].mlp.down_proj.weight)
        seen.append(is_parent)
        return {"text": "2" if is_parent else "9", "terminated": True, "generation_calls": 1}
    monkeypatch.setattr(runner, "generate", fake_generate)
    result = runner.evaluate(model, None, data, plan, tmp_path, "control-evaluate", {})
    assert seen == [True, False]
    assert result["retention"][0]["text"] == "2"
    assert result["retention_forced"][0]["text"] == "9"


def test_failed_child_is_billed_and_not_retried(tmp_path, monkeypatch):
    previous_shared_study = runner.shared.study
    monkeypatch.setattr(study, "bind_freeze", lambda **kwargs: "frozen")
    monkeypatch.setattr(runner, "verify", lambda path: None)
    calls = []
    def fail(arm, seed, home, seconds):
        calls.append(arm)
        save(home / (arm + "-launch.json"), {"cpu_seconds": 7.5, "outcome": "failed"})
        raise subprocess.TimeoutExpired(["fake-worker"], seconds)
    monkeypatch.setattr(runner, "isolated", fail)
    result = runner.run(tmp_path, tmp_path / "run")
    assert calls == ["baseline"]
    assert not result["execution_completed"] and not result["passed"]
    assert result["child_cpu_seconds"] == 7.5
    assert runner.shared.study is previous_shared_study
    assert json.loads((tmp_path / "run" / "status.json").read_text())["state"] == "failed-execution"
