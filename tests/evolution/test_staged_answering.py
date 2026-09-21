import copy
import json

import pytest
import torch

from neuroshard.evolution import staged_answering as candidate
from neuroshard.evolution.staged_answer_format import evaluate
from neuroshard.evolution.staged_integration_run import install
from scripts import run_staged_answering as runner


def test_successor_excludes_every_old_and_calibration_pair():
    spec = candidate.load_spec()
    data = candidate.load_data(spec)
    excluded = set()
    for name in ("config/experiments/staged-integration-data.json", candidate.CALIBRATION):
        previous = json.loads((candidate.root() / name).read_text())
        excluded.update((r["a"], r["b"]) for rows in previous["roles"].values() for r in rows)
    pairs = [(r["a"], r["b"]) for rows in data["roles"].values() for r in rows]
    assert len(pairs) == len(set(pairs)) == 160
    assert not set(pairs) & excluded
    assert len(excluded) == 180
    assert spec["gates"]["minimum_parent_retention_correct"] == 8
    assert spec["gates"]["minimum_new_gain"] == 2
    assert not spec["calibration"]["screens_passed"]
    assert not spec["gpu_launch_authorized"]


def arm(spec, correct, added=0):
    result = {"peak_rss_bytes": 1000}
    for role, rows in candidate.make_data(spec)["roles"].items():
        if role not in candidate.EVAL_ROLES:
            continue
        result[role] = []
        for index, row in enumerate(rows):
            passed = index < (correct if role == "development" else 8)
            text = row["answer"] if passed else "invalid"
            result[role].append({
                "id": row["id"], "text": text, "answer": row["answer"], "passed": passed,
                "parsed_answer": row["answer"] if passed else None, "terminated": True,
                "automatic": True, "seconds": .1, "generated_tokens": 1,
                "added_answer_tokens": added, "routes": [{"choices": [[added]],
                    "added_margin": [[1.0 if added else -1.0]], "answer_choices": [added],
                    "answer_margins": [1.0 if added else -1.0]}],
            })
    return result


def score(parent, expanded, control):
    training = {"expert_training_signal": True, "incumbent_unchanged": True,
                "added_unchanged_during_gate": True, "comparison_cpu_seconds": 1.0}
    spent = {"matched_budget": True, "training_cpu_seconds": 1.1}
    return candidate.score(candidate.load_spec(), parent, expanded, control, training, spent)


def test_new_scorer_preserves_every_answer_and_rechecks_text_and_routing():
    spec = candidate.load_spec()
    parent, expanded, control = arm(spec, 0), arm(spec, 3, added=1), arm(spec, 1)
    assert score(parent, expanded, control)["passed"]
    bad = copy.deepcopy(expanded)
    bad["retention"][0].update(text="invalid", parsed_answer=None, passed=False)
    result = score(parent, bad, control)
    assert not result["passed"] and len(result["retention"]["lost"]) == 1
    for change in ({"terminated": False}, {"text": "Answer is 3 or 4"}, {"parsed_answer": "wrong"}):
        bad = copy.deepcopy(expanded)
        bad["development"][0].update(change)
        with pytest.raises(ValueError, match="generated answer"):
            score(parent, bad, control)
    bad = copy.deepcopy(expanded)
    bad["development"][0]["added_answer_tokens"] = 0
    with pytest.raises(ValueError, match="recorded trace"):
        score(parent, bad, control)


def test_fresh_baseline_still_stops_without_any_training(tmp_path, monkeypatch):
    baseline = arm(candidate.load_spec(), 0)
    for row in baseline["retention"]:
        row.update(text="invalid", parsed_answer=None, passed=False)
    calls = []
    def isolated(name, *args):
        calls.append(name)
        return baseline
    monkeypatch.setattr(runner, "bind_freeze", lambda **kwargs: "test")
    monkeypatch.setattr(runner, "verify", lambda seed: None)
    monkeypatch.setattr(runner, "run_isolated", isolated)
    result = runner.run_study(tmp_path, tmp_path / "study")
    assert calls == ["baseline"]
    assert result["next"] == "stop-baseline-uninformative"


def test_actual_generation_records_routes_and_rejects_missing_eos(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, eos_token_id=31, pad_token_id=0)
    model = LlamaForCausalLM(config)
    with torch.no_grad():
        model.lm_head.weight.zero_()  # Always generate token 0, never EOS.
    install(model, expansion=True)
    spec = copy.deepcopy(candidate.load_spec())
    spec["training"]["generation_tokens"] = 2
    row = {"id": "test", "a": 1, "b": 2, "family": "addition", "answer": "3", "messages": []}
    class Tokenizer:
        eos_token_id = 31
        def apply_chat_template(self, *args, **kwargs):
            return torch.tensor([[1, 2, 3]])
        def decode(self, *args, **kwargs):
            return "3"
    result = evaluate(model, Tokenizer(), {role: [row] for role in candidate.EVAL_ROLES}, spec,
                      tmp_path, "tiny")
    for role in candidate.EVAL_ROLES:
        reply = result[role][0]
        assert reply["generated_tokens"] == len(reply["routes"]) == 2
        assert not reply["terminated"] and not reply["passed"]


def test_freeze_preserves_predecessor_and_rejects_uncommitted_execution(monkeypatch):
    from neuroshard.evolution.staged_integration import bind_freeze as previous_freeze
    assert previous_freeze() == "a03f68aa2742508f8a1d527508c9a5523c5375144a8d91115ea0a6d4aa718cbb"
    assert candidate.bind_freeze() == candidate.identity(candidate.freeze_inventory())
    monkeypatch.setattr(candidate.subprocess, "check_output", lambda *args, **kwargs: b"different")
    with pytest.raises(ValueError, match="Commit the answering candidate"):
        candidate.bind_freeze(committed=True)
