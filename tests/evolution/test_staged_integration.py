import copy
import json

import pytest
import torch
from torch import nn

from neuroshard.evolution import staged_integration as candidate
from neuroshard.evolution import staged_integration_run as runner


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)

    def forward(self, hidden):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


def mixture():
    torch.manual_seed(31)
    return candidate.StagedMixture(TinyMLP())


def test_dataset_is_new_disjoint_and_has_no_previous_benchmark_dependency():
    spec = candidate.load_spec()
    data = candidate.load_data(spec)
    pairs = []
    for role in candidate.ROLES:
        assert len(data["roles"][role]) == spec["data"]["counts"][role]
        for row in data["roles"][role]:
            pairs.append(tuple(sorted((row["a"], row["b"]))))
            result = row["a"] + row["b"]
            assert row["answer"] == str(result % 7 if row["family"] == "modular-addition" else result)
    assert len(set(pairs)) == len(pairs)
    assert candidate.make_data(spec) == data
    assert data["confirmation"] is None
    assert spec["model"]["repo"].endswith("135M-Instruct")
    assert spec["gpu_launch_authorized"] is False
    assert spec["parent_result"]["development"] == "failed"
    assert spec["parent_result"]["confirmation"] == "never-opened"


def test_freeze_binds_source_and_does_not_authorize_gpu():
    assert candidate.bind_freeze() == candidate.identity(candidate.freeze_inventory())
    assert candidate.freeze_inventory()["gpu_launch_authorized"] is False


def test_dirty_execution_source_is_refused_before_a_run(monkeypatch):
    monkeypatch.setattr(candidate.subprocess, "check_output", lambda *args, **kwargs: b"different HEAD")
    with pytest.raises(ValueError, match="Commit the staged candidate"):
        candidate.bind_freeze(committed=True)


def test_initial_serving_matches_parent_and_executes_only_one_expert():
    module = mixture().eval()
    hidden = torch.randn(1, 3, 4)
    calls = [0, 0]
    handles = [expert.register_forward_hook(lambda m, args, output, i=i: calls.__setitem__(i, calls[i] + 1))
               for i, expert in enumerate((module.incumbent, module.added))]
    module.record_routes = True
    expected = module.incumbent(hidden)
    calls[:] = [0, 0]
    assert torch.allclose(module(hidden), expected)
    assert calls == [1, 0]
    assert module.trace[0]["answer_choices"] == [0]
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("phase", ["expert", "gate"])
def test_training_routing_cannot_be_used_for_serving(phase):
    module = mixture()
    module.set_phase(phase)
    module.eval()
    with pytest.raises(ValueError, match="training-only"):
        module(torch.randn(1, 2, 4))


def test_expert_then_gate_updates_disjoint_parameters_and_records_training_routes():
    module = mixture()
    hidden = torch.randn(1, 6, 4)
    original = candidate.tensor_identity(module.incumbent)
    added_before = candidate.tensor_identity(module.added)
    router_before = candidate.tensor_identity(module.router)
    module.set_phase("expert")
    module.train()
    optimizer = torch.optim.SGD([p for p in module.parameters() if p.requires_grad], lr=.1)
    module(hidden).square().sum().backward()
    optimizer.step()
    assert candidate.tensor_identity(module.incumbent) == original
    assert candidate.tensor_identity(module.added) != added_before
    assert candidate.tensor_identity(module.router) == router_before
    assert module.training_routes["expert_evaluations_per_token"] == 1

    added_after = candidate.tensor_identity(module.added)
    module.set_phase("gate")
    optimizer = torch.optim.SGD([p for p in module.parameters() if p.requires_grad], lr=.1)
    optimizer.zero_grad()
    module(hidden).square().sum().backward()
    assert module.router.weight.grad.abs().max() > 0
    optimizer.step()
    assert candidate.tensor_identity(module.router) != router_before
    assert candidate.tensor_identity(module.added) == added_after
    assert candidate.tensor_identity(module.incumbent) == original
    assert module.training_routes["expert_evaluations_per_token"] == 2


def test_checkpoint_round_trip_and_tamper_rejection(tmp_path):
    module = mixture()
    module.set_phase("gate")
    optimizer = torch.optim.AdamW(module.router.parameters(), lr=.01)
    binding = {"freeze": "a" * 64, "tokenizer": "b" * 64}
    path = tmp_path / "checkpoint"
    candidate.write_checkpoint(path, module, optimizer, binding=binding, phase="gate", step=3)
    restored = mixture()
    manifest = candidate.read_checkpoint(path, restored, binding=binding, phase="gate")
    assert manifest["module"] == candidate.tensor_identity(module)
    state = torch.load(path / "training.pt", weights_only=True)
    assert "optimizer" in state and "torch_rng" in state and "python_rng" in state
    with pytest.raises(FileExistsError):
        candidate.write_checkpoint(path, module, optimizer, binding=binding, phase="gate", step=3)
    with pytest.raises(ValueError, match="another phase"):
        candidate.read_checkpoint(path, restored, binding=binding, phase="expert")
    with (path / "weights.safetensors").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        candidate.read_checkpoint(path, restored, binding=binding, phase="gate")


def test_retention_rejects_replacement_success_and_empty_baseline():
    parent = [{"id": "old", "passed": True}, {"id": "new", "passed": False}]
    swapped = [{"id": "old", "passed": False}, {"id": "new", "passed": True}]
    result = candidate.preservation(parent, swapped, 1)
    assert result == {"protected": ["old"], "lost": ["old"], "gained": ["new"],
                      "nonempty_baseline": True, "passed": False}
    assert not candidate.preservation(swapped, swapped, 2)["passed"]
    empty = [{"id": "old", "passed": False}, {"id": "new", "passed": False}]
    assert not candidate.preservation(empty, empty, 1)["passed"]


def scored_arm(spec, new_correct, *, added=0):
    data = candidate.make_data(spec)
    result = {"peak_rss_bytes": 1000}
    for role in candidate.EVAL_ROLES:
        result[role] = []
        for index, row in enumerate(data["roles"][role]):
            passed = index < (new_correct if role == "development" else 8)
            result[role].append({"id": row["id"], "passed": passed, "answer": row["answer"],
                                 "text": row["answer"] if passed else "wrong", "seconds": .1,
                                 "automatic": True, "added_answer_tokens": added, "generated_tokens": 1,
                                 "routes": [{"choices": [[added]], "added_margin": [[1.0 if added else -1.0]],
                                             "answer_choices": [added], "answer_margins": [1.0 if added else -1.0]}]})
    return result


def evidence(signal=True):
    return {"expert_training_signal": signal, "incumbent_unchanged": True,
            "added_unchanged_during_gate": True, "comparison_cpu_seconds": 1.0}


def test_three_way_diagnosis_and_automatic_only_success():
    spec = candidate.load_spec()
    parent = scored_arm(spec, 1)
    control = scored_arm(spec, 2)
    poor = scored_arm(spec, 1)
    good = scored_arm(spec, 4, added=1)
    spent = {"matched_budget": True, "training_cpu_seconds": 1.1}
    first = candidate.score(spec, parent, poor, control, evidence(False), spent)
    assert first["diagnosis"] == "expert_learning_not_observed"
    second = candidate.score(spec, parent, poor, control, evidence(), spent)
    assert second["diagnosis"] == "automatic_integration_not_effective"
    third = candidate.score(spec, parent, good, control, evidence(), spent)
    assert third["diagnosis"] == "complete_system_evaluated"
    assert third["passed"] and not third["admission_evidence"]
    assert not third["gpu_launch_authorized"] and not third["item4_complete"]
    good["development"][0]["automatic"] = False
    with pytest.raises(ValueError, match="Only automatic"):
        candidate.score(spec, parent, good, control, evidence(), spent)


def test_loss_of_old_answer_or_insufficient_control_spend_rejects_gain():
    spec = candidate.load_spec()
    parent, control, good = scored_arm(spec, 1), scored_arm(spec, 2), scored_arm(spec, 4, added=1)
    spent = {"matched_budget": True, "training_cpu_seconds": 1.1}
    good["retention"][0].update(passed=False, text="wrong")
    good["retention"][9].update(passed=True, text=good["retention"][9]["answer"])
    result = candidate.score(spec, parent, good, control, evidence(), spent)
    assert not result["passed"] and result["retention"]["lost"]
    good = scored_arm(spec, 4, added=1)
    result = candidate.score(spec, parent, good, control, evidence(), {**spent, "training_cpu_seconds": .9})
    assert not result["gates"]["training_budget_control"]


def test_scorer_rejects_forged_labels_and_changed_split():
    spec = candidate.load_spec()
    parent, control, good = scored_arm(spec, 1), scored_arm(spec, 2), scored_arm(spec, 4, added=1)
    spent = {"matched_budget": True, "training_cpu_seconds": 1.1}
    bad = copy.deepcopy(good)
    bad["development"][0]["text"] = "not the answer"
    with pytest.raises(ValueError, match="generated text"):
        candidate.score(spec, parent, bad, control, evidence(), spent)
    bad = copy.deepcopy(good)
    bad["development"][0]["id"] = "replacement"
    with pytest.raises(ValueError, match="identities"):
        candidate.score(spec, parent, bad, control, evidence(), spent)
    bad = copy.deepcopy(good)
    bad["development"][0]["added_answer_tokens"] = 0
    with pytest.raises(ValueError, match="recorded trace"):
        candidate.score(spec, parent, bad, control, evidence(), spent)


def test_isolated_launcher_uses_separate_processes_and_captures_transient_peak(tmp_path, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run_staged_integration.py").write_text("""import argparse, gc, json, os, resource
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--worker'); p.add_argument('--seed'); p.add_argument('--home')
a = p.parse_args()
payload = bytearray(64 * 1024 * 1024 if a.worker == 'baseline' else 1024)
del payload
gc.collect()
peak = next(int(line.split()[1]) * 1024 for line in Path('/proc/self/status').read_text().splitlines()
            if line.startswith('VmHWM:'))
result = {'pid': os.getpid(), 'peak': peak,
          'gpu': os.environ['CUDA_VISIBLE_DEVICES']}
(Path(a.home) / (a.worker + '.json')).write_text(json.dumps(result))
""")
    home = tmp_path / "study"
    home.mkdir()
    monkeypatch.setattr(runner, "root", lambda: tmp_path)
    first = runner.run_isolated("baseline", tmp_path, home, 10)
    second = runner.run_isolated("control-evaluate", tmp_path, home, 10)
    assert first["pid"] != second["pid"]
    assert first["gpu"] == second["gpu"] == ""
    assert first["peak"] > 64 * 1024 * 1024
    assert second["peak"] < first["peak"]
    assert json.loads((home / "baseline-launch.json").read_text())["cpu_seconds"] > 0


def test_timeout_preserves_actual_child_spend(tmp_path, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run_staged_integration.py").write_text("while True: pass\n")
    home = tmp_path / "study"
    home.mkdir()
    monkeypatch.setattr(runner, "root", lambda: tmp_path)
    import subprocess
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run_isolated("baseline", tmp_path, home, .1)
    receipt = json.loads((home / "baseline-launch.json").read_text())
    assert receipt["outcome"] == "failed"
    assert receipt["cpu_seconds"] > 0


def test_uninformative_baseline_stops_before_any_training(tmp_path, monkeypatch):
    spec = candidate.load_spec()
    baseline = scored_arm(spec, 0)
    for row in baseline["retention"]:
        row.update(passed=False, text="wrong")
    calls = []
    def isolated(arm, *args):
        calls.append(arm)
        assert arm == "baseline"
        return baseline
    monkeypatch.setattr(runner, "bind_freeze", lambda **kwargs: "unit-freeze")
    monkeypatch.setattr(runner, "verify", lambda seed: None)
    monkeypatch.setattr(runner, "run_isolated", isolated)
    home = tmp_path / "study"
    result = runner.run_study(tmp_path, home)
    assert calls == ["baseline"]
    assert result["next"] == "stop-baseline-uninformative"
    assert not result["passed"]
    assert json.loads((home / "protected-before-training.json").read_text())["ids"] == []


def test_tiny_llama_runs_both_training_phases_and_automatic_generation(tmp_path, monkeypatch):
    from transformers import LlamaConfig, LlamaForCausalLM
    spec = copy.deepcopy(candidate.load_spec())
    spec["training"].update(expert_steps=2, gate_steps=2, maximum_control_steps=4,
                             generation_tokens=2)
    config = LlamaConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                         eos_token_id=None, pad_token_id=0, bos_token_id=1)
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4]]),
             "labels": torch.tensor([[-100, -100, 3, 4]]),
             "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    monkeypatch.setattr(runner, "batches", lambda *args: [batch])
    model = LlamaForCausalLM(config)
    initial = copy.deepcopy(model.state_dict())
    rows = candidate.make_data(spec)["roles"]
    binding = {"test": "tiny-llama-not-mechanism-evidence"}
    training = runner.train_expansion(model, object(), rows, spec, tmp_path, binding)
    assert training["steps"] == 4
    assert training["added_changed"]
    assert training["incumbent_unchanged"]
    assert training["added_unchanged_during_gate"]
    assert training["comparison_cpu_seconds"] > training["training_cpu_seconds"] > 0
    assert set(training["checkpoints"]) == {"expert", "gate"}

    restored = LlamaForCausalLM(config)
    restored.load_state_dict(initial)
    module = runner.install(restored, expansion=True)
    candidate.read_checkpoint(tmp_path / "checkpoints/gate", module, binding=binding, phase="gate")
    class Tokenizer:
        eos_token_id = 31
        def apply_chat_template(self, *args, **kwargs):
            return torch.tensor([[1, 2, 3]])
        def decode(self, *args, **kwargs):
            return "test-output"
    small = {role: rows[role][:1] for role in candidate.EVAL_ROLES}
    outputs = runner.evaluate(restored, Tokenizer(), small, spec, tmp_path, "tiny")
    for role in candidate.EVAL_ROLES:
        answer = outputs[role][0]
        assert answer["automatic"]
        assert len(answer["routes"]) == answer["generated_tokens"] == 2
        assert answer["added_answer_tokens"] == sum(sum(call["answer_choices"]) for call in answer["routes"])

    control = LlamaForCausalLM(config)
    control.load_state_dict(initial)
    budget = runner.train_control(control, object(), rows, spec, tmp_path, binding, .000001)
    assert budget["matched_budget"] and budget["steps"] == 1
    assert budget["training_cpu_seconds"] >= budget["target_cpu_seconds"]
