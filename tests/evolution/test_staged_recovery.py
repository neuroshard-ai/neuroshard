import copy
import json
import random

import pytest
import torch

from neuroshard.evolution import staged_recovery as recovery
from neuroshard.evolution.reference_data import identity, save
from neuroshard.evolution.staged_integration import read_checkpoint, tensor_identity
from neuroshard.evolution import staged_integration_run as original


def test_recovery_freeze_pins_method_input_objects_and_protected_answers():
    plan = recovery.load_plan()
    record = json.loads((recovery.method.root() / recovery.RECORD).read_text())
    assert plan["protected_ids"] == record["artifacts"]["protected-before-training.json"]["ids"]
    assert len(plan["protected_ids"]) == 15
    assert plan["expert_checkpoint"]["step"] == plan["gate"]["steps"] == 64
    assert plan["gate"]["verify_recorded_prefix_steps"] == 63
    for name, expected in record["raw_sha256"].items():
        assert plan["inputs"][name] == expected
    assert not plan["gpu_launch_authorized"] and not plan["method_changed"]
    assert not plan["item4_complete"]
    assert recovery.bind_freeze() == identity(recovery.inventory())


def test_uncommitted_amendment_cannot_launch(monkeypatch):
    monkeypatch.setattr(recovery.subprocess, "check_output", lambda *args, **kwargs: b"different")
    with pytest.raises(ValueError, match="Commit the recovery amendment"):
        recovery.bind_freeze(committed=True)


def test_complete_failed_worker_and_recovery_bill_define_control_target():
    record = {"artifacts": {
        "expansion-training.json": [{"phase": "expert", "cpu_seconds": 3},
                                    {"phase": "gate", "cpu_seconds": 4}],
        "expansion-train-launch.json": {"cpu_seconds": 10}}}
    gate = {"training_cpu_seconds": 5, "probe_cpu_seconds": 1}
    launch = {"cpu_seconds": 8, "outcome": "completed"}
    cost = recovery.cost_record(record, gate, launch)
    assert cost["comparison_cpu_seconds"] == 18
    assert cost["discarded_gate_optimization_cpu_seconds"] == 4
    assert cost["prior_setup_probes_interruption_unattributed_cpu_seconds"] == 3
    assert cost["recovery_other_cpu_seconds"] == 2
    with pytest.raises(ValueError, match="complete recovery"):
        recovery.cost_record(record, gate, {**launch, "outcome": "failed"})


def test_prefix_verification_does_not_accept_changed_numerical_training():
    receipt = {"step": 0, "loss": 1.2, "gradient_norm": .3, "input_tokens": 8,
               "answer_tokens": 3, "padded_tokens": 8, "routes": {"mean_added_probability": .5},
               "cpu_seconds": 10}
    recovery.verify_prefix({**receipt, "cpu_seconds": 20}, receipt)
    with pytest.raises(ValueError, match="numerical prefix"):
        recovery.verify_prefix({**receipt, "loss": 1.3}, receipt)


def test_recovery_matches_uninterrupted_gate_with_dropout_and_rng_restoration(tmp_path, monkeypatch):
    from transformers import LlamaConfig, LlamaForCausalLM
    spec = copy.deepcopy(recovery.method.load_spec())
    spec["training"].update(expert_steps=2, gate_steps=3, probe_documents=1)
    config = LlamaConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                         attention_dropout=.2, eos_token_id=31)
    torch.manual_seed(7)
    model = LlamaForCausalLM(config)
    initial = copy.deepcopy(model.state_dict())
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4]]),
             "labels": torch.tensor([[-100, -100, 3, 4]]),
             "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    monkeypatch.setattr(original, "batches", lambda *args: [batch])
    monkeypatch.setattr(recovery, "batches", lambda *args: [batch])
    rows = recovery.method.load_data(recovery.method.load_spec())["roles"]
    old_binding = {"test": "old"}
    new_binding = {"test": "amendment"}
    previous = tmp_path / "previous"
    previous.mkdir()
    torch.manual_seed(29)
    random.seed(29)
    uninterrupted = original.train_expansion(model, object(), rows, spec, previous, old_binding)
    expected = tensor_identity(model.model.layers[-1].mlp)
    old_history = json.loads((previous / "expansion-training.json").read_text())
    # An interrupted gate leaves only its first two recorded steps.
    save(previous / "expansion-training.json", old_history[:-1])
    restored = LlamaForCausalLM(config)
    restored.load_state_dict(initial)
    torch.manual_seed(29)
    random.seed(29)
    home = tmp_path / "recovered"
    home.mkdir()
    result = recovery.restart_gate(restored, object(), rows, spec, home, previous, old_binding, new_binding)
    assert result["verified_prefix_steps"] == 2
    assert result["steps"] == 3
    assert result["incumbent_unchanged"] and result["added_unchanged_during_gate"]
    assert tensor_identity(restored.model.layers[-1].mlp) == expected
    assert json.loads((home / "gate-start.json").read_text())["fresh_optimizer"]
    reference = json.loads((previous / "checkpoints/gate/manifest.json").read_text())
    actual = json.loads((home / "checkpoints/gate/manifest.json").read_text())
    assert reference["module"] == actual["module"]
    assert uninterrupted["steps"] == 5
    # The original expert file remains unchanged and its hashes still verify.
    clean = LlamaForCausalLM(config)
    clean.load_state_dict(initial)
    module = original.install(clean, expansion=True)
    read_checkpoint(previous / "checkpoints/expert", module, binding=old_binding, phase="expert")


def test_failure_accounting_retains_prior_and_failed_child_spend(tmp_path):
    save(tmp_path / "restart-gate-launch.json", {"cpu_seconds": 12, "outcome": "failed"})
    result = recovery.total_spend({"measured_process_cpu_seconds": 766}, tmp_path, 0)
    assert result["total_process_cpu_seconds"] >= 778
    assert result["launches"]["restart-gate"]["outcome"] == "failed"
