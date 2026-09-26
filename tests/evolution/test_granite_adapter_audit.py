import copy
import importlib.util
import json

import pytest
import torch

from neuroshard.evolution import granite_adapter_audit as audit


def fixture_rows():
    tasks = [{"id": "case", "kind": "json", "expected": {"score": "yes"}}]
    rows = [{"id": "case", "text": '{"score":"no"}', "token_ids": [3, 1],
             "input_token_ids": [27, 4], "route_trace": [[[0, 11]], [[11]]],
             "terminated": True, "passed": False}]
    return tasks, rows


def test_reproducing_wrong_answers_never_changes_quality_or_admits():
    tasks, rows = fixture_rows()
    result = audit.summarize(tasks, rows, rows, rows)
    assert result["decision"] == "same-published-adapter-errors"
    assert result["correct"] == {"standalone": 0, "modular": 0}
    assert result["old_failure_unchanged"]
    assert not result["admission_evidence"] and not result["checklist_credit"]


def test_replay_and_pair_disagreement_are_distinct_and_cannot_hide_changed_routes():
    tasks, rows = fixture_rows()
    changed = copy.deepcopy(rows)
    changed[0].update(text='{"score":"yes"}', token_ids=[2, 1], passed=True)
    assert audit.summarize(tasks, changed, rows, rows)["decision"] == "standalone-and-switch-differ"
    assert audit.summarize(tasks, changed, changed, rows)["decision"] == "prior-switch-output-not-reproduced"
    changed = copy.deepcopy(rows)
    changed[0]["route_trace"] = [[[11, 11]], [[11]]]
    assert audit.summarize(tasks, rows, changed, rows)["replay_mismatch_ids"] == ["case"]
    for invalid in ([], rows + rows):
        with pytest.raises(ValueError, match="scope"):
            audit.summarize(tasks, invalid, rows, rows)
    forged = copy.deepcopy(rows)
    forged[0]["passed"] = True
    with pytest.raises(ValueError, match="rescore"):
        audit.summarize(tasks, forged, rows, rows)


def adapter_tensors():
    raw, state = {}, {}
    for projection in ("q", "k", "v", "o"):
        for letter in ("A", "B"):
            key = f"base_model.model.model.layers.0.self_attn.{projection}_proj.lora_{letter}.weight"
            target = (f"model.layers.0.self_attn.o_proj.lora_{letter}" if projection == "o" else
                      f"model.layers.0.self_attn.qkv_proj.lora_{letter}_slices.{('q', 'k', 'v').index(projection)}")
            shape = (16, 3) if letter == "A" else (3, 16)
            raw[key] = torch.ones(shape)
            state[target] = torch.zeros((12, 1, 32, 3) if letter == "A" else (12, 1, 3, 32), dtype=torch.bfloat16)
            if letter == "A":
                state[target][10, 0, :16] = 1
            else:
                state[target][10, 0, :, :16] = 4
    state["model.layers.0.shared_mlp.output_linear.lora_A"] = torch.zeros(12, 1, 32, 3)
    return raw, state


@pytest.mark.parametrize("damage", ["scale", "padding", "extra-mlp", "missing"])
def test_adapter_mapping_rejects_wrong_scaling_padding_scope_and_missing_weights(damage):
    raw, state = adapter_tensors()
    report = audit.compare_adapter_weights(state, raw, layers=1)
    assert len(report["matched"]) == 8 and len(report["zero_targets"]) == 1
    if damage == "scale":
        state["model.layers.0.self_attn.o_proj.lora_B"][10, 0, :, :16] = 1
    elif damage == "padding":
        state["model.layers.0.self_attn.o_proj.lora_A"][10, 0, 20, 0] = 1
    elif damage == "extra-mlp":
        state["model.layers.0.shared_mlp.output_linear.lora_A"][10, 0, 0, 0] = 1
    else:
        raw["unclaimed_tensor"] = torch.ones(1)
    with pytest.raises(ValueError):
        audit.compare_adapter_weights(state, raw, layers=1)


def test_backbone_map_covers_every_tensor_once_and_fuses_in_order():
    state = {"q": torch.tensor([[1., 2.]]), "k": torch.tensor([[3., 4.]])}
    mapping = [{"source": ["q", "k"], "target": "qk", "type": "fused_qkv_proj"}]
    assert audit.mapped_base_digests(state, mapping) == {"qk": audit.tensor_digest(torch.cat(list(state.values())))}
    mapping[0]["source"] = ["q", "q"]
    with pytest.raises(ValueError, match="exactly once"):
        audit.mapped_base_digests(state, mapping)


def test_standalone_trace_must_activate_at_declared_boundary(monkeypatch):
    from types import SimpleNamespace
    projection = torch.nn.Identity()
    model = SimpleNamespace(base_model=SimpleNamespace(model=SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(q_proj=projection))]))))
    def generate(*args, **kwargs):
        assert kwargs["generation_kwargs"] == {"alora_offsets": [2]}
        projection(torch.zeros(1, 5, 2), alora_offsets=[2])
        projection(torch.zeros(1, 1, 2), alora_offsets=[3])
        return {"text": "fixture"}
    # Identity does not accept offsets; this projection fixture preserves actual hook semantics.
    projection.forward = lambda x, **kwargs: x
    monkeypatch.setattr(audit.reference, "generate", generate)
    assert audit.standalone_generate(model, None, {}, {}, 3, 2)["alora_trace"][0]["active_from"] == 3
    with pytest.raises(ValueError, match="boundary"):
        audit.standalone_generate(model, None, {}, {}, 2, 2)


def test_contract_keeps_opened_scope_closed_results_and_runtime_separate():
    plan = audit.read(audit.ROOT / audit.PLAN)
    execution = audit.read(audit.ROOT / audit.EXECUTION)
    original = audit.read(audit.ROOT / audit.reference.PLAN)
    assert plan["task_ids"] == [t["id"] for t in original["reference_tasks"]]
    assert plan["resources"]["generation_calls"] == 2 * len(plan["task_ids"]) == 32
    assert not plan["training_authorized"] and not plan["gpu_launch_authorized"]
    assert not plan["admission_evidence"] and not plan["checklist_credit"]
    assert execution["packages"]["peft"] == "0.19.0"
    assert audit.SCRIPT in execution["sources"]
    for path, digest in execution["contracts"].items():
        assert audit.sha256(audit.ROOT / path) == digest
    assert audit.sha256(audit.ROOT / plan["old_result"]) == "774f68e6fbf57286fa1a0eca0692acf811af181c64fc473f39fc0321c7beb361"


def test_audit_cloud_profile_stays_cpu_only_and_budgeted(monkeypatch):
    spec = importlib.util.spec_from_file_location("cloud_audit", audit.ROOT / "scripts/modular_reference_cloud.py")
    cloud = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cloud)
    resource = cloud.resources("granite-adapter-audit")
    assert not resource["gpu"] and resource["hours"] == 2
    assert "run_granite_adapter_audit.py" in " ".join(cloud.remote_command("granite-adapter-audit"))
    assert cloud.GRANITE_PROFILES["granite-reference"][1] != cloud.GRANITE_PROFILES["granite-adapter-audit"][1]
    original = cloud.read
    path = cloud.ROOT / cloud.RESOURCE_PROFILES["granite-adapter-audit"]
    for changes in ({"hours": 3}, {"instances": 2}, {"gpu": True}, {"planning_cap_usd": 7}):
        monkeypatch.setattr(cloud, "read", lambda p: {**resource, **changes} if p == path else original(p))
        with pytest.raises(ValueError, match="allowance"):
            cloud.resources("granite-adapter-audit")


def test_wrong_standalone_adapter_family_is_rejected():
    config = {"r": 16, "lora_alpha": 64, "alora_invocation_tokens": [27, 71226, 29],
              "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
              "base_model_name_or_path": "ibm-granite/granite-4.1-3b",
              "rank_pattern": {}, "alpha_pattern": {}, "modules_to_save": None}
    audit.validate_adapter(config)
    for changes in ({"r": 64}, {"alora_invocation_tokens": None}, {"lora_alpha": 16}):
        with pytest.raises(ValueError, match="configuration"):
            audit.validate_adapter({**config, **changes})


def test_audit_execution_refuses_uncommitted_sources(tmp_path):
    import subprocess
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.invalid")
    source = tmp_path / "worker.py"
    source.write_text("original\n")
    audit.save(tmp_path / audit.EXECUTION, {"contracts": {}, "sources": ["worker.py", audit.EXECUTION]})
    git("add", ".")
    git("commit", "-qm", "fixture")
    assert audit.committed_sources(tmp_path)["commit"]
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="uncommitted"):
        audit.committed_sources(tmp_path)
