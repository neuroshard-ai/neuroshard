import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers import FlexOlmoConfig, FlexOlmoForCausalLM, Olmo2Config, Olmo2ForCausalLM
from transformers.cache_utils import DynamicCache

from neuroshard.evolution import modular_reference_execution as execution
from neuroshard.evolution.modular_reference import canonical_call, load_plan, score_reply
from neuroshard.evolution.modular_reference_run import forward_logits, layer_classes


ROOT = Path(__file__).resolve().parents[2]


def plan():
    return load_plan(ROOT / execution.PLAN)


def reply(task, model="baseline", phase="primary", binding=None):
    text = (task["accept"][0] if task["kind"] == "exact" else
            "<function_calls>" + canonical_call(task["expect"][0]) + "</function_calls>")
    return {"id": task["id"], "model": model, "phase": phase, "category": task["category"],
            "binding": execution.identity(binding or {}), "task_sha256": execution.identity(task),
            "text": text, "terminated": True, "token_ids": [1, 2], "stopped": False,
            "seconds": 1, "max_rss_bytes": 1000, **score_reply(task, text, True)}


def test_old_reply_cannot_follow_a_changed_prompt_or_input_binding():
    contract = plan()
    task = contract["tasks"][0]
    row = reply(task)
    assert execution.checked(contract, row, {}, "baseline", "primary", task) == row
    changed = copy.deepcopy(task)
    changed["messages"][-1]["content"] = "A different question under the same task ID"
    with pytest.raises(ValueError, match="provenance"):
        execution.checked(contract, row, {}, "baseline", "primary", changed)
    for binding in ({"weights": "different"}, {"tokenizer": "different"}, {"source": "different"}):
        with pytest.raises(ValueError, match="provenance"):
            execution.checked(contract, row, binding, "baseline", "primary", task)
    with pytest.raises(ValueError, match="provenance"):
        execution.checked(contract, row, {}, "baseline", "replay", task)


def test_missing_or_disagreeing_independent_replay_cannot_open_gate():
    contract = plan()
    primary = [reply(task, model) for model in ("baseline", "modular") for task in contract["tasks"]]
    assert execution.gate(contract, primary, [])["quality_ready"] is False
    replays = []
    for row in primary:
        replay = dict(row, phase="replay")
        comparison = execution.compare_replay(row, replay)
        replays.append({"model": row["model"], "id": row["id"], **comparison})
    assert execution.gate(contract, primary, replays)["quality_ready"] is True
    assert execution.gate(contract, primary, replays)["milestone_complete"] is False
    different = dict(primary[0], text="different answer", phase="replay")
    assert execution.compare_replay(primary[0], different)["matched"] is False
    replays[0]["matched"] = False
    assert execution.gate(contract, primary, replays)["quality_ready"] is False
    assert execution.gate(contract, primary, replays[1:])["quality_ready"] is False


def test_failure_and_interruption_charge_actual_or_reserved_work(tmp_path):
    interrupted = tmp_path / "attempts/baseline-primary-a"
    execution.save(interrupted / "request.json", {"seconds": 50, "phase": "primary"})
    assert execution.charged_seconds(interrupted) == 50
    execution.save(interrupted / "outcome.json", {"seconds": 12.5, "completed": False})
    assert execution.charged_seconds(interrupted) == 12.5
    preparation = tmp_path / "attempts/baseline-prepare"
    execution.save(preparation / "request.json", {"seconds": 100, "phase": "prepare"})
    execution.save(preparation / "outcome.json", {"seconds": 3, "completed": True})
    assert execution.spent(tmp_path, "baseline", 40) == 52.5
    assert execution.spent(tmp_path, "baseline", 40, preparation=True) == 3


def test_failed_launch_preserves_receipt_and_cannot_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(execution, "supervised", lambda *args: {
        "completed": False, "reason": "time-limit", "seconds": 2.25})
    binding = {"freeze": {"commit": "test"}}
    with pytest.raises(RuntimeError, match="time-limit"):
        execution.launch(tmp_path, tmp_path, binding, "baseline", "prepare", 2, 100)
    attempt = tmp_path / "attempts/baseline-prepare"
    assert execution.read(attempt / "outcome.json")["seconds"] == 2.25
    with pytest.raises(FileExistsError):
        execution.launch(tmp_path, tmp_path, binding, "baseline", "prepare", 2, 100)


def test_supervisor_sets_kernel_memory_limit_and_no_extra_twenty_minutes(tmp_path, monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(execution.subprocess, "run", run)
    result = execution.supervised([sys.executable, "-c", "pass"], tmp_path / "log", 2, 64 * 1024 ** 2, "test-unit")
    argv, kwargs = calls[0]
    assert "RuntimeMaxSec=2000ms" in argv
    assert "MemoryMax=67108864" in argv and "MemorySwapMax=0" in argv
    assert "KillMode=control-group" in argv and "TimeoutStopSec=0" in argv
    assert kwargs["timeout"] == 7  # five seconds to clean up; still billed, never a passing overrun
    assert calls[-1][0] == ["systemctl", "--user", "stop", "test-unit"]
    assert result["completed"] is True


def test_immutable_artifact_inventory_detects_same_size_tampering(tmp_path):
    payloads = {"model.safetensors": b"abc", "tokenizer.json": b"{}",
                "model.safetensors.index.json": json.dumps({"weight_map": {"model.weight": "model.safetensors"}}).encode()}
    inventory = {"files": {}}
    for name, payload in payloads.items():
        (tmp_path / name).write_bytes(payload)
        if name == "tokenizer.json":
            algorithm = "git-blob-sha1"
            digest = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        else:
            algorithm = "sha256"
            digest = hashlib.sha256(payload).hexdigest()
        inventory["files"][name] = {"bytes": len(payload), "algorithm": algorithm, "digest": digest}
    original = execution.verify_artifacts(tmp_path, inventory)
    assert set(original) == set(payloads)
    (tmp_path / "model.safetensors").write_bytes(b"abd")
    with pytest.raises(ValueError, match="hash mismatch"):
        execution.verify_artifacts(tmp_path, inventory)
    # Back-to-back writes can share a filesystem timestamp. Hashing above
    # detects those too; the inexpensive metadata check assumes a changed stat.
    path = tmp_path / "model.safetensors"
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert execution.file_state(tmp_path, inventory) != original
    (tmp_path / "added_tokens.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        execution.file_state(tmp_path, inventory)


def test_amendment_preserves_original_tasks_and_pins_every_runtime_asset():
    amendment = execution.read(ROOT / execution.AMENDMENT)
    inventory = execution.read(ROOT / execution.ARTIFACTS)
    contract = plan()
    assert amendment["plan_sha256"] == execution.sha256(ROOT / execution.PLAN)
    assert amendment["artifacts_sha256"] == execution.sha256(ROOT / execution.ARTIFACTS)
    assert amendment["method_changed"] is False
    assert amendment["gpu_launch_authorized"] is False
    for model in ("baseline", "modular"):
        assets = inventory["models"][model]
        assert assets["revision"] == contract["models"][model]["revision"]
        assert {"tokenizer.json", "vocab.json", "merges.txt", "model.safetensors.index.json",
                "special_tokens_map.json", "chat_template.jinja"} <= set(assets["files"])
        assert sum(spec["bytes"] for name, spec in assets["files"].items()
                   if name.endswith(".safetensors")) == contract["models"][model]["storage_bytes"]


def test_execution_freeze_rejects_dirty_committed_source(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    execution.save(tmp_path / execution.PLAN, {"tasks": "unchanged"})
    execution.save(tmp_path / execution.ARTIFACTS, {"assets": "unchanged"})
    (tmp_path / "engine.py").write_text("# original\n")
    amendment = {"plan_sha256": execution.sha256(tmp_path / execution.PLAN),
                 "artifacts_sha256": execution.sha256(tmp_path / execution.ARTIFACTS),
                 "sources": [execution.PLAN, execution.ARTIFACTS, execution.AMENDMENT, "engine.py"],
                 "packages": {}, "python": execution.platform.python_version()}
    execution.save(tmp_path / execution.AMENDMENT, amendment)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "-c", "commit.gpgsign=false", "commit", "-qm", "freeze"], cwd=tmp_path, check=True)
    assert execution.freeze(tmp_path)["sources"]["engine.py"]
    (tmp_path / "engine.py").write_text("# changed after freeze\n")
    with pytest.raises(ValueError, match="uncommitted execution source"):
        execution.freeze(tmp_path)


def test_reply_overruns_are_execution_failures():
    contract = plan()
    task = contract["tasks"][0]
    for field, excess in (("seconds", contract["limits"]["per_task_seconds"] + 1),
                          ("max_rss_bytes", contract["limits"]["max_rss_bytes"] + 1)):
        row = reply(task)
        row[field] = excess
        with pytest.raises(ValueError, match="resource budget"):
            execution.checked(contract, row, {}, "baseline", "primary", task)


@pytest.mark.parametrize("mismatch", [False, True])
def test_controller_generates_separate_replay_and_stops_on_disagreement(tmp_path, monkeypatch, mismatch):
    contract = plan()
    root = tmp_path / "source"
    execution.save(root / execution.PLAN, contract)
    execution.save(root / execution.AMENDMENT, {})
    frozen = {"plan_sha256": execution.sha256(root / execution.PLAN), "commit": "fixture"}
    monkeypatch.setattr(execution, "ROOT", root)
    monkeypatch.setattr(execution, "freeze", lambda **kwargs: frozen)
    legacy = tmp_path / "legacy.json"
    execution.save(legacy, {"which": "baseline", "plan_sha256": frozen["plan_sha256"], "seconds": 42})
    calls = []

    def launch(home, models, binding, which, phase, seconds, memory_bytes, *, task=None, stats=None):
        calls.append((which, phase, task["id"] if task else None))
        if phase == "prepare":
            return {"file_state": {}, "verified": True}
        row = reply(task, which, phase, binding)
        if mismatch and phase == "replay":
            row["text"] = "a different replay"
            row.update(score_reply(task, row["text"], True))
        return row

    monkeypatch.setattr(execution, "launch", launch)
    home = tmp_path / "study"
    result = execution.run(home, tmp_path / "models", legacy)
    if mismatch:
        assert result["execution_completed"] is False
        assert "replay mismatch" in result["error"]
        assert not any(which == "modular" for which, _, _ in calls)
    else:
        assert result["quality_ready"] is True
        assert len([row for row in calls if row[1] == "primary"]) == 18
        assert len([row for row in calls if row[1] == "replay"]) == 18
    assert result["accounting"]["baseline"]["evaluation_seconds"] == 42
    previous_calls = list(calls)
    assert execution.run(home, tmp_path / "models", legacy) == result
    assert calls == previous_calls  # Reading a finished record never repeats inference.


@pytest.mark.parametrize("config_cls,model_cls", [(Olmo2Config, Olmo2ForCausalLM), (FlexOlmoConfig, FlexOlmoForCausalLM)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_streamed_prefill_and_cached_decode_match_transformers(config_cls, model_cls, dtype, tmp_path):
    previous_threads = torch.get_num_threads()
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng():
            torch.manual_seed(123)
            config = config_cls(hidden_size=32, intermediate_size=64, num_attention_heads=4,
                                num_key_value_heads=4, num_hidden_layers=2, vocab_size=128, head_dim=8,
                                num_experts=5, num_experts_per_tok=5, attention_dropout=0.0,
                                tie_word_embeddings=False, pad_token_id=0, bos_token_id=1, eos_token_id=2)
            config._attn_implementation = "eager"
            torch.set_default_dtype(dtype)
            model = model_cls(config).eval()
            torch.set_default_dtype(previous_dtype)
        # Exercise the real safetensors index/loader, not weights borrowed from
        # the oracle object. Loading also preserves upstream FP32 RoPE buffers.
        model.save_pretrained(tmp_path, max_shard_size="10KB")
        model = model_cls.from_pretrained(tmp_path, dtype=dtype, attn_implementation="eager").eval()
        from neuroshard.evolution.modular_reference_run import Checkpoint, assign_weights
        checkpoint = Checkpoint(tmp_path)
        layer_cls, norm_cls, rotary_cls = layer_classes(config)
        norm = assign_weights(norm_cls(config.hidden_size, eps=config.rms_norm_eps),
                              {"weight": checkpoint.tensor("model.norm.weight")})
        rotary = rotary_cls(config)
        reference_cache, streamed_cache = DynamicCache(config=config), DynamicCache(config=config)
        tokens, seen = torch.tensor([[11, 9, 4, 31]]), 0
        with torch.no_grad():
            for _ in range(4):
                native = model(input_ids=tokens, attention_mask=torch.ones((1, seen + tokens.shape[1]), dtype=torch.long),
                               past_key_values=reference_cache, use_cache=True).logits
                streamed = forward_logits(config, layer_cls, checkpoint, checkpoint.tensor("model.embed_tokens.weight"),
                                          checkpoint.tensor("lm_head.weight"), norm, rotary, streamed_cache, tokens, seen)
                assert torch.equal(native, streamed)
                seen += tokens.shape[1]
                tokens = native[:, -1].argmax(-1).reshape(1, 1)
    finally:
        torch.set_default_dtype(previous_dtype)
        torch.set_num_threads(previous_threads)
