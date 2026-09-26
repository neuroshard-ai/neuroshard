import copy
import hashlib
import importlib.util
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Olmo2Config

from neuroshard.evolution import modular_decoder_parity as parity
from neuroshard.evolution import modular_reference_execution as execution


ROOT = Path(__file__).resolve().parents[2]


def test_audit_uses_only_opened_pinned_outputs_and_stock_source():
    plan, amendment, prior, tasks, rows = parity.inputs()
    assert len(rows) == 24 and set(tasks) == set(rows)
    assert amendment["plan_sha256"] == prior["binding"]["freeze"]["plan_sha256"]
    assert amendment["model_order"] == ["baseline"]
    assert not amendment["training_authorized"] and not amendment["gpu_launch_authorized"]
    assert not amendment["checklist_credit"]
    assert plan["limits"]["max_new_tokens"] == amendment["generation_rule"]["max_new_tokens"] == 128
    assert parity.SCRIPT in amendment["sources"]
    changed = copy.deepcopy(amendment)
    changed["upstream_sources"]["models/olmo2/modeling_olmo2.py"] = "0" * 64
    with pytest.raises(ValueError, match="upstream implementation changed"):
        parity.verify_upstream(changed)


def test_equal_argmax_cannot_hide_different_logits_or_nonfinite_values():
    expected = torch.tensor([[[1.0, 2.0, 3.0]]])
    actual = expected.clone()
    assert parity.logit_comparison(actual, expected)["equal"]
    actual[0, 0, 0] += .01
    comparison = parity.logit_comparison(actual, expected)
    assert comparison["argmax_equal"] and not comparison["equal"]
    assert comparison["maximum_absolute_difference"] > 0
    actual[0, 0, 0] = float("nan")
    comparison = parity.logit_comparison(actual, expected)
    assert not comparison["finite"] and not comparison["equal"]


def test_partial_or_disagreeing_receipt_cannot_claim_complete_parity():
    amendment = {"logit_ids": ["a"], "generation_ids": ["a", "b"]}
    rows = {"a": {"token_ids": [1, 2]}}
    result = {"execution_completed": True, "decoder_agreement": True, "stop_reason": None,
              "logit_checks": [{"id": "a", "prediction_index": i, "equal": True,
                                "argmax_equal": True, "finite": True} for i in range(2)],
              "generations": [{"id": identity, "matched": True} for identity in ("a", "b")]}
    parity.validate_completion(result, amendment, rows)
    for field in ("logit_checks", "generations"):
        partial = copy.deepcopy(result)
        partial[field].pop()
        with pytest.raises(ValueError, match="cannot pass"):
            parity.validate_completion(partial, amendment, rows)
    result["logit_checks"][0]["equal"] = False
    with pytest.raises(ValueError, match="cannot pass"):
        parity.validate_completion(result, amendment, rows)


def test_new_worker_script_must_be_in_the_committed_inventory(tmp_path):
    with pytest.raises(ValueError, match="outside the execution freeze"):
        execution.launch(tmp_path, tmp_path, {"freeze": {"sources": {}}}, "baseline", "parity", 1, 1024,
                         worker_script=parity.SCRIPT)
    assert not (tmp_path / "attempts").exists()


def test_cloud_selects_only_the_audit_and_enforces_its_own_small_budget(monkeypatch):
    spec = importlib.util.spec_from_file_location("audit_cloud", ROOT / "scripts/modular_reference_cloud.py")
    cloud = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cloud)
    limits = cloud.resources("decoder-parity")
    assert limits["hours"] == 2 and limits["planning_cap_usd"] == 6 and not limits["gpu"]
    command = cloud.remote_command("decoder-parity")
    assert any(s.endswith(parity.SCRIPT) for s in command) and "--legacy" not in command
    original_read = cloud.read
    monkeypatch.setattr(cloud, "read", lambda path: {**limits, "hours": 3, "planning_cap_usd": 7}
                        if path == cloud.ROOT / cloud.RESOURCE_PROFILES["decoder-parity"] else original_read(path))
    with pytest.raises(ValueError, match="two-hour six-dollar"):
        cloud.resources("decoder-parity")


@pytest.mark.parametrize("mismatch", [False, True])
def test_full_audit_uses_independent_loaded_model_and_standard_generation(tmp_path, monkeypatch, mismatch):
    previous_threads, previous_grad = torch.get_num_threads(), torch.is_grad_enabled()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(key, os.environ.get(key, "1"))
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng():
            torch.manual_seed(17)
            config = Olmo2Config(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                num_attention_heads=2, num_key_value_heads=2, vocab_size=32,
                                pad_token_id=0, bos_token_id=1, eos_token_id=2,
                                attention_dropout=0., tie_word_embeddings=False)
            config._attn_implementation = "eager"
            model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16).eval()
        model.save_pretrained(tmp_path / "model", max_shard_size="4KB")
        oracle = AutoModelForCausalLM.from_pretrained(tmp_path / "model", dtype=torch.bfloat16,
                                                    attn_implementation="eager").eval()

        class Tokenizer:
            def apply_chat_template(self, messages, *, tokenize, **kwargs):
                return torch.tensor([[4, 5, 6]]) if tokenize else "fixture prompt"

            def decode(self, tokens, **kwargs):
                return " ".join(map(str, tokens))

        tokenizer = Tokenizer()
        monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
        with torch.inference_mode():
            generated = oracle.generate(input_ids=torch.tensor([[4, 5, 6]]),
                attention_mask=torch.ones((1, 3), dtype=torch.long), do_sample=False, num_beams=1,
                use_cache=True, logits_to_keep=0, max_new_tokens=3)[0, 3:].tolist()
        terminated = generated[-1] == 2
        text = tokenizer.decode(generated[:-1] if terminated else generated)
        task = {"id": "fixture", "category": "conversation", "kind": "exact",
                "messages": [{"role": "user", "content": "fixture"}], "accept": ["unrelated expected answer"]}
        prior = {"text": text, "token_ids": generated, "terminated": terminated,
                 "rendered_sha256": hashlib.sha256(b"fixture prompt").hexdigest(),
                 "passed": False, "reason": "exact-mismatch" if terminated else "unterminated"}
        plan = {"limits": {"threads": 1, "max_new_tokens": 3}}
        amendment = {"logit_ids": ["fixture"], "generation_ids": ["fixture"]}
        control_calls = []
        if mismatch:
            from neuroshard.evolution import modular_reference_run
            control = {**prior, "stopped": False}

            def same_host_control(*args):
                control_calls.append(args[3]["id"])
                return control

            monkeypatch.setattr(modular_reference_run, "generate_task", same_host_control)
            prior = {**prior, "text": "a different earlier reply"}
        result = parity.diagnose(tmp_path / "model", plan, amendment, {"fixture": task},
                                 {"fixture": prior}, tmp_path / "progress.json")
        parity.validate_completion(result, amendment, {"fixture": prior})
        assert result["decoder_agreement"] is (not mismatch) and not result["admission_evidence"]
        assert len(result["logit_checks"]) == len(generated)
        assert all(row["maximum_absolute_difference"] == 0 for row in result["logit_checks"])
        assert result["generations"][0]["matched"] is (not mismatch)
        assert not result["generations"][0]["passed"]
        if mismatch:
            assert control_calls == ["fixture"]
            assert result["execution_completed"] and result["stop_reason"] == "upstream-generation-disagreement"
            assert result["same_host_streamed_matches_upstream"] and not result["same_host_streamed_matches_recorded"]
    finally:
        torch.set_num_threads(previous_threads)
        torch.set_grad_enabled(previous_grad)
