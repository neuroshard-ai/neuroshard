import copy

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from neuroshard.evolution import block_expert as study
from neuroshard.evolution import block_expert_run as runner


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny():
    torch.manual_seed(9)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, attention_dropout=0.0,
        eos_token_id=31, pad_token_id=0))
    model.eval()
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 0]]),
             "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]),
             "labels": torch.tensor([[-100, -100, -100, 4, 5], [-100, -100, 4, 5, -100]])}
    return model, batch


def test_identity_blocks_preserve_full_logits_and_cached_generation():
    model, batch = tiny()
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    with torch.no_grad():
        before = model(**inputs).logits
        generation = model.generate(batch["input_ids"][:1, :3], do_sample=False, max_new_tokens=4)
    base_parameters = sum(p.numel() for p in model.parameters())
    blocks = study.install_blocks(model, 2, expansion=True)
    with torch.no_grad():
        after = model(**inputs).logits
        expanded_generation = model.generate(batch["input_ids"][:1, :3], do_sample=False, max_new_tokens=4)
    assert torch.equal(before, after)
    assert torch.equal(generation, expanded_generation)
    assert sum(p.numel() for p in model.parameters()) == base_parameters + sum(p.numel() for p in blocks.parameters())
    assert [b.self_attn.layer_idx for b in blocks] == [4, 5]


@pytest.mark.parametrize("expansion", [False, True])
def test_cached_training_matches_full_loss_gradients_and_update(expansion):
    model, batch = tiny()
    prefixes = study.capture_prefixes(model, batch, 2)
    full = copy.deepcopy(model)
    cached = copy.deepcopy(model)
    full_blocks = study.install_blocks(full, 2, expansion=expansion)
    cached_blocks = study.install_blocks(cached, 2, expansion=expansion)
    frozen = [(name, p) for name, p in full.named_parameters() if not p.requires_grad]
    before = study.parameters_identity(frozen)
    cached.model.layers = cached_blocks
    cached.config.num_hidden_layers = 2
    full.train()
    cached.train()
    full_loss = study.loss(full, batch)
    cached_loss = study.loss(cached, batch, hidden=prefixes["expert" if expansion else "control"])
    assert torch.equal(full_loss, cached_loss)
    full_loss.backward()
    cached_loss.backward()
    for a, b in zip(full_blocks.parameters(), cached_blocks.parameters()):
        assert torch.equal(a.grad, b.grad)
    for blocks in (full_blocks, cached_blocks):
        torch.optim.AdamW(blocks.parameters(), lr=0.001).step()
    assert study.parameters_identity(full_blocks.named_parameters()) == study.parameters_identity(cached_blocks.named_parameters())
    assert study.parameters_identity(frozen) == before


def test_selective_head_matches_complete_masked_causal_loss():
    model, batch = tiny()
    selected = study.loss(model, batch)
    standard = model(**batch, use_cache=False).loss
    assert torch.allclose(selected, standard, atol=1e-6, rtol=0)
    changed = dict(batch, labels=batch["labels"].clone())
    changed["labels"][0, -1] = 6
    assert not torch.equal(study.loss(model, changed), selected)


def test_preparation_checks_both_paths_without_mutating_parent():
    model, batch = tiny()
    before = study.parameters_identity(model.named_parameters())
    prefixes = study.capture_prefixes(model, batch, 2)
    result = runner.verify_cached_execution(model, batch, prefixes, 2, 2e-5)
    assert result["initial_parent_logits_exact"]
    assert result["maximum_logit_error"] == {"expert": 0, "control": 0}
    assert len(model.model.layers) == model.config.num_hidden_layers == 4
    assert study.parameters_identity(model.named_parameters()) == before


def test_fresh_pairs_and_balanced_prompt_coverage():
    plan = study.load_plan()
    data = study.load_data(plan)
    pairs = [(row["a"], row["b"]) for rows in data["roles"].values() for row in rows]
    assert len(pairs) == len(set(pairs)) == 1280
    assert not set(pairs) & study.excluded_pairs()
    for role, rows in data["roles"].items():
        assert len(rows) == plan["data"]["counts"][role]
        for row in rows:
            total = row["a"] + row["b"]
            assert row["answer"] == str(total % 7 if row["family"] == "modular-addition" else total)
    assert not plan["selector_training_authorized"]


def receipt(data, new_correct, retained_correct):
    result = {"peak_rss_bytes": 1000}
    for role, correct in (("development", new_correct), ("retention", retained_correct)):
        result[role] = []
        for index, truth in enumerate(data["roles"][role]):
            value = truth["answer"] if index < correct else "999"
            result[role].append({"id": truth["id"], "answer": truth["answer"], "text": value,
                                 "parsed_answer": value, "passed": value == truth["answer"],
                                 "terminated": True, "seconds": 1.0, "selection": "declared-arm"})
    return result


def test_generated_answers_and_actual_control_spend_govern_handoff():
    plan = study.load_plan()
    data = study.load_data(plan)
    parent, expert, control = receipt(data, 0, 15), receipt(data, 32, 4), receipt(data, 10, 3)
    training = {"frozen_unchanged": True}
    control_training = {"matched_budget": True, "optimization_cpu_seconds": 100}
    result = study.score(plan, data, parent, expert, control, training, control_training, 100)
    assert result["passed"]
    assert len(result["retention"]["expert"]["lost"]) == 11
    assert not result["automatic_serving_proven"] and not result["admission_evidence"]
    assert not result["selector_training_authorized"]
    weak = receipt(data, 0, 4)
    assert not study.score(plan, data, parent, weak, control, training, control_training, 100)["passed"]
    assert not study.score(plan, data, parent, expert, control, training, control_training, 101)["passed"]
    bad = copy.deepcopy(expert)
    bad["development"][0]["terminated"] = False
    with pytest.raises(ValueError, match="complete generated answer"):
        study.score(plan, data, parent, bad, control, training, control_training, 100)
    bad = copy.deepcopy(expert)
    bad["development"][0]["id"] = "another-question"
    with pytest.raises(ValueError, match="identities"):
        study.score(plan, data, parent, bad, control, training, control_training, 100)


def test_checkpoint_binding_and_corruption_are_rejected(tmp_path):
    model, _ = tiny()
    blocks = study.install_blocks(model, 2, expansion=True)
    optimizer = torch.optim.AdamW(blocks.parameters())
    directory = tmp_path / "checkpoint"
    expected = runner.checkpoint(directory, blocks, optimizer, {"test": 1}, "expert-train", 0)
    runner.restore(directory, blocks, {"test": 1}, "expert-train", expected)
    with pytest.raises(ValueError, match="binding"):
        runner.restore(directory, blocks, {"test": 2}, "expert-train", expected)
    with (directory / "weights.safetensors").open("ab") as output:
        output.write(b"changed")
    with pytest.raises(ValueError, match="bytes"):
        runner.restore(directory, blocks, {"test": 1}, "expert-train", expected)


def test_no_training_without_protected_baseline(tmp_path, monkeypatch):
    data = study.load_data(study.load_plan())
    calls = []
    monkeypatch.setattr(study, "bind_freeze", lambda **kwargs: "test")
    monkeypatch.setattr(runner, "verify", lambda path: None)
    def isolated(arm, *args):
        calls.append(arm)
        return {**receipt(data, 0, 0), "binding": {}}
    monkeypatch.setattr(runner, "isolated", isolated)
    result = runner.run(tmp_path, tmp_path / "study")
    assert calls == ["baseline"] and not result["passed"]
    assert "no training" in result["error"]


def test_sources_must_be_committed_before_execution(monkeypatch):
    assert study.bind_freeze() == study.identity(study.inventory())
    monkeypatch.setattr(study.subprocess, "check_output", lambda *args, **kwargs: b"changed")
    with pytest.raises(ValueError, match="Commit every frozen source"):
        study.bind_freeze(committed=True)
