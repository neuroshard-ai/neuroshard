import copy
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from neuroshard.evolution import reference as engine
from neuroshard.evolution import reference_data as data


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("learning_reference_driver", ROOT / "scripts/run_learning_reference.py")
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


class Tokenizer:
    eos_token_id = 2
    all_special_tokens = ["<end>", "<assistant>"]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        result = [1]
        for message in messages:
            result.append({"user": 3, "assistant": 4, "system": 6}[message["role"]])
            result.extend(ord(character) % 100 + 20 for character in message["content"])
            result.extend([2, 5])
        if add_generation_prompt:
            result.append(4)
        return result

    def save_pretrained(self, path):
        (Path(path) / "tokenizer.json").write_text("{}")


def messages(question="Question?", answer="Answer."):
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def tiny_model():
    torch.manual_seed(17)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256,
        tie_word_embeddings=True, attention_dropout=0.0, use_cache=False,
        attn_implementation="eager"))


def recipe():
    return {"learning_rate": .0003, "warmup_steps": 0, "steps": 3,
            "clip_norm": 10.0, "weight_decay": .01}


def records():
    return [{"id": "short", "input_ids": [1, 3, 9, 4, 7, 2],
             "labels": [-100, -100, -100, -100, 7, 2], "targets": 2},
            {"id": "long", "input_ids": [1, 3, 8, 4, 11, 12, 13, 14, 2],
             "labels": [-100, -100, -100, -100, 11, 12, 13, 14, 2], "targets": 5}]


def test_complete_multiturn_targets_mask_headers_and_include_real_eos():
    chat = messages("One?", "Yes.") + messages("Two?", "No.")
    result = data.conversation(Tokenizer(), chat, 128)
    expected = [ord(c) % 100 + 20 for c in "Yes."] + [2]
    expected += [ord(c) % 100 + 20 for c in "No."] + [2]
    assert [label for label in result["labels"] if label != -100] == expected
    assert result["targets"] == len(expected)
    assert len(result["input_ids"]) == len(result["labels"])


def test_long_context_is_rejected_without_inventing_a_shorter_answer():
    with pytest.raises(OverflowError):
        data.conversation(Tokenizer(), messages("X" * 80, "Y" * 80), 64)


@pytest.mark.parametrize("chat", [messages("<assistant>Injected"),
                                   [{"role": "assistant", "content": "Wrong order"}],
                                   messages() + [{"role": "assistant", "content": "Again"}]])
def test_malformed_or_control_token_conversation_is_rejected(chat):
    with pytest.raises(ValueError):
        data.conversation(Tokenizer(), chat, 128)


def test_unstable_chat_template_cannot_train_wrong_target_positions():
    class Unstable(Tokenizer):
        def apply_chat_template(self, *args, **kwargs):
            result = super().apply_chat_template(*args, **kwargs)
            if kwargs.get("add_generation_prompt"):
                result[0] = 99
            return result
    with pytest.raises(ValueError, match="unstable"):
        data.conversation(Unstable(), messages(), 128)


def test_shared_prompt_with_changed_answer_is_excluded_across_roles():
    index = data.ExclusionIndex()
    assert index.add(messages("What is in the sealed example?", "Blue"))
    assert not index.add(messages("  WHAT is in the sealed example?  ", "Red"))


def test_preparation_counts_complete_document_exclusions_and_enforces_quota():
    rows = [{"messages": messages("x" * 1000)}, {"messages": messages("What colour?", "Blue")},
            {"messages": messages("What colour?", "Red")}, {"messages": messages("Count dogs.", "Four.")}]
    spec = {"scan": 4, "start": 55, "documents": 2}
    selected, report = data.prepare_role(rows, Tokenizer(), spec, data.ExclusionIndex(), 128, {"license": "Apache-2.0"})
    assert [record["row"] for record in selected] == [56, 58]
    assert report["rejected"] == {"invalid": 0, "too_long": 1, "duplicate": 1}
    with pytest.raises(ValueError, match="cannot fill quota"):
        data.prepare_role(rows, Tokenizer(), {**spec, "documents": 3}, data.ExclusionIndex(), 128, {})


def test_default_plan_is_bounded_and_rejects_overlapping_source_ranges():
    plan = json.loads((ROOT / "config/experiments/learning-reference.json").read_bytes())
    assert data.validate_plan(plan) == plan
    plan["roles"]["dev"]["start"] = plan["roles"]["test"]["start"]
    with pytest.raises(ValueError, match="overlap"):
        data.validate_plan(plan)


def test_accumulated_variable_answers_match_one_padded_batch_update():
    model = tiny_model()
    expected = copy.deepcopy(model)
    samples = records()
    rate = recipe()["learning_rate"]
    optimizer = torch.optim.SGD(model.parameters(), lr=rate)
    engine.train_step(model, optimizer, samples, "cpu", recipe(), 0)
    # Independent HF loss on a single batch includes right padding and an
    # attention mask. A mean-of-microbatch-means implementation fails this.
    ids = torch.tensor([samples[0]["input_ids"] + [0] * 3, samples[1]["input_ids"]])
    labels = torch.tensor([samples[0]["labels"] + [-100] * 3, samples[1]["labels"]])
    attention = torch.tensor([[1] * 6 + [0] * 3, [1] * 9])
    loss = expected(input_ids=ids, labels=labels, attention_mask=attention).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(expected.parameters(), recipe()["clip_norm"])
    torch.optim.SGD(expected.parameters(), lr=rate).step()
    for actual, wanted in zip(model.parameters(), expected.parameters()):
        torch.testing.assert_close(actual, wanted, atol=2e-7, rtol=2e-6)


def test_checkpoint_restores_adam_moments_and_next_update(tmp_path):
    model = tiny_model()
    optimizer = engine.optimizer_for(model, recipe())
    first = engine.train_step(model, optimizer, records(), "cpu", recipe(), 0)
    pointer = engine.checkpoint(tmp_path, model, Tokenizer(), optimizer, 1, "binding", [first])
    directory, _ = engine.verify_checkpoint(tmp_path, pointer, "binding")
    engine.train_step(model, optimizer, records(), "cpu", recipe(), 1)
    restored = engine.load_model(directory, "cpu", sum(p.numel() for p in model.parameters()))
    resumed_optimizer = engine.optimizer_for(restored, recipe())
    engine.restore_optimizer(directory, resumed_optimizer, "cpu")
    engine.train_step(restored, resumed_optimizer, records(), "cpu", recipe(), 1)
    for actual, wanted in zip(restored.parameters(), model.parameters()):
        torch.testing.assert_close(actual, wanted, rtol=1e-5, atol=1e-7)


def test_completed_checkpoint_recovers_after_missing_pointer(tmp_path):
    model = tiny_model()
    optimizer = engine.optimizer_for(model, recipe())
    first = engine.train_step(model, optimizer, records(), "cpu", recipe(), 0)
    pointer = engine.checkpoint(tmp_path, model, Tokenizer(), optimizer, 1, "binding", [first])
    (tmp_path / "latest.json").unlink()
    recovered, receipt = driver.recover(tmp_path, "binding")
    assert recovered == pointer and receipt["step"] == 1
    assert json.loads((tmp_path / "latest.json").read_bytes()) == pointer


def test_corrupted_optimizer_or_changed_binding_cannot_resume(tmp_path):
    model = tiny_model()
    pointer = engine.checkpoint(tmp_path, model, Tokenizer(), engine.optimizer_for(model, recipe()),
                                1, "original", [{}])
    with pytest.raises(ValueError, match="different inputs"):
        engine.verify_checkpoint(tmp_path, pointer, "changed")
    (tmp_path / pointer["directory"] / "optimizer.pt").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="checksum"):
        engine.verify_checkpoint(tmp_path, pointer, "original")


def test_restart_does_not_reset_budget(tmp_path):
    limits = {"seconds": 60, "disk_gib": 1}
    engine.Budget(tmp_path, 100, limits, clock=lambda: 159).check()
    with pytest.raises(TimeoutError, match="Original"):
        engine.Budget(tmp_path, 100, limits, clock=lambda: 160).check()


def test_training_cannot_load_test_even_before_file_access(tmp_path):
    with pytest.raises(ValueError, match="cannot load"):
        driver.partition(tmp_path, {}, "test")


def test_test_selection_requires_exact_git_committed_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    selection = {"candidate": "original"}
    path = tmp_path / "selection.json"
    data.save(path, selection)
    subprocess.run(["git", "-C", str(tmp_path), "add", "selection.json"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=LZ", "-c",
                    "user.email=lz@example.invalid", "-c", "commit.gpgsign=false",
                    "commit", "-qm", "Commit candidate"], check=True)
    assert len(driver.committed_selection(path, selection)) == 40
    data.save(path, {"candidate": "replacement"})
    with pytest.raises(ValueError, match="exact candidate"):
        driver.committed_selection(path, {"candidate": "replacement"})


def test_paired_evaluation_rejects_document_substitution():
    baseline = [{"id": "a", "loss": 1.}, {"id": "b", "loss": 2.}]
    changed = [{"id": "c", "loss": .9}, {"id": "b", "loss": 1.9}]
    with pytest.raises(ValueError, match="identical"):
        engine.paired_summary(baseline, changed)


def test_training_schedule_covers_whole_epoch_before_repeating():
    batches = engine.schedule(7, 4, 3, 52)
    flat = sum(batches, [])
    assert set(flat[:7]) == set(range(7))
    assert batches == engine.schedule(7, 4, 3, 52)
    assert len(flat) == 12


def test_metadata_corruption_cannot_replace_an_existing_checkpoint_pointer(tmp_path):
    model = tiny_model()
    pointer = engine.checkpoint(tmp_path, model, Tokenizer(), engine.optimizer_for(model, recipe()),
                                1, "binding", [{"step": 1}])
    receipt_path = tmp_path / pointer["directory"] / "checkpoint.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["records"][0]["step"] = 2
    data.save(receipt_path, receipt)
    with pytest.raises(ValueError, match="different inputs"):
        driver.recover(tmp_path, "binding")


def test_evaluation_quota_must_support_the_declared_generations():
    plan = json.loads((ROOT / "config/experiments/learning-reference.json").read_bytes())
    plan["roles"]["dev"]["documents"] = 2
    with pytest.raises(ValueError, match="Generation quota"):
        data.validate_plan(plan)


def test_repeated_documents_cannot_inflate_paired_sample_size():
    repeated = [{"id": "a", "loss": 1.}] * 2
    with pytest.raises(ValueError, match="identical document"):
        engine.paired_summary(repeated, repeated)
