"""Strict arithmetic answer reading for the successor CPU mechanism study.

Read one complete answer, never search prose for a convenient number, compute
an answer, or use the expected label while interpreting the generated text.
"""
import re
import time

import torch

from neuroshard.evolution.reference_data import save
from neuroshard.evolution.staged_integration import EVAL_ROLES, StagedMixture


GENERATION_TOKENS = 32
NUMBER = r"(0|[1-9][0-9]*)"


def answer_value(text, row, *, terminated):
    if not terminated:
        return None
    value = text.strip()
    # A single optional terminal full stop is punctuation, not a decimal.
    numeric = re.fullmatch(NUMBER + r"\.?", value)
    if numeric:
        return numeric.group(1)
    left = rf"{row['a']}\s*\+\s*{row['b']}"
    if row["family"] == "modular-addition":
        left = rf"\({left}\)\s*(?:%|mod)\s*7"
    equation = re.fullmatch(left + r"\s*=\s*" + NUMBER + r"\.?", value)
    return equation.group(1) if equation else None


def evaluate(model, tokenizer, rows, spec, home, arm):
    module = model.model.layers[-1].mlp
    if isinstance(module, StagedMixture):
        module.set_phase("serve")
        module.record_routes = True
    model.eval()
    model.config.use_cache = True
    result = {}
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [tokenizer.eos_token_id])
    for role in EVAL_ROLES:
        result[role] = []
        for row in rows[role]:
            ids = tokenizer.apply_chat_template(row["messages"], tokenize=True,
                                                add_generation_prompt=True, return_tensors="pt")
            if ids.shape[1] > spec["training"]["max_length"]:
                raise ValueError("Prompt exceeds the frozen context budget")
            if isinstance(module, StagedMixture):
                module.trace = []
            started = time.monotonic()
            with torch.inference_mode():
                output = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                        max_new_tokens=spec["training"]["generation_tokens"],
                                        pad_token_id=tokenizer.eos_token_id)
            seconds = time.monotonic() - started
            generated = output[0, ids.shape[1]:]
            terminated = int(generated[-1]) in eos_ids
            text = tokenizer.decode(generated, skip_special_tokens=True)
            value = answer_value(text, row, terminated=terminated)
            trace = module.trace if isinstance(module, StagedMixture) else []
            result[role].append({
                "id": row["id"], "text": text, "answer": row["answer"],
                "parsed_answer": value, "terminated": terminated,
                "passed": value == row["answer"], "seconds": seconds,
                "generated_tokens": len(generated), "automatic": True, "routes": trace,
                "added_answer_tokens": sum(sum(call["answer_choices"]) for call in trace),
            })
            save(home / (arm + "-answers.json"), result)
    return result
