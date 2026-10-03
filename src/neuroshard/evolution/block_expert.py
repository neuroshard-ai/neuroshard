"""CPU-only competence study for an identity-initialized block expert.

This experiment has no selector. It measures an explicitly selected expert,
not automatic serving, network admission, or distributed execution.
"""
import copy
import hashlib
import json
import random
import subprocess
import time

import torch
from torch import nn

from neuroshard.evolution.reference_data import identity, sha256
from neuroshard.evolution.seed import FILES, MODEL_REPO, MODEL_REVISION
from neuroshard.evolution.staged_answer_format import answer_value
from neuroshard.evolution.staged_answering import SOURCES as EARLIER_SOURCES
from neuroshard.evolution.staged_integration import p95, root


FORMAT = "neuroshard-block-expert-v1"
PLAN = "config/experiments/block-expert.json"
DATA = "config/experiments/block-expert-data.json"
FREEZE = "config/experiments/block-expert-freeze.json"
CONTRACT = "6cc2cae343b7f0309c85ecc9096885646395b73b4f8d1dceac0f13904fcf7c2c"
EXCLUDED = (
    "config/experiments/staged-integration-data.json",
    "config/experiments/staged-answering-data.json",
    "config/experiments/staged-answer-calibration-system.json",
)
SOURCES = tuple(dict.fromkeys((*EARLIER_SOURCES, *EXCLUDED, PLAN, DATA,
    "src/neuroshard/evolution/block_expert.py",
    "src/neuroshard/evolution/block_expert_run.py",
    "scripts/run_block_expert.py", "tests/evolution/test_block_expert.py",
    "docs/BLOCK_EXPERT.md")))
ROLES = ("train_new", "train_replay", "development", "retention")
EVAL_ROLES = ROLES[2:]


def load_plan():
    plan = json.loads((root() / PLAN).read_text())
    if identity(plan) != CONTRACT:
        raise ValueError("Block expert contract changed; declare a separate experiment")
    if (plan["model"] != {"repo": MODEL_REPO, "revision": MODEL_REVISION, "files": FILES}
            or plan["host"] != "cpu" or plan["gpu_launch_authorized"]
            or plan["selector_training_authorized"] or plan["admission_evidence"]):
        raise ValueError("Only the CPU expert-competence experiment is authorized")
    return plan


def excluded_pairs():
    pairs = set()
    for name in EXCLUDED:
        data = json.loads((root() / name).read_text())
        for rows in data["roles"].values():
            pairs.update(tuple(sorted((row["a"], row["b"]))) for row in rows)
    return pairs


def make_data(plan):
    excluded = excluded_pairs()
    limit = plan["data"]["operand_limit_exclusive"]
    pairs = [(a, b) for a in range(limit) for b in range(a, limit) if (a, b) not in excluded]
    pairs.sort(key=lambda pair: identity({"namespace": FORMAT, "seed": plan["data"]["seed"],
                                        "pair": pair}))
    offset, roles = 0, {}
    for role in ROLES:
        count = plan["data"]["counts"][role]
        selected = pairs[offset:offset + count]
        if len(selected) != count:
            raise ValueError("Fresh operand pool exhausted")
        offset += count
        rows = []
        for index, (a, b) in enumerate(selected):
            modular = role in {"train_new", "development"}
            family = "modular-addition" if modular else "addition"
            templates = plan["data"]["prompts"][family]
            prompt = templates[index % len(templates)].format(a=a, b=b)
            task = {"family": family, "a": a, "b": b}
            rows.append({"id": identity({"namespace": FORMAT, **task}), **task,
                         "messages": [{"role": "system", "content": plan["data"]["system"]},
                                      {"role": "user", "content": prompt}],
                         "answer": str((a + b) % 7 if modular else a + b)})
        roles[role] = rows
    return {"format": FORMAT + "/data", "roles": roles, "excluded_pairs": len(excluded),
            "exclusions": {name: sha256(root() / name) for name in EXCLUDED},
            "license": "Apache-2.0", "source": "generated-arithmetic", "confirmation": None}


def load_data(plan):
    data = json.loads((root() / DATA).read_text())
    if data != make_data(plan):
        raise ValueError("Data differ from the frozen generator")
    return data


def inventory():
    plan = load_plan()
    load_data(plan)
    return {"format": FORMAT + "/freeze", "contract": identity(plan),
            "files": {name: sha256(root() / name) for name in SOURCES},
            "gpu_launch_authorized": False, "admission_evidence": False}


def bind_freeze(*, committed=False):
    saved = json.loads((root() / FREEZE).read_text())
    if saved != inventory():
        raise ValueError("Block expert sources differ from the freeze")
    if committed:
        for name in (*SOURCES, FREEZE):
            value = subprocess.check_output(["git", "show", "HEAD:" + name], cwd=root(),
                                            stderr=subprocess.DEVNULL)
            if value != (root() / name).read_bytes():
                raise ValueError("Commit every frozen source before running: " + name)
    return identity(saved)


def install_blocks(model, count, *, expansion):
    """Return the sole trainable partition; both arms leave the head frozen."""
    if (not 0 < count < len(model.model.layers) or model.config.attention_dropout != 0
            or any(p.device.type != "cpu" for p in model.parameters())):
        raise ValueError("CPU blocks require a frozen prefix and zero attention dropout")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    original_depth = len(model.model.layers)
    if expansion:
        blocks = nn.ModuleList(copy.deepcopy(list(model.model.layers[-count:])))
        for index, block in enumerate(blocks):
            block.self_attn.layer_idx = original_depth + index
            nn.init.zeros_(block.self_attn.o_proj.weight)
            nn.init.zeros_(block.mlp.down_proj.weight)
        model.model.layers.extend(blocks)
        model.config.num_hidden_layers = original_depth + count
    else:
        blocks = nn.ModuleList(list(model.model.layers[-count:]))
    for parameter in blocks.parameters():
        parameter.requires_grad_(True)
    return blocks


def capture_prefixes(model, batch, count):
    """Cache only frozen activations, before final normalization, for both arms."""
    captured = {}
    control_ready = None

    def capture(name):
        def hook(module, arguments):
            nonlocal control_ready
            captured[name] = arguments[0].detach().clone()
            if name == "control":
                control_ready = time.process_time()
            else:
                captured["expert_extension_cpu_seconds"] = time.process_time() - control_ready
        return hook

    hooks = [model.model.layers[-count].register_forward_pre_hook(capture("control")),
             model.model.norm.register_forward_pre_hook(capture("expert"))]
    try:
        with torch.no_grad():
            model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    return captured


def answer_logits(model, batch, *, hidden=None):
    """Exactly the supervised causal positions; never project masked prompt tokens."""
    inputs = {"input_ids": batch["input_ids"]} if hidden is None else {"inputs_embeds": hidden}
    outputs = model.model(**inputs, attention_mask=batch["attention_mask"], use_cache=False)
    targets = batch["labels"][:, 1:]
    active = targets != -100
    if not bool(active.any()):
        raise ValueError("No assistant targets")
    return model.lm_head(outputs.last_hidden_state[:, :-1][active]).float(), targets[active]


def loss(model, batch, *, hidden=None):
    logits, targets = answer_logits(model, batch, hidden=hidden)
    return nn.functional.cross_entropy(logits, targets)


def parameters_identity(parameters):
    digest = hashlib.sha256()
    for name, parameter in parameters:
        tensor = parameter.detach().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def paired_gain(left, right, plan):
    if len(left) != len(right) or not left:
        raise ValueError("Unpaired answers")
    differences = [int(a["passed"]) - int(b["passed"]) for a, b in zip(left, right)]
    rng = random.Random(plan["gates"]["bootstrap_seed"])
    draws = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences)
                   for _ in range(plan["gates"]["bootstrap_draws"]))
    return {"net": sum(differences), "unique_gains": differences.count(1),
            "unique_losses": differences.count(-1),
            "lower_95": draws[int(0.025 * len(draws))]}


def score(plan, data, parent, expert, control, training, control_training, target):
    counts = {}
    for name, arm in (("parent", parent), ("expert", expert), ("control", control)):
        counts[name] = {}
        for role in EVAL_ROLES:
            rows, expected = arm[role], data["roles"][role]
            if [r["id"] for r in rows] != [r["id"] for r in expected]:
                raise ValueError("Evaluation identities changed")
            for row, truth in zip(rows, expected):
                parsed = answer_value(row["text"], truth, terminated=row["terminated"])
                if (type(row["terminated"]) is not bool or type(row["passed"]) is not bool
                        or row["answer"] != truth["answer"] or row["parsed_answer"] != parsed
                        or row["passed"] != (parsed == truth["answer"])):
                    raise ValueError("Scoring differs from the complete generated answer")
            counts[name][role] = sum(row["passed"] for row in rows)
    gains = {name: paired_gain(expert["development"], arm["development"], plan)
             for name, arm in (("parent", parent), ("control", control))}
    protection = {}
    for name, arm in (("expert", expert), ("control", control)):
        protection[name] = {
            "lost": [a["id"] for a, b in zip(parent["retention"], arm["retention"])
                     if a["passed"] and not b["passed"]],
            "gained": [a["id"] for a, b in zip(parent["retention"], arm["retention"])
                       if not a["passed"] and b["passed"]],
        }
    latencies = {name: p95([row["seconds"] for role in EVAL_ROLES for row in arm[role]])
                 for name, arm in (("expert", expert), ("control", control))}
    gates = {
        "competence": counts["expert"]["development"] >= plan["gates"]["minimum_new_correct"],
        "gain_over_both": all(g["net"] >= plan["gates"]["minimum_gain"] and g["lower_95"] > 0
                              for g in gains.values()),
        "nonempty_protection": counts["parent"]["retention"] >= plan["gates"]["minimum_protected"],
        "parent_unchanged": training["frozen_unchanged"],
        "control_budget": control_training["matched_budget"]
                          and control_training["optimization_cpu_seconds"] >= target,
        "latency": latencies["expert"] <= min(plan["gates"]["maximum_p95_seconds"],
                                               plan["gates"]["maximum_latency_ratio"] * latencies["control"]),
        "memory": expert["peak_rss_bytes"] <= min(plan["budget"]["maximum_peak_rss_bytes"],
                       plan["gates"]["maximum_memory_ratio"] * control["peak_rss_bytes"]),
    }
    passed = all(gates.values())
    return {"format": FORMAT + "/result", "passed": passed, "gates": gates, "counts": counts,
            "gain": gains, "retention": protection, "p95_seconds": latencies,
            "next": "review-expert-before-separate-selector-contract" if passed else "stop-this-candidate",
            "scope": "Explicit expert competence only; retention losses are reported, not repaired by an oracle.",
            "automatic_serving_proven": False, "selector_training_authorized": False,
            "gpu_launch_authorized": False, "admission_evidence": False, "item4_complete": False}
