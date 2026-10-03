"""Measure block-expert competence without a retention start gate.

The stopped block-expert baseline stays closed. Parent retention is saved and
reported. An unfinished reply is incorrect. It does not authorize a selector,
a GPU, or checklist credit.
"""
import json
import subprocess

from neuroshard.evolution.block_expert import (
    EVAL_ROLES, ROLES, answer_logits, capture_prefixes, install_blocks, loss,
    paired_gain, parameters_identity,
)
from neuroshard.evolution.reference_data import identity, sha256
from neuroshard.evolution.seed import FILES, MODEL_REPO, MODEL_REVISION
from neuroshard.evolution.staged_integration import p95, root


FORMAT = "neuroshard-block-expert-measure-v1"
PLAN = "config/experiments/block-expert-measure.json"
DATA = "config/experiments/block-expert-measure-data.json"
FREEZE = "config/experiments/block-expert-measure-freeze.json"
CONTRACT = "4dd616028de2fd6019677a5634c754bbcf6535db400ac74e70bcc937aa4e1ecf"
PARENT_CONTRACT = "6cc2cae343b7f0309c85ecc9096885646395b73b4f8d1dceac0f13904fcf7c2c"
EXCLUDED = (
    "config/experiments/staged-integration-data.json",
    "config/experiments/staged-answering-data.json",
    "config/experiments/staged-answer-calibration-system.json",
    "config/experiments/block-expert-data.json",
)
SOURCES = (
    PLAN, DATA, "docs/BLOCK_EXPERT_MEASURE.md",
    "src/neuroshard/evolution/block_expert_measure.py",
    "src/neuroshard/evolution/block_expert_measure_run.py",
    "scripts/run_block_expert_measure.py",
    "tests/evolution/test_block_expert_measure.py",
    "src/neuroshard/evolution/block_expert.py",
    "src/neuroshard/evolution/block_expert_run.py",
    *EXCLUDED,
)


def load_plan():
    plan = json.loads((root() / PLAN).read_text())
    if identity(plan) != CONTRACT:
        raise ValueError("Block-expert measurement contract changed; declare a separate experiment")
    if (plan["model"] != {"repo": MODEL_REPO, "revision": MODEL_REVISION, "files": FILES}
            or plan["host"] != "cpu" or plan["gpu_launch_authorized"]
            or plan["selector_training_authorized"] or plan["admission_evidence"]):
        raise ValueError("Only the CPU expert-competence measurement is authorized")
    if plan["gates"]["retention_blocks_training"] or plan["reuses_block_expert_cases"]:
        raise ValueError("Retention does not block this measurement, and opened cases stay closed")
    if plan["parent_stop"]["contract"] != PARENT_CONTRACT or plan["parent_stop"]["outcome"] != "stopped-before-training":
        raise ValueError("This measurement is bound to the stopped block-expert baseline")
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
    pairs.sort(key=lambda pair: identity({"namespace": FORMAT, "seed": plan["data"]["seed"], "pair": pair}))
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
            prompt = plan["data"]["prompts"][family][index % 4].format(a=a, b=b)
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
            "gpu_launch_authorized": False, "admission_evidence": False,
            "retention_blocks_training": False}


def bind_freeze(*, committed=False):
    saved = json.loads((root() / FREEZE).read_text())
    if saved != inventory():
        raise ValueError("Block-expert measurement sources differ from the freeze")
    if committed:
        for name in (*SOURCES, FREEZE):
            value = subprocess.check_output(["git", "show", "HEAD:" + name], cwd=root(),
                                            stderr=subprocess.DEVNULL)
            if value != (root() / name).read_bytes():
                raise ValueError("Commit every frozen source before running: " + name)
    return identity(saved)


def score(plan, data, parent, expert, control, training, control_training, target):
    """Competence is generated new answers. Retention is a report, not a start gate."""
    counts, incomplete = {}, {}
    for name, arm in (("parent", parent), ("expert", expert), ("control", control)):
        counts[name], incomplete[name] = {}, {}
        for role in EVAL_ROLES:
            rows, expected = arm[role], data["roles"][role]
            if [row["id"] for row in rows] != [row["id"] for row in expected]:
                raise ValueError("Evaluation identities changed")
            for row, truth in zip(rows, expected):
                from neuroshard.evolution.staged_answer_format import answer_value
                parsed = answer_value(row["text"], truth, terminated=row["terminated"])
                if (type(row["terminated"]) is not bool or type(row["passed"]) is not bool
                        or row["answer"] != truth["answer"] or row["parsed_answer"] != parsed
                        or row["passed"] != (parsed == truth["answer"])):
                    raise ValueError("Scoring differs from the complete generated answer")
            counts[name][role] = sum(row["passed"] for row in rows)
            incomplete[name][role] = sum(not row["terminated"] for row in rows)
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
        "gain_over_both": all(gain["net"] >= plan["gates"]["minimum_gain"] and gain["lower_95"] > 0
                              for gain in gains.values()),
        "parent_unchanged": training["frozen_unchanged"],
        "control_budget": control_training["matched_budget"]
                          and control_training["optimization_cpu_seconds"] >= target,
        "latency": latencies["expert"] <= min(plan["gates"]["maximum_p95_seconds"],
                                               plan["gates"]["maximum_latency_ratio"] * latencies["control"]),
        "memory": expert["peak_rss_bytes"] <= min(plan["budget"]["maximum_peak_rss_bytes"],
                       plan["gates"]["maximum_memory_ratio"] * control["peak_rss_bytes"]),
    }
    return {"format": FORMAT + "/result", "passed": all(gates.values()), "gates": gates,
            "counts": counts, "incomplete": incomplete, "gain": gains, "retention": protection,
            "parent_retention_correct": counts["parent"]["retention"],
            "retention_blocks_training": False, "p95_seconds": latencies,
            "next": "review-expert-before-separate-selector-contract" if all(gates.values()) else "stop-this-candidate",
            "scope": "Explicit expert competence. Unfinished replies are incorrect. Retention does not block training.",
            "automatic_serving_proven": False, "selector_training_authorized": False,
            "gpu_launch_authorized": False, "admission_evidence": False, "item4_complete": False}
