"""CPU execution of append-only growth. Fresh questions. No GPU. No 0.4.0 upgrade."""
import json
import subprocess

from neuroshard.evolution.append_only_growth import CONTRACT_IDENTITY as RULE, score_served
from neuroshard.evolution.block_expert import (
    EVAL_ROLES, ROLES, answer_logits, capture_prefixes, install_blocks, loss, parameters_identity,
)
from neuroshard.evolution.reference_data import identity, sha256
from neuroshard.evolution.seed import FILES, MODEL_REPO, MODEL_REVISION
from neuroshard.evolution.staged_integration import p95, root


FORMAT = "neuroshard-append-only-execution-v1"
PLAN = "config/experiments/append-only-execution.json"
DATA = "config/experiments/append-only-execution-data.json"
FREEZE = "config/experiments/append-only-execution-freeze.json"
CONTRACT = "3bffe777d2bc752568e0b83d59d36cd7f4d8fcec866d16925d8af4dea6a38370"
EXCLUDED = (
    "config/experiments/staged-integration-data.json",
    "config/experiments/staged-answering-data.json",
    "config/experiments/staged-answer-calibration-system.json",
    "config/experiments/block-expert-data.json",
    "config/experiments/block-expert-measure-data.json",
)
SOURCES = (
    PLAN, DATA, "docs/APPEND_ONLY_EXECUTION.md", "docs/APPEND_ONLY_GROWTH.md",
    "config/experiments/append-only-growth.json",
    "src/neuroshard/evolution/append_only_execution.py",
    "src/neuroshard/evolution/append_only_execution_run.py",
    "src/neuroshard/evolution/append_only_growth.py",
    "scripts/run_append_only_execution.py",
    "tests/evolution/test_append_only_execution.py",
    "src/neuroshard/evolution/block_expert.py",
    "src/neuroshard/evolution/block_expert_run.py",
    *EXCLUDED,
)


def load_plan():
    plan = json.loads((root() / PLAN).read_text())
    if identity(plan) != CONTRACT:
        raise ValueError("Append-only execution contract changed; declare a separate experiment")
    if (plan["model"] != {"repo": MODEL_REPO, "revision": MODEL_REVISION, "files": FILES}
            or plan["host"] != "cpu" or plan["gpu_launch_authorized"] or plan["upgrade_public_0_4_0"]):
        raise ValueError("Only a CPU execution is authorized, and it does not upgrade 0.4.0")
    if plan["rule"] != RULE or plan["reuses_opened_measurement"]:
        raise ValueError("Execution is bound to the append-only rule and fresh questions")
    if plan["train"] is not True:
        raise ValueError("This execution freeze is the one that trains")
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
            "license": "Apache-2.0", "source": "generated-arithmetic", "confirmation": None}


def load_data(plan):
    data = json.loads((root() / DATA).read_text())
    if data != make_data(plan):
        raise ValueError("Data differ from the frozen generator")
    return data


def inventory():
    plan = load_plan()
    load_data(plan)
    return {"format": FORMAT + "/freeze", "contract": identity(plan), "rule": RULE,
            "files": {name: sha256(root() / name) for name in SOURCES},
            "gpu_launch_authorized": False, "upgrade_public_0_4_0": False}


def bind_freeze(*, committed=False):
    saved = json.loads((root() / FREEZE).read_text())
    if saved != inventory():
        raise ValueError("Append-only execution sources differ from the freeze")
    if committed:
        for name in (*SOURCES, FREEZE):
            value = subprocess.check_output(["git", "show", "HEAD:" + name], cwd=root(),
                                            stderr=subprocess.DEVNULL)
            if value != (root() / name).read_bytes():
                raise ValueError("Commit every frozen source before running: " + name)
    return identity(saved)


def _check_rows(arm, expected):
    from neuroshard.evolution.staged_answer_format import answer_value
    if [row["id"] for row in arm] != [row["id"] for row in expected]:
        raise ValueError("Evaluation identities changed")
    for row, truth in zip(arm, expected):
        parsed = answer_value(row["text"], truth, terminated=row["terminated"])
        if row["parsed_answer"] != parsed or row["passed"] != (parsed == truth["answer"]):
            raise ValueError("Scoring differs from the complete generated answer")


def score(plan, data, parent, expert, control, training, control_training, target):
    for role in EVAL_ROLES:
        for arm in (parent, expert, control):
            _check_rows(arm[role], data["roles"][role])
    protected = [row["id"] for row in parent["retention"] if row["passed"]]
    if len(protected) < plan["gates"]["minimum_protected"]:
        raise ValueError("Baseline has no protected answer to preserve")
    served_new = score_served(parent["development"], expert["development"], protected)
    served_old = score_served(parent["retention"], expert["retention"], protected)
    latencies = {name: p95([row["seconds"] for role in EVAL_ROLES for row in arm[role]])
                 for name, arm in (("expert", expert), ("control", control))}
    gates = {
        "protected_kept": served_old["passed_growth_rule"] and not served_old["protected_lost"],
        "served_gain": len(served_new["gained"]) >= plan["gates"]["minimum_served_gain"],
        "parent_unchanged": training["frozen_unchanged"],
        "control_budget": control_training["matched_budget"]
                          and control_training["optimization_cpu_seconds"] >= target,
        "latency": latencies["expert"] <= min(plan["gates"]["maximum_p95_seconds"],
                                               plan["gates"]["maximum_latency_ratio"] * latencies["control"]),
        "memory": expert["peak_rss_bytes"] <= min(plan["budget"]["maximum_peak_rss_bytes"],
                       plan["gates"]["maximum_memory_ratio"] * control["peak_rss_bytes"]),
    }
    passed = all(gates.values())
    return {"format": FORMAT + "/result", "passed": passed, "gates": gates,
            "protected": protected, "served_new": served_new, "served_retention": served_old,
            "control_new": sum(row["passed"] for row in control["development"]),
            "control_retention": sum(row["passed"] for row in control["retention"]),
            "next": "review-before-any-ledger-settlement" if passed else "stop-this-candidate",
            "settlement_authorized": False, "upgrade_public_0_4_0": False,
            "gpu_launch_authorized": False, "admission_evidence": False, "item4_complete": False}
