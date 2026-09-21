"""Successor to the stopped staged-integration baseline; historical files stay fixed."""
import json
import subprocess

from neuroshard.evolution.reference_data import identity, sha256
from neuroshard.evolution.seed import FILES, MODEL_REPO, MODEL_REVISION
from neuroshard.evolution.staged_answer_format import GENERATION_TOKENS, answer_value
from neuroshard.evolution.staged_integration import (
    EVAL_ROLES, ROLES, SOURCES as PREDECESSOR_SOURCES,
    p95, preservation, root, validate_routes,
)


FORMAT = "neuroshard-staged-answering-v1"
PLAN = "config/experiments/staged-answering.json"
DATA = "config/experiments/staged-answering-data.json"
FREEZE = "config/experiments/staged-answering-freeze.json"
CALIBRATION = "config/experiments/staged-answer-calibration-system.json"
CONTRACT_IDENTITY = "ac66cb34bc0c529685c3271e17478b4a50965da582e346ad74b2875af9077007"
SOURCES = tuple(dict.fromkeys((*PREDECESSOR_SOURCES, PLAN, DATA, CALIBRATION,
    "config/experiments/staged-answer-calibration.json",
    "config/experiments/staged-answer-calibration-results.json",
    "docs/STAGED_ANSWERING.md", "scripts/run_staged_answering.py",
    "src/neuroshard/evolution/staged_answering.py",
    "src/neuroshard/evolution/staged_answer_format.py")))


def load_spec():
    spec = json.loads((root() / PLAN).read_text())
    if identity(spec) != CONTRACT_IDENTITY:
        raise ValueError("Declare a separate candidate instead of changing this contract")
    if (spec["model"] != {"repo": MODEL_REPO, "revision": MODEL_REVISION, "files": FILES}
            or spec["host"] != "cpu" or spec["gpu_launch_authorized"] is not False
            or spec["admission_evidence"] is not False
            or spec["training"]["generation_tokens"] != GENERATION_TOKENS):
        raise ValueError("Only the frozen 135M CPU answering study is allowed")
    if spec["confirmation_opened"] or spec["original_final_opened"] or spec["reuses_opened_64"]:
        raise ValueError("Historical evaluation sets remain excluded")
    return spec


def make_data(spec):
    previous = json.loads((root() / "config/experiments/staged-integration-data.json").read_text())
    calibration = json.loads((root() / CALIBRATION).read_text())
    excluded = {(r["a"], r["b"]) for data in (previous, calibration)
                for rows in data["roles"].values() for r in rows}
    limit = spec["data"]["operand_limit_exclusive"]
    pairs = [(a, b) for a in range(limit) for b in range(a, limit) if (a, b) not in excluded]
    pairs.sort(key=lambda pair: identity({"namespace": FORMAT, "seed": spec["data"]["seed"],
                                        "pair": pair}))
    roles, offset = {}, 0
    for role in ROLES:
        count = spec["data"]["counts"][role]
        selected = pairs[offset:offset + count]
        if len(selected) != count:
            raise ValueError("Fresh operand pool exhausted")
        offset += count
        rows = []
        for a, b in selected:
            modular = role in {"expert_new", "gate_new", "development"}
            task = {"a": a, "b": b, "family": "modular-addition" if modular else "addition"}
            prompt = (f"What is the remainder when {a} + {b} is divided by 7? Reply with only the number."
                      if modular else f"What is {a} + {b}? Reply with only the number.")
            rows.append({"id": identity({"namespace": FORMAT, **task}), **task,
                         "messages": [{"role": "system", "content": spec["answer_protocol"]["system"]},
                                      {"role": "user", "content": prompt}],
                         "answer": str((a + b) % 7 if modular else a + b)})
        roles[role] = rows
    return {"format": FORMAT + "/data", "roles": roles, "confirmation": None,
            "license": "Apache-2.0", "excluded_pairs": len(excluded),
            "previous_data": identity(previous), "calibration": identity(calibration)}


def load_data(spec):
    data = json.loads((root() / DATA).read_text())
    if data != make_data(spec):
        raise ValueError("Answering data differ from the frozen generator")
    return data


def freeze_inventory():
    spec = load_spec()
    load_data(spec)
    return {"format": FORMAT + "/freeze", "contract": identity(spec),
            "files": {name: sha256(root() / name) for name in SOURCES},
            "host": "cpu", "gpu_launch_authorized": False, "admission_evidence": False}


def bind_freeze(*, committed=False):
    saved = json.loads((root() / FREEZE).read_text())
    if saved != freeze_inventory():
        raise ValueError("Answering source or data changed after the freeze")
    if committed:
        for name in (*SOURCES, FREEZE):
            try:
                content = subprocess.check_output(["git", "show", "HEAD:" + name], cwd=root(),
                                                  stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as error:
                raise ValueError("Commit the answering candidate before running: " + name) from error
            if content != (root() / name).read_bytes():
                raise ValueError("Commit the answering candidate before running: " + name)
    return identity(saved)


def score(spec, parent, expansion, control, training, control_training):
    expected = make_data(spec)["roles"]
    for role in EVAL_ROLES:
        for arm in (parent, expansion, control):
            rows = arm[role]
            if [r["id"] for r in rows] != [r["id"] for r in expected[role]]:
                raise ValueError("Frozen evaluation identities differ")
            for row, truth in zip(rows, expected[role]):
                if row.get("automatic") is not True or type(row.get("terminated")) is not bool:
                    raise ValueError("Only complete automatic serving receipts may be scored")
                parsed = answer_value(row["text"], truth, terminated=row["terminated"])
                if (row.get("answer") != truth["answer"] or row.get("parsed_answer") != parsed
                        or type(row.get("passed")) is not bool
                        or row["passed"] != (parsed == truth["answer"])):
                    raise ValueError("Score disagrees with the complete generated answer")
        for row in expansion[role]:
            validate_routes(row, spec["training"]["generation_tokens"])
    kept = preservation(parent["retention"], expansion["retention"],
                        spec["gates"]["minimum_parent_retention_correct"])
    counts = {name: sum(r["passed"] for r in arm["development"])
              for name, arm in (("parent", parent), ("expansion", expansion), ("control", control))}
    gain = counts["expansion"] >= max(counts["parent"], counts["control"]) + spec["gates"]["minimum_new_gain"]
    used = any(b["passed"] and not a["passed"] and b["added_answer_tokens"] > 0
               for a, b in zip(parent["development"], expansion["development"]))
    latency = {name: p95([r["seconds"] for role in EVAL_ROLES for r in arm[role]])
               for name, arm in (("expansion", expansion), ("control", control))}
    gates = {
        "automatic_gain": gain, "per_answer_preservation": kept["passed"],
        "added_module_used_on_gain": used,
        "latency": latency["expansion"] <= min(spec["gates"]["maximum_p95_seconds"],
                                              latency["control"] * spec["gates"]["maximum_latency_ratio"]),
        "isolated_peak_memory": expansion["peak_rss_bytes"] <= min(
            spec["gates"]["maximum_peak_rss_bytes"],
            control["peak_rss_bytes"] * spec["gates"]["maximum_memory_ratio"]),
        "training_budget_control": control_training["matched_budget"] and (
            control_training["training_cpu_seconds"] >= training["comparison_cpu_seconds"]),
        "frozen_incumbent": training["incumbent_unchanged"],
        "gate_only_phase": training["added_unchanged_during_gate"],
    }
    diagnosis = ("complete_system_evaluated" if gain and used else
                 "expert_learning_not_observed" if not training["expert_training_signal"] else
                 "automatic_integration_not_effective")
    passed = all(gates.values())
    return {"format": FORMAT + "/result", "passed": passed, "gates": gates, "counts": counts,
            "retention": kept, "diagnosis": diagnosis, "p95_seconds": latency,
            "diagnosis_scope": "Training loss is a mechanism signal, not a generalization result.",
            "admission_evidence": False, "gpu_launch_authorized": False,
            "confirmation_opened": False, "item4_complete": False,
            "next": "review-cpu-mechanism-evidence" if passed else "stop-this-candidate"}
