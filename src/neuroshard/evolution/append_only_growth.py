"""Append-only serving: the parent keeps protected answers.

A new shard is consulted only where the parent missed. This module does not
train, authorize a GPU, or reopen the failed block-expert measurement.
"""
import json
from pathlib import Path

from neuroshard.evolution.reference_data import identity


FORMAT = "neuroshard-append-only-growth-v1"
CONTRACT_IDENTITY = "fbc5b5a90694f6b20d4754959d5c9eab1df2fcba20be6cf8737e5b8c77975834"
PLAN = "config/experiments/append-only-growth.json"


def spec_path():
    marker = Path(PLAN)
    for parent in Path(__file__).resolve().parents:
        candidate = parent / marker
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("append-only-growth.json is not next to this source tree")


def load_spec():
    spec = json.loads(spec_path().read_text())
    if identity(spec) != CONTRACT_IDENTITY:
        raise ValueError("Append-only contract changed; declare a separate experiment")
    if spec.get("train") is not False or spec.get("gpu_launch_authorized") is not False:
        raise ValueError("This specification does not authorize training or a GPU launch")
    if spec.get("later_execution", {}).get("authorized") is not False:
        raise ValueError("A later execution freeze is required before any run")
    if spec.get("reuses_opened_measurement") is not False:
        raise ValueError("The failed measurement cases stay closed")
    return spec


def serve(parent, added, protected):
    """Return the served reply and which arm produced it.

    `parent` and `added` are scored rows with `id` and `passed`. Protected ids
    always keep the parent reply, including when the parent was wrong: those
    identities were reserved before training and cannot be swapped afterward.
    Elsewhere the added shard is used only when the parent missed and the shard
    is correct.
    """
    if parent["id"] != added["id"]:
        raise ValueError("Parent and added rows must be the same question")
    if parent["id"] in protected or parent["passed"]:
        return parent, "parent"
    if added["passed"]:
        return added, "added"
    return parent, "parent"


def score_served(parent_rows, added_rows, protected):
    """Monotonic growth: protected parent successes remain, misses may be filled."""
    spec = load_spec()
    if [row["id"] for row in parent_rows] != [row["id"] for row in added_rows]:
        raise ValueError("Served rows changed identity or order")
    if not protected:
        raise ValueError("Name the protected parent answers before serving")
    served = []
    for parent, added in zip(parent_rows, added_rows):
        reply, source = serve(parent, added, set(protected))
        served.append({"id": parent["id"], "passed": reply["passed"], "source": source})
    protected_lost = [
        row["id"] for row, parent in zip(served, parent_rows)
        if parent["id"] in protected and parent["passed"] and not row["passed"]
    ]
    gained = [
        row["id"] for row, parent in zip(served, parent_rows)
        if not parent["passed"] and row["passed"]
    ]
    return {
        "format": FORMAT + "/served",
        "passed_growth_rule": not protected_lost and all(
            row["source"] == "parent" for row, parent in zip(served, parent_rows)
            if parent["id"] in protected),
        "protected_lost": protected_lost,
        "gained": gained,
        "served": served,
        "train": False,
        "gpu_launch_authorized": False,
        "admission_evidence": False,
        "item4_complete": False,
        "opened_measurement_is_not_a_pass": True,
        "spec": identity(spec),
    }
