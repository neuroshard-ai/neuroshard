"""Paired reference comparison; completion and useful growth are separate claims."""

import math

from neuroshard.evolution.modular_reference import CATEGORIES


def percentile95(values):
    return sorted(values)[math.ceil(.95 * len(values)) - 1] if values else None


def compare(plan, rows, replay_complete=False):
    tasks = {task["id"]: task for task in plan["tasks"]}
    by_model = {name: {} for name in ("baseline", "modular")}
    for row in rows:
        model, task_id = row["model"], row["id"]
        if model not in by_model or task_id not in tasks or task_id in by_model[model]:
            raise ValueError("unknown or duplicate comparison row")
        by_model[model][task_id] = row
    complete = {name: set(records) == set(tasks) and all(not r.get("stopped") for r in records.values())
                for name, records in by_model.items()}
    correct = {name: {key for key, row in records.items() if row["passed"] and not row.get("stopped")}
               for name, records in by_model.items()}
    categories = {category: {name: sum(tasks[key]["category"] == category for key in successes)
                             for name, successes in correct.items()} for category in CATEGORIES}
    rule = plan["comparison"]
    baseline_gate = (complete["baseline"] and len(correct["baseline"]) >= rule["minimum_baseline_total"]
                     and all(row["baseline"] >= rule["minimum_baseline_per_category"] for row in categories.values()))
    both = all(complete.values())
    gains = sorted(correct["modular"] - correct["baseline"]) if both else []
    losses = sorted(correct["baseline"] - correct["modular"]) if both else []
    latency = {}
    for name, records in by_model.items():
        latency[name] = {
            "p95_seconds": percentile95([r["seconds"] for r in records.values()]),
            "p95_first_token_seconds": percentile95([r["first_token_seconds"] for r in records.values()
                                                     if r.get("first_token_seconds") is not None]),
            "peak_rss_bytes": max((r["max_rss_bytes"] for r in records.values()), default=0)}
    baseline_p95 = latency["baseline"]["p95_seconds"]
    modular_p95 = latency["modular"]["p95_seconds"]
    ratio = modular_p95 / baseline_p95 if baseline_p95 and modular_p95 is not None else None
    latency_gate = (both and ratio is not None and ratio <= rule["maximum_p95_ratio"]
                    and modular_p95 <= rule["maximum_modular_p95_seconds"])
    reference_ready = baseline_gate and both and replay_complete
    return {"baseline_gate": baseline_gate, "comparison_complete": both,
            "reference_ready": reference_ready, "category_correct": categories,
            "correct": {name: len(ids) for name, ids in correct.items()},
            "protected_ids": sorted(correct["baseline"]), "gained_ids": gains, "lost_ids": losses,
            "net_gain": len(gains) - len(losses) if both else None,
            "latency": latency, "p95_ratio": ratio, "latency_gate": latency_gate,
            "growth_screen_passed": bool(reference_ready and not losses
                                         and len(gains) >= rule["minimum_new_successes"] and latency_gate),
            "quality_ready": False, "admission_evidence": False, "milestone_complete": False}
