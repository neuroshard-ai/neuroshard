#!/usr/bin/env python3
"""Fetch and score the committed A1 BAR reference without training."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

from neuroshard.evolution.modular_reference import (
    assess, load_plan, plan_hash, route_estimate, score_reply)
from neuroshard.evolution.modular_reference_run import file_sha256, generate_task


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fetch", "generate-one", "evaluate"))
    parser.add_argument("--plan", type=Path, default=ROOT / "config/experiments/modular-reference-a1.json")
    parser.add_argument("--home", type=Path, default=ROOT / ".neuroshard/modular-reference")
    parser.add_argument("--which", choices=("baseline", "modular"))
    parser.add_argument("--task")
    args = parser.parse_args()
    if args.which is None:
        parser.error("--which is required")
    plan = load_plan(args.plan)
    if args.command == "fetch":
        fetch(plan, args.which, args.home)
    elif args.command == "generate-one":
        task = next(item for item in plan["tasks"] if item["id"] == args.task)
        model_dir = args.home / args.which
        result = generate_task(plan, args.which, model_dir, task)
        destination = args.home / "replies" / args.which / f"{args.task}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        evaluate(plan, args.plan, args.which, args.home)


def fetch(plan, which, home):
    spec = plan["models"][which]
    home.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(home).free
    if free < spec["storage_bytes"] + 5 * 1024 ** 3:
        raise SystemExit(f"need {spec['storage_bytes']} weight bytes plus 5 GiB, found {free} free")
    destination = home / which
    destination.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    snapshot_download(spec["repo"], revision=spec["revision"], local_dir=destination)
    if time.monotonic() - started > plan["limits"]["fetch_seconds"]:
        raise SystemExit("fetch exceeded its declared budget")
    weights = sorted(destination.glob("model-*.safetensors"))
    manifest = {
        "which": which,
        "revision": spec["revision"],
        "weights": {path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)} for path in weights},
    }
    (home / f"{which}-weights.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"which": which, "files": len(manifest["weights"]), "seconds": round(time.monotonic() - started, 3)}), flush=True)


def evaluate(plan, plan_path, which, home):
    started = time.monotonic()
    replies = home / "replies" / which
    replies.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in plan["tasks"]:
        destination = replies / f"{task['id']}.json"
        if destination.exists():
            row = checked(plan, json.loads(destination.read_text(encoding="utf-8")))
            if row.get("reason") == "rescore-mismatch":
                destination.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
            rows.append(row)
            if row.get("stopped"):
                break
            continue
        if time.monotonic() - started > plan["limits"]["evaluate_seconds"]:
            rows.append(stopped(task, which, "evaluate-budget"))
            break
        log_path = replies / f"{task['id']}.log"
        remaining = plan["limits"]["per_task_seconds"] + 1200
        with log_path.open("w", encoding="utf-8") as log:
            try:
                completed = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "generate-one",
                     "--plan", str(plan_path), "--home", str(home), "--which", which, "--task", task["id"]],
                    cwd=ROOT, timeout=remaining, check=False, stdout=log, stderr=subprocess.STDOUT)
                returncode = completed.returncode
            except subprocess.TimeoutExpired:
                log.write("\nper-task budget exceeded\n")
                returncode = "timeout"
        if destination.exists() and returncode == 0:
            row = checked(plan, json.loads(destination.read_text(encoding="utf-8")))
        else:
            row = stopped(task, which, f"exit-{returncode}")
            destination.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        if row.get("max_rss_bytes", 0) > plan["limits"]["max_rss_bytes"]:
            row["stopped"] = True
            row["passed"] = False
            row["reason"] = "rss-limit"
            destination.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        rows.append(row)
        if row.get("stopped"):
            break
    summary = {
        "format": "neuroshard-modular-reference-a1/result",
        "which": which,
        "plan_sha256": plan_hash(plan_path),
        "rows": rows,
        "seconds": round(time.monotonic() - started, 3),
    }
    (home / f"{which}-result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_gate(plan, home)
    print(json.dumps({"which": which, "rows": len(rows), "stopped": any(row.get("stopped") for row in rows)}), flush=True)


def checked(plan, row):
    if row.get("stopped"):
        return row
    task = next(item for item in plan["tasks"] if item["id"] == row["id"])
    scored = score_reply(task, row["text"], row["terminated"])
    if scored["passed"] != row["passed"] or scored["reason"] != row["reason"]:
        row["stopped"] = True
        row["passed"] = False
        row["reason"] = "rescore-mismatch"
    return row


def stopped(task, which, reason):
    return {
        "id": task["id"], "model": which, "category": task["category"], "stopped": True,
        "passed": False, "reason": reason, "text": "", "terminated": False,
        "seconds": 0, "max_rss_bytes": 0, "generated_tokens": 0,
    }


def write_gate(plan, home):
    rows = []
    for which in ("baseline", "modular"):
        path = home / f"{which}-result.json"
        if path.exists():
            rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
    if rows:
        decision = assess(plan, rows)
        decision["route"] = route_estimate()
        (home / "a1-gate.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
