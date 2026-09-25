"""Run the competence measurement. Parent retention does not stop training."""
import json
import subprocess
import time

from neuroshard.evolution import block_expert_measure as measure
from neuroshard.evolution import block_expert_run as runner
from neuroshard.evolution.reference_data import identity, save
from neuroshard.evolution.seed import verify


SCRIPT = "scripts/run_block_expert_measure.py"
ARMS = runner.ARMS


def bind_runner():
    runner.study = measure
    runner.SCRIPT = SCRIPT
    return runner


def run(seed, home):
    previous_study, previous_script = runner.study, runner.SCRIPT
    bind_runner()
    try:
        return _run(seed, home)
    finally:
        runner.study, runner.SCRIPT = previous_study, previous_script


def _run(seed, home):
    plan, freeze = measure.load_plan(), measure.bind_freeze(committed=True)
    verify(seed)
    home.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.monotonic(), time.process_time()
    save(home / "study.json", {"plan": identity(plan), "freeze": freeze,
         "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=measure.root()).decode().strip(),
         "admission_evidence": False, "gpu_launch_authorized": False,
         "retention_blocks_training": False})
    receipts = {}
    try:
        for arm in ARMS:
            remaining = plan["budget"]["total_wall_seconds"] - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Total study budget exhausted")
            receipts[arm] = runner.isolated(arm, seed, home, min(remaining, plan["budget"]["arms"][arm]))
            if receipts[arm]["peak_rss_bytes"] > plan["budget"]["maximum_peak_rss_bytes"]:
                raise ValueError("Worker exceeded memory budget")
            if arm == "baseline":
                protected = [row["id"] for row in receipts[arm]["retention"] if row["passed"]]
                unfinished = {role: sum(not row["terminated"] for row in receipts[arm][role])
                              for role in ("development", "retention")}
                save(home / "protected-before-training.json", {
                    "ids": protected, "baseline": identity(receipts[arm]),
                    "unfinished": unfinished, "blocks_training": False})
            if arm == "control-train" and not receipts[arm]["matched_budget"]:
                raise ValueError("Control failed to spend the matched optimization budget")
        target = (json.loads((home / "expert-train-launch.json").read_text())["cpu_seconds"]
                  + receipts["prepare"]["expert_extension_cpu_seconds"])
        result = measure.score(plan, measure.load_data(plan), receipts["baseline"],
                               receipts["expert-evaluate"], receipts["control-evaluate"],
                               receipts["expert-train"], receipts["control-train"], target)
    except (ValueError, TimeoutError, subprocess.SubprocessError) as error:
        result = {"passed": False, "error": str(error), "next": "stop-execution-no-quality-conclusion",
                  "completed_arms": list(receipts), "admission_evidence": False,
                  "gpu_launch_authorized": False, "selector_training_authorized": False}
    launches = {arm: json.loads((home / (arm + "-launch.json")).read_text())
                for arm in ARMS if (home / (arm + "-launch.json")).exists()}
    result.update({"contract": identity(plan), "freeze": freeze,
                   "receipts": {arm: identity(value) for arm, value in receipts.items()},
                   "launches": launches, "wall_seconds": time.monotonic() - started,
                   "coordinator_cpu_seconds": time.process_time() - cpu_started,
                   "child_cpu_seconds": sum(launch["cpu_seconds"] for launch in launches.values()),
                   "retention_blocks_training": False})
    save(home / "result.json", result)
    return result
