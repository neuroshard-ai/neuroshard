"""Run append-only execution. Stop before training when nothing is protected."""
import json
import subprocess
import time

from neuroshard.evolution import append_only_execution as study
from neuroshard.evolution import block_expert_run as runner
from neuroshard.evolution.reference_data import identity, save
from neuroshard.evolution.seed import verify


SCRIPT = "scripts/run_append_only_execution.py"
ARMS = runner.ARMS


def bind_runner():
    runner.study = study
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
    plan, freeze = study.load_plan(), study.bind_freeze(committed=True)
    verify(seed)
    home.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.monotonic(), time.process_time()
    save(home / "study.json", {"plan": identity(plan), "freeze": freeze, "rule": plan["rule"],
         "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=study.root()).decode().strip(),
         "gpu_launch_authorized": False, "upgrade_public_0_4_0": False})
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
                save(home / "protected-before-training.json", {"ids": protected,
                                                              "baseline": identity(receipts[arm])})
                if len(protected) < plan["gates"]["minimum_protected"]:
                    raise ValueError("Baseline has no protected answer to preserve; no training authorized")
            if arm == "control-train" and not receipts[arm]["matched_budget"]:
                raise ValueError("Control failed to spend the matched optimization budget")
        target = (json.loads((home / "expert-train-launch.json").read_text())["cpu_seconds"]
                  + receipts["prepare"]["expert_extension_cpu_seconds"])
        result = study.score(plan, study.load_data(plan), receipts["baseline"], receipts["expert-evaluate"],
                             receipts["control-evaluate"], receipts["expert-train"],
                             receipts["control-train"], target)
    except (ValueError, TimeoutError, subprocess.SubprocessError) as error:
        result = {"passed": False, "error": str(error), "next": "stop-execution-no-quality-conclusion",
                  "completed_arms": list(receipts), "settlement_authorized": False,
                  "gpu_launch_authorized": False, "upgrade_public_0_4_0": False}
    launches = {arm: json.loads((home / (arm + "-launch.json")).read_text())
                for arm in ARMS if (home / (arm + "-launch.json")).exists()}
    result.update({"contract": identity(plan), "freeze": freeze,
                   "wall_seconds": time.monotonic() - started,
                   "child_cpu_seconds": sum(launch["cpu_seconds"] for launch in launches.values())})
    save(home / "result.json", result)
    return result
