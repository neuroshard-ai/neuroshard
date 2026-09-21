"""Run the separately frozen 135M CPU staged-answering study."""
import argparse
import json
import os
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch

from neuroshard.evolution.reference_data import identity, save, tokenizer_identity
from neuroshard.evolution.seed import verify
from neuroshard.evolution.staged_answer_format import evaluate
from neuroshard.evolution.staged_answering import (
    FORMAT, bind_freeze, load_data, load_spec, root, score,
)
from neuroshard.evolution.staged_integration import read_checkpoint, tensor_identity
from neuroshard.evolution.staged_integration_run import (
    ARMS, environment, install, load_parent, peak_rss_bytes,
    read_receipt, train_control, train_expansion,
)


def worker(arm, seed, home):
    spec, freeze = load_spec(), bind_freeze(committed=True)
    data = load_data(spec)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(spec["training"]["seed"])
    random.seed(spec["training"]["seed"])
    resource.setrlimit(resource.RLIMIT_CPU, (spec["budget"]["worker_cpu_seconds"],
                                           spec["budget"]["worker_cpu_seconds"] + 1))
    started = time.monotonic()
    model, tokenizer = load_parent(seed)
    binding = {"contract": identity(spec), "freeze": freeze, "data": identity(data),
               "tokenizer": tokenizer_identity(tokenizer)}
    if arm != "baseline":
        baseline = read_receipt(home, "baseline", binding)
        if sum(r["passed"] for r in baseline["retention"]) < spec["gates"]["minimum_parent_retention_correct"]:
            raise ValueError("Baseline has too few protected answers; do not train")
    if arm == "expansion-train":
        result = train_expansion(model, tokenizer, data["roles"], spec, home, binding)
    elif arm == "control-train":
        training = read_receipt(home, "expansion-train", binding)
        result = train_control(model, tokenizer, data["roles"], spec, home, binding,
                               training["comparison_cpu_seconds"])
    else:
        if arm != "baseline":
            expanded = arm == "expansion-evaluate"
            module = install(model, expansion=expanded)
            original = tensor_identity(module.incumbent) if expanded else None
            phase = "gate" if expanded else "control"
            manifest = read_checkpoint(home / "checkpoints" / phase, module, binding=binding, phase=phase)
            training = read_receipt(home, "expansion-train" if expanded else "control-train", binding)
            expected = training["checkpoints"][phase] if expanded else training["checkpoint"]
            if identity(manifest) != expected:
                raise ValueError("Evaluation checkpoint differs from the trained module")
            if expanded and tensor_identity(module.incumbent) != original:
                raise ValueError("Frozen incumbent was modified")
        result = evaluate(model, tokenizer, data["roles"], spec, home, arm)
    result.update({"format": FORMAT + "/arm", "arm": arm, "binding": binding,
                   "process_cpu_seconds": time.process_time(), "wall_seconds": time.monotonic() - started,
                   "peak_rss_bytes": peak_rss_bytes(), "environment": environment(),
                   "admission_evidence": False, "gpu_launch_authorized": False})
    save(home / (arm + ".json"), result)
    return result


def run_isolated(arm, seed, home, seconds):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONPATH=str(root() / "src"))
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", arm,
               "--seed", str(seed.resolve()), "--home", str(home.resolve())]
    before, started = resource.getrusage(resource.RUSAGE_CHILDREN), time.monotonic()
    outcome = "failed"
    try:
        with (home / (arm + ".log")).open("x") as log:
            subprocess.run(command, env=env, cwd=root(), check=True, timeout=seconds,
                           stdout=log, stderr=subprocess.STDOUT)
        outcome = "completed"
    finally:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        save(home / (arm + "-launch.json"), {
            "arm": arm, "outcome": outcome, "wall_seconds": time.monotonic() - started,
            "cpu_seconds": after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
            "scope": "complete isolated child, including loading and artifact writes",
        })
    return json.loads((home / (arm + ".json")).read_text())


def run_study(seed, home):
    spec, freeze = load_spec(), bind_freeze(committed=True)
    verify(seed)
    home.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    save(home / "study.json", {"contract": identity(spec), "freeze": freeze,
                               "admission_evidence": False, "gpu_launch_authorized": False})
    receipts = {}
    try:
        for arm in ARMS:
            remaining = spec["budget"]["total_wall_seconds"] - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("CPU study wall budget exhausted")
            receipts[arm] = run_isolated(arm, seed, home, min(remaining, spec["budget"]["worker_wall_seconds"]))
            if receipts[arm]["peak_rss_bytes"] > spec["gates"]["maximum_peak_rss_bytes"]:
                raise ValueError("Process exceeded the frozen memory gate")
            if arm == "baseline":
                protected = [r["id"] for r in receipts[arm]["retention"] if r["passed"]]
                save(home / "protected-before-training.json", {"ids": protected, "baseline": identity(receipts[arm])})
                if len(protected) < spec["gates"]["minimum_parent_retention_correct"]:
                    result = {"passed": False, "next": "stop-baseline-uninformative", "protected": protected,
                              "admission_evidence": False, "gpu_launch_authorized": False,
                              "confirmation_opened": False, "item4_complete": False}
                    save(home / "result.json", result)
                    return result
        result = score(spec, receipts["baseline"], receipts["expansion-evaluate"], receipts["control-evaluate"],
                       receipts["expansion-train"], receipts["control-train"])
    except (subprocess.SubprocessError, TimeoutError, ValueError) as error:
        save(home / "failure.json", {"error": str(error), "completed_arms": list(receipts),
                                     "wall_seconds": time.monotonic() - started, "passed": False,
                                     "admission_evidence": False, "gpu_launch_authorized": False})
        raise
    result["binding"] = receipts["baseline"]["binding"]
    result["receipts"] = {arm: identity(receipt) for arm, receipt in receipts.items()}
    result["launches"] = {arm: json.loads((home / (arm + "-launch.json")).read_text()) for arm in ARMS}
    result["total_process_cpu_seconds"] = sum(r["cpu_seconds"] for r in result["launches"].values())
    result["total_wall_seconds"] = time.monotonic() - started
    save(home / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--worker", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--home", type=Path)
    args = parser.parse_args()
    if args.run or args.worker:
        if args.seed is None or args.home is None:
            parser.error("CPU execution requires --seed and --home")
        result = worker(args.worker, args.seed, args.home) if args.worker else run_study(args.seed, args.home)
    else:
        result = {"freeze": bind_freeze(), "executed": False, "gpu_launch_authorized": False}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
