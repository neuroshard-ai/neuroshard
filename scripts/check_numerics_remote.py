#!/usr/bin/env python3
"""Compare full training-step commitments on this host and a prepared SSH host."""

import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]


def emit(steps):
    os.environ.update(ATEN_CPU_CAPABILITY="default", MKL_ENABLE_INSTRUCTIONS="SSE4_2")
    sys.path.insert(0, str(REPO / "src"))
    import torch
    from neuroshard.demo import work
    from neuroshard.lab.app import execution_manifest

    work.configure_cpu()
    data = work.read_data(REPO / "docs/eval/data/input.txt")
    weights = work.encode_weights(work.make_model())
    manifest = execution_manifest(data)
    vectors = []
    for step in range(steps):
        result = work.replay(weights, data, step, "neuroshard/physical-host-conformance/v1")
        weights = result["weights"]
        vectors.append({"round": step, "model_root": work.digest(weights),
                        "gradient_root": work.digest(result["gradients"]),
                        "stage_receipts_root": work.digest(result["receipts"]),
                        "loss_hex": float(result["loss"]).hex()})
    cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
               if line.startswith("model name"))
    return {"python": platform.python_version(), "torch": torch.__version__,
            "cpu": cpu, "architecture": platform.machine(), "os": platform.freedesktop_os_release()["PRETTY_NAME"],
            "manifest": manifest, "training": vectors,
            "inference": work.infer(weights, "ROMEO:", 12)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", action="store_true")
    parser.add_argument("--host")
    parser.add_argument("--remote-root", default="/home/ubuntu/neuroshard-lab")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000:
        parser.error("steps must be between 1 and 1000")
    if args.emit:
        result = emit(args.steps)
    else:
        if not args.host or args.host.startswith("-"):
            parser.error("--host must name the prepared SSH host")
        local = emit(args.steps)
        root = Path(args.remote_root)
        remote_output = root / ".neuroshard/physical-host-conformance.json"
        command = shlex.join([str(root / "venv_build/bin/python"), str(root / "scripts/check_numerics_remote.py"),
                              "--emit", "--steps", str(args.steps), "--output", str(remote_output)])
        subprocess.run(["ssh", "-o", "BatchMode=yes", args.host, command], check=True)
        raw = subprocess.check_output(["ssh", "-o", "BatchMode=yes", args.host,
                                       shlex.join(["cat", str(remote_output)])])
        remote = json.loads(raw)
        result = {"scope": "Two separate physical hosts under the same operator; CPU execution only",
                  "steps": args.steps, "local": local, "remote": remote,
                  "genesis_manifest_equal": local["manifest"] == remote["manifest"],
                  "all_training_commitments_equal": local["training"] == remote["training"],
                  "inference_equal": local["inference"] == remote["inference"],
                  "interpretation": "Finite hardware conformance, not universal arithmetic portability or independent ownership"}
        result["passed"] = all(result[key] for key in (
            "genesis_manifest_equal", "all_training_commitments_equal", "inference_equal"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not args.emit:
        print(json.dumps({key: result[key] for key in ("steps", "genesis_manifest_equal",
                         "all_training_commitments_equal", "inference_equal", "passed")}))
        if not result["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
