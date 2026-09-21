"""One CPU-only output-format calibration. No training or evaluation-set selection."""
import argparse
import json
import resource
import time
from pathlib import Path

import torch

from neuroshard.evolution.reference_data import identity, save, sha256
from neuroshard.evolution.staged_answer_format import evaluate
from neuroshard.evolution.staged_integration import root
from neuroshard.evolution.staged_integration_run import environment, load_parent, peak_rss_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--plan", type=Path,
                        default=Path("config/experiments/staged-answer-calibration.json"))
    args = parser.parse_args()
    plan_path = root() / args.plan
    plan = json.loads(plan_path.read_text())
    args.home.mkdir(parents=True, exist_ok=False)
    files = (str(plan_path.relative_to(root())), "scripts/calibrate_staged_answers.py",
             "src/neuroshard/evolution/staged_answer_format.py",
             "src/neuroshard/evolution/staged_integration_run.py",
             "src/neuroshard/evolution/staged_integration.py",
             "src/neuroshard/evolution/seed.py")
    save(args.home / "before-generation.json", {
        "plan": identity(plan), "files": {name: sha256(root() / name) for name in files},
        "training": False, "admission_evidence": False, "gpu_launch_authorized": False,
    })
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    resource.setrlimit(resource.RLIMIT_CPU, (120, 121))
    started = time.monotonic()
    model, tokenizer = load_parent(args.seed)
    result = evaluate(model, tokenizer, plan["roles"], plan, args.home, "calibration")
    counts = {role: sum(row["passed"] for row in rows) for role, rows in result.items()}
    receipt = {"plan": identity(plan), "counts": counts,
               "passed": counts["retention"] >= plan["minimum_retention_correct"],
               "wall_seconds": time.monotonic() - started,
               "process_cpu_seconds": time.process_time(), "peak_rss_bytes": peak_rss_bytes(),
               "environment": environment(), "training": False, "admission_evidence": False,
               "gpu_launch_authorized": False}
    save(args.home / "result.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
