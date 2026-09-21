"""Verify or run the separate 135M CPU staged-integration mechanism candidate."""
import argparse
import json
from pathlib import Path

from neuroshard.evolution.staged_integration import bind_freeze
from neuroshard.evolution.staged_integration_run import ARMS, run_study, worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Run only after the CPU candidate is committed")
    mode.add_argument("--worker", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=Path, help="Existing pinned local SmolLM2-135M-Instruct files")
    parser.add_argument("--home", type=Path, help="New study directory; existing evidence is never overwritten")
    args = parser.parse_args()
    if args.run or args.worker:
        if args.seed is None or args.home is None:
            parser.error("CPU execution requires --seed and --home")
        result = worker(args.worker, args.seed, args.home) if args.worker else run_study(args.seed, args.home)
    else:
        result = {"freeze": bind_freeze(), "verified": True, "executed": False,
                  "gpu_launch_authorized": False, "admission_evidence": False}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
