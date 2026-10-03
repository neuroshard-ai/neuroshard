"""Verify or execute the committed CPU-only staged-answering recovery amendment."""
import argparse
import json
from pathlib import Path

from neuroshard.evolution.staged_recovery import ARMS, bind_freeze, run_study, worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--worker", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--home", type=Path)
    args = parser.parse_args()
    if args.run or args.worker:
        if args.seed is None or args.previous is None or args.home is None:
            parser.error("Execution requires --seed, --previous and a new --home")
        result = (worker(args.worker, args.seed, args.previous, args.home) if args.worker else
                  run_study(args.seed, args.previous, args.home))
    else:
        result = {"freeze": bind_freeze(), "executed": False, "gpu_launch_authorized": False}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
