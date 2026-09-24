"""Verify or run the committed CPU append-only execution."""
import argparse
import json
from pathlib import Path

from neuroshard.evolution.append_only_execution import bind_freeze
from neuroshard.evolution.append_only_execution_run import ARMS, bind_runner, run
from neuroshard.evolution.block_expert_run import worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--worker", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--home", type=Path)
    args = parser.parse_args()
    bind_runner()
    if args.run or args.worker:
        if args.seed is None or args.home is None:
            parser.error("Execution requires --seed and a new --home")
        result = worker(args.worker, args.seed, args.home) if args.worker else run(args.seed, args.home)
    else:
        result = {"freeze": bind_freeze(), "executed": False, "gpu_launch_authorized": False,
                  "settlement_authorized": False}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
