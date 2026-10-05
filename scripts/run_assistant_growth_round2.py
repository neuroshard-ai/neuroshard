#!/usr/bin/env python3
"""A3 stage 1, round 2, on one GPU host: budget-aware coaching, then stage 1's training and integration."""

import argparse
from pathlib import Path

from neuroshard.evolution.assistant_growth_round2 import run, worker


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "worker"))
    parser.add_argument("--home", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    if args.command == "worker":
        if args.request is None:
            parser.error("worker requires --request")
        worker(args.request)
    else:
        if args.home is None or args.models is None:
            parser.error("run requires --home and --models")
        if not run(args.home, args.models)["execution_completed"]:
            raise SystemExit(1)
