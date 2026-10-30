#!/usr/bin/env python3
"""Step 0: the served a4-general on the general-capability development split, on one canonical CPU host."""

import argparse
from pathlib import Path

from neuroshard.evolution.general_eval.development_run import PROFILE, PROFILES, run, worker


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "worker"))
    parser.add_argument("--home", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--profile", choices=tuple(PROFILES), default=PROFILE)
    args = parser.parse_args()
    if args.command == "worker":
        if args.request is None:
            parser.error("worker requires --request")
        worker(args.request)
    else:
        if args.home is None or args.models is None:
            parser.error("run requires --home and --models")
        if not run(args.home, args.models, args.profile)["execution_completed"]:
            raise SystemExit(1)
