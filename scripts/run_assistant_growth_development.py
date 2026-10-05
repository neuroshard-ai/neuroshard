#!/usr/bin/env python3
"""A3 stage 1 development: one version routed turn by turn on the opened development cases; no confirmation."""

import argparse
from pathlib import Path

from neuroshard.evolution.assistant_growth_eval import VERSIONS, run, worker


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "worker"))
    parser.add_argument("--home", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--version", choices=tuple(VERSIONS))
    args = parser.parse_args()
    if args.command == "worker":
        if args.request is None:
            parser.error("worker requires --request")
        worker(args.request)
    else:
        if args.home is None or args.models is None or args.version is None:
            parser.error("run requires --home, --models and --version")
        if not run(args.home, args.models, args.version)["execution_completed"]:
            raise SystemExit(1)
