#!/usr/bin/env python3
"""Confirmation of one verified-experience system on the sealed split; opens only after a development pass."""

import argparse
from pathlib import Path

from neuroshard.evolution.assistant_experience_confirm import SYSTEMS, run, worker


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "worker"))
    parser.add_argument("--home", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--system", choices=SYSTEMS)
    args = parser.parse_args()
    if args.command == "worker":
        if args.request is None:
            parser.error("worker requires --request")
        worker(args.request)
    else:
        if args.home is None or args.models is None or args.system is None:
            parser.error("run requires --home, --models and --system")
        if not run(args.home, args.models, args.system)["execution_completed"]:
            raise SystemExit(1)
