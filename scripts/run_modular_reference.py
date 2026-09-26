#!/usr/bin/env python3
"""Run the committed A1 execution amendment; retain the old freeze in git."""

import argparse
import json
from pathlib import Path

from neuroshard.evolution.modular_reference_execution import ROOT, read, run, worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "status", "worker"))
    parser.add_argument("--home", type=Path)
    parser.add_argument("--models", type=Path, default=ROOT / ".neuroshard/modular-reference")
    parser.add_argument("--legacy", type=Path)
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    if args.command == "worker":
        if args.request is None:
            parser.error("worker requires --request")
        worker(args.request)
    elif args.command == "status":
        if args.home is None:
            parser.error("status requires --home")
        print(json.dumps(read(args.home / "status.json"), indent=2))
    else:
        if args.home is None or args.legacy is None:
            parser.error("run requires a new --home and the completed --legacy baseline summary")
        result = run(args.home, args.models, args.legacy)
        print(json.dumps({key: result.get(key) for key in
                          ("execution_completed", "quality_ready", "milestone_complete", "error", "accounting")}, indent=2))
        if not result["execution_completed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
