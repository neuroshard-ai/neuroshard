#!/usr/bin/env python3
"""Run a committed reference or interface diagnostic without changing old freezes."""

import argparse
import json
from pathlib import Path

from neuroshard.evolution.modular_reference_execution import ROOT, PROFILES, configure_runtime, freeze, read, run, save, wait_for_ci, worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "status", "worker"))
    parser.add_argument("--home", type=Path)
    parser.add_argument("--models", type=Path, default=ROOT / ".neuroshard/modular-reference")
    parser.add_argument("--legacy", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="reference")
    parser.add_argument("--wait-for-ci", action="store_true",
                        help="Wait at most one hour for successful push CI at this committed freeze")
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
        configure_runtime(args.profile)
        if args.wait_for_ci:
            try:
                frozen = freeze(profile=args.profile)
                wait_for_ci(args.home, frozen["commit"])
                if freeze(profile=args.profile) != frozen:
                    raise ValueError("execution freeze changed while waiting for CI")
            except Exception as error:
                save(args.home / "status.json", {"state": "stopped-before-inference", "error": str(error)})
                raise
        result = run(args.home, args.models, args.legacy, profile=args.profile)
        print(json.dumps({key: result.get(key) for key in
                          ("execution_completed", "reference_ready", "growth_screen_passed", "net_gain",
                           "quality_ready", "interface_confirmed", "correct_calls",
                           "milestone_complete", "error", "accounting")}, indent=2))
        if not result["execution_completed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
