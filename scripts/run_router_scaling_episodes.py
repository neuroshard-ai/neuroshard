#!/usr/bin/env python3
"""Reworded real requests served end to end by the accepted A3 system under the pinned and the context router."""

import argparse
from pathlib import Path

from neuroshard.evolution.router_scaling_episodes import PROFILE, PROFILES, run, worker


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
