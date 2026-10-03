#!/usr/bin/env python3
"""Run the committed CPU decoder audit without opening a new quality study."""

import argparse
from pathlib import Path

from neuroshard.evolution.modular_decoder_parity import run, worker


def main():
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
        result = run(args.home, args.models)
        if not result["execution_completed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
