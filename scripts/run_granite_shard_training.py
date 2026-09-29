#!/usr/bin/env python3
"""One owner or the reference host for one declared phase of the A4 shard training execution."""

import argparse
import os
from pathlib import Path

from neuroshard.evolution.granite_shard_training import PHASES, REFERENCE_PHASES, owner, reference


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", required=True, help="owner rank, or 'reference'")
    parser.add_argument("--address", default="")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--phase", choices=PHASES + REFERENCE_PHASES, required=True)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    args = parser.parse_args()
    if args.rank == "reference":
        result = reference(args.phase, args.home, args.store)
    else:
        result = owner(int(args.rank), args.address, args.port, args.phase, args.home, args.store)
    # Gloo threads can outlive a broken peer group; the result file is already durable.
    os._exit(0 if result.get("completed") else 1)
