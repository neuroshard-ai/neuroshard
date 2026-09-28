#!/usr/bin/env python3
"""One Granite shard owner for one declared phase of the A4 shard execution."""

import argparse
import os
from pathlib import Path

from neuroshard.evolution.granite_shard_execution import PHASES, owner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--address", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    args = parser.parse_args()
    result = owner(args.rank, args.address, args.port, args.phase, args.home, args.store)
    # Gloo threads can outlive a broken peer group; the result file is already durable.
    os._exit(0 if result.get("completed") else 1)
