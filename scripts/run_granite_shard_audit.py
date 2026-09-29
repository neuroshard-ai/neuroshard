#!/usr/bin/env python3
"""One owner or auditor for one declared phase of the A4 audited serving execution."""

import argparse
import os
from pathlib import Path
import traceback

from neuroshard.evolution.granite_shard_audit import AUDITOR_PHASES, OWNER_PHASES, auditor, owner
from neuroshard.evolution.modular_reference_execution import save


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", required=True, help="owner rank, or auditor-RANK")
    parser.add_argument("--address", default="")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--phase", choices=sorted(set(OWNER_PHASES + AUDITOR_PHASES)), required=True)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.rank.startswith("auditor-"):
            result = auditor(int(args.rank.split("-", 1)[1]), args.phase, args.home, args.store)
        else:
            result = owner(int(args.rank), args.address, args.port, args.phase, args.home, args.store)
    except BaseException:
        # Host journals are lost at retirement; the evidence copy includes this file.
        save(args.home / "errors" / f"{args.phase}.json", {"rank": args.rank, "error": traceback.format_exc()})
        os._exit(1)
    # Gloo threads can outlive a broken peer group; the result file is already durable.
    os._exit(0 if result.get("completed") else 1)
