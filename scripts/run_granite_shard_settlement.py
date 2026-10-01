#!/usr/bin/env python3
"""One owner, the auditor, or a validator for one declared phase of the A4/A5 settlement execution."""

import argparse
import os
from pathlib import Path
import traceback

from neuroshard.evolution.granite_shard_settlement import (
    AUDITOR_PHASES, OWNER_PHASES, VALIDATOR_PHASES, auditor, owner, validator,
)
from neuroshard.evolution.modular_reference_execution import save


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, help="owner-RANK, auditor, or validator-INDEX")
    parser.add_argument("--address", default="")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--phase", choices=sorted(set(OWNER_PHASES + AUDITOR_PHASES + VALIDATOR_PHASES)), required=True)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.role.startswith("owner-"):
            result = owner(int(args.role.split("-", 1)[1]), args.address, args.port, args.phase, args.home, args.store)
        elif args.role == "auditor":
            result = auditor(args.phase, args.home, args.store)
        elif args.role.startswith("validator-"):
            result = validator(int(args.role.split("-", 1)[1]), args.phase, args.home, args.store)
        else:
            raise ValueError("unknown settlement role")
    except BaseException:
        # Host journals are lost at retirement; the evidence copy includes this file.
        save(args.home / "errors" / f"{args.phase}.json", {"role": args.role, "error": traceback.format_exc()})
        os._exit(1)
    # Gloo threads can outlive a broken peer group; the result file is already durable.
    os._exit(0 if result.get("completed") else 1)
