"""Verify, run, or inspect the separately committed CPU reasoning experiment."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--home", type=Path)
    args = parser.parse_args()
    if args.status:
        if args.home is None:
            parser.error("--status needs --home")
        result = {}
        for name in ("status", "active-worker", "worker-progress", "preparation-progress", "result"):
            path = args.home / (name + ".json")
            if path.exists():
                record = json.loads(path.read_text())
                if name == "result":
                    record = {key: value for key, value in record.items() if key not in ("answers", "launches", "protected")}
                result[name] = record
        for arm in ("expert-train", "control-train"):
            path = args.home / (arm + "-history.json")
            if path.exists():
                rows = json.loads(path.read_text())
                result[arm] = {"completed_steps": len(rows), "last": rows[-1] if rows else None}
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    from neuroshard.evolution.observable_reasoning import bind_freeze
    from neuroshard.evolution.observable_reasoning_run import ARMS, run, worker
    if args.run or args.worker:
        if args.seed is None or args.home is None:
            parser.error("Execution needs --seed and a new --home")
        if args.worker and args.worker not in ARMS:
            parser.error("Unknown worker arm")
        result = worker(args.worker, args.seed, args.home) if args.worker else run(args.seed, args.home)
        # Full results are written durably to the study home, not repeated in logs.
        print(json.dumps({key: result[key] for key in ("passed", "execution_completed", "arm", "error") if key in result}))
    else:
        print(json.dumps({"freeze": bind_freeze(committed=True), "executed": False}))


if __name__ == "__main__":
    main()
