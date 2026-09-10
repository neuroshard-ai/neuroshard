#!/usr/bin/env python3
"""Contribute with outbound connections or sponsor a bounded number of tasks."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import threading
import time
from contextlib import ExitStack

from neuroshard.demo import work
from neuroshard.publicnet.pool import Coordinator, Worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--home", type=Path, required=True)
    worker.add_argument("--stage", type=int, choices=(0, 1), required=True)
    worker.add_argument("--coordinator", required=True)
    worker.add_argument("--max-tasks", type=int, default=0, help="Stop after returning this many stage gradients; 0 runs continuously")
    sponsor = commands.add_parser("sponsor")
    sponsor.add_argument("--home", type=Path, required=True)
    sponsor.add_argument("--tasks", type=int, required=True, help="Maximum reservations; stop on first failure")
    sponsor.add_argument("--port", type=int, default=26660)
    sponsor.add_argument("--wait-seconds", type=int, default=180, help="Wait for workers before reserving; 0 waits indefinitely")
    sponsor.add_argument("--budget-file", type=Path, help="Persist the remaining attempt budget across process restarts")
    sponsor.add_argument("--interval-seconds", type=float, default=0, help="Pause between settled tasks")
    args = parser.parse_args()
    work.configure_cpu()
    worker_class, coordinator_class = Worker, Coordinator
    config = json.loads((args.home / "node.json").read_text())
    if config.get("profile") == "llm-testnet":
        from neuroshard.inference.pool import Worker as LLMWorker, Coordinator as LLMCoordinator
        worker_class, coordinator_class = LLMWorker, LLMCoordinator
    role = f"worker-{args.stage}" if args.command == "worker" else "sponsor"
    with (args.home / f"{role}.lock").open("a") as lock, ExitStack() as resources:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "worker":
            if args.max_tasks < 0:
                parser.error("max-tasks cannot be negative")
            worker_class(args.home, args.stage).run(args.coordinator, max_tasks=args.max_tasks)
        else:
            if not 1 <= args.tasks <= 100000:
                parser.error("tasks must be 1–100000")
            if args.wait_seconds < 0:
                parser.error("wait-seconds cannot be negative")
            if not 0 <= args.interval_seconds <= 3600:
                parser.error("interval-seconds must be 0–3600")
            if args.budget_file:
                args.budget_file.parent.mkdir(parents=True, exist_ok=True)
                budget_lock = resources.enter_context(args.budget_file.with_suffix('.lock').open('a'))
                fcntl.flock(budget_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            coordinator = coordinator_class(args.home)
            budget = {"chain_id": config["chain_id"], "total": args.tasks, "remaining": args.tasks}
            if args.budget_file and args.budget_file.exists():
                budget = json.loads(args.budget_file.read_text())
                if (budget.get("chain_id") != config["chain_id"] or budget.get("total") != args.tasks
                        or type(budget.get("remaining")) is not int or not 0 <= budget["remaining"] <= args.tasks):
                    parser.error("Budget file belongs to another chain or attempt limit")
            coordinator.remaining = budget["remaining"]
            server = coordinator.server("127.0.0.1", args.port)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for _ in range(budget["remaining"]):
                    # Charge the attempt before reserving; a restart cannot silently replenish it.
                    budget["remaining"] -= 1
                    if args.budget_file:
                        args.budget_file.parent.mkdir(parents=True, exist_ok=True)
                        temporary = args.budget_file.with_suffix(".tmp")
                        with temporary.open("w") as f:
                            json.dump(budget, f); f.flush(); os.fsync(f.fileno())
                        temporary.replace(args.budget_file)
                        directory = os.open(args.budget_file.parent, os.O_RDONLY | os.O_DIRECTORY)
                        try: os.fsync(directory)
                        finally: os.close(directory)
                    print(json.dumps(coordinator.mine(args.wait_seconds)), flush=True)
                    coordinator.remaining = budget["remaining"]
                    if args.interval_seconds: time.sleep(args.interval_seconds)
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    main()
