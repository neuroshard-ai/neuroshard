#!/usr/bin/env python3
"""Contribute with outbound connections or sponsor a bounded number of tasks."""
import argparse
import fcntl
import json
from pathlib import Path
import threading

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
    args = parser.parse_args()
    work.configure_cpu()
    role = f"worker-{args.stage}" if args.command == "worker" else "sponsor"
    with (args.home / f"{role}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "worker":
            if args.max_tasks < 0:
                parser.error("max-tasks cannot be negative")
            Worker(args.home, args.stage).run(args.coordinator, max_tasks=args.max_tasks)
        else:
            if not 1 <= args.tasks <= 1000:
                parser.error("tasks must be 1–1000")
            if args.wait_seconds < 0:
                parser.error("wait-seconds cannot be negative")
            coordinator = Coordinator(args.home)
            coordinator.remaining = args.tasks
            server = coordinator.server("127.0.0.1", args.port)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for _ in range(args.tasks):
                    print(json.dumps(coordinator.mine(args.wait_seconds)), flush=True)
                    coordinator.remaining -= 1
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    main()
