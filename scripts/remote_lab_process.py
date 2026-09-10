#!/usr/bin/env python3
"""Start/stop only the experiment processes recorded in a dedicated directory."""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rpartition(") ")[2].split()
        return fields[19] if fields[0] != "Z" else None
    except (FileNotFoundError, IndexError):
        return None


def stop(path):
    record = json.loads(path.read_text())
    pid = record["pid"]
    if identity(pid) == record["start"]:
        os.killpg(pid, signal.SIGTERM)
        deadline = time.monotonic() + 8
        while identity(pid) == record["start"] and time.monotonic() < deadline:
            time.sleep(0.1)
        if identity(pid) == record["start"]:
            os.killpg(pid, signal.SIGKILL)
    path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop", "stop-all"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--name")
    args, command = parser.parse_known_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.action == "stop-all":
        for path in args.directory.glob("*.pid.json"):
            stop(path)
        return
    if not args.name or not re.fullmatch(r"[a-z][a-z0-9]*", args.name):
        parser.error("invalid process name")
    path = args.directory / f"{args.name}.pid.json"
    if args.action == "stop":
        if path.exists():
            stop(path)
        return
    if path.exists():
        record = json.loads(path.read_text())
        if identity(record["pid"]) == record["start"]:
            print(record["pid"])
            return
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a process command is required")
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(PYTHONPATH=str(root / "src"), ATEN_CPU_CAPABILITY="default",
               MKL_ENABLE_INSTRUCTIONS="SSE4_2", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with (args.directory / f"{args.name}.log").open("ab") as log:
        child = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = {"pid": child.pid, "start": identity(child.pid)}
    if record["start"] is None:
        raise RuntimeError(f"Process {args.name} failed at launch; inspect its log")
    path.write_text(json.dumps(record))
    print(child.pid)


if __name__ == "__main__":
    main()
