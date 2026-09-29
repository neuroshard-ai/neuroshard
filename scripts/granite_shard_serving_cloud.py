#!/usr/bin/env python3
"""Owner hosts for the Granite shard serving execution; collect evidence and retire every host.

Each owner is its own single-host allocation from the reference controller.
Owners reach each other by private address only. The arm and its gate are
uploaded only to owner 2 (the arm) and owner 0 (the gate).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import shlex
import time

from neuroshard.evolution import granite_shard_serving as serving
from neuroshard.evolution.modular_reference_execution import ROOT, read, save, sha256, wait_for_ci

spec = importlib.util.spec_from_file_location("granite_shard_cloud", ROOT / "scripts/granite_shard_cloud.py")
ring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ring)
cloud = ring.cloud


def unit(phase, index=None):
    return f"granite-owner-{phase}" + ("" if index is None else f"-{index}")


def start(home, allocation, rank, phase, plan, address="", index=None):
    memory = plan["fetch_memory_limit_bytes"] if phase == "fetch" else plan["memory_limit_bytes"]
    command = [cloud.PYTHON, cloud.REMOTE + "/" + serving.SCRIPT, "--rank", str(rank), "--address", address,
               "--port", str(ring.PORT), "--phase", phase, "--home", ring.OWNER_HOME, "--store", ring.STORE,
               "--index", str(index or 0)]
    cloud.ssh(home, allocation, ["systemd-run", "--user", f"--unit={unit(phase, index)}",
        "-p", f"RuntimeMaxSec={plan['phase_seconds'][phase]}", "-p", f"MemoryMax={memory}",
        "-p", "MemorySwapMax=0", "-p", "TimeoutStopSec=0", "-p", "KillMode=control-group",
        "--working-directory=" + cloud.REMOTE, "--setenv=PYTHONPATH=" + cloud.REMOTE + "/src", *command])


def wait(units, plan, phase):
    deadline = time.monotonic() + plan["phase_seconds"][phase] + 120
    while time.monotonic() < deadline:
        states = [cloud.ssh(h, a, ["systemctl", "--user", "show", "--value", "-p", "ActiveState", name]).stdout.strip()
                  for h, a, name in units]
        if all(state not in (b"active", b"activating") for state in states):
            return
        time.sleep(15)
    raise TimeoutError(f"shard serving phase {phase} exhausted its allowance")


def upload(home, allocation, plan):
    payload = cloud.upload_bundle(plan["upload"])
    target = cloud.REMOTE + "/" + serving.UPLOADED
    cloud.ssh(home, allocation, ["bash", "-c", f"mkdir -p {shlex.quote(target)} && tar -xzf - -C {shlex.quote(target)}"],
              data=payload, timeout=600)


def run(home):
    home.mkdir(parents=True, exist_ok=True)
    plan = read(ROOT / serving.PLAN)
    source = serving.committed_sources()
    wait_for_ci(home, source["commit"])
    if serving.committed_sources() != source:
        raise ValueError("source changed while waiting for CI")
    world = len(plan["boundaries"]) - 1
    homes = [home / f"owner-{rank}" for rank in range(world)]
    allocations, failure = [], None
    result = {"binding": {"freeze": source, "plan_sha256": sha256(ROOT / serving.PLAN)}, "execution_completed": False}
    try:
        save(home / "status.json", {"state": "allocating", "commit": source["commit"]})
        for host in homes:
            host.mkdir(parents=True, exist_ok=True)
            allocations.append(cloud.allocate(host, source["commit"], serving.PROFILE))
        owners = list(zip(homes, allocations))
        ring.open_ring(allocations)
        save(home / "status.json", {"state": "setup"})
        with ThreadPoolExecutor(max_workers=world) as pool:
            list(pool.map(lambda pair: cloud.bootstrap(*pair, source, serving.PROFILE), owners))
        for rank in (0, world - 1):
            upload(*owners[rank], plan)
        save(home / "status.json", {"state": "running", "phase": "fetch"})
        for rank, (h, a) in enumerate(owners):
            start(h, a, rank, "fetch", plan)
        wait([(h, a, unit("fetch")) for h, a in owners], plan, "fetch")
        fetches = [ring.remote_json(h, a, f"{ring.OWNER_HOME}/fetch.json") for h, a in owners]
        save(home / "fetch.json", fetches)
        if not all(f and f.get("completed") for f in fetches):
            raise RuntimeError("an owner could not fetch its shard")
        rows = []
        for index in range(plan["determinism"]["processes"]):
            save(home / "status.json", {"state": "running", "phase": "determinism", "process": index})
            for rank, (h, a) in enumerate(owners):
                start(h, a, rank, "determinism", plan, index=index)
            wait([(h, a, unit("determinism", index)) for h, a in owners], plan, "determinism")
            rows += [ring.remote_json(h, a, f"{ring.OWNER_HOME}/determinism/process-{index}.json") for h, a in owners]
        save(home / "determinism.json", rows)
        save(home / "status.json", {"state": "running", "phase": "serve"})
        address = allocations[0]["private_ip"]
        for rank, (h, a) in enumerate(owners):
            start(h, a, rank, "serve", plan, address)
        wait([(h, a, unit("serve")) for h, a in owners], plan, "serve")
        served = [ring.remote_json(h, a, f"{ring.OWNER_HOME}/serve/result.json") for h, a in owners]
        save(home / "serve.json", served)
        result.update(fetches=fetches, determinism=rows, served=served,
                      report=serving.assess(plan, fetches, rows, served), execution_completed=True)
    except BaseException as error:
        failure = str(error) or type(error).__name__
        result["error"] = failure
    finally:
        for host, allocation in zip(homes, allocations):
            if allocation.get("private_ip"):
                try:
                    cloud.collect(host, allocation)
                except Exception as copy_error:
                    save(host / "copy-failure.json", {"error": str(copy_error)})
        save(home / "result.json", result)
        receipts, errors = [], {}
        for host in homes:
            if (host / "allocation.json").exists():
                try:
                    receipts.append(cloud.retire(host))
                except Exception as retire_error:
                    errors[host.name] = str(retire_error)
        result.update(resources_finished=receipts, retirement_errors=errors)
        save(home / "result.json", result)
        if errors and not failure:
            failure = f"retirement incomplete: {errors}"
        save(home / "status.json", {"state": "failed" if failure else "finished", "error": failure})
    if failure:
        raise RuntimeError(failure)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "retire"))
    parser.add_argument("--home", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "retire":
        for host in sorted(p for p in args.home.resolve().iterdir() if (p / "allocation.json").exists()):
            cloud.retire(host)
    else:
        run(args.home.resolve())
