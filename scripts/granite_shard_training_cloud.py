#!/usr/bin/env python3
"""Owner hosts and a reference host for the Granite shard training execution; retire every host.

Owners and the reference are separate single-host allocations from the reference
controller. Owners reach each other by private address only; the reference
never joins the ring. SSH stays limited to this controller.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import time

from neuroshard.evolution import granite_shard_training as training
from neuroshard.evolution.modular_reference_execution import ROOT, read, save, sha256, wait_for_ci

spec = importlib.util.spec_from_file_location("granite_shard_cloud", ROOT / "scripts/granite_shard_cloud.py")
ring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ring)
cloud = ring.cloud


def start(home, allocation, rank, phase, plan, address=""):
    memory = (plan["fetch_memory_limit_bytes"] if phase in ("fetch", "download")
              else plan["reference_memory_limit_bytes"] if rank == "reference" else plan["memory_limit_bytes"])
    command = [cloud.PYTHON, cloud.REMOTE + "/" + training.SCRIPT, "--rank", str(rank), "--address", address,
               "--port", str(ring.PORT), "--phase", phase, "--home", ring.OWNER_HOME, "--store", ring.STORE]
    cloud.ssh(home, allocation, ["systemd-run", "--user", f"--unit=granite-owner-{phase}",
        "-p", f"RuntimeMaxSec={plan['phase_seconds'][phase]}", "-p", f"MemoryMax={memory}",
        "-p", "MemorySwapMax=0", "-p", "TimeoutStopSec=0", "-p", "KillMode=control-group",
        "--working-directory=" + cloud.REMOTE, "--setenv=PYTHONPATH=" + cloud.REMOTE + "/src", *command])


def wait(units, plan):
    deadline = time.monotonic() + max(plan["phase_seconds"][phase] for _, _, phase in units) + 120
    while time.monotonic() < deadline:
        if all(ring.state(h, a, phase) not in (b"active", b"activating") for h, a, phase in units):
            return
        time.sleep(30)
    raise TimeoutError("shard training phase exhausted its allowance")


def owner_result(home, allocation, phase):
    return ring.remote_json(home, allocation, f"{ring.OWNER_HOME}/{phase}/result.json")


def run(home):
    home.mkdir(parents=True, exist_ok=True)
    plan = read(ROOT / training.PLAN)
    source = training.committed_sources()
    wait_for_ci(home, source["commit"])
    if training.committed_sources() != source:
        raise ValueError("source changed while waiting for CI")
    world = len(plan["boundaries"]) - 1
    names = [f"owner-{rank}" for rank in range(world)] + ["reference"]
    homes = [home / name for name in names]
    allocations, failure = [], None
    result = {"binding": {"freeze": source, "plan_sha256": sha256(ROOT / training.PLAN)}, "execution_completed": False}
    try:
        save(home / "status.json", {"state": "allocating", "commit": source["commit"]})
        for host in homes:
            host.mkdir(parents=True, exist_ok=True)
            allocations.append(cloud.allocate(host, source["commit"], training.PROFILE))
        owners = list(zip(homes[:world], allocations[:world]))
        control = (homes[world], allocations[world])
        ring.open_ring(allocations[:world])
        save(home / "status.json", {"state": "setup"})
        with ThreadPoolExecutor(max_workers=len(homes)) as pool:
            list(pool.map(lambda pair: cloud.bootstrap(*pair, source, training.PROFILE), zip(homes, allocations)))
        address = allocations[0]["private_ip"]
        save(home / "status.json", {"state": "running", "phase": "fetch"})
        for rank, (h, a) in enumerate(owners):
            start(h, a, rank, "fetch", plan)
        start(*control, "reference", "download", plan)
        wait([(h, a, "fetch") for h, a in owners] + [(*control, "download")], plan)
        fetches = [ring.remote_json(h, a, f"{ring.OWNER_HOME}/fetch.json") for h, a in owners]
        download = ring.remote_json(*control, f"{ring.OWNER_HOME}/download.json")
        save(home / "fetch.json", {"owners": fetches, "reference": download})
        if not all(f and f.get("completed") for f in fetches) or not (download and download.get("completed")):
            raise RuntimeError("an owner or the reference could not obtain its tensors")
        phases = {}
        for name in training.PHASES[1:]:
            save(home / "status.json", {"state": "running", "phase": name})
            for rank, (h, a) in enumerate(owners):
                start(h, a, rank, name, plan, address)
            units = [(h, a, name) for h, a in owners]
            if name == "train":
                start(*control, "reference", "reference", plan)
                units.append((*control, "reference"))
            wait(units, plan)
            phases[name] = [owner_result(h, a, name) for h, a in owners]
            save(home / f"{name}.json", phases[name])
        reference = ring.remote_json(*control, f"{ring.OWNER_HOME}/reference.json")
        save(home / "reference.json", reference)
        result.update(fetches=fetches, download=download, phases=phases, reference=reference,
                      report=training.assess(plan, fetches, phases, reference), execution_completed=True)
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
