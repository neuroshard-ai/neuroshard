#!/usr/bin/env python3
"""Owner and auditor hosts for the audited serving execution; move logs to auditors; retire every host.

Each host is its own single-host allocation from the reference controller.
Owners reach each other by private address only; auditors never join the ring
and receive an owner's signed log only through this controller.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import shlex
import time

from neuroshard.evolution import granite_shard_audit as audited
from neuroshard.evolution.modular_reference_execution import ROOT, read, save, sha256, wait_for_ci

spec = importlib.util.spec_from_file_location("granite_shard_serving_cloud", ROOT / "scripts/granite_shard_serving_cloud.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
ring, cloud = base.ring, base.cloud


def start(home, allocation, role, phase, plan, address=""):
    memory = plan["fetch_memory_limit_bytes"] if phase == "fetch" else plan["memory_limit_bytes"]
    command = [cloud.PYTHON, cloud.REMOTE + "/" + audited.SCRIPT, "--rank", str(role), "--address", address,
               "--port", str(ring.PORT), "--phase", phase, "--home", ring.OWNER_HOME, "--store", ring.STORE]
    cloud.ssh(home, allocation, ["systemd-run", "--user", f"--unit=granite-{phase}",
        "-p", f"RuntimeMaxSec={base.seconds(plan, phase)}", "-p", f"MemoryMax={memory}",
        "-p", "MemorySwapMax=0", "-p", "TimeoutStopSec=0", "-p", "KillMode=control-group",
        "--working-directory=" + cloud.REMOTE, "--setenv=PYTHONPATH=" + cloud.REMOTE + "/src", *command])


def wait(units, plan, phase):
    deadline = time.monotonic() + base.seconds(plan, phase) + 120
    while time.monotonic() < deadline:
        states = [cloud.ssh(h, a, ["systemctl", "--user", "show", "--value", "-p", "ActiveState",
                                   f"granite-{phase}"]).stdout.strip() for h, a in units]
        if all(state not in (b"active", b"activating") for state in states):
            return
        time.sleep(15)
    raise TimeoutError(f"audited phase {phase} exhausted its allowance")


def transfer(source, target, directory, name):
    """Copy one owner's signed log directory to its auditor through this controller."""
    payload = cloud.ssh(*source, ["tar", "-czf", "-", "-C", f"{ring.OWNER_HOME}/{directory}", name], timeout=1800).stdout
    destination = f"{ring.OWNER_HOME}/{directory}"
    cloud.ssh(*target, ["bash", "-c", f"mkdir -p {shlex.quote(destination)} && tar -xzf - -C {shlex.quote(destination)}"],
              data=payload, timeout=1800)
    return len(payload)


def run(home):
    home.mkdir(parents=True, exist_ok=True)
    plan = read(ROOT / audited.PLAN)
    source = audited.committed_sources()
    wait_for_ci(home, source["commit"])
    if audited.committed_sources() != source:
        raise ValueError("source changed while waiting for CI")
    world = len(plan["boundaries"]) - 1
    names = [f"owner-{r}" for r in range(world)] + [f"auditor-{r}" for r in plan["audited_owners"]]
    homes = {name: home / name for name in names}
    allocations, failure = {}, None
    result = {"binding": {"freeze": source, "plan_sha256": sha256(ROOT / audited.PLAN)}, "execution_completed": False}
    try:
        save(home / "status.json", {"state": "allocating", "commit": source["commit"]})
        for name in names:
            homes[name].mkdir(parents=True, exist_ok=True)
            allocations[name] = cloud.allocate(homes[name], source["commit"], audited.PROFILE)
        hosts = {name: (homes[name], allocations[name]) for name in names}
        owners = [hosts[f"owner-{r}"] for r in range(world)]
        auditors = {r: hosts[f"auditor-{r}"] for r in plan["audited_owners"]}
        ring.open_ring([allocations[f"owner-{r}"] for r in range(world)])
        save(home / "status.json", {"state": "setup"})
        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            list(pool.map(lambda name: cloud.bootstrap(*hosts[name], source, audited.PROFILE), names))
        for name in ("owner-0", f"owner-{world - 1}", f"auditor-{world - 1}"):
            base.upload(audited, *hosts[name], plan)
        save(home / "status.json", {"state": "running", "phase": "fetch"})
        for r, host in enumerate(owners):
            start(*host, r, "fetch", plan)
        for r, host in auditors.items():
            start(*host, f"auditor-{r}", "fetch", plan)
        wait(owners + list(auditors.values()), plan, "fetch")
        fetches = {"owners": [ring.remote_json(*h, f"{ring.OWNER_HOME}/fetch.json") for h in owners],
                   "auditors": {r: ring.remote_json(*h, f"{ring.OWNER_HOME}/fetch.json") for r, h in auditors.items()}}
        save(home / "fetch.json", fetches)
        if not all(f and f.get("completed") for f in fetches["owners"] + list(fetches["auditors"].values())):
            raise RuntimeError("an owner or auditor could not fetch its shard")
        address = allocations["owner-0"]["private_ip"]
        phases, transferred = {}, {}
        for serve, audit in (("serve-honest", "audit-honest"), ("serve-cheat", "audit-cheat")):
            save(home / "status.json", {"state": "running", "phase": serve})
            for r, host in enumerate(owners):
                start(*host, r, serve, plan, address)
            wait(owners, plan, serve)
            phases[serve] = [ring.remote_json(*h, f"{ring.OWNER_HOME}/{serve}/result.json") for h in owners]
            save(home / f"{serve}.json", phases[serve])
            for r, host in auditors.items():
                transferred[f"{serve}-{r}"] = transfer(owners[r], host, serve, f"log-{r}")
            save(home / "status.json", {"state": "running", "phase": audit})
            for r, host in auditors.items():
                start(*host, f"auditor-{r}", audit, plan)
            wait(list(auditors.values()), plan, audit)
            phases[audit] = {f"auditor-{r}": ring.remote_json(*h, f"{ring.OWNER_HOME}/{audit}/result.json")
                             for r, h in auditors.items()}
            save(home / f"{audit}.json", phases[audit])
        accused = plan["fault"]["rank"]
        verifier = auditors[accused]
        cloud.ssh(*verifier, ["bash", "-c", f"cat > {shlex.quote(ring.OWNER_HOME)}/accused.json"],
                  data=json.dumps({"public_key": fetches["owners"][accused]["public_key"]}).encode())
        save(home / "status.json", {"state": "running", "phase": "verify"})
        start(*verifier, f"auditor-{accused}", "verify", plan)
        wait([verifier], plan, "verify")
        phases["verify"] = {f"auditor-{accused}": ring.remote_json(*verifier, f"{ring.OWNER_HOME}/verify/result.json")}
        result.update(fetches=fetches, phases=phases, transferred_bytes=transferred,
                      report=audited.assess(plan, fetches, phases), execution_completed=True)
    except BaseException as error:
        failure = str(error) or type(error).__name__
        result["error"] = failure
    finally:
        for name, allocation in allocations.items():
            if allocation.get("private_ip"):
                try:
                    cloud.collect(homes[name], allocation)
                except Exception as copy_error:
                    save(homes[name] / "copy-failure.json", {"error": str(copy_error)})
        save(home / "result.json", result)
        receipts, errors = [], {}
        for name in names:
            if (homes[name] / "allocation.json").exists():
                try:
                    receipts.append(cloud.retire(homes[name]))
                except Exception as retire_error:
                    errors[name] = str(retire_error)
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
