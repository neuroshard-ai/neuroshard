#!/usr/bin/env python3
"""Disposable owner hosts for the Granite shard execution; collect evidence and retire every host.

Each owner is its own single-host allocation (guard timer, expiry timer, tags,
security group) from the reference controller. The owners may reach each other
over TCP by private address only; SSH stays limited to this controller.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import shlex
import time

import boto3

from neuroshard.evolution import granite_shard_execution as execution
from neuroshard.evolution.modular_reference_execution import ROOT, read, save, sha256, wait_for_ci

spec = importlib.util.spec_from_file_location("reference_cloud", ROOT / "scripts/modular_reference_cloud.py")
cloud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cloud)

PORT = 29500
OWNER_HOME = cloud.STUDY + "/owner"
STORE = cloud.REMOTE + "/.shard"


def open_ring(allocations):
    """Allow TCP between owner hosts' private addresses, nothing else."""
    ec2 = boto3.client("ec2", region_name=allocations[0]["resources"]["region"])
    for allocation in allocations:
        peers = [a["private_ip"] + "/32" for a in allocations if a is not allocation]
        ec2.authorize_security_group_ingress(GroupId=allocation["security_group"], IpPermissions=[{
            "IpProtocol": "tcp", "FromPort": 0, "ToPort": 65535, "IpRanges": [{"CidrIp": ip} for ip in peers]}])


def start(home, allocation, rank, address, phase, plan):
    command = [cloud.PYTHON, cloud.REMOTE + "/" + execution.SCRIPT, "--rank", str(rank), "--address", address,
               "--port", str(PORT), "--phase", phase, "--home", OWNER_HOME, "--store", STORE]
    cloud.ssh(home, allocation, ["systemd-run", "--user", f"--unit=granite-owner-{phase}",
        "-p", f"RuntimeMaxSec={plan['phase_seconds'][phase]}",
        "-p", f"MemoryMax={plan['fetch_memory_limit_bytes'] if phase == 'fetch' else plan['memory_limit_bytes']}",
        "-p", "MemorySwapMax=0", "-p", "TimeoutStopSec=0", "-p", "KillMode=control-group",
        "--working-directory=" + cloud.REMOTE, "--setenv=PYTHONPATH=" + cloud.REMOTE + "/src", *command])


def state(home, allocation, phase):
    return cloud.ssh(home, allocation, ["systemctl", "--user", "show", "--value", "-p", "ActiveState",
                                        f"granite-owner-{phase}"]).stdout.strip()


def remote_json(home, allocation, path):
    try:
        return json.loads(cloud.ssh(home, allocation, ["cat", path]).stdout)
    except Exception:
        return None


def phase(homes, allocations, name, plan):
    address = allocations[0]["private_ip"]
    for rank, (home, allocation) in enumerate(zip(homes, allocations)):
        start(home, allocation, rank, address, name, plan)
    deadline = time.monotonic() + plan["phase_seconds"][name] + 120
    while time.monotonic() < deadline:
        states = [state(h, a, name) for h, a in zip(homes, allocations)]
        if all(s not in (b"active", b"activating") for s in states):
            break
        time.sleep(30)
    else:
        raise TimeoutError(f"shard phase {name} exhausted its allowance")
    path = f"{OWNER_HOME}/fetch.json" if name == "fetch" else f"{OWNER_HOME}/{name}/result.json"
    return [remote_json(h, a, path) for h, a in zip(homes, allocations)]


def run(home):
    home.mkdir(parents=True, exist_ok=True)
    plan = read(ROOT / execution.PLAN)
    source = execution.committed_sources()
    wait_for_ci(home, source["commit"])
    if execution.committed_sources() != source:
        raise ValueError("source changed while waiting for CI")
    world = len(plan["boundaries"]) - 1
    homes = [home / f"owner-{rank}" for rank in range(world)]
    allocations, failure, result = [], None, {"binding": {"freeze": source, "plan_sha256": sha256(ROOT / execution.PLAN)},
                                              "execution_completed": False}
    try:
        save(home / "status.json", {"state": "allocating", "commit": source["commit"]})
        for owner_home in homes:
            owner_home.mkdir(parents=True, exist_ok=True)
            allocations.append(cloud.allocate(owner_home, source["commit"], execution.PROFILE))
        open_ring(allocations)
        save(home / "status.json", {"state": "setup", "owners": [a["instance_ids"] for a in allocations]})
        with ThreadPoolExecutor(max_workers=world) as pool:
            list(pool.map(lambda pair: cloud.bootstrap(*pair, source, execution.PROFILE), zip(homes, allocations)))
        phases = {}
        for name in execution.PHASES:
            save(home / "status.json", {"state": "running", "phase": name})
            if name == "resume":
                committed = (phases["outage"][0] or {}).get("inflight", {}).get("token_ids")
                if not committed:
                    raise RuntimeError("outage phase left no committed tokens")
                cloud.ssh(homes[0], allocations[0], ["bash", "-c", f"cat > {shlex.quote(OWNER_HOME)}/committed.json"],
                          data=json.dumps(committed).encode())
            phases[name] = phase(homes, allocations, name, plan)
            save(home / f"{name}.json", phases[name])
            if name == "fetch" and not all(r and r.get("completed") for r in phases[name]):
                raise RuntimeError("an owner could not fetch its shard")
        fetches = phases.pop("fetch")
        result.update(phases=phases, fetches=fetches, report=execution.assess(plan, fetches, phases),
                      execution_completed=True)
    except BaseException as error:
        failure = str(error) or type(error).__name__
        result["error"] = failure
    finally:
        for owner_home, allocation in zip(homes, allocations):
            if allocation.get("private_ip"):
                try:
                    cloud.collect(owner_home, allocation)
                except Exception as copy_error:
                    save(owner_home / "copy-failure.json", {"error": str(copy_error)})
        save(home / "result.json", result)
        # Each owner is retired on its own, so one slow retirement never strands the others.
        receipts, errors = [], {}
        for owner_home in homes:
            if (owner_home / "allocation.json").exists():
                try:
                    receipts.append(cloud.retire(owner_home))
                except Exception as retire_error:
                    errors[owner_home.name] = str(retire_error)
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
        for owner_home in sorted(args.home.resolve().glob("owner-*")):
            cloud.retire(owner_home)
    else:
        run(args.home.resolve())
