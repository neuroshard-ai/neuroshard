#!/usr/bin/env python3
"""Owners, the auditor and two validators for the settlement execution; order the blocks; retire every host.

Each host is its own single-host allocation from the reference controller. The
controller holds the user account, orders signed transactions into blocks, moves
logs to the auditor and blocks with proof bundles to the validators.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import time

from neuroshard.evolution import granite_shard_settlement as settlement
from neuroshard.evolution.modular_reference_execution import ROOT, read, save, sha256, wait_for_ci

spec = importlib.util.spec_from_file_location("granite_shard_serving_cloud", ROOT / "scripts/granite_shard_serving_cloud.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
ring, cloud = base.ring, base.cloud


def start(home, allocation, role, phase, plan, address=""):
    memory = plan["fetch_memory_limit_bytes"] if phase == "fetch" else plan["memory_limit_bytes"]
    command = [cloud.PYTHON, cloud.REMOTE + "/" + settlement.SCRIPT, "--role", role, "--address", address,
               "--port", str(ring.PORT), "--phase", phase, "--home", ring.OWNER_HOME, "--store", ring.STORE]
    cloud.ssh(home, allocation, ["systemd-run", "--user", f"--unit=granite-{phase}",
        "-p", f"RuntimeMaxSec={base.seconds(plan, phase)}", "-p", f"MemoryMax={memory}",
        "-p", "MemorySwapMax=0", "-p", "TimeoutStopSec=0", "-p", "KillMode=control-group",
        "--working-directory=" + cloud.REMOTE, "--setenv=PYTHONPATH=" + cloud.REMOTE + "/src", *command])


def wait(hosts, plan, phase):
    deadline = time.monotonic() + base.seconds(plan, phase) + 120
    while time.monotonic() < deadline:
        states = [cloud.ssh(h, a, ["systemctl", "--user", "show", "--value", "-p", "ActiveState",
                                   f"granite-{phase}"]).stdout.strip() for h, a in hosts]
        if all(state not in (b"active", b"activating") for state in states):
            return
        time.sleep(15)
    raise TimeoutError(f"settlement phase {phase} exhausted its allowance")


def run_phase(hosts, plan, phase, address=""):
    """Start one phase on (host, role) pairs, wait, and read each result."""
    for host, role in hosts:
        start(*host, role, phase, plan, address)
    wait([host for host, _ in hosts], plan, phase)
    path = "fetch.json" if phase == "fetch" else f"{phase}/result.json"
    return {role: ring.remote_json(*host, f"{ring.OWNER_HOME}/{path}") for host, role in hosts}


def put(host, relative, value):
    cloud.ssh(*host, ["bash", "-c", f"cat > {shlex.quote(ring.OWNER_HOME + '/' + relative)}"],
              data=json.dumps(value).encode())


def transfer(source, target, directory, name):
    payload = cloud.ssh(*source, ["tar", "-czf", "-", "-C", f"{ring.OWNER_HOME}/{directory}", name], timeout=1800).stdout
    destination = f"{ring.OWNER_HOME}/{directory}"
    cloud.ssh(*target, ["bash", "-c", f"mkdir -p {shlex.quote(destination)} && tar -xzf - -C {shlex.quote(destination)}"],
              data=payload, timeout=1800)
    return len(payload)


def user_account(home):
    return settlement.account(home, "user")


def run(home):
    from neuroshard.inference import optimistic as ledger

    home.mkdir(parents=True, exist_ok=True)
    plan = read(ROOT / settlement.PLAN)
    terms = plan["ledger"]
    source = settlement.committed_sources()
    wait_for_ci(home, source["commit"])
    if settlement.committed_sources() != source:
        raise ValueError("source changed while waiting for CI")
    world = len(plan["boundaries"]) - 1
    names = [f"owner-{r}" for r in range(world)] + ["auditor", "validator-1", "validator-2"]
    homes = {name: home / name for name in names}
    allocations, failure = {}, None
    result = {"binding": {"freeze": source, "plan_sha256": sha256(ROOT / settlement.PLAN)}, "execution_completed": False}
    try:
        save(home / "status.json", {"state": "allocating", "commit": source["commit"]})
        for name in names:
            homes[name].mkdir(parents=True, exist_ok=True)
            allocations[name] = cloud.allocate(homes[name], source["commit"], settlement.PROFILE)
        hosts = {name: (homes[name], allocations[name]) for name in names}
        owners = [hosts[f"owner-{r}"] for r in range(world)]
        ring.open_ring([allocations[f"owner-{r}"] for r in range(world)])
        save(home / "status.json", {"state": "setup"})
        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            list(pool.map(lambda name: cloud.bootstrap(*hosts[name], source, settlement.PROFILE), names))
        for name in ("owner-0", f"owner-{world - 1}"):
            base.upload(settlement, *hosts[name], plan)
        phases, transferred = {}, {}
        save(home / "status.json", {"state": "running", "phase": "fetch"})
        fetched = run_phase([(owners[r], f"owner-{r}") for r in range(world)]
                            + [(hosts["auditor"], "auditor"), (hosts["validator-1"], "validator-1"),
                               (hosts["validator-2"], "validator-2")], plan, "fetch")
        if not all(f and f.get("completed") for f in fetched.values()):
            raise RuntimeError("a host could not fetch its shard")
        accounts = {r: ring.remote_json(*owners[r], f"{ring.OWNER_HOME}/account.json") for r in (1, 2)}
        fetches = {"owners": [{**fetched[f"owner-{r}"], **(accounts.get(r) or {})} for r in range(world)],
                   "auditor": fetched["auditor"], "validators": [fetched["validator-1"], fetched["validator-2"]]}
        auditor_accounts = ring.remote_json(*hosts["auditor"], f"{ring.OWNER_HOME}/accounts.json")
        user_key, user_public = user_account(home)
        parties = {"user": user_public, "owner-1": accounts[1]["account"], "owner-2": accounts[2]["account"],
                   "auditor": auditor_accounts["auditor"], "accuser": auditor_accounts["accuser"]}
        log_keys = [accounts[1]["log_key"], accounts[2]["log_key"]]
        model_root = sha256(ROOT / settlement.MODEL_INVENTORY)
        genesis = ledger.genesis(terms["chain_id"], model_root, terms["shards"],
                                 {account: terms["allocation"] for account in parties.values()}, terms["params"])
        save(home / "parties.json", {"parties": parties, "log_keys": log_keys, "model_root": model_root})

        def owner_envelopes(phase, nonce, **request):
            for r in (1, 2):
                put(owners[r], f"{phase}-request.json", {"chain_id": terms["chain_id"], "nonce": nonce, **request})
            replies = run_phase([(owners[r], f"owner-{r}") for r in (1, 2)], plan, phase)
            phases[phase] = replies
            if not all(reply and reply.get("completed") for reply in replies.values()):
                raise RuntimeError(f"an owner could not complete {phase}")
            return [replies[f"owner-{r}"]["envelope"] for r in (1, 2)]

        session_key = fetched["owner-0"]["session_key"]

        def open_job(nonce, label):
            body = {"kind": "serve_open", "chain_id": terms["chain_id"], "nonce": nonce, "model_root": model_root,
                    "owners": log_keys, "request_root": hashlib.sha256(f"{plan['target']}:{label}".encode()).hexdigest(),
                    "session_key": session_key, "price": terms["price"]}
            envelope = settlement.signed(user_key, body)
            return envelope, ledger.transaction_id(envelope)

        def bind_serving(serve, envelope, job_id):
            """Every owner serves the opened job: owner 0 signs under the session key, owners under their log keys."""
            session = {"chain_id": terms["chain_id"], "job_id": job_id, "request_root": envelope["body"]["request_root"],
                       "session_key": session_key, "log_keys": log_keys}
            for r in range(world):
                put(owners[r], f"{serve}-request.json", {"session": session})

        address = allocations["owner-0"]["private_ip"]
        blocks = [owner_envelopes("sign-bond", 0, model_root=model_root, amount=terms["owner_bond"])]
        jobs = {}
        for nonce, label in ((1, "honest"), (2, "cheat")):
            opened, jobs[label] = open_job(nonce - 1, label)
            blocks.append([opened])
            serve = f"serve-{label}"
            bind_serving(serve, opened, jobs[label])
            save(home / "status.json", {"state": "running", "phase": serve})
            served = run_phase([(owners[r], f"owner-{r}") for r in range(world)], plan, serve, address)
            phases[serve] = [served[f"owner-{r}"] for r in range(world)]
            save(home / f"{serve}.json", phases[serve])
            transferred[serve] = transfer(owners[1], hosts["auditor"], serve, "log-1")
            blocks.append(owner_envelopes(f"commit-{label}", nonce, job_id=jobs[label]))
            audit = f"audit-{label}"
            save(home / "status.json", {"state": "running", "phase": audit})
            phases[audit] = run_phase([(hosts["auditor"], "auditor")], plan, audit)["auditor"]
            save(home / f"{audit}.json", phases[audit])
        put(hosts["auditor"], "challenge-request.json",
            {"chain_id": terms["chain_id"], "honest_job": jobs["honest"], "cheated_job": jobs["cheat"], "log_key": log_keys[0]})
        save(home / "status.json", {"state": "running", "phase": "challenge"})
        phases["challenge"] = run_phase([(hosts["auditor"], "auditor")], plan, "challenge")["auditor"]
        if not (phases["challenge"] or {}).get("completed"):
            raise RuntimeError("the auditor could not prepare its challenges")
        # The honest job settles at block 8, before the fraud proof against its owner arrives at block 9.
        blocks += [[phases["challenge"]["framing"]], [], [], [phases["challenge"]["proven"]], [], []]
        chain = {"genesis": genesis, "blocks": blocks}
        save(home / "blocks.json", chain)
        payload = cloud.ssh(*hosts["auditor"], ["tar", "-czf", "-", "-C", ring.OWNER_HOME, "bundles"], timeout=1800).stdout
        transferred["bundles"] = len(payload)
        for name in ("validator-1", "validator-2"):
            cloud.ssh(*hosts[name], ["bash", "-c", f"mkdir -p {ring.OWNER_HOME} && tar -xzf - -C {ring.OWNER_HOME}"],
                      data=payload, timeout=1800)
            put(hosts[name], "blocks.json", chain)
        save(home / "status.json", {"state": "running", "phase": "validate"})
        phases["validate"] = run_phase([(hosts["validator-1"], "validator-1"), (hosts["validator-2"], "validator-2")],
                                       plan, "validate")
        result.update(fetches=fetches, parties=parties, jobs=jobs, phases=phases, transferred_bytes=transferred,
                      report=settlement.assess(plan, fetches, phases, parties,
                                               {"honest": jobs["honest"], "cheated": jobs["cheat"]}),
                      execution_completed=True)
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
