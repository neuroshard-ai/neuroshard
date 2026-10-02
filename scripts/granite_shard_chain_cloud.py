#!/usr/bin/env python3
"""Owners, the auditor and four CometBFT validators for the chain settlement execution; retire every host.

The controller holds the user account and submits signed transactions to validator
mempools over SSH; blocks come from CometBFT consensus among the validator hosts.
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import time

from neuroshard.evolution import granite_shard_chain as chain
from neuroshard.evolution import granite_shard_settlement as settlement
from neuroshard.evolution.modular_reference_execution import ROOT, read, save, sha256, wait_for_ci

spec = importlib.util.spec_from_file_location("granite_shard_settlement_cloud", ROOT / "scripts/granite_shard_settlement_cloud.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
base, ring, cloud = helpers.base, helpers.ring, helpers.cloud
VALIDATORS = [f"validator-{i}" for i in (1, 2, 3, 4)]
RPC = """
import json, sys, urllib.request
request = urllib.request.Request("http://127.0.0.1:%d", data=sys.stdin.buffer.read(), headers={"Content-Type": "application/json"})
print(urllib.request.urlopen(request, timeout=320).read().decode())
""" % chain.PORTS["rpc"]


class Rejected(Exception):
    pass


def start(home, allocation, role, phase, plan, address=""):
    memory = plan["fetch_memory_limit_bytes"] if phase == "fetch" else plan["memory_limit_bytes"]
    command = [cloud.PYTHON, cloud.REMOTE + "/" + chain.SCRIPT, "--role", role, "--address", address,
               "--port", str(ring.PORT), "--phase", phase, "--home", ring.OWNER_HOME, "--store", ring.STORE]
    cloud.ssh(home, allocation, ["systemd-run", "--user", f"--unit=granite-{phase}",
        "-p", f"RuntimeMaxSec={base.seconds(plan, phase)}", "-p", f"MemoryMax={memory}",
        "-p", "MemorySwapMax=0", "-p", "TimeoutStopSec=0", "-p", "KillMode=control-group",
        "--working-directory=" + cloud.REMOTE, "--setenv=PYTHONPATH=" + cloud.REMOTE + "/src", *command])


def run_phase(hosts, plan, phase, address=""):
    for host, role in hosts:
        start(*host, role, phase, plan, address)
    helpers.wait([host for host, _ in hosts], plan, phase)
    path = "fetch.json" if phase == "fetch" else f"{phase}/result.json"
    return {role: ring.remote_json(*host, f"{ring.OWNER_HOME}/{path}") for host, role in hosts}


def rpc(host, method, params=None):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}).encode()
    value = json.loads(cloud.ssh(*host, [cloud.PYTHON, "-c", RPC], data=body, timeout=360).stdout)
    if "error" in value:
        raise Rejected(json.dumps(value["error"]))
    return value["result"]


def height(host):
    return int(rpc(host, "status")["sync_info"]["latest_block_height"])


def state(host):
    response = rpc(host, "abci_query", {"path": "/state", "prove": False})["response"]
    if response.get("code", 0):
        raise Rejected(response.get("log"))
    return json.loads(base64.b64decode(response["value"]))


def wait_height(validators, target, timeout=3600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(height(v) >= target for v in validators):
            return
        time.sleep(2)
    raise TimeoutError(f"validators did not reach height {target}")


def submit(validators, envelope, via=0, timeout=900):
    """Mempool admission, and for an admitted transaction the height at which every validator committed it."""
    from neuroshard.inference import optimistic as ledger

    raw = ledger.canonical(envelope)
    try:
        admitted = rpc(validators[via], "broadcast_tx_sync", {"tx": base64.b64encode(raw).decode()})
        receipt = {"code": admitted.get("code", 0), "log": admitted.get("log", ""), "height": None}
        if receipt["code"]:
            return receipt
    except Exception as error:
        # A reply can be lost after the transaction was admitted; inclusion decides.
        receipt = {"code": None, "log": f"admission reply lost: {type(error).__name__}", "height": None}
    digest = base64.b64encode(hashlib.sha256(raw).digest()).decode()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            found = rpc(validators[0], "tx", {"hash": digest})
            receipt.update(height=int(found["height"]), code=found["tx_result"].get("code", 0),
                           log=found["tx_result"].get("log", ""))
            wait_height(validators, receipt["height"])
            return receipt
        except Rejected as error:
            if "not found" not in str(error):
                raise
        time.sleep(2)
    raise TimeoutError("transaction was not committed")


def agreed(validators, timeout=600):
    """Every validator's state at one common height."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        states = [state(v) for v in validators]
        if len({s["height"] for s in states}) == 1:
            return states
        time.sleep(0.5)
    raise TimeoutError("validators never reported one common height")


def run(home):
    from neuroshard.inference import optimistic as ledger
    from neuroshard.inference import optimistic_network as network

    home.mkdir(parents=True, exist_ok=True)
    plan = read(ROOT / chain.PLAN)
    terms = plan["ledger"]
    source = chain.committed_sources()
    wait_for_ci(home, source["commit"])
    if chain.committed_sources() != source:
        raise ValueError("source changed while waiting for CI")
    binary = Path(plan["cometbft"]["local"]).read_bytes()
    if hashlib.sha256(binary).hexdigest() != plan["cometbft"]["sha256"]:
        raise ValueError("local CometBFT binary differs from the declaration")
    world = len(plan["boundaries"]) - 1
    names = [f"owner-{r}" for r in range(world)] + ["auditor"] + VALIDATORS
    homes = {name: home / name for name in names}
    allocations, failure, started_chain = {}, None, []
    result = {"binding": {"freeze": source, "plan_sha256": sha256(ROOT / chain.PLAN)}, "execution_completed": False}
    admissions, phases = {}, {}
    try:
        save(home / "status.json", {"state": "allocating", "commit": source["commit"]})
        for name in names:
            homes[name].mkdir(parents=True, exist_ok=True)
            allocations[name] = cloud.allocate(homes[name], source["commit"], chain.PROFILE)
        hosts = {name: (homes[name], allocations[name]) for name in names}
        owners = [hosts[f"owner-{r}"] for r in range(world)]
        validators = [hosts[name] for name in VALIDATORS]
        ring.open_ring([allocations[f"owner-{r}"] for r in range(world)])
        ring.open_ring([allocations[name] for name in VALIDATORS])
        save(home / "status.json", {"state": "setup"})
        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            list(pool.map(lambda name: cloud.bootstrap(*hosts[name], source, chain.PROFILE), names))
        for name in ("owner-0", f"owner-{world - 1}"):
            base.upload(chain, *hosts[name], plan)
        for host in validators:
            cloud.ssh(*host, ["bash", "-c", f"mkdir -p {ring.STORE} && cat > {ring.STORE}/cometbft && chmod +x {ring.STORE}/cometbft"],
                      data=binary, timeout=900)
        save(home / "status.json", {"state": "running", "phase": "fetch"})
        fetched = run_phase([(owners[r], f"owner-{r}") for r in range(world)] + [(hosts["auditor"], "auditor")]
                            + [(hosts[name], name) for name in VALIDATORS], plan, "fetch")
        if not all(f and f.get("completed") for f in fetched.values()):
            raise RuntimeError("a host could not fetch its shard")
        accounts = {r: ring.remote_json(*owners[r], f"{ring.OWNER_HOME}/account.json") for r in (1, 2)}
        fetches = {"owners": [{**fetched[f"owner-{r}"], **(accounts.get(r) or {})} for r in range(world)],
                   "auditor": fetched["auditor"], "validators": [fetched[name] for name in VALIDATORS]}
        auditor_accounts = ring.remote_json(*hosts["auditor"], f"{ring.OWNER_HOME}/accounts.json")
        user_key, user_public = settlement.account(home, "user")
        parties = {"user": user_public, "owner-1": accounts[1]["account"], "owner-2": accounts[2]["account"],
                   "auditor": auditor_accounts["auditor"], "accuser": auditor_accounts["accuser"]}
        log_keys = [accounts[1]["log_key"], accounts[2]["log_key"]]
        model_root = sha256(ROOT / chain.MODEL_INVENTORY)
        save(home / "parties.json", {"parties": parties, "log_keys": log_keys, "model_root": model_root})

        save(home / "status.json", {"state": "running", "phase": "chain"})
        identities = run_phase([(hosts[name], name) for name in VALIDATORS], plan, "chain-init")
        if not all(i and i.get("completed") for i in identities.values()):
            raise RuntimeError("a validator could not create its consensus identity")
        genesis = network.genesis_document(identities["validator-1"]["template"],
                                           {**terms, "allocations": {a: terms["allocation"] for a in parties.values()},
                                            "model_root": model_root},
                                           [identities[name] for name in VALIDATORS],
                                           identities["validator-1"]["template"]["genesis_time"])
        save(home / "genesis.json", genesis)
        for name in VALIDATORS:
            peers = ",".join(f'{identities[other]["node_id"]}@{allocations[other]["private_ip"]}:{chain.PORTS["p2p"]}'
                             for other in VALIDATORS if other != name)
            helpers.put(hosts[name], "chain-configure-request.json", {"genesis": genesis, "peers": peers})
        configured = run_phase([(hosts[name], name) for name in VALIDATORS], plan, "chain-configure")
        if len({c.get("genesis_sha256") for c in configured.values()}) != 1:
            raise RuntimeError("validators configured different genesis documents")
        runtime = str(plan["chain_seconds"])
        # Process logs go to the evidence directory: host journals are lost at retirement.
        node_home = f"{ring.OWNER_HOME}/chain"
        app = (f"exec {cloud.PYTHON} -m neuroshard.inference.optimistic_app --home {node_home} "
               f"--port {chain.PORTS['abci']} >> {node_home}/app.log 2>&1")
        node = f"exec {ring.STORE}/cometbft start --home {node_home} >> {node_home}/node.log 2>&1"
        for host in validators:
            started_chain.append(host)
            cloud.ssh(*host, ["systemd-run", "--user", "--unit=settlement-app", "-p", f"RuntimeMaxSec={runtime}",
                              "--working-directory=" + cloud.REMOTE, "--setenv=PYTHONPATH=" + cloud.REMOTE + "/src",
                              "bash", "-c", app])
        time.sleep(30)
        for host in validators:
            cloud.ssh(*host, ["systemd-run", "--user", "--unit=settlement-node", "-p", f"RuntimeMaxSec={runtime}",
                              "bash", "-c", node])
        wait_height(validators, 2, timeout=900)

        def owner_envelopes(phase, nonce, **request):
            for r in (1, 2):
                helpers.put(owners[r], f"{phase}-request.json", {"chain_id": terms["chain_id"], "nonce": nonce, **request})
            replies = run_phase([(owners[r], f"owner-{r}") for r in (1, 2)], plan, phase)
            phases[phase] = replies
            if not all(reply and reply.get("completed") for reply in replies.values()):
                raise RuntimeError(f"an owner could not complete {phase}")
            return {r: replies[f"owner-{r}"]["envelope"] for r in (1, 2)}

        def open_job(nonce, label):
            body = {"kind": "serve_open", "chain_id": terms["chain_id"], "nonce": nonce, "model_root": model_root,
                    "owners": log_keys, "request_root": hashlib.sha256(f"{plan['target']}:{label}".encode()).hexdigest(),
                    "price": terms["price"]}
            envelope = settlement.signed(user_key, body)
            return envelope, ledger.transaction_id(envelope)

        save(home / "status.json", {"state": "running", "phase": "sign-bond"})
        for r, envelope in owner_envelopes("sign-bond", 0, model_root=model_root, amount=terms["owner_bond"]).items():
            admissions[f"bond-{r}"] = submit(validators, envelope, via=r)
        jobs = {}
        address = allocations["owner-0"]["private_ip"]
        for nonce, label in ((1, "honest"), (2, "cheat")):
            opened, jobs[label] = open_job(nonce - 1, label)
            admissions[f"open-{label}"] = submit(validators, opened)
            serve = f"serve-{label}"
            save(home / "status.json", {"state": "running", "phase": serve})
            served = run_phase([(owners[r], f"owner-{r}") for r in range(world)], plan, serve, address)
            phases[serve] = [served[f"owner-{r}"] for r in range(world)]
            save(home / f"{serve}.json", phases[serve])
            helpers.transfer(owners[1], hosts["auditor"], serve, "log-1")
            for r, envelope in owner_envelopes(f"commit-{label}", nonce, job_id=jobs[label]).items():
                admissions[f"commit-{label}-{r}"] = submit(validators, envelope, via=r)
            audit = f"audit-{label}"
            save(home / "status.json", {"state": "running", "phase": audit})
            phases[audit] = run_phase([(hosts["auditor"], "auditor")], plan, audit)["auditor"]
            save(home / f"{audit}.json", phases[audit])
            challenge = "frame" if label == "honest" else "prove"
            if label == "cheat":
                window = max(admissions[f"commit-honest-{r}"]["height"] for r in (1, 2)) + terms["params"]["challenge_blocks"]
                wait_height(validators, window + 1)
            helpers.put(hosts["auditor"], f"{challenge}-request.json",
                        {"chain_id": terms["chain_id"], "honest_job": jobs["honest"], "cheated_job": jobs.get("cheat"),
                         "log_key": log_keys[0]})
            phases[challenge] = run_phase([(hosts["auditor"], "auditor")], plan, challenge)["auditor"]
            if not (phases[challenge] or {}).get("completed"):
                raise RuntimeError(f"the auditor could not complete {challenge}")
            payload = cloud.ssh(*hosts["auditor"], ["tar", "-czf", "-", "-C", ring.OWNER_HOME, "bundles"], timeout=900).stdout
            for host in validators:
                cloud.ssh(*host, ["bash", "-c", f"tar -xzf - -C {ring.OWNER_HOME}"], data=payload, timeout=900)
            if challenge == "frame":
                admissions["framing"] = submit(validators, phases["frame"]["framing"], via=3)
            else:
                admissions["proven"] = submit(validators, phases["prove"]["proven"], via=2)
        save(home / "status.json", {"state": "running", "phase": "agree"})
        states = agreed(validators)
        save(home / "states.json", states)
        result.update(fetches=fetches, parties=parties, jobs=jobs, phases=phases, admissions=admissions,
                      genesis_sha256=hashlib.sha256(json.dumps(genesis, sort_keys=True).encode()).hexdigest(),
                      report=chain.assess(plan, fetches, phases, parties, {"honest": jobs["honest"], "cheated": jobs["cheat"]},
                                          admissions, states),
                      execution_completed=True)
    except BaseException as error:
        failure = str(error) or type(error).__name__
        result.update(error=failure, admissions=admissions)
    finally:
        for host in started_chain:
            try:
                cloud.ssh(*host, ["systemctl", "--user", "stop", "settlement-node", "settlement-app"], timeout=120)
            except Exception:
                pass
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
