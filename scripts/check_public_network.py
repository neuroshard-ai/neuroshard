#!/usr/bin/env python3
"""Exercise public genesis, direct native P2P, joining, and quorum recovery."""

import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

os.environ.update(ATEN_CPU_CAPABILITY="default", MKL_ENABLE_INSTRUCTIONS="SSE4_2")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from neuroshard.demo import client as wire, protocol, work
from neuroshard.lab import client
from neuroshard.publicnet import bootstrap


def wait_for(predicate, timeout=180):
    deadline, last = time.monotonic() + timeout, None
    while time.monotonic() < deadline:
        try:
            value = predicate()
            if value:
                return value
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            last = error
        time.sleep(0.5)
    raise AssertionError(f"Network condition timed out: {last}")


class Experiment:
    def __init__(self, host, remote_root):
        self.host, self.remote_root = host, Path(remote_root)
        self.home = Path(tempfile.mkdtemp(prefix="public-network-", dir=REPO / ".neuroshard"))
        self.remote_home = self.remote_root / ".neuroshard" / self.home.name
        self.children, self.homes = {}, []
        self.engine = bootstrap.engine_path()

    def ssh(self, args):
        return subprocess.check_output(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", self.host,
                                       shlex.join([str(arg) for arg in args])], text=True, stderr=subprocess.PIPE)

    def remote_cli(self, args):
        return self.ssh([self.remote_root / "venv_build/bin/python", self.remote_root / "scripts/neuroshard_chain.py", *args])

    def remote_control(self, action):
        command = [self.remote_root / "venv_build/bin/python", self.remote_root / "scripts/remote_lab_process.py",
                   action, "--directory", self.remote_home / "processes", "--name", "seed"]
        if action == "start":
            command += ["--", self.remote_root / "venv_build/bin/python", self.remote_root / "scripts/neuroshard_chain.py",
                        "run", "--home", self.remote_home / "seed"]
        return self.ssh(command)

    def remote_get(self, path):
        code = "import urllib.request; print(urllib.request.urlopen(" + repr("http://127.0.0.1:26659" + path) + ",timeout=5).read().decode())"
        return json.loads(self.ssh(["python3", "-c", code]))

    def sync(self):
        archive = self.home / "source.tar.gz"
        subprocess.run(["tar", "--exclude=__pycache__", "--exclude=*.pyc", "-czf", str(archive), "src",
            "scripts/neuroshard_chain.py", "scripts/remote_lab_process.py", "docs/demo-requirements.txt",
            "tests/test_public_node.py", "tests/test_protocol_candidate.py", "tests/test_verified_demo.py"], cwd=REPO, check=True)
        self.ssh(["mkdir", "-p", self.remote_home])
        subprocess.run(["scp", "-q", str(archive), f"{self.host}:{self.remote_home / 'source.tar.gz'}"], check=True)
        self.ssh(["tar", "-xzf", self.remote_home / "source.tar.gz", "-C", self.remote_root])

    def launch(self, name, command):
        with (self.home / f"{name}.log").open("ab") as log:
            env = os.environ.copy()
            env.update(PYTHONPATH=str(REPO / "src"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            self.children[name] = subprocess.Popen([str(arg) for arg in command], stdout=log, stderr=log,
                env=env, start_new_session=True)

    def start(self, index):
        self.launch(f"node{index}", [sys.executable, REPO / "scripts/neuroshard_chain.py", "run", "--home", self.homes[index]])

    def stop(self, name):
        child = self.children.pop(name, None)
        if child and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()

    def api(self, index):
        config = json.loads((self.homes[index] / "node.json").read_text())
        return f'http://127.0.0.1:{config["base_port"] + 3}'

    def status(self, index=0):
        return wire.http(self.api(index) + "/api/network", timeout=5)

    def cleanup(self):
        try:
            self.remote_control("stop")
        finally:
            for name in list(self.children):
                self.stop(name)


def run(args):
    work.configure_cpu()
    experiment = Experiment(args.host, args.remote_root)
    x = experiment
    passed = False
    report = {"recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Native public-port P2P between two machines; SSH used only for setup/process administration",
        "ownership": "One operator controls both machines; independent ownership is not established"}
    try:
        x.sync()
        chain_id = "neuroshard-stage-" + x.home.name.rsplit("-", 1)[-1].replace("_", "0")
        declarations = []
        for i in range(3):
            home = x.home / f"node{i}"
            x.homes.append(home)
            declarations.append(bootstrap.declaration(home, chain_id, 2500000, 20000000, x.engine))
        x.remote_cli(["declare", "--home", x.remote_home / "seed", "--chain-id", chain_id,
                      "--output", x.remote_home / "declaration.json"])
        declarations.append(json.loads(x.ssh(["cat", x.remote_home / "declaration.json"])))
        bundle = x.home / "bundle"
        made = bootstrap.make_genesis(chain_id, declarations, bundle, "testnet")
        report.update(made)
        report["manifest"] = json.loads((bundle / "genesis.json").read_text())["app_state"]["manifest"]
        subprocess.run(["scp", "-q", str(bundle / "genesis.json"), f"{x.host}:{x.remote_home / 'genesis.json'}"], check=True)
        remote = json.loads(x.remote_cli(["init", "--home", x.remote_home / "seed", "--genesis", x.remote_home / "genesis.json",
            "--genesis-sha256", made["genesis_sha256"], "--base-port", "26656", "--advertise", args.peer_host, "--private-network"]))
        remote_peer = f'{remote["node_id"]}@{args.peer_host}:26656'
        ids = [subprocess.check_output([x.engine, "show-node-id", "--home", str(home)], text=True).strip() for home in x.homes]
        for i, home in enumerate(x.homes):
            local_peers = [f"{key}@127.0.0.1:{args.base_port + j * 10}" for j, key in enumerate(ids) if i != j]
            bootstrap.initialize(home, str(bundle / "genesis.json"), made["genesis_sha256"], [remote_peer, *local_peers],
                x.engine, args.base_port + i * 10, private=True)
        x.remote_control("start")
        for i in range(3):
            x.start(i)
        wait_for(lambda: all(x.status(i)["ready"] for i in range(3)) and x.remote_get("/healthz")["ready"])
        print("PUBLIC: native validators connected over the public peer port", flush=True)

        joiner = x.home / "joiner"
        x.homes.append(joiner)
        before = x.status()["height"]
        bootstrap.initialize(joiner, str(bundle / "genesis.json"), made["genesis_sha256"], [remote_peer],
                             x.engine, args.base_port + 30, private=True)
        x.start(3)
        wait_for(lambda: x.status(3)["ready"] and x.status(3)["height"] >= before)
        joined_identity = protocol.Identity.load_or_create(joiner / "account.key")
        fresh = wire.http(x.api(3) + "/api/account?public_key=" + joined_identity.public_key)
        assert fresh["balance"] == "0"
        report["fresh_join"] = {"initial_balance": fresh["balance"], "only_configured_peer": remote_peer,
                                "joined_after_height": before, "keys_created_on_joiner": True}
        print("PUBLIC: a fresh zero-balance node joined from genesis and one public peer", flush=True)

        workers = [f"http://127.0.0.1:{args.base_port + 100 + i}" for i in range(2)]
        for i in range(2):
            key = joiner / "account.key" if i == 0 else x.home / "worker1.key"
            x.launch(f"worker{i}", [sys.executable, "-m", "neuroshard.demo.worker", "--stage", i,
                                   "--port", args.base_port + 100 + i, "--key", key])
        wait_for(lambda: all(wire.http(url + "/identity") for url in workers))
        sponsor = protocol.Identity.load_or_create(x.homes[0] / "account.key")
        rpc = x.api(0) + "/rpc"
        for _ in range(2):
            client.mine(rpc, workers, sponsor)
        # A commit response concerns the submitting node. Peers replay it later.
        wait_for(lambda: wire.http(x.api(3) + "/api/account?public_key=" + joined_identity.public_key)["balance"] == "800000")
        receipt, new_key = client.bond(rpc, joined_identity, joiner)
        report["earned_then_bonded"] = {"earned": "800000", "bond_height": int(receipt["height"])}
        def activated():
            validators = wire.http(x.api(0) + "/api/validators")["validators"]
            return len(validators) == 5 and any(v["public_key"] == new_key for v in validators)
        wait_for(activated, timeout=240)
        report["earned_then_bonded"]["observed_active_height"] = x.status()["height"]
        print("PUBLIC: earned-stake admission passed with the longer testnet activation delay", flush=True)

        x.remote_control("stop")
        progress = x.status()["height"]
        wait_for(lambda: x.status()["height"] >= progress + 4)
        report["seed_outage_preserves_progress"] = True
        x.stop("node2")
        time.sleep(4)
        halted = x.status()["height"]
        time.sleep(4)
        assert x.status()["height"] == halted
        report["insufficient_quorum_halts"] = {"height": halted, "remaining_power": 21, "total_power": 41}
        x.start(2)
        wait_for(lambda: x.status()["height"] > halted + 3)
        x.remote_control("start")
        wait_for(lambda: x.remote_get("/healthz")["ready"] and x.remote_get("/api/network")["height"] >= halted + 3)
        print("PUBLIC: seed outage, quorum halt, and restart recovery passed", flush=True)

        client.mine(rpc, workers, sponsor)
        wait_for(lambda: x.status()["round"] == 3)
        final = x.status()
        wait_for(lambda: x.remote_get("/api/network")["round"] == 3)
        final_remote = x.remote_get("/api/network")
        assert final["model_root"] == final_remote["model_root"] and final["issued"] == "3000000"
        common = min(final["height"], final_remote["height"])
        local_block = wire.http(x.api(0) + f"/api/block/{common}")
        remote_block = x.remote_get(f"/api/block/{common}")
        assert local_block["hash"] == remote_block["hash"]
        report.update(final_model_root=final["model_root"], issued=final["issued"], final_round=3,
            common_block={"height": common, "hash": local_block["hash"]}, all_nodes_agree=True, passed=True)
        report["home"] = str(x.home)
        report["remote_home"] = str(x.remote_home)
        report["preview_api"] = x.api(0)
        passed = True
    finally:
        if not (passed and args.keep_running):
            x.cleanup()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    (x.home / "experiment.json").write_text(json.dumps({"host": x.host, "homes": [str(p) for p in x.homes],
        "remote_home": str(x.remote_home), "remote_root": str(x.remote_root),
        "pids": {name: child.pid for name, child in x.children.items()}}, indent=2))
    print(json.dumps({"passed": True, "common_block": report["common_block"], "preview_api": report["preview_api"],
                      "kept_running": args.keep_running, "home": str(x.home)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--peer-host", required=True)
    parser.add_argument("--remote-root", default="/home/ubuntu/neuroshard-lab")
    parser.add_argument("--base-port", type=int, default=38656)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-running", action="store_true")
    run(parser.parse_args())
