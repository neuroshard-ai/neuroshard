#!/usr/bin/env python3
"""A clean-installed remote full node earns its first reward through outbound HTTPS."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time

os.environ.update(ATEN_CPU_CAPABILITY="default", MKL_ENABLE_INSTRUCTIONS="SSE4_2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from neuroshard.demo import client as wire, work
from neuroshard.publicnet.pool import Coordinator, Worker
from check_public_network import wait_for


def run(args):
    work.configure_cpu()
    report = json.loads(args.network_result.read_text())
    experiment = json.loads((Path(report["home"]) / "experiment.json").read_text())
    remote_root = Path(args.remote_root)
    remote_home = remote_root / f".neuroshard/outbound-node-{time.time_ns()}"
    controller = Path(experiment["remote_root"]) / "scripts/remote_lab_process.py"
    processes = remote_root / ".neuroshard/outbound-processes"

    def ssh(command):
        return subprocess.check_output(["ssh", "-o", "BatchMode=yes", experiment["host"],
            shlex.join([str(p) for p in command])], text=True, stderr=subprocess.PIPE)

    def remote_cli(*command):
        return ssh([remote_root / "venv_build/bin/python", remote_root / "scripts/neuroshard_chain.py", *command])

    def control(action, name, command=()):
        return ssh(["python3", controller, action, "--directory", processes, "--name", name,
                    *(["--", *command] if command else [])])

    # The public genesis came from HTTPS; the independently recorded test digest is pinned.
    peer = report["fresh_join"]["only_configured_peer"].split("@", 1)[0] + "@127.0.0.1:26656"
    config = json.loads(remote_cli("init", "--home", remote_home, "--genesis", args.site + "/network/genesis.json",
        "--genesis-sha256", report["genesis_sha256"], "--peer", peer, "--private-network", "--base-port", "27656"))
    coordinator = Coordinator(Path(experiment["homes"][0]))
    initial_round = wire.query(coordinator.rpc, "/summary")["round"]
    server = coordinator.server("127.0.0.1", 38660)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    stop = threading.Event()
    local_worker = Worker(Path(experiment["homes"][0]), 0)
    worker_thread = threading.Thread(target=local_worker.run, args=("http://127.0.0.1:38660", stop), daemon=True)
    succeeded = False
    try:
        control("start", "workerchain", [remote_root / "venv_build/bin/python", remote_root / "scripts/neuroshard_chain.py",
            "run", "--home", remote_home])
        def remote_status():
            return json.loads(remote_cli("status", "--home", remote_home))
        wait_for(lambda: remote_status()["round"] == initial_round)
        fresh = json.loads(remote_cli("account", "--home", remote_home))
        assert fresh["balance"] == 0
        worker_thread.start()
        control("start", "worker", [remote_root / "venv_build/bin/python", remote_root / "scripts/neuroshard_work.py",
            "worker", "--home", remote_home, "--stage", "1", "--coordinator", args.site, "--max-tasks", "1"])
        receipt = coordinator.mine()
        wait_for(lambda: json.loads(remote_cli("account", "--home", remote_home))["balance"] == 400000)
        state = remote_status()
        local = wire.query(coordinator.rpc, "/summary")
        assert state["model_root"] == local["model_root"] and state["round"] == initial_round + 1
        result = {"recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "passed": True,
            "chain_id": report["chain_id"], "genesis_sha256": report["genesis_sha256"],
            "clean_install_root": str(remote_root), "worker_public_key": config["public_key"],
            "initial_balance": "0", "earned_balance": "400000", "worker_transport": "Outbound HTTPS to public site",
            "worker_inbound_port": None, "local_lease_and_model_verification": True,
            "training_block_height": int(receipt["height"]), "round": state["round"], "model_root": state["model_root"],
            "ownership": "One operator; this verifies transport and protocol behavior, not independent ownership"}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        succeeded = True
    finally:
        stop.set()
        control("stop", "worker")
        control("stop", "workerchain")
        if worker_thread.is_alive():
            worker_thread.join(timeout=15)
        server.shutdown()
        server.server_close()
    return succeeded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-result", type=Path, default=ROOT / "docs/eval/results/public_network_candidate.json")
    parser.add_argument("--remote-root", default="/home/ubuntu/neuroshard-native-candidate")
    parser.add_argument("--site", default="https://neuroshard.com/native-preview")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/eval/results/outbound_work_candidate.json")
    run(parser.parse_args())
