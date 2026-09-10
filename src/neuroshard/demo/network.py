"""Launch a local four-validator NeuroShard chain and two training workers."""

import argparse
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from neuroshard.demo import client, protocol, work


REPO = Path(__file__).resolve().parents[3]
DEFAULT_HOME = REPO / ".neuroshard" / "reference"
COMET_VERSION = "0.38.26"


def engine_path(explicit=None):
    candidate = explicit or os.environ.get("COMETBFT_BINARY")
    if not candidate:
        local = REPO / ".neuroshard" / "tools" / "cometbft"
        candidate = str(local) if local.exists() else shutil.which("cometbft")
    if not candidate:
        raise ValueError("Install CometBFT v0.38.26 using scripts/install_demo_consensus.sh")
    candidate = str(Path(candidate).resolve())
    version = subprocess.check_output([candidate, "version"], text=True).strip()
    if version != COMET_VERSION:
        raise ValueError(f"Expected CometBFT {COMET_VERSION}, found {version}")
    return candidate


def edit_config(text, section, key, value):
    current, found, lines = "", False, []
    for line in text.splitlines():
        if re.match(r"^\[.*\]$", line):
            current = line[1:-1]
        if current == section and re.match(rf"^{re.escape(key)}\s*=", line):
            line, found = f"{key} = {value}", True
        lines.append(line)
    if not found:
        raise ValueError(f"Missing upstream configuration option [{section}] {key}")
    return "\n".join(lines) + "\n"


def initialize(home=DEFAULT_HOME, base_port=27650, engine=None, data=None):
    home = Path(home).resolve()
    if home.exists() and any(home.iterdir()):
        raise ValueError(f"Use an empty home directory: {home}")
    if not 1024 <= base_port <= 65480:
        raise ValueError("Base port must be in [1024, 65480]")
    binary = engine_path(engine)
    data = Path(data or REPO / "docs/eval/data/input.txt").resolve()
    work.configure_cpu()
    spec = work.manifest(work.read_data(data))
    home.mkdir(parents=True, exist_ok=True)
    subprocess.run([binary, "testnet", "--v", "4", "--o", str(home / "chain"),
                    "--home", str(home / "initializer"),
                    "--populate-persistent-peers=false"], check=True)
    nodes = []
    for i in range(4):
        node_home = home / "chain" / f"node{i}"
        node_id = subprocess.check_output([binary, "show-node-id", "--home", str(node_home)], text=True).strip()
        nodes.append({"home": str(node_home), "id": node_id, "p2p": base_port + i * 10,
                      "rpc": base_port + i * 10 + 1, "abci": base_port + i * 10 + 2})
    chain_id = "neuroshard-dev-" + secrets.token_hex(6)
    genesis = json.loads((Path(nodes[0]["home"]) / "config/genesis.json").read_text())
    genesis.update(chain_id=chain_id, app_state=spec)
    # No enormous blocks: at most one 16-KiB application transaction is accepted.
    genesis["consensus_params"]["block"]["max_bytes"] = "65536"
    genesis["consensus_params"]["evidence"]["max_bytes"] = "16384"
    for i, node in enumerate(nodes):
        path = Path(node["home"]) / "config/config.toml"
        text = path.read_text()
        peers = ",".join(f'{other["id"]}@127.0.0.1:{other["p2p"]}' for j, other in enumerate(nodes) if i != j)
        changes = [("", "proxy_app", json.dumps(f'127.0.0.1:{node["abci"]}')),
                   ("", "abci", '"grpc"'), ("", "log_level", '"error"'),
                   ("rpc", "laddr", json.dumps(f'tcp://127.0.0.1:{node["rpc"]}')),
                   ("p2p", "laddr", json.dumps(f'tcp://127.0.0.1:{node["p2p"]}')),
                   ("p2p", "persistent_peers", json.dumps(peers)),
                   ("p2p", "allow_duplicate_ip", "true"),
                   ("consensus", "timeout_commit", '"500ms"'),
                   ("consensus", "timeout_propose", '"1s"'),
                   ("consensus", "timeout_prevote", '"500ms"'),
                   ("consensus", "timeout_precommit", '"500ms"')]
        for section, key, value in changes:
            text = edit_config(text, section, key, value)
        path.write_text(text)
        (path.parent / "genesis.json").write_bytes(work.canonical(genesis))
    config = {"home": str(home), "engine": binary, "data": str(data), "chain_id": chain_id,
              "nodes": nodes, "workers": [base_port + 50, base_port + 51]}
    (home / "network.json").write_bytes(work.canonical(config))
    return config


def read_config(home=DEFAULT_HOME):
    return json.loads((Path(home) / "network.json").read_text())


def read_pids(config):
    path = Path(config["home"]) / "processes.json"
    return json.loads(path.read_text()) if path.exists() else {}


def owned_process(config, pid):
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        prefix = config["home"].encode() + b"/"
        return any(arg.startswith(prefix) for arg in command)
    except (FileNotFoundError, ProcessLookupError):
        return False


def launch(config, name, command):
    pids = read_pids(config)
    if name in pids and owned_process(config, pids[name]):
        return
    log_path = Path(config["home"]) / "logs"
    log_path.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "1"
    with (log_path / f"{name}.log").open("ab") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, env=env, start_new_session=True)
    pids[name] = process.pid
    (Path(config["home"]) / "processes.json").write_bytes(work.canonical(pids))


def start_validator(config, index):
    node = config["nodes"][index]
    launch(config, f"app{index}", [sys.executable, "-m", "neuroshard.demo.app",
           "--home", node["home"], "--data", config["data"], "--port", str(node["abci"])])
    launch(config, f"node{index}", [config["engine"], "start", "--home", node["home"]])


def stop_process(config, name):
    pids = read_pids(config)
    pid = pids.pop(name, None)
    if pid and owned_process(config, pid):
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 5
        while owned_process(config, pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        if owned_process(config, pid):
            os.kill(pid, signal.SIGKILL)
    (Path(config["home"]) / "processes.json").write_bytes(work.canonical(pids))


def stop_validator(config, index):
    stop_process(config, f"node{index}")
    stop_process(config, f"app{index}")


def stop(config):
    for name in list(read_pids(config)):
        stop_process(config, name)


def urls(config, index=0):
    return (f'http://127.0.0.1:{config["nodes"][index]["rpc"]}',
            [f"http://127.0.0.1:{port}" for port in config["workers"]])


def wait_ready(config, timeout=60, indices=range(4)):
    deadline, last_error = time.monotonic() + timeout, None
    while time.monotonic() < deadline:
        try:
            states = [client.query(urls(config, i)[0]) for i in indices]
            syncing = [client.rpc(urls(config, i)[0], "status")["sync_info"]["catching_up"] for i in indices]
            commitments = {(s["round"], s["model_root"], work.digest(s["lease"])) for s in states}
            if all(state["height"] >= 1 for state in states) and not any(syncing) and len(commitments) == 1:
                for worker in urls(config)[1]:
                    client.http(worker + "/identity")
                return states
        except Exception as exc:
            last_error = exc
        time.sleep(0.2)
    raise RuntimeError(f"Network did not become ready; see {config['home']}/logs: {last_error}")


def start(config):
    for i in range(4):
        start_validator(config, i)
    for i, port in enumerate(config["workers"]):
        launch(config, f"worker{i}", [sys.executable, "-m", "neuroshard.demo.worker", "--stage", str(i),
               "--port", str(port), "--key", str(Path(config["home"]) / f"worker{i}.key")])
    try:
        return wait_ready(config)
    except Exception:
        stop(config)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME)
    sub = parser.add_subparsers(dest="command", required=True)
    up = sub.add_parser("up", help="Initialize if needed and start a local development network")
    up.add_argument("--base-port", type=int, default=27650)
    up.add_argument("--engine")
    sub.add_parser("down", help="Stop only the processes recorded for this network")
    sub.add_parser("status")
    mine = sub.add_parser("mine")
    mine.add_argument("--steps", type=int, default=10)
    infer = sub.add_parser("infer")
    infer.add_argument("prompt")
    infer.add_argument("--max-tokens", type=int, default=24)
    check = sub.add_parser("check", help="Run training, reward, rejection, and outage checks on a fresh chain")
    check.add_argument("--steps", type=int, default=12)
    check.add_argument("--base-port", type=int, default=28650)
    check.add_argument("--engine")
    check.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "check":
        from neuroshard.demo.check import run
        if not 2 <= args.steps <= 50:
            parser.error("Check steps must be between two and fifty")
        result = run(args.steps, args.base_port, args.engine, args.output)
        print(json.dumps(result, indent=2))
        return
    config = (read_config(args.home) if (args.home / "network.json").exists()
              else initialize(args.home, args.base_port, args.engine) if args.command == "up" else None)
    if config is None:
        parser.error("Run up first")
    url, workers = urls(config)
    if args.command == "up":
        states = start(config)
        result = {"chain_id": config["chain_id"], "rpc": url, "workers": workers,
                  "validators": len(states), "round": states[0]["round"], "home": config["home"]}
    elif args.command == "down":
        stop(config)
        result = {"stopped": config["home"]}
    elif args.command == "status":
        result = client.query(url)
    elif args.command == "mine":
        if not 1 <= args.steps <= work.MAX_REWARDED_TASKS:
            parser.error("Steps must be between one and the development emission limit")
        identity = protocol.Identity.load_or_create(Path(config["home"]) / "miner.key")
        result = client.mine(url, workers, identity, args.steps)
    else:
        result = client.query(url, "/infer", {"prompt": args.prompt, "max_tokens": args.max_tokens})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
