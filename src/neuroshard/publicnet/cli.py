"""Operate a native full node with local signing keys and public genesis."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from neuroshard.demo import client as wire, protocol, work
from neuroshard.lab import client
from neuroshard.publicnet import bootstrap


def run(home):
    config = json.loads((home / "node.json").read_text())
    bootstrap.engine_path(config["engine"])
    logs = home / "logs"
    logs.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(ATEN_CPU_CAPABILITY="default",
               MKL_ENABLE_INSTRUCTIONS="SSE4_2", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if (bootstrap.REPO / "src/neuroshard").is_dir():
        env["PYTHONPATH"] = str(bootstrap.REPO / "src")
    commands = {"application": [sys.executable, "-m", "neuroshard.lab.app", "--home", str(home),
        "--data", str(home / "corpus.txt"), "--port", str(config["base_port"] + 2), "--profile", config["profile"]],
        "consensus": [config["engine"], "start", "--home", str(home)],
        "gateway": [sys.executable, "-m", "neuroshard.publicnet.gateway", "--home", str(home)]}
    children, stopping = [], False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for name, command in commands.items():
            with (logs / f"{name}.log").open("ab") as log:
                children.append(subprocess.Popen(command, stdout=log, stderr=log, env=env, start_new_session=True))
        print(json.dumps({"chain_id": config["chain_id"], "peer_port": config["base_port"],
                          "api": f'http://{config["api_host"]}:{config["base_port"] + 3}', "home": str(home)}), flush=True)
        while not stopping:
            if any(child.poll() is not None for child in children):
                raise RuntimeError(f"A node component exited; inspect {logs}")
            checkpoint = config["trusted_checkpoint"]
            if checkpoint:
                try:
                    rpc = f'http://127.0.0.1:{config["base_port"] + 1}'
                    current = int(wire.rpc(rpc, "status")["sync_info"]["latest_block_height"])
                    if current >= checkpoint["height"]:
                        block = wire.rpc(rpc, "block", {"height": str(checkpoint["height"])})
                        if block["block_id"]["hash"] != checkpoint["hash"]:
                            raise RuntimeError("Trusted checkpoint mismatch; stopping node")
                        config["trusted_checkpoint"] = None
                except (OSError, wire.Rejected):
                    pass
            time.sleep(0.5)
    finally:
        for child in reversed(children):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    declare = sub.add_parser("declare", help="Generate local keys and a public genesis declaration")
    declare.add_argument("--home", type=Path, required=True)
    declare.add_argument("--chain-id", required=True)
    declare.add_argument("--bond", type=int, default=2500000)
    declare.add_argument("--liquid", type=int, default=20000000)
    declare.add_argument("--output", type=Path, required=True)
    declare.add_argument("--engine")
    genesis = sub.add_parser("genesis", help="Assemble public declarations into a reviewable genesis bundle")
    genesis.add_argument("--chain-id", required=True)
    genesis.add_argument("--declarations", nargs="+", type=Path, required=True)
    genesis.add_argument("--output", type=Path, required=True)
    genesis.add_argument("--profile", choices=("lab", "testnet"), default="testnet")
    init = sub.add_parser("init", help="Join from a checksum-pinned public genesis")
    init.add_argument("--home", type=Path, required=True)
    init.add_argument("--genesis", required=True)
    init.add_argument("--genesis-sha256", required=True)
    init.add_argument("--peer", action="append", default=[])
    init.add_argument("--engine")
    init.add_argument("--base-port", type=int, default=26656)
    init.add_argument("--advertise")
    init.add_argument("--private-network", action="store_true")
    init.add_argument("--api-host", default="127.0.0.1")
    init.add_argument("--trusted-height", type=int, default=0)
    init.add_argument("--trusted-hash")
    start = sub.add_parser("run", help="Run supervised consensus, application, and public ledger API")
    start.add_argument("--home", type=Path, required=True)
    for name in ("status", "account", "transfer", "bond", "unbond", "withdraw", "mine"):
        command = sub.add_parser(name)
        command.add_argument("--home", type=Path, required=True)
        command.add_argument("--rpc", help="Defaults to this node's local native RPC")
        if name in ("transfer", "bond"):
            command.add_argument("--amount", type=int, required=True)
        if name == "transfer":
            command.add_argument("--to", required=True)
        if name == "mine":
            command.add_argument("--workers", nargs=2, required=True)
    args = parser.parse_args()
    work.configure_cpu()
    if args.command == "declare":
        value = bootstrap.declaration(args.home, args.chain_id, args.bond, args.liquid, bootstrap.engine_path(args.engine))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(work.canonical(value))
        print(json.dumps({"public_key": value["public_key"], "declaration": str(args.output)}))
        return
    if args.command == "genesis":
        result = bootstrap.make_genesis(args.chain_id, [protocol.parse_json(p.read_bytes()) for p in args.declarations], args.output, args.profile)
    elif args.command == "init":
        result = bootstrap.initialize(args.home, args.genesis, args.genesis_sha256, args.peer, args.engine,
            args.base_port, args.advertise, args.private_network, args.api_host, args.trusted_height, args.trusted_hash)
    elif args.command == "run":
        run(args.home.resolve())
        return
    else:
        config = json.loads((args.home / "node.json").read_text())
        rpc = args.rpc or f'http://127.0.0.1:{config["base_port"] + 1}'
        identity = protocol.Identity.load_or_create(args.home / "account.key")
        status = wire.query(rpc, "/summary")
        if status["chain_id"] != config["chain_id"]:
            raise ValueError("RPC belongs to a different chain")
        account = wire.query(rpc, "/account", {"public_key": identity.public_key})
        if args.command == "status":
            result = status
        elif args.command == "account":
            result = account
        elif args.command == "mine":
            result = client.mine(rpc, args.workers, identity)
        else:
            fields = {}
            if args.command == "transfer":
                fields = {"to": args.to, "amount": args.amount}
            else:
                key, secret = client.consensus_identity(args.home)
                fields = {"consensus_key": key}
                if args.command == "bond":
                    fields.update(amount=args.amount, possession=secret.sign(client.state.possession_message(
                        config["chain_id"], identity.public_key, key, args.amount, account["nonce"])).hex())
            envelope = identity.sign({"kind": args.command, "chain_id": config["chain_id"], "nonce": account["nonce"], **fields})
            result = wire.broadcast(rpc, envelope)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
