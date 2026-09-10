#!/usr/bin/env python3
"""Run the v2 acceptance scenario across two prepared physical SSH hosts."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

os.environ.update(ATEN_CPU_CAPABILITY="default", MKL_ENABLE_INSTRUCTIONS="SSE4_2")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from neuroshard.demo import network as base
from neuroshard.lab import network


class RemoteLab:
    def __init__(self, host, root):
        self.host, self.root = host, Path(root)
        self.original_launch, self.original_stop_process = base.launch, base.stop_process
        self.original_stop, self.original_initialize = base.stop, network.initialize
        self.config, self.remote_home, self.tunnel = None, None, None
        self.copied = set()
        self.remote_names = {"app3", "node3", "app4", "node4", "worker1", "provider"}

    def ssh(self, command):
        return subprocess.check_output(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                                        self.host, shlex.join([str(arg) for arg in command])], text=True)

    def translate(self, path):
        value = str(path)
        return str(self.root) + value[len(str(REPO)):] if value.startswith(str(REPO) + "/") else value

    def control(self, action, name=None, command=()):
        args = [self.root / "venv_build/bin/python", self.root / "scripts/remote_lab_process.py", action,
                "--directory", self.remote_home / "processes"]
        if name:
            args += ["--name", name]
        if command:
            args += ["--", *[self.translate(arg) for arg in command]]
        return self.ssh(args)

    def initialize(self, base_port, engine=None):
        config = self.original_initialize(base_port, engine)
        self.config = config
        self.remote_home = Path(self.translate(config["home"]))
        self.ssh(["mkdir", "-p", self.remote_home / "chain"])
        # Application/worker RPCs stay on loopback. SSH carries P2P and the remote stage's HTTP traffic.
        command = ["ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
                   "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3"]
        remote_ports = [config["nodes"][i][kind] for i in (3, 4) for kind in ("p2p", "rpc")]
        remote_ports += [config["workers"][1], base_port + 52]
        for port in remote_ports:
            command += ["-L", f"127.0.0.1:{port}:127.0.0.1:{port}"]
        for i in (0, 1, 2):
            port = config["nodes"][i]["p2p"]
            command += ["-R", f"127.0.0.1:{port}:127.0.0.1:{port}"]
        command.append(self.host)
        with (Path(config["home"]) / "ssh-transport.log").open("wb") as log:
            self.tunnel = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        time.sleep(1)
        if self.tunnel.poll() is not None:
            raise RuntimeError("SSH transport could not bind; inspect ssh-transport.log")
        return config

    def launch(self, config, name, command):
        if name not in self.remote_names:
            return self.original_launch(config, name, command)
        if name in ("app3", "app4") and name not in self.copied:
            node = config["nodes"][int(name[-1])]
            subprocess.run(["scp", "-q", "-r", node["home"],
                            f"{self.host}:{self.remote_home / 'chain'}"], check=True)
            self.copied.add(name)
        if name == "provider" and name not in self.copied:
            subprocess.run(["scp", "-q", str(Path(config["home"]) / "provider.key"),
                            f"{self.host}:{self.remote_home / 'provider.key'}"], check=True)
            self.copied.add(name)
        pid = int(self.control("start", name, command).strip())
        config.setdefault("remote_processes", {})[name] = pid
        return pid

    def stop_process(self, config, name):
        if name not in self.remote_names:
            return self.original_stop_process(config, name)
        self.control("stop", name)
        config.get("remote_processes", {}).pop(name, None)

    def stop(self, config):
        try:
            if self.remote_home:
                self.control("stop-all")
        finally:
            try:
                self.original_stop(config)
            finally:
                if self.tunnel:
                    self.tunnel.terminate()
                    self.tunnel.wait(timeout=10)
                    self.tunnel = None

    def __enter__(self):
        base.launch, base.stop_process, base.stop = self.launch, self.stop_process, self.stop
        network.initialize = self.initialize
        return self

    def __exit__(self, *_):
        try:
            if self.config:
                self.stop(self.config)
        finally:
            base.launch, base.stop_process, base.stop = self.original_launch, self.original_stop_process, self.original_stop
            network.initialize = self.original_initialize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-root", default="/home/ubuntu/neuroshard-lab")
    parser.add_argument("--base-port", type=int, default=32650)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.host.startswith("-") or not Path(args.remote_root).is_absolute():
        parser.error("provide an SSH host and an absolute remote root")
    with RemoteLab(args.host, args.remote_root):
        report = network.run(base_port=args.base_port)
    report["scope"] = "Two physical hosts, five native nodes, two pipeline stages, one inference provider; one operator"
    report["placement"] = {"local_full_nodes": [0, 1, 2], "remote_full_nodes": [3, 4],
                           "local_pipeline_stages": [0], "remote_pipeline_stages": [1],
                           "inference_provider": "remote", "transport": "SSH forwards for native P2P and remote RPC/HTTP"}
    report["limitations"] = ["Both machines belong to one operator", "SSH carries actual inter-host traffic; open Internet P2P discovery is not tested",
                             "The tiny execution profile and short lab evidence windows are unchanged"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "all_five_nodes_agree": report["all_five_nodes_agree"],
                      "final_round": report["final_round"], "shared_block": report["shared_block"]}, indent=2))


if __name__ == "__main__":
    main()
