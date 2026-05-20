#!/usr/bin/env python3
"""
Docker LAN gate for NeuroShard.

Runs CPU-only tiny NeuroShard nodes in separate Docker containers on a bridge
network. This is a single-machine LAN emulation: it does not prove GPU
throughput or real router/NAT behavior, but it does exercise separate network
identities, tracker discovery, HTTP/gRPC reachability, training, and PoNW.

Example:
    python scripts/lan_gate.py --nodes 2 --duration 90 --report reports/lan-gate.json --check
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import List, Tuple

import requests


DEFAULT_NODE_IMAGE = "neuroshard-observer"
DEFAULT_TRACKER_IMAGE = "neuroshard-tracker"


def run(cmd: List[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a command."""
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def docker_exists(kind: str, name: str) -> bool:
    """Return True if a Docker object exists."""
    result = run(["docker", kind, "inspect", name], check=False, capture=True)
    return result.returncode == 0


def docker_rm_container(name: str) -> None:
    """Remove an old container if present."""
    if docker_exists("container", name):
        run(["docker", "rm", "-f", name], check=False)


def docker_run_args(keep: bool) -> List[str]:
    """Return docker run cleanup flags."""
    return [] if keep else ["--rm"]


def generate_token(index: int) -> str:
    """Generate a deterministic token for a LAN gate node."""
    return hashlib.sha256(f"neuroshard_lan_gate_node_{index}".encode()).hexdigest()


def poll_node(host_port: int, timeout: float = 2.0) -> dict:
    """Poll a node exposed on localhost."""
    result = {"port": host_port, "status": "offline"}
    try:
        health = requests.get(f"http://localhost:{host_port}/api/v1/health", timeout=timeout)
        if health.status_code == 200:
            result["healthy"] = health.json().get("healthy", False)
            result["status"] = "healthy" if result["healthy"] else "starting"
        else:
            result["status"] = "starting"
            return result
    except Exception:
        return result

    try:
        stats = requests.get(f"http://localhost:{host_port}/api/stats", timeout=timeout)
        if stats.status_code == 200:
            data = stats.json()
            result["layers"] = data.get("my_layers", [])
            result["role"] = data.get("role", "?")
            result["loss"] = data.get("current_loss")
            result["steps"] = data.get("training_batches", data.get("training_rounds", 0))
            result["training"] = data.get("training_status", "idle")
            result["neuro"] = data.get("neuro_balance", 0.0)
            result["peers"] = data.get("peer_count", 0)
            result["mode"] = data.get("contribution_mode", "?")
            result["status"] = "training" if result["steps"] > 0 else "ready"
    except Exception:
        pass

    try:
        metrics = requests.get(f"http://localhost:{host_port}/api/v1/metrics", timeout=timeout)
        if metrics.status_code == 200:
            data = metrics.json()
            result["neuro"] = data.get("rewards", {}).get("earned_total", result.get("neuro", 0.0))
            metric_steps = data.get("training", {}).get("steps_total", 0)
            result["steps"] = max(result.get("steps", 0), metric_steps)
    except Exception:
        pass

    return result


def summarize(nodes: List[dict], expected_down: List[str] = None) -> dict:
    """Summarize node state."""
    expected_down = set(expected_down or [])
    return {
        "online": sum(
            1
            for n in nodes
            if n.get("status") not in {"offline", "pending", "crashed", "expected_down"}
        ),
        "training": sum(1 for n in nodes if n.get("status") == "training"),
        "total_steps": sum(n.get("steps", 0) for n in nodes),
        "total_neuro": sum(n.get("neuro", 0.0) for n in nodes),
        "max_peers": max((n.get("peers", 0) for n in nodes), default=0),
        "crashed": sum(
            1
            for n in nodes
            if n.get("status") == "crashed" and n.get("container") not in expected_down
        ),
        "expected_down": sum(1 for n in nodes if n.get("container") in expected_down),
    }


def evaluate(report: dict, args: argparse.Namespace) -> Tuple[bool, List[str]]:
    """Evaluate report thresholds."""
    summary = report["summary"]
    failures: List[str] = []
    thresholds = {
        "min_online": args.min_online if args.min_online is not None else args.nodes,
        "min_training_nodes": args.min_training_nodes,
        "min_total_steps": args.min_total_steps,
        "min_max_peers": args.min_max_peers,
        "max_crashed": args.max_crashed,
        "min_post_churn_steps_delta": args.min_post_churn_steps_delta,
        "min_post_restart_steps_delta": args.min_post_restart_steps_delta,
    }

    if summary["online"] < thresholds["min_online"]:
        failures.append(f"online {summary['online']} < {thresholds['min_online']}")
    if summary["training"] < thresholds["min_training_nodes"]:
        failures.append(f"training nodes {summary['training']} < {thresholds['min_training_nodes']}")
    if summary["total_steps"] < thresholds["min_total_steps"]:
        failures.append(f"total steps {summary['total_steps']} < {thresholds['min_total_steps']}")
    if summary["max_peers"] < thresholds["min_max_peers"]:
        failures.append(f"max peers {summary['max_peers']} < {thresholds['min_max_peers']}")
    if summary["crashed"] > thresholds["max_crashed"]:
        failures.append(f"crashed nodes {summary['crashed']} > {thresholds['max_crashed']}")

    churn = report.get("churn") or {}
    if churn.get("enabled") and churn.get("event"):
        before = churn.get("pre_churn_summary") or {}
        after = churn.get("post_churn_summary") or {}
        delta = after.get("total_steps", 0) - before.get("total_steps", 0)
        if delta < thresholds["min_post_churn_steps_delta"]:
            failures.append(
                f"post-churn steps delta {delta} < {thresholds['min_post_churn_steps_delta']}"
            )
        if after.get("training", 0) < thresholds["min_training_nodes"]:
            failures.append(
                f"post-churn training nodes {after.get('training', 0)} < {thresholds['min_training_nodes']}"
            )
    elif churn.get("enabled"):
        failures.append("churn was enabled but event did not fire")

    restart = report.get("restart") or {}
    if restart.get("enabled") and restart.get("event"):
        before = restart.get("pre_restart_summary") or {}
        after = restart.get("post_restart_summary") or summary
        delta = after.get("total_steps", 0) - before.get("total_steps", 0)
        if delta < thresholds["min_post_restart_steps_delta"]:
            failures.append(
                f"post-restart steps delta {delta} < {thresholds['min_post_restart_steps_delta']}"
            )
        if after.get("online", 0) < thresholds["min_online"]:
            failures.append(f"post-restart online {after.get('online', 0)} < {thresholds['min_online']}")
        if after.get("training", 0) < thresholds["min_training_nodes"]:
            failures.append(
                f"post-restart training nodes {after.get('training', 0)} < {thresholds['min_training_nodes']}"
            )
    elif restart.get("enabled"):
        failures.append("restart was enabled but event did not fire")

    report["checks"] = {
        "passed": not failures,
        "thresholds": thresholds,
        "failures": failures,
    }
    return not failures, failures


def write_report(path: str, report: dict) -> None:
    """Write JSON report."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"[LAN] Report written to {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Docker-emulated NeuroShard LAN gate")
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--duration", type=int, default=90)
    parser.add_argument("--poll-interval", type=int, default=5)
    parser.add_argument("--memory", type=int, default=512)
    parser.add_argument("--base-host-port", type=int, default=8200)
    parser.add_argument("--base-container-port", type=int, default=8100)
    parser.add_argument("--tracker-host-port", type=int, default=3900)
    parser.add_argument("--network", default="neuroshard-lan-gate")
    parser.add_argument("--prefix", default="neuroshard-lan")
    parser.add_argument("--node-image", default=DEFAULT_NODE_IMAGE)
    parser.add_argument("--tracker-image", default=DEFAULT_TRACKER_IMAGE)
    parser.add_argument("--report", default="reports/lan-gate.json")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--keep", action="store_true", help="Keep containers running after the gate")
    parser.add_argument("--min-online", type=int, default=None)
    parser.add_argument("--min-training-nodes", type=int, default=1)
    parser.add_argument("--min-total-steps", type=int, default=50)
    parser.add_argument("--min-max-peers", type=int, default=1)
    parser.add_argument("--max-crashed", type=int, default=0)
    parser.add_argument("--churn-at", type=int, default=None,
                        help="Kill one node after N seconds to test churn resilience")
    parser.add_argument("--churn-node", type=int, default=-1,
                        help="Node index to kill for churn (default: last node)")
    parser.add_argument("--restart-after", type=int, default=None,
                        help="Restart the churned node N seconds after churn")
    parser.add_argument("--min-post-churn-steps-delta", type=int, default=50,
                        help="Minimum extra steps after churn for --check")
    parser.add_argument("--min-post-restart-steps-delta", type=int, default=0,
                        help="Minimum extra steps after restart for --check")
    args = parser.parse_args()
    if args.churn_at is not None:
        churn_node = args.churn_node if args.churn_node >= 0 else args.nodes - 1
        if churn_node < 0 or churn_node >= args.nodes:
            print(f"[LAN] Invalid --churn-node {args.churn_node} for {args.nodes} nodes")
            return 2
        args.churn_node = churn_node

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tracker_name = f"{args.prefix}-tracker"
    node_names = [f"{args.prefix}-node-{i}" for i in range(args.nodes)]

    if not docker_exists("image", args.node_image):
        print(f"[LAN] Node image {args.node_image!r} not found. Build website/Dockerfile.node first.")
        return 2
    if not docker_exists("image", args.tracker_image):
        print(f"[LAN] Tracker image {args.tracker_image!r} not found. Build website/Dockerfile.tracker first.")
        return 2

    for name in [tracker_name, *node_names]:
        docker_rm_container(name)

    if not docker_exists("network", args.network):
        run(["docker", "network", "create", args.network])

    containers: List[str] = []
    snapshots: List[dict] = []
    expected_down: List[str] = []
    churn_event = None
    pre_churn_summary = None
    post_churn_summary = None
    restart_event = None
    pre_restart_summary = None
    post_restart_summary = None
    start_time = time.time()

    try:
        print(f"[LAN] Starting tracker {tracker_name} on Docker network {args.network}")
        run([
            "docker", "run", "-d",
            *docker_run_args(args.keep),
            "--name", tracker_name,
            "--network", args.network,
            "-p", f"{args.tracker_host_port}:3000",
            args.tracker_image,
        ])
        containers.append(tracker_name)
        time.sleep(2)

        def start_node(i: int, seed_indices: List[int], restarting: bool = False):
            """Start or restart a LAN gate node container."""
            name = node_names[i]
            host_port = args.base_host_port + i
            container_port = args.base_container_port
            grpc_host_port = host_port + 1000
            container_grpc_port = container_port + 1000
            seed_peers = ",".join(f"{node_names[j]}:{container_port}" for j in seed_indices)
            cmd = [
                "python", "-m", "neuroshard",
                "--port", str(container_port),
                "--token", generate_token(i),
                "--tracker", f"http://{tracker_name}:3000",
                "--device", "cpu",
                "--memory", str(args.memory),
                "--max-storage", "1",
                "--diloco-steps", "20",
                "--announce-ip", name,
                "--announce-port", str(container_port),
            ]
            if seed_peers:
                cmd.extend(["--seed-peers", seed_peers])

            if restarting:
                docker_rm_container(name)

            action = "Restarting" if restarting else "Starting"
            print(f"[LAN] {action} node {name}: host http://localhost:{host_port}")
            run([
                "docker", "run", "-d",
                *docker_run_args(args.keep),
                "--name", name,
                "--hostname", name,
                "--network", args.network,
                "-p", f"{host_port}:{container_port}",
                "-p", f"{grpc_host_port}:{container_grpc_port}",
                "-v", f"{root}/src/neuroshard:/app/neuroshard:ro",
                "-e", "PYTHONPATH=/app",
                "-e", "NEUROSHARD_HTTP_HOST=0.0.0.0",
                "-e", "NEUROSHARD_LOCAL_TEST=1",
                "-e", "NEUROSHARD_DISABLE_DEFAULT_BOOTSTRAP=1",
                "-e", "NEUROSHARD_TINY_VOCAB_SIZE=2048",
                "-e", "NEUROSHARD_TINY_LAYERS=2",
                "-e", "NEUROSHARD_TINY_HIDDEN_DIM=128",
                "-e", "NEUROSHARD_TINY_INTERMEDIATE_DIM=512",
                "-e", "NEUROSHARD_TINY_NUM_HEADS=4",
                "-e", "NEUROSHARD_TINY_NUM_KV_HEADS=1",
                "-e", "NEUROSHARD_TINY_SEQ_LEN=64",
                "-e", "NEUROSHARD_TINY_BATCH_SIZE=1",
                "-e", "NEUROSHARD_TINY_ASYNC_STEPS=2",
                "-e", "NEUROSHARD_TINY_ASYNC_INTERVAL=5",
                "-e", "NEUROSHARD_PONW_INTERVAL_SECONDS=15",
                args.node_image,
                *cmd,
            ])
            if name not in containers:
                containers.append(name)

        for i, _name in enumerate(node_names):
            start_node(i, seed_indices=list(range(i)))
            time.sleep(5)

        final_nodes: List[dict] = []
        while True:
            elapsed = time.time() - start_time
            nodes = []
            for i, name in enumerate(node_names):
                info = poll_node(args.base_host_port + i)
                info["container"] = name
                inspect = run(
                    ["docker", "inspect", "-f", "{{.State.Running}} {{.State.ExitCode}}", name],
                    check=False,
                    capture=True,
                )
                if inspect.returncode == 0:
                    running, exit_code = inspect.stdout.strip().split()
                    info["container_running"] = running == "true"
                    info["container_exit_code"] = int(exit_code)
                    if running != "true":
                        info["status"] = "expected_down" if name in expected_down else "crashed"
                elif name in expected_down:
                    info["container_running"] = False
                    info["container_exit_code"] = None
                    info["status"] = "expected_down"
                nodes.append(info)

            final_nodes = nodes
            current_summary = summarize(nodes, expected_down=expected_down)
            if (
                args.churn_at is not None
                and churn_event is None
                and elapsed >= args.churn_at
            ):
                churn_name = node_names[args.churn_node]
                pre_churn_summary = current_summary
                print(f"[LAN] Churn: stopping {churn_name} at {elapsed:.1f}s")
                run(["docker", "stop", churn_name], check=False)
                expected_down.append(churn_name)
                churn_event = {
                    "elapsed_seconds": round(elapsed, 2),
                    "node_index": args.churn_node,
                    "container": churn_name,
                    "action": "stop",
                }

            if (
                args.restart_after is not None
                and churn_event is not None
                and restart_event is None
                and elapsed >= (churn_event["elapsed_seconds"] + args.restart_after)
            ):
                restart_name = node_names[args.churn_node]
                pre_restart_summary = current_summary
                print(f"[LAN] Restart: starting {restart_name} at {elapsed:.1f}s")
                start_node(
                    args.churn_node,
                    seed_indices=[i for i in range(args.nodes) if i != args.churn_node],
                    restarting=True,
                )
                expected_down = [name for name in expected_down if name != restart_name]
                restart_event = {
                    "elapsed_seconds": round(elapsed, 2),
                    "node_index": args.churn_node,
                    "container": restart_name,
                    "action": "restart",
                }

            snapshot = {
                "elapsed_seconds": round(elapsed, 2),
                "summary": current_summary,
                "nodes": nodes,
            }
            snapshots.append(snapshot)
            summary = snapshot["summary"]
            if churn_event is not None:
                post_churn_summary = current_summary
            if restart_event is not None:
                post_restart_summary = current_summary
            print(
                f"[LAN] {elapsed:5.1f}s online={summary['online']}/{args.nodes} "
                f"training={summary['training']} steps={summary['total_steps']} "
                f"neuro={summary['total_neuro']:.4f} peers={summary['max_peers']} "
                f"expected_down={summary['expected_down']}"
            )

            if elapsed >= args.duration:
                break
            time.sleep(args.poll_interval)

        report = {
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
            "ended_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "duration_seconds": round(time.time() - start_time, 2),
            "mode": "docker-lan-tiny",
            "config": vars(args),
            "summary": summarize(final_nodes, expected_down=expected_down),
            "nodes": final_nodes,
            "snapshots": snapshots,
            "churn": {
                "enabled": args.churn_at is not None,
                "event": churn_event,
                "expected_down": expected_down,
                "pre_churn_summary": pre_churn_summary,
                "post_churn_summary": post_churn_summary,
            },
            "restart": {
                "enabled": args.restart_after is not None,
                "event": restart_event,
                "pre_restart_summary": pre_restart_summary,
                "post_restart_summary": post_restart_summary,
            },
        }

        exit_code = 0
        if args.check:
            passed, failures = evaluate(report, args)
            if passed:
                print("[LAN] Checks passed")
            else:
                print(f"[LAN] Checks failed: {', '.join(failures)}")
                exit_code = 1

        if args.report:
            write_report(args.report, report)

        return exit_code
    finally:
        if not args.keep:
            for name in reversed(containers):
                docker_rm_container(name)


if __name__ == "__main__":
    sys.exit(main())
