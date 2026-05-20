#!/usr/bin/env python3
"""
NeuroShard Network Simulation — Live Multi-Node Environment

Launches real NeuroShard nodes as separate processes on different ports,
with a local tracker for peer discovery. Nodes discover each other via
DHT, form quorums, train with DiLoCo, and serve inference.

Usage:
    # Default: 4 nodes, 30s stagger between launches
    python simulate_network.py

    # Quick test: 2 nodes, 10s stagger
    python simulate_network.py --nodes 2 --delay 10

    # Lightweight local training test: tiny model + synthetic data
    python simulate_network.py --tiny --nodes 2 --delay 10

    # Fastest smoke test: boot/network only, no training
    python simulate_network.py --smoke --nodes 2 --delay 5

    # CI-style gate: run for 90s, write report, fail if thresholds are not met
    python simulate_network.py --tiny --nodes 2 --duration 90 --report reports/tiny.json --check

    # Custom ports
    python simulate_network.py --base-port 9000 --nodes 3

    # With more memory per node (larger model)
    python simulate_network.py --memory 1024

What you can observe:
    - Terminal dashboard updates every 5 seconds showing all nodes
    - Open http://localhost:{base_port} in browser for each node's dashboard
    - Watch nodes discover each other (peer count increases)
    - Watch quorum formation and training in real time
    - Ctrl+C to gracefully shut down all nodes
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Lightweight Local Tracker
# ---------------------------------------------------------------------------
# This is a minimal peer bootstrap server. Nodes POST /announce to register
# and GET /peers to discover each other. Identical to the production tracker
# API contract used by P2PManager.
# ---------------------------------------------------------------------------

TRACKER_PEERS: Dict[str, dict] = {}
TRACKER_LOCK = threading.Lock()


def run_tracker(port: int):
    """Run a minimal tracker server in a background thread."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        import uvicorn
    except ImportError:
        print("[TRACKER] FastAPI/uvicorn not available, install with: pip install fastapi uvicorn")
        sys.exit(1)

    tracker_app = FastAPI(title="NeuroShard Local Tracker")

    @tracker_app.post("/announce")
    async def announce(data: dict):
        """Register a node in the peer list."""
        ip = data.get("ip", "127.0.0.1")
        node_port = data.get("port", 8000)
        url = f"http://{ip}:{node_port}"

        with TRACKER_LOCK:
            TRACKER_PEERS[url] = {
                "url": url,
                "ip": ip,
                "port": node_port,
                "shard_range": data.get("shard_range", "unknown"),
                "is_entry": data.get("is_entry", False),
                "is_exit": data.get("is_exit", False),
                "tps": data.get("tps", 0),
                "training_enabled": data.get("training_enabled", False),
                "last_seen": time.time(),
            }

        return {"status": "ok", "peer_count": len(TRACKER_PEERS)}

    @tracker_app.get("/peers")
    async def get_peers(limit: int = 100):
        """Return list of known peers."""
        with TRACKER_LOCK:
            peers = list(TRACKER_PEERS.values())
        # Filter stale peers (not seen in 120s)
        now = time.time()
        active = [p for p in peers if now - p.get("last_seen", 0) < 120]
        return JSONResponse(content=active[:limit])

    @tracker_app.get("/health")
    async def health():
        return {"status": "ok", "peers": len(TRACKER_PEERS)}

    # Run in a thread (non-blocking)
    config = uvicorn.Config(
        tracker_app,
        host="0.0.0.0",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server.run()


# ---------------------------------------------------------------------------
# Node Launcher
# ---------------------------------------------------------------------------

def generate_token(index: int) -> str:
    """Generate a deterministic 64-char hex token for a node."""
    import hashlib
    seed = f"neuroshard_simulation_node_{index}_v1"
    return hashlib.sha256(seed.encode()).hexdigest()


def launch_node(
    index: int,
    port: int,
    tracker_url: str,
    memory_mb: int,
    diloco_steps: int,
    seed_peers: List[str],
    log_dir: str,
    tiny: bool = False,
    no_training: bool = False,
) -> subprocess.Popen:
    """Launch a NeuroShard node as a subprocess."""
    token = generate_token(index)

    cmd = [
        sys.executable, "-m", "neuroshard",
        "--port", str(port),
        "--token", token,
        "--tracker", tracker_url,
        "--device", "cpu",
        "--memory", str(memory_mb),
        "--max-storage", "50",
        "--diloco-steps", str(diloco_steps),
    ]

    if seed_peers:
        cmd.extend(["--seed-peers", ",".join(seed_peers)])
    if no_training:
        cmd.append("--no-training")

    # Log to file
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"node_{index}_port_{port}.log")
    log_fh = open(log_file, "w")

    env = os.environ.copy()
    # Suppress verbose torch warnings
    env["PYTHONWARNINGS"] = "ignore"
    env["TOKENIZERS_PARALLELISM"] = "false"
    if tiny:
        env.update({
            "NEUROSHARD_LOCAL_TEST": "1",
            "NEUROSHARD_TINY_VOCAB_SIZE": "2048",
            "NEUROSHARD_TINY_LAYERS": "2",
            "NEUROSHARD_TINY_HIDDEN_DIM": "128",
            "NEUROSHARD_TINY_INTERMEDIATE_DIM": "512",
            "NEUROSHARD_TINY_NUM_HEADS": "4",
            "NEUROSHARD_TINY_NUM_KV_HEADS": "1",
            "NEUROSHARD_TINY_SEQ_LEN": "64",
            "NEUROSHARD_TINY_BATCH_SIZE": "1",
            "NEUROSHARD_TINY_ASYNC_STEPS": "2",
            "NEUROSHARD_TINY_ASYNC_INTERVAL": "5",
            "NEUROSHARD_PONW_INTERVAL_SECONDS": "15",
            "NEUROSHARD_DISABLE_DEFAULT_BOOTSTRAP": "1",
        })

    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,  # New process group for clean shutdown
    )

    return proc


# ---------------------------------------------------------------------------
# Node Monitor
# ---------------------------------------------------------------------------

def poll_node(port: int, timeout: float = 3.0) -> Optional[dict]:
    """Poll a node's health and stats endpoints."""
    import requests

    result = {"port": port, "status": "offline"}

    try:
        # Health check
        health = requests.get(f"http://localhost:{port}/api/v1/health", timeout=timeout)
        if health.status_code == 200:
            result["healthy"] = health.json().get("healthy", False)
            result["status"] = "healthy" if result["healthy"] else "starting"
        else:
            result["status"] = "starting"
    except Exception:
        return result

    try:
        # Stats (more detailed)
        stats = requests.get(f"http://localhost:{port}/api/stats", timeout=timeout)
        if stats.status_code == 200:
            data = stats.json()
            result["peers"] = data.get("peer_count", 0)
            result["layers"] = data.get("my_layers", [])
            result["role"] = data.get("role", "?")
            result["loss"] = data.get("current_loss")
            result["batches"] = data.get("training_batches", data.get("training_rounds", 0))
            result["steps"] = result["batches"]
            result["training"] = data.get("training_status", "idle")
            result["neuro"] = data.get("neuro_balance", 0.0)
            result["mode"] = data.get("contribution_mode", "?")
            result["params_m"] = data.get("my_params_m", 0)
            result["net_layers"] = data.get("network_layers", 0)
            result["status"] = "training" if result["batches"] > 0 else "ready"
    except Exception:
        pass

    try:
        # Metrics (rewards)
        metrics = requests.get(f"http://localhost:{port}/api/v1/metrics", timeout=timeout)
        if metrics.status_code == 200:
            data = metrics.json()
            result["neuro"] = data.get("rewards", {}).get("earned_total", 0.0)
            metric_steps = data.get("training", {}).get("steps_total", 0)
            result["steps"] = max(result.get("steps", 0), metric_steps)
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Terminal Dashboard
# ---------------------------------------------------------------------------

def summarize_nodes(nodes_info: List[dict], launched: int) -> dict:
    """Summarize a node snapshot for report/check evaluation."""
    online = sum(1 for n in nodes_info if n.get("status") not in ("offline", "pending", "crashed"))
    training = sum(1 for n in nodes_info if n.get("status") == "training")
    total_steps = sum(n.get("steps", n.get("batches", 0)) for n in nodes_info)
    total_neuro = sum(n.get("neuro", 0.0) for n in nodes_info)
    max_peers = max((n.get("peers", 0) for n in nodes_info), default=0)
    crashed = sum(1 for n in nodes_info if n.get("status") == "crashed")
    return {
        "online": online,
        "launched": launched,
        "training": training,
        "total_steps": total_steps,
        "total_neuro": total_neuro,
        "max_peers": max_peers,
        "crashed": crashed,
    }


def build_report(
    args: argparse.Namespace,
    start_time: float,
    snapshots: List[dict],
    final_nodes: List[dict],
    launched_ports: List[int],
    processes: List[subprocess.Popen],
    checks: Optional[dict] = None,
) -> dict:
    """Build a JSON-serializable simulation report."""
    end_time = time.time()
    process_info = []
    for idx, proc in enumerate(processes):
        process_info.append({
            "index": idx,
            "port": launched_ports[idx] if idx < len(launched_ports) else None,
            "pid": proc.pid,
            "exit_code": proc.poll(),
        })

    return {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
        "ended_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end_time)),
        "duration_seconds": round(end_time - start_time, 2),
        "mode": {
            "tiny": args.tiny,
            "smoke": args.smoke,
            "training_enabled": not args.no_training,
        },
        "config": {
            "nodes": args.nodes,
            "base_port": args.base_port,
            "tracker_port": args.tracker_port,
            "memory_mb": args.memory,
            "diloco_steps": args.diloco_steps,
            "delay_seconds": args.delay,
            "poll_interval_seconds": args.poll_interval,
            "duration_seconds": args.duration,
        },
        "summary": summarize_nodes(final_nodes, len(launched_ports)),
        "nodes": final_nodes,
        "processes": process_info,
        "snapshots": snapshots,
        "checks": checks or {},
    }


def evaluate_report(report: dict, args: argparse.Namespace) -> Tuple[bool, List[str]]:
    """Evaluate a report against promotion-gate thresholds."""
    summary = report.get("summary", {})
    failures: List[str] = []

    min_online = args.min_online
    if min_online is None:
        min_online = args.nodes

    min_training_nodes = args.min_training_nodes
    if min_training_nodes is None:
        min_training_nodes = 0 if args.no_training else 1

    min_total_steps = args.min_total_steps
    if min_total_steps is None:
        min_total_steps = 0 if args.no_training else 1

    min_max_peers = args.min_max_peers
    if min_max_peers is None:
        min_max_peers = 1 if args.nodes > 1 else 0

    min_total_neuro = args.min_total_neuro
    if min_total_neuro is None:
        min_total_neuro = 0.0

    thresholds = {
        "min_online": min_online,
        "min_training_nodes": min_training_nodes,
        "min_total_steps": min_total_steps,
        "min_max_peers": min_max_peers,
        "min_total_neuro": min_total_neuro,
        "max_crashed": args.max_crashed,
    }

    if summary.get("online", 0) < min_online:
        failures.append(f"online {summary.get('online', 0)} < {min_online}")
    if summary.get("training", 0) < min_training_nodes:
        failures.append(f"training nodes {summary.get('training', 0)} < {min_training_nodes}")
    if summary.get("total_steps", 0) < min_total_steps:
        failures.append(f"total steps {summary.get('total_steps', 0)} < {min_total_steps}")
    if summary.get("max_peers", 0) < min_max_peers:
        failures.append(f"max peers {summary.get('max_peers', 0)} < {min_max_peers}")
    if summary.get("total_neuro", 0.0) < min_total_neuro:
        failures.append(f"total NEURO {summary.get('total_neuro', 0.0):.6f} < {min_total_neuro}")
    if summary.get("crashed", 0) > args.max_crashed:
        failures.append(f"crashed nodes {summary.get('crashed', 0)} > {args.max_crashed}")

    report["checks"] = {
        "passed": not failures,
        "thresholds": thresholds,
        "failures": failures,
    }
    return not failures, failures


def write_report(path: str, report: dict) -> None:
    """Write a simulation report to disk."""
    report_dir = os.path.dirname(path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"\033[1;32m[SIM] Report written to {path}\033[0m")

def render_dashboard(
    nodes_info: List[dict],
    elapsed: float,
    total_nodes: int,
    launched: int,
    log_dir: str,
):
    """Render a live terminal dashboard."""
    # Clear screen
    print("\033[2J\033[H", end="")

    width = 78

    print(f"\033[1;36m{'=' * width}\033[0m")
    print(f"\033[1;37m  NEUROSHARD NETWORK SIMULATION\033[0m")
    print(f"\033[0;37m  {launched}/{total_nodes} nodes launched | "
          f"elapsed {int(elapsed)}s | Ctrl+C to stop\033[0m")
    print(f"\033[1;36m{'=' * width}\033[0m")

    # Header
    print(f"\033[1;33m {'#':>2}  {'Port':<6} {'Status':<10} {'Layers':<10} "
          f"{'Role':<12} {'Loss':<8} {'Steps':<7} {'NEURO':<7} {'Peers':<5}\033[0m")
    print(f" {'--':>2}  {'------':<6} {'----------':<10} {'----------':<10} "
          f"{'------------':<12} {'--------':<8} {'-------':<7} {'-------':<7} {'-----':<5}")

    for i, info in enumerate(nodes_info):
        port = info.get("port", "?")
        status = info.get("status", "offline")

        # Color status
        if status == "training":
            status_str = f"\033[1;32m{'TRAIN':<10}\033[0m"
        elif status == "ready":
            status_str = f"\033[1;34m{'READY':<10}\033[0m"
        elif status == "healthy":
            status_str = f"\033[1;34m{'ONLINE':<10}\033[0m"
        elif status == "starting":
            status_str = f"\033[1;33m{'STARTING':<10}\033[0m"
        elif status == "pending":
            status_str = f"\033[0;90m{'PENDING':<10}\033[0m"
        else:
            status_str = f"\033[0;31m{'OFFLINE':<10}\033[0m"

        layers = info.get("layers", [])
        if layers:
            layers_str = f"{min(layers)}-{max(layers)}"
        else:
            layers_str = "--"

        role = info.get("role", "--")
        if len(role) > 11:
            role = role[:11]

        loss = info.get("loss")
        if loss is not None and isinstance(loss, (int, float)):
            loss_str = f"{loss:.4f}"
        else:
            loss_str = "--"

        steps = info.get("steps", info.get("batches", 0))
        neuro = info.get("neuro", 0)
        peers = info.get("peers", 0)

        print(f" {i+1:>2}  {port:<6} {status_str} {layers_str:<10} "
              f"{role:<12} {loss_str:<8} {steps:<7} {neuro:<7.2f} {peers:<5}")

    print(f"\033[1;36m{'=' * width}\033[0m")

    # Summary
    summary = summarize_nodes(nodes_info, launched)
    online = summary["online"]
    training = summary["training"]
    total_steps = summary["total_steps"]
    total_neuro = summary["total_neuro"]
    max_peers = summary["max_peers"]

    print(f"  Online: {online}/{launched} | Training: {training} | "
          f"Steps: {total_steps} | NEURO: {total_neuro:.2f} | "
          f"Max peers: {max_peers}")
    print(f"  Dashboards: ", end="")
    active_ports = [n["port"] for n in nodes_info
                    if n.get("status") not in ("offline", "pending")]
    if active_ports:
        print(f"http://localhost:{active_ports[0]}", end="")
        if len(active_ports) > 1:
            print(f" .. {active_ports[-1]}", end="")
    print()
    print(f"  Logs: {log_dir}/node_*.log")
    print(f"\033[1;36m{'=' * width}\033[0m")


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="NeuroShard Live Network Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--nodes", type=int, default=4,
                        help="Number of nodes to launch (default: 4)")
    parser.add_argument("--delay", type=int, default=30,
                        help="Seconds between launching each node (default: 30)")
    parser.add_argument("--base-port", type=int, default=8100,
                        help="Base port for nodes (default: 8100)")
    parser.add_argument("--tracker-port", type=int, default=7999,
                        help="Port for local tracker (default: 7999)")
    parser.add_argument("--memory", type=int, default=512,
                        help="Memory per node in MB (default: 512)")
    parser.add_argument("--diloco-steps", type=int, default=20,
                        help="DiLoCo inner steps (default: 20, production: 500)")
    parser.add_argument("--tiny", action="store_true",
                        help="Use tiny local-test model, synthetic data, and fast PoNW cadence")
    parser.add_argument("--smoke", action="store_true",
                        help="Network smoke test: tiny mode with training disabled")
    parser.add_argument("--no-training", action="store_true",
                        help="Launch nodes with training disabled")
    parser.add_argument("--poll-interval", type=int, default=5,
                        help="Dashboard refresh interval in seconds (default: 5)")
    parser.add_argument("--log-dir", type=str, default="simulation_logs",
                        help="Directory for node logs (default: simulation_logs)")
    parser.add_argument("--duration", type=int, default=None,
                        help="Stop automatically after N seconds and write report if configured")
    parser.add_argument("--report", type=str, default=None,
                        help="Write JSON report to this path on exit")
    parser.add_argument("--check", action="store_true",
                        help="Exit non-zero if report thresholds are not met")
    parser.add_argument("--min-online", type=int, default=None,
                        help="Minimum online nodes for --check (default: all nodes)")
    parser.add_argument("--min-training-nodes", type=int, default=None,
                        help="Minimum training nodes for --check (default: 1 when training enabled)")
    parser.add_argument("--min-total-steps", type=int, default=None,
                        help="Minimum total training steps for --check (default: 1 when training enabled)")
    parser.add_argument("--min-max-peers", type=int, default=None,
                        help="Minimum max peer count for --check (default: 1 when nodes > 1)")
    parser.add_argument("--min-total-neuro", type=float, default=None,
                        help="Minimum total earned NEURO for --check (default: 0)")
    parser.add_argument("--max-crashed", type=int, default=0,
                        help="Maximum crashed nodes for --check (default: 0)")
    args = parser.parse_args()
    if args.smoke:
        args.tiny = True
        args.no_training = True

    processes: List[subprocess.Popen] = []
    log_handles: List = []
    tracker_thread = None
    shutdown_event = threading.Event()
    snapshots: List[dict] = []
    latest_nodes_info: List[dict] = []
    launched_ports: List[int] = []
    start_time = time.time()

    def stop_processes():
        """Gracefully stop all launched node processes."""
        print(f"\n\033[1;33m[SIM] Shutting down {len(processes)} node(s)...\033[0m")

        # Send SIGTERM to all process groups
        for proc in processes:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass

        # Wait up to 5 seconds
        deadline = time.time() + 5
        for proc in processes:
            remaining = max(0.1, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass

        print(f"\033[1;32m[SIM] All nodes stopped.\033[0m")

    def finish(exit_code: int = 0):
        """Write report if requested, stop nodes, and exit."""
        nonlocal latest_nodes_info
        if args.report or args.check:
            report = build_report(
                args=args,
                start_time=start_time,
                snapshots=snapshots,
                final_nodes=latest_nodes_info,
                launched_ports=launched_ports,
                processes=processes,
            )
            if args.check:
                passed, failures = evaluate_report(report, args)
                exit_code = 0 if passed else 1
                if passed:
                    print("\033[1;32m[SIM] Checks passed\033[0m")
                else:
                    print(f"\033[1;31m[SIM] Checks failed: {', '.join(failures)}\033[0m")
            if args.report:
                write_report(args.report, report)

        stop_processes()
        sys.exit(exit_code)

    def cleanup(signum=None, frame=None):
        """Gracefully shut down all nodes."""
        if shutdown_event.is_set():
            return
        shutdown_event.set()
        finish(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # ------------------------------------------------------------------
    # Step 1: Start local tracker
    # ------------------------------------------------------------------
    print(f"\033[1;36m[SIM] Starting local tracker on port {args.tracker_port}...\033[0m")
    tracker_thread = threading.Thread(
        target=run_tracker,
        args=(args.tracker_port,),
        daemon=True,
        name="tracker",
    )
    tracker_thread.start()
    time.sleep(1)  # Let tracker bind

    tracker_url = f"http://localhost:{args.tracker_port}"
    print(f"\033[1;32m[SIM] Tracker ready at {tracker_url}\033[0m")

    # ------------------------------------------------------------------
    # Step 2: Launch nodes with staggered timing
    # ------------------------------------------------------------------
    for i in range(args.nodes):
        if shutdown_event.is_set():
            break

        port = args.base_port + i

        # Seed peers: all previously launched nodes
        seed_peers = [f"localhost:{p}" for p in launched_ports]

        print(f"\033[1;36m[SIM] Launching Node {i+1} on port {port}...\033[0m")
        proc = launch_node(
            index=i,
            port=port,
            tracker_url=tracker_url,
            memory_mb=args.memory,
            diloco_steps=args.diloco_steps,
            seed_peers=seed_peers,
            log_dir=args.log_dir,
            tiny=args.tiny,
            no_training=args.no_training,
        )
        processes.append(proc)
        launched_ports.append(port)

        print(f"\033[1;32m[SIM] Node {i+1} launched (PID {proc.pid})\033[0m")

        # Wait between launches (except after last node)
        if i < args.nodes - 1:
            print(f"\033[0;37m[SIM] Waiting {args.delay}s before launching next node...\033[0m")
            for _ in range(args.delay):
                if shutdown_event.is_set():
                    break
                time.sleep(1)

                # Show brief dashboard during wait
                elapsed = time.time() - start_time
                if int(elapsed) % args.poll_interval == 0:
                    nodes_info = []
                    for j, p in enumerate(launched_ports):
                        info = poll_node(p)
                        if info:
                            nodes_info.append(info)
                    # Add pending nodes
                    for k in range(len(launched_ports), args.nodes):
                        nodes_info.append({
                            "port": args.base_port + k,
                            "status": "pending",
                        })
                    latest_nodes_info = nodes_info
                    snapshots.append({
                        "elapsed_seconds": round(elapsed, 2),
                        "summary": summarize_nodes(nodes_info, len(launched_ports)),
                        "nodes": nodes_info,
                    })
                    render_dashboard(nodes_info, elapsed, args.nodes, len(launched_ports), args.log_dir)

    # ------------------------------------------------------------------
    # Step 3: Monitor loop — live dashboard
    # ------------------------------------------------------------------
    print(f"\033[1;32m[SIM] All {args.nodes} nodes launched. Monitoring...\033[0m")

    while not shutdown_event.is_set():
        elapsed = time.time() - start_time

        # Poll all nodes
        nodes_info = []
        for port in launched_ports:
            info = poll_node(port)
            if info:
                nodes_info.append(info)

        # Check for crashed processes
        for idx, proc in enumerate(processes):
            if proc.poll() is not None:
                port = launched_ports[idx]
                for info in nodes_info:
                    if info.get("port") == port:
                        info["status"] = "crashed"
                        break

        latest_nodes_info = nodes_info
        snapshots.append({
            "elapsed_seconds": round(elapsed, 2),
            "summary": summarize_nodes(nodes_info, len(launched_ports)),
            "nodes": nodes_info,
        })
        render_dashboard(nodes_info, elapsed, args.nodes, len(launched_ports), args.log_dir)

        if args.duration is not None and elapsed >= args.duration:
            print(f"\033[1;36m[SIM] Duration reached ({args.duration}s).\033[0m")
            shutdown_event.set()
            finish(0)

        # Sleep with interruptibility
        for _ in range(args.poll_interval):
            if shutdown_event.is_set():
                break
            time.sleep(1)


if __name__ == "__main__":
    main()
