#!/usr/bin/env python3
"""
NeuroShard CLI - Friendly entry point for running a node.

Quick Start:
    neuroshard setup                  # One-time interactive setup
    neuroshard start                  # Start node in background
    neuroshard stop                   # Stop the node
    neuroshard status                 # Check if running
    neuroshard logs -f                # Follow live logs

Legacy (still works):
    neuroshard --token YOUR_TOKEN --device cuda
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
import webbrowser

from neuroshard.version import __version__

# ─── Paths ────────────────────────────────────────────────────────────────────
NEUROSHARD_DIR = os.path.expanduser("~/.neuroshard")
PID_FILE = os.path.join(NEUROSHARD_DIR, "node.pid")
LOG_FILE = os.path.join(NEUROSHARD_DIR, "node.log")
CONFIG_FILE = os.path.join(NEUROSHARD_DIR, "config.json")
SUPERVISOR_STATE_FILE = os.path.join(NEUROSHARD_DIR, "supervisor.json")

# ─── Supervisor settings ──────────────────────────────────────────────────────
RESTART_DELAY_BASE = 10     # seconds
RESTART_DELAY_MAX = 300     # 5 minutes cap
RESTART_BACKOFF_RESET = 600 # reset backoff after 10 min of stable running

# ─── ANSI colors ──────────────────────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# Disable colors when not a TTY (e.g. piped output, log files)
if not sys.stdout.isatty():
    GREEN = YELLOW = RED = CYAN = BOLD = DIM = RESET = ""


# ═══════════════════════════════════════════════════════════════════════════════
#  Config management
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    "token": None,
    "device": "auto",
    "memory": None,           # MB, null = auto-detect
    "max_storage": 100,       # MB
    "port": 8000,
    "cpu_threads": None,      # null = all cores
    "tracker": "https://neuroshard.com/api/tracker",
    "no_training": False,
    "diloco_steps": 500,
    "announce_ip": None,
    "announce_port": None,
    "seed_peers": None,
    "auto_open_browser": True,
}


def ensure_dir():
    """Ensure ~/.neuroshard directory exists."""
    os.makedirs(NEUROSHARD_DIR, exist_ok=True)


def load_config() -> dict:
    """Load config from ~/.neuroshard/config.json, merged with defaults."""
    config = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                saved = json.load(f)
            config.update(saved)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  {YELLOW}Warning: Could not read config: {e}{RESET}")
    return config


def save_config(config: dict):
    """Save config to ~/.neuroshard/config.json."""
    ensure_dir()
    # Only save non-default values to keep the file clean
    to_save = {}
    for k, v in config.items():
        if k in DEFAULT_CONFIG and v != DEFAULT_CONFIG[k]:
            to_save[k] = v
        elif k not in DEFAULT_CONFIG:
            to_save[k] = v
    # Always save token even if None (so the key exists for clarity)
    if "token" in config:
        to_save["token"] = config["token"]

    with open(CONFIG_FILE, "w") as f:
        json.dump(to_save, f, indent=2)
    # Protect config file (contains token)
    try:
        os.chmod(CONFIG_FILE, 0o600)
    except OSError:
        pass


def mask_token(token: str) -> str:
    """Show first 8 and last 4 chars of a token."""
    if not token or len(token) < 16:
        return token or "(not set)"
    return f"{token[:8]}...{token[-4:]}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Process management
# ═══════════════════════════════════════════════════════════════════════════════

def get_running_pid() -> int | None:
    """Get PID of running daemon, or None if not running."""
    if not os.path.exists(PID_FILE):
        return None
    try:
        with open(PID_FILE, "r") as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)  # Check if alive
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        _cleanup_pid_file()
        return None


def _cleanup_pid_file():
    try:
        os.remove(PID_FILE)
    except OSError:
        pass


def _get_uptime(pid: int) -> str:
    """Get human-readable uptime for a process."""
    try:
        import psutil
        proc = psutil.Process(pid)
        elapsed = time.time() - proc.create_time()
        if elapsed < 60:
            return f"{int(elapsed)}s"
        elif elapsed < 3600:
            return f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
        elif elapsed < 86400:
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            return f"{h}h {m}m"
        else:
            d = int(elapsed // 86400)
            h = int((elapsed % 86400) // 3600)
            return f"{d}d {h}h"
    except Exception:
        return "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
#  Supervisor (auto-restart)
# ═══════════════════════════════════════════════════════════════════════════════

def _save_supervisor_state(restarts: int, last_restart: float | None = None):
    """Persist restart count and timestamp for status display."""
    state = {"restarts": restarts, "last_restart": last_restart}
    try:
        with open(SUPERVISOR_STATE_FILE, "w") as f:
            json.dump(state, f)
    except OSError:
        pass


def _load_supervisor_state() -> dict:
    try:
        with open(SUPERVISOR_STATE_FILE, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"restarts": 0, "last_restart": None}


def _run_supervisor(cmd: list[str]):
    """Supervisor loop: run the node subprocess and restart on crash.

    This function runs in the daemon process (detached from the terminal).
    It never returns under normal operation — only exits when SIGTERM/SIGINT
    is received or the child exits with code 0 (clean shutdown).
    """
    child = None
    restarts = 0
    backoff = RESTART_DELAY_BASE
    stop_requested = False

    def _forward_signal(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        if child and child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, _forward_signal)
    signal.signal(signal.SIGINT, _forward_signal)

    log_fd = open(LOG_FILE, "a")

    while not stop_requested:
        log_fd.write(f"[SUPERVISOR] Starting node process (restart #{restarts})\n")
        log_fd.flush()

        start_time = time.time()
        child = subprocess.Popen(
            cmd,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )

        _save_supervisor_state(restarts, time.time() if restarts > 0 else None)

        child.wait()
        exit_code = child.returncode
        uptime = time.time() - start_time

        if stop_requested or exit_code == 0:
            log_fd.write(
                f"[SUPERVISOR] Node exited cleanly (code={exit_code}). "
                f"Not restarting.\n"
            )
            log_fd.flush()
            break

        restarts += 1
        _save_supervisor_state(restarts, time.time())

        if uptime > RESTART_BACKOFF_RESET:
            backoff = RESTART_DELAY_BASE

        log_fd.write(
            f"[SUPERVISOR] Node crashed (code={exit_code}) after "
            f"{int(uptime)}s. Restarting in {backoff}s... "
            f"(restart #{restarts})\n"
        )
        log_fd.flush()

        # Interruptible sleep so SIGTERM during backoff works
        for _ in range(backoff):
            if stop_requested:
                break
            time.sleep(1)

        backoff = min(backoff * 2, RESTART_DELAY_MAX)

    # Kill child if still alive (e.g. stop_requested during child.wait)
    if child and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()

    log_fd.write(f"[SUPERVISOR] Exiting. Total restarts: {restarts}\n")
    log_fd.flush()
    log_fd.close()
    _cleanup_pid_file()
    sys.exit(0)


# ═══════════════════════════════════════════════════════════════════════════════
#  GPU detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_gpu() -> tuple[str, str]:
    """Detect GPU and return (description, color)."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory
            vram_gb = vram / (1024 ** 3)
            return f"CUDA - {name} ({vram_gb:.1f} GB)", GREEN
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple Metal (MPS)", GREEN
        else:
            return "CPU only (no GPU detected)", YELLOW
    except ImportError:
        return "PyTorch not installed", RED


def detect_memory() -> int | None:
    """Auto-detect available memory in MB (70% of system RAM)."""
    try:
        import psutil
        total = psutil.virtual_memory().total
        return int((total * 0.7) / (1024 * 1024))
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Banner
# ═══════════════════════════════════════════════════════════════════════════════

def print_banner(compact: bool = False):
    """Print the NeuroShard banner."""
    if compact:
        print(f"\n  {BOLD}{CYAN}NeuroShard{RESET} v{__version__}")
        print()
        return

    print(f"""
{CYAN}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗               ║
║   ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗              ║
║   ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║              ║
║   ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║              ║
║   ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝              ║
║   ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝              ║
║                                                              ║
║   ███████╗██╗  ██╗ █████╗ ██████╗ ██████╗                   ║
║   ██╔════╝██║  ██║██╔══██╗██╔══██╗██╔══██╗                  ║
║   ███████╗███████║███████║██████╔╝██║  ██║                  ║
║   ╚════██║██╔══██║██╔══██║██╔══██╗██║  ██║                  ║
║   ███████║██║  ██║██║  ██║██║  ██║██████╔╝                  ║
║   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝                  ║
║                                                              ║
║            Decentralized AI Training Network                 ║
║                     v{__version__:<10s}                            ║
╚══════════════════════════════════════════════════════════════╝{RESET}
    """)


# ═══════════════════════════════════════════════════════════════════════════════
#  Mnemonic handling
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_token(raw_token: str) -> str:
    """Resolve a token - could be hex or 12-word mnemonic."""
    if not raw_token:
        return raw_token
    words = raw_token.strip().split()
    if len(words) == 12:
        try:
            from mnemonic import Mnemonic
            mnemo = Mnemonic("english")
            if mnemo.check(raw_token):
                seed = mnemo.to_seed(raw_token, passphrase="")
                print(f"  {GREEN}Wallet recovered from mnemonic{RESET}")
                return seed[:32].hex()
            else:
                print(f"  {YELLOW}Warning: Invalid mnemonic - treating as raw token{RESET}")
        except ImportError:
            print(f"  {YELLOW}Warning: 'mnemonic' package not installed{RESET}")
        except Exception as e:
            print(f"  {YELLOW}Warning: Mnemonic error: {e}{RESET}")
    return raw_token.strip()


# ═══════════════════════════════════════════════════════════════════════════════
#  Subcommand: setup
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_setup(args, _called_from_start=False):
    """Interactive first-time setup."""
    if not _called_from_start:
        print_banner(compact=True)
    print(f"  {BOLD}Interactive Setup{RESET}")
    print(f"  {DIM}Configure your node. Press Enter to keep defaults.{RESET}")
    print()

    config = load_config()

    # ── Token ─────────────────────────────────────────────────────────────
    current = mask_token(config.get("token"))
    print(f"  {BOLD}1. Wallet Token{RESET}")
    print(f"     Get yours at: {CYAN}https://neuroshard.com/wallet{RESET}")
    if config.get("token"):
        print(f"     Current: {DIM}{current}{RESET}")
        token_input = input(f"     Token (Enter to keep current): ").strip()
    else:
        token_input = input(f"     Token: ").strip()

    if token_input:
        config["token"] = resolve_token(token_input)
    print()

    # ── Device ────────────────────────────────────────────────────────────
    gpu_desc, gpu_color = detect_gpu()
    print(f"  {BOLD}2. Compute Device{RESET}")
    print(f"     Detected: {gpu_color}{gpu_desc}{RESET}")
    print(f"     Options: auto, cuda, mps, cpu")
    current_dev = config.get("device", "auto")
    device_input = input(f"     Device [{current_dev}]: ").strip().lower()
    if device_input in ("auto", "cuda", "mps", "cpu"):
        config["device"] = device_input
    elif device_input:
        print(f"     {YELLOW}Invalid choice, keeping '{current_dev}'{RESET}")
    print()

    # ── Memory ────────────────────────────────────────────────────────────
    auto_mem = detect_memory()
    print(f"  {BOLD}3. Memory Limit (MB){RESET}")
    if auto_mem:
        print(f"     Auto-detected: {auto_mem} MB (70% of system RAM)")
    current_mem = config.get("memory")
    mem_hint = str(current_mem) if current_mem else "auto"
    mem_input = input(f"     Memory [{mem_hint}]: ").strip()
    if mem_input:
        if mem_input.lower() == "auto":
            config["memory"] = None
        else:
            try:
                config["memory"] = int(mem_input)
            except ValueError:
                print(f"     {YELLOW}Invalid number, keeping current{RESET}")
    print()

    # ── Storage ───────────────────────────────────────────────────────────
    print(f"  {BOLD}4. Max Storage for Training Data (MB){RESET}")
    current_storage = config.get("max_storage", 100)
    storage_input = input(f"     Storage [{current_storage}]: ").strip()
    if storage_input:
        try:
            config["max_storage"] = int(storage_input)
        except ValueError:
            print(f"     {YELLOW}Invalid number, keeping current{RESET}")
    print()

    # ── Port ──────────────────────────────────────────────────────────────
    print(f"  {BOLD}5. HTTP Port{RESET}")
    current_port = config.get("port", 8000)
    port_input = input(f"     Port [{current_port}]: ").strip()
    if port_input:
        try:
            config["port"] = int(port_input)
        except ValueError:
            print(f"     {YELLOW}Invalid number, keeping current{RESET}")
    print()

    # ── Save ──────────────────────────────────────────────────────────────
    save_config(config)

    print(f"  {GREEN}{BOLD}Configuration saved!{RESET}")
    print(f"  {DIM}Config file: {CONFIG_FILE}{RESET}")
    print()
    if not _called_from_start:
        print(f"  {BOLD}Next steps:{RESET}")
        print(f"    {CYAN}neuroshard start{RESET}       Start node in background")
        print(f"    {CYAN}neuroshard run{RESET}         Start in foreground (debug)")
        print(f"    {CYAN}neuroshard status{RESET}      Check if running")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Subcommand: start  (background daemon)
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_start(args):
    """Start the node as a background daemon."""
    config = load_config()
    _apply_cli_overrides(config, args)

    token = config.get("token")
    if not token:
        # Interactive terminal → run setup automatically
        if sys.stdin.isatty():
            print(f"\n  {YELLOW}No configuration found. Let's set up your node first!{RESET}")
            print()
            cmd_setup(args, _called_from_start=True)
            # Reload config after setup
            config = load_config()
            _apply_cli_overrides(config, args)
            token = config.get("token")

        if not token:
            print(f"\n  {RED}No wallet token configured!{RESET}")
            print()
            print(f"  Run {CYAN}neuroshard setup{RESET} to configure, or pass {CYAN}--token YOUR_TOKEN{RESET}")
            print(f"  Get your token at: {CYAN}https://neuroshard.com/wallet{RESET}")
            print()
            sys.exit(1)

    # Check if already running
    existing_pid = get_running_pid()
    if existing_pid:
        uptime = _get_uptime(existing_pid)
        print(f"\n  {YELLOW}Node is already running{RESET} (PID {existing_pid}, uptime {uptime})")
        print(f"  Use {CYAN}neuroshard restart{RESET} to restart, or {CYAN}neuroshard stop{RESET} first.")
        print()
        sys.exit(1)

    # Check platform
    if sys.platform == "win32":
        print(f"\n  {RED}Background mode not supported on Windows.{RESET}")
        print(f"  Use {CYAN}neuroshard run{RESET} to start in foreground.")
        sys.exit(1)

    ensure_dir()

    port = config.get("port", 8000)

    print_banner(compact=True)
    gpu_desc, gpu_color = detect_gpu()
    print(f"  {gpu_color}Device: {gpu_desc}{RESET}")
    print(f"  Token:  {DIM}{mask_token(token)}{RESET}")
    mem = config.get("memory")
    print(f"  Memory: {mem} MB" if mem else f"  Memory: auto-detect")
    print()

    # Build the command to run the node in background
    cmd = [sys.executable, "-m", "neuroshard", "run", "--from-daemon"]
    # Pass all config as CLI args to the subprocess
    cmd.extend(["--token", token])
    cmd.extend(["--port", str(port)])
    cmd.extend(["--device", config.get("device", "auto")])
    cmd.extend(["--max-storage", str(config.get("max_storage", 100))])
    cmd.extend(["--tracker", config.get("tracker", "https://neuroshard.com/api/tracker")])
    cmd.extend(["--diloco-steps", str(config.get("diloco_steps", 500))])
    if config.get("memory"):
        cmd.extend(["--memory", str(config["memory"])])
    if config.get("cpu_threads"):
        cmd.extend(["--cpu-threads", str(config["cpu_threads"])])
    if config.get("announce_ip"):
        cmd.extend(["--announce-ip", config["announce_ip"]])
    if config.get("announce_port"):
        cmd.extend(["--announce-port", str(config["announce_port"])])
    if config.get("no_training"):
        cmd.append("--no-training")
    if config.get("seed_peers"):
        cmd.extend(["--seed-peers", config["seed_peers"]])

    # Open log file for appending
    log_fd = open(LOG_FILE, "a")

    # Write startup marker to log
    log_fd.write(f"\n{'='*60}\n")
    log_fd.write(f"[DAEMON] Starting NeuroShard at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_fd.write(f"[DAEMON] Port: {port}\n")
    log_fd.write(f"[DAEMON] Auto-restart: enabled\n")
    log_fd.write(f"{'='*60}\n\n")
    log_fd.flush()
    log_fd.close()

    # Reset supervisor state for fresh start
    _save_supervisor_state(0)

    # Launch supervisor process (which manages the node subprocess)
    supervisor_cmd = [
        sys.executable, "-m", "neuroshard", "_supervisor", "--",
    ] + cmd
    proc = subprocess.Popen(
        supervisor_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Write PID file
    with open(PID_FILE, "w") as f:
        f.write(str(proc.pid))

    # Give it a moment to check if it crashes immediately
    time.sleep(1.5)
    if proc.poll() is not None:
        print(f"  {RED}Node failed to start!{RESET} (exit code {proc.returncode})")
        print(f"  Check logs: {CYAN}neuroshard logs{RESET}")
        _cleanup_pid_file()
        sys.exit(1)

    print(f"  {GREEN}{BOLD}Node started!{RESET} (PID {proc.pid})")
    print(f"  {DIM}Auto-restart enabled — node will recover from crashes.{RESET}")
    print()
    print(f"  {DIM}Dashboard:{RESET}  http://localhost:{port}/")
    print(f"  {DIM}Logs:{RESET}       {LOG_FILE}")
    print()
    print(f"  {BOLD}Useful commands:{RESET}")
    print(f"    {CYAN}neuroshard status{RESET}      Check node status")
    print(f"    {CYAN}neuroshard logs -f{RESET}     Follow live logs")
    print(f"    {CYAN}neuroshard stop{RESET}        Stop the node")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Subcommand: stop
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_stop(args):
    """Stop the running node."""
    pid = get_running_pid()
    if pid is None:
        print(f"\n  {YELLOW}No running node found.{RESET}")
        print()
        return

    uptime = _get_uptime(pid)
    print(f"\n  Stopping node (PID {pid}, uptime {uptime})...")

    try:
        # Kill the entire process group (supervisor + child) so nothing orphans.
        # The supervisor is started with start_new_session=True, so it leads
        # its own process group. os.killpg sends to all members.
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            # Fallback to single-PID kill
            os.kill(pid, signal.SIGTERM)

        # Wait for graceful shutdown (supervisor forwards SIGTERM to child,
        # child saves checkpoint, then supervisor exits).
        # Checkpoint save can take 60-120s on slow storage (Jetson eMMC).
        for i in range(600):  # Up to 60 seconds for checkpoint save
            time.sleep(0.1)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
        else:
            # Still running - force kill the whole group
            print(f"  {YELLOW}Graceful shutdown timed out, sending SIGKILL...{RESET}")
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGKILL)
                time.sleep(0.5)
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(pid, signal.SIGKILL)
                    time.sleep(0.5)
                except ProcessLookupError:
                    pass

        _cleanup_pid_file()
        print(f"  {GREEN}Node stopped.{RESET}")
        print()
    except Exception as e:
        print(f"  {RED}Error stopping node: {e}{RESET}")
        _cleanup_pid_file()


# ═══════════════════════════════════════════════════════════════════════════════
#  Subcommand: restart
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_restart(args):
    """Restart the node."""
    pid = get_running_pid()
    if pid:
        print(f"\n  Stopping current node (PID {pid})...")
        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(60):
                time.sleep(0.1)
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
            else:
                try:
                    os.kill(pid, signal.SIGKILL)
                    time.sleep(0.5)
                except ProcessLookupError:
                    pass
            _cleanup_pid_file()
            print(f"  {GREEN}Stopped.{RESET}")
        except Exception as e:
            print(f"  {RED}Error stopping: {e}{RESET}")
            _cleanup_pid_file()

        # Brief pause before restart
        time.sleep(1)
    else:
        print(f"\n  {DIM}No running node found, starting fresh...{RESET}")

    cmd_start(args)


# ═══════════════════════════════════════════════════════════════════════════════
#  Subcommand: _supervisor (internal — not user-facing)
# ═══════════════════════════════════════════════════════════════════════════════

def _cmd_supervisor(args):
    """Entry point for the supervisor daemon process."""
    child_cmd = args.child_cmd
    # Strip leading "--" separator from REMAINDER
    if child_cmd and child_cmd[0] == "--":
        child_cmd = child_cmd[1:]
    if not child_cmd:
        sys.exit(1)
    _run_supervisor(child_cmd)


# ═══════════════════════════════════════════════════════════════════════════════
#  Subcommand: status
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_status(args):
    """Show node status."""
    pid = get_running_pid()
    config = load_config()
    port = config.get("port", 8000)

    print()
    if pid:
        uptime = _get_uptime(pid)
        print(f"  {GREEN}{BOLD}Node is running{RESET}")
        print()
        print(f"    PID:        {pid}")
        print(f"    Uptime:     {uptime}")

        sup = _load_supervisor_state()
        if sup["restarts"] > 0:
            last = ""
            if sup.get("last_restart"):
                ago = time.time() - sup["last_restart"]
                if ago < 60:
                    last = f" (last: {int(ago)}s ago)"
                elif ago < 3600:
                    last = f" (last: {int(ago // 60)}m ago)"
                else:
                    last = f" (last: {int(ago // 3600)}h ago)"
            print(f"    Restarts:   {sup['restarts']}{last}")

        print(f"    Dashboard:  http://localhost:{port}/")
        print(f"    Logs:       {LOG_FILE}")

        # Try to get more info from the node's API
        try:
            import requests
            resp = requests.get(f"http://localhost:{port}/api/status", timeout=3)
            if resp.ok:
                data = resp.json()
                if data.get("node_id"):
                    print(f"    Node ID:    {data['node_id'][:16]}...")
                if data.get("peers"):
                    print(f"    Peers:      {data['peers']}")
                if data.get("balance") is not None:
                    print(f"    Balance:    {data['balance']:.4f} NEURO")
        except Exception:
            pass

        print()
        print(f"  {BOLD}Commands:{RESET}")
        print(f"    {CYAN}neuroshard logs -f{RESET}     Follow live logs")
        print(f"    {CYAN}neuroshard stop{RESET}        Stop the node")
        print(f"    {CYAN}neuroshard restart{RESET}     Restart the node")
    else:
        print(f"  {DIM}Node is not running.{RESET}")
        print()
        if config.get("token"):
            print(f"  Start with: {CYAN}neuroshard start{RESET}")
        else:
            print(f"  Set up first: {CYAN}neuroshard setup{RESET}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Subcommand: logs
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_logs(args):
    """Show or follow node logs."""
    if not os.path.exists(LOG_FILE):
        print(f"\n  {DIM}No log file found.{RESET} Start a node first: {CYAN}neuroshard start{RESET}")
        print()
        return

    lines = getattr(args, "lines", 50)
    follow = getattr(args, "follow", False)

    if follow:
        _follow_logs(lines)
    else:
        _tail_logs(lines)


def _tail_logs(lines: int):
    """Show last N lines of logs."""
    try:
        with open(LOG_FILE, "r") as f:
            all_lines = f.readlines()
        for line in all_lines[-lines:]:
            print(line, end="")
        if not all_lines:
            print(f"  {DIM}(log file is empty){RESET}")
    except Exception as e:
        print(f"  {RED}Error reading logs: {e}{RESET}")


def _follow_logs(initial_lines: int):
    """Follow logs like tail -f."""
    print(f"  {DIM}Following {LOG_FILE} (Ctrl+C to stop){RESET}")
    print()

    try:
        with open(LOG_FILE, "r") as f:
            # Print last N lines first
            all_lines = f.readlines()
            for line in all_lines[-initial_lines:]:
                print(line, end="")

            # Now follow new content
            while True:
                line = f.readline()
                if line:
                    print(line, end="", flush=True)
                else:
                    time.sleep(0.3)
    except KeyboardInterrupt:
        print(f"\n  {DIM}Stopped following logs.{RESET}")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Subcommand: config
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_config(args):
    """Show or modify configuration."""
    action = getattr(args, "config_action", "show")

    if action == "show":
        _config_show()
    elif action == "set":
        _config_set(args.key, args.value)
    elif action == "path":
        print(CONFIG_FILE)
    elif action == "reset":
        _config_reset()
    else:
        _config_show()


def _config_show():
    """Display current configuration."""
    config = load_config()
    print()
    print(f"  {BOLD}NeuroShard Configuration{RESET}")
    print(f"  {DIM}File: {CONFIG_FILE}{RESET}")
    print()

    display = {
        "token": mask_token(config.get("token")),
        "device": config.get("device", "auto"),
        "memory": f"{config['memory']} MB" if config.get("memory") else "auto-detect",
        "max_storage": f"{config.get('max_storage', 100)} MB",
        "port": config.get("port", 8000),
        "cpu_threads": config.get("cpu_threads") or "auto",
        "tracker": config.get("tracker", ""),
        "no_training": config.get("no_training", False),
        "diloco_steps": config.get("diloco_steps", 500),
        "auto_open_browser": config.get("auto_open_browser", True),
    }

    max_key_len = max(len(k) for k in display)
    for key, val in display.items():
        print(f"    {key:<{max_key_len + 2}} {val}")
    print()
    print(f"  {DIM}Change with: neuroshard config set <key> <value>{RESET}")
    print(f"  {DIM}Example:     neuroshard config set memory 8000{RESET}")
    print()


def _config_set(key: str, value: str):
    """Set a config value."""
    config = load_config()

    # Type coercion based on known keys
    int_keys = {"memory", "max_storage", "port", "cpu_threads", "announce_port", "diloco_steps"}
    bool_keys = {"no_training", "auto_open_browser"}
    none_ok_keys = {"memory", "cpu_threads", "announce_ip", "announce_port", "seed_peers"}

    if key not in DEFAULT_CONFIG:
        print(f"\n  {RED}Unknown config key: '{key}'{RESET}")
        print(f"  Valid keys: {', '.join(sorted(DEFAULT_CONFIG.keys()))}")
        print()
        sys.exit(1)

    if value.lower() in ("none", "null", "auto") and key in none_ok_keys:
        config[key] = None
    elif key in bool_keys:
        config[key] = value.lower() in ("true", "1", "yes", "on")
    elif key in int_keys:
        try:
            config[key] = int(value)
        except ValueError:
            print(f"\n  {RED}'{key}' must be a number, got '{value}'{RESET}")
            sys.exit(1)
    else:
        config[key] = value

    save_config(config)
    display_val = mask_token(value) if key == "token" else value
    print(f"\n  {GREEN}Set {key} = {display_val}{RESET}")
    print()


def _config_reset():
    """Reset config to defaults."""
    if os.path.exists(CONFIG_FILE):
        os.remove(CONFIG_FILE)
        print(f"\n  {GREEN}Configuration reset to defaults.{RESET}")
    else:
        print(f"\n  {DIM}No config file to reset.{RESET}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Subcommand: run  (foreground)
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_run(args):
    """Run the node in foreground (blocking)."""
    from_daemon = getattr(args, "from_daemon", False)

    if from_daemon:
        # Daemon subprocess: use ONLY the explicit CLI args passed by cmd_start.
        # Don't re-read config — cmd_start already resolved everything.
        config = _config_from_cli_args(args)
    else:
        # Interactive foreground: merge config file + CLI overrides
        config = load_config()
        _apply_cli_overrides(config, args)

    token = config.get("token")
    if not token:
        # Interactive terminal → run setup automatically
        if sys.stdin.isatty() and not from_daemon:
            print(f"\n  {YELLOW}No configuration found. Let's set up your node first!{RESET}")
            print()
            cmd_setup(args, _called_from_start=True)
            # Reload config after setup
            config = load_config()
            _apply_cli_overrides(config, args)
            token = config.get("token")

        if not token:
            print(f"\n  {RED}No wallet token configured!{RESET}")
            print()
            print(f"  Run {CYAN}neuroshard setup{RESET} to configure, or pass {CYAN}--token YOUR_TOKEN{RESET}")
            print(f"  Get your token at: {CYAN}https://neuroshard.com/wallet{RESET}")
            print()
            sys.exit(1)

    if not from_daemon:
        print_banner()
        gpu_desc, gpu_color = detect_gpu()
        print(f"  {gpu_color}Device: {gpu_desc}{RESET}")
        print()

    # Write PID file so `neuroshard status` works even in foreground mode
    ensure_dir()
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    # Register cleanup
    import atexit
    def cleanup():
        _cleanup_pid_file()
    atexit.register(cleanup)

    # Resolve mnemonic
    resolved_token = resolve_token(token)

    port = config.get("port", 8000)

    if not from_daemon:
        print(f"  Starting on port {port}...")
        print(f"  Dashboard: http://localhost:{port}/")
        print(f"  {DIM}Press Ctrl+C to stop{RESET}")
        print()

    # Auto-open browser (foreground only, not daemon)
    if not from_daemon and config.get("auto_open_browser", True):
        if not getattr(args, "no_browser", False) and not getattr(args, "headless", False):
            _open_browser_delayed(port)

    # Parse seed peers
    seed_peers = None
    sp = config.get("seed_peers")
    if sp:
        if isinstance(sp, str):
            seed_peers = [p.strip() for p in sp.split(",") if p.strip()]
        elif isinstance(sp, list):
            seed_peers = sp

    # Import and run
    from neuroshard.runner import run_node

    try:
        run_node(
            port=port,
            tracker=config.get("tracker", "https://neuroshard.com/api/tracker"),
            node_token=resolved_token,
            announce_ip=config.get("announce_ip"),
            announce_port=config.get("announce_port"),
            enable_training=not config.get("no_training", False),
            observer_mode=config.get("observer", False),
            available_memory_mb=config.get("memory"),
            max_storage_mb=config.get("max_storage", 100),
            max_cpu_threads=config.get("cpu_threads"),
            diloco_inner_steps=config.get("diloco_steps", 500),
            device=config.get("device", "auto"),
            seed_peers=seed_peers,
        )
    except KeyboardInterrupt:
        pass  # Shutdown already handled by request_shutdown() in runner.py


def _config_from_cli_args(args) -> dict:
    """Build a config dict directly from CLI args (for daemon subprocess).
    
    This avoids re-reading config.json in the subprocess — cmd_start already
    resolved all values and passed them as explicit CLI flags.
    """
    return {
        "token": getattr(args, "token", None),
        "port": getattr(args, "port", None) or 8000,
        "device": getattr(args, "device", None) or "auto",
        "memory": getattr(args, "memory", None),
        "max_storage": getattr(args, "max_storage", None) or 100,
        "cpu_threads": getattr(args, "cpu_threads", None),
        "tracker": getattr(args, "tracker", None) or "https://neuroshard.com/api/tracker",
        "no_training": getattr(args, "no_training", False),
        "observer": getattr(args, "observer", False),
        "diloco_steps": getattr(args, "diloco_steps", None) or 500,
        "announce_ip": getattr(args, "announce_ip", None),
        "announce_port": getattr(args, "announce_port", None),
        "seed_peers": getattr(args, "seed_peers", None),
        "auto_open_browser": False,  # Never in daemon mode
    }


def _open_browser_delayed(port: int, delay: float = 3.0):
    """Open dashboard in browser after a short delay."""
    def opener():
        time.sleep(delay)
        url = f"http://localhost:{port}/"
        try:
            webbrowser.open(url)
        except Exception:
            pass
    threading.Thread(target=opener, daemon=True).start()


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI argument overrides (merge --flags into config)
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_cli_overrides(config: dict, args):
    """Apply CLI --flag overrides on top of saved config."""
    if getattr(args, "token", None):
        config["token"] = args.token
    if getattr(args, "device", None) and args.device != "auto":
        config["device"] = args.device
    if getattr(args, "memory", None) is not None:
        config["memory"] = args.memory
    if getattr(args, "max_storage", None) is not None:
        config["max_storage"] = args.max_storage
    if getattr(args, "port", None) is not None:
        config["port"] = args.port
    if getattr(args, "cpu_threads", None) is not None:
        config["cpu_threads"] = args.cpu_threads
    if getattr(args, "tracker", None):
        config["tracker"] = args.tracker
    if getattr(args, "no_training", False):
        config["no_training"] = True
    if getattr(args, "observer", False):
        config["observer"] = True
    if getattr(args, "diloco_steps", None) is not None:
        config["diloco_steps"] = args.diloco_steps
    if getattr(args, "announce_ip", None):
        config["announce_ip"] = args.announce_ip
    if getattr(args, "announce_port", None) is not None:
        config["announce_port"] = args.announce_port
    if getattr(args, "seed_peers", None):
        config["seed_peers"] = args.seed_peers


# ═══════════════════════════════════════════════════════════════════════════════
#  Common node-option arguments (reused by start, run, restart)
# ═══════════════════════════════════════════════════════════════════════════════

def _add_node_args(parser):
    """Add common node arguments to a subparser."""
    parser.add_argument("--token", "-t", type=str, default=None,
                        help="Wallet token (overrides saved config)")
    parser.add_argument("--port", "-p", type=int, default=None,
                        help="HTTP port (default: from config or 8000)")
    parser.add_argument("--device", type=str, default=None,
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Compute device (default: from config or auto)")
    parser.add_argument("--memory", type=int, default=None,
                        help="Max memory in MB (default: from config or auto)")
    parser.add_argument("--max-storage", type=int, default=None,
                        help="Max disk storage in MB (default: from config or 100)")
    parser.add_argument("--cpu-threads", type=int, default=None,
                        help="Max CPU threads (default: from config or all)")
    parser.add_argument("--tracker", type=str, default=None,
                        help="Tracker URL for peer discovery")
    parser.add_argument("--no-training", action="store_true",
                        help="Disable training (inference only)")
    parser.add_argument("--observer", action="store_true",
                        help="Observer mode")
    parser.add_argument("--diloco-steps", type=int, default=None,
                        help="DiLoCo inner steps (default: 500)")
    parser.add_argument("--announce-ip", type=str, default=None,
                        help="Force IP for peer announcements")
    parser.add_argument("--announce-port", type=int, default=None,
                        help="Force port for peer announcements")
    parser.add_argument("--seed-peers", type=str, default=None,
                        help="Comma-separated seed peers")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main entry point & argument parser
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main CLI entry point with subcommands + backward compatibility."""

    # ── Backward compatibility: detect legacy --flag-only usage ────────────
    # If the user passes --token, --stop, --status, --logs, --daemon without
    # a subcommand, handle it the old way so existing scripts don't break.
    if len(sys.argv) > 1 and sys.argv[1].startswith("--"):
        _handle_legacy_args()
        return

    # ── No arguments: show friendly help ──────────────────────────────────
    if len(sys.argv) == 1:
        _show_quick_help()
        return

    # ── Subcommand-based CLI ──────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        prog="neuroshard",
        description="NeuroShard - Decentralized AI Training Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version",
                        version=f"NeuroShard {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── neuroshard setup ──────────────────────────────────────────────────
    sub_setup = subparsers.add_parser(
        "setup", help="Interactive first-time setup",
        description="Walk through interactive configuration for your node."
    )
    sub_setup.set_defaults(func=cmd_setup)

    # ── neuroshard start ──────────────────────────────────────────────────
    sub_start = subparsers.add_parser(
        "start", help="Start node in background",
        description="Start the node as a background daemon. Uses saved config by default."
    )
    _add_node_args(sub_start)
    sub_start.set_defaults(func=cmd_start)

    # ── neuroshard stop ───────────────────────────────────────────────────
    sub_stop = subparsers.add_parser(
        "stop", help="Stop the running node",
        description="Gracefully stop the running node."
    )
    sub_stop.set_defaults(func=cmd_stop)

    # ── neuroshard restart ────────────────────────────────────────────────
    sub_restart = subparsers.add_parser(
        "restart", help="Restart the node",
        description="Stop the current node and start a new one."
    )
    _add_node_args(sub_restart)
    sub_restart.set_defaults(func=cmd_restart)

    # ── neuroshard status ─────────────────────────────────────────────────
    sub_status = subparsers.add_parser(
        "status", help="Check node status",
        description="Show whether the node is running and its details."
    )
    sub_status.set_defaults(func=cmd_status)

    # ── neuroshard logs ───────────────────────────────────────────────────
    sub_logs = subparsers.add_parser(
        "logs", help="View node logs",
        description="Show recent logs or follow live log output."
    )
    sub_logs.add_argument("-f", "--follow", action="store_true",
                          help="Follow log output (like tail -f)")
    sub_logs.add_argument("-n", "--lines", type=int, default=50,
                          help="Number of lines to show (default: 50)")
    sub_logs.set_defaults(func=cmd_logs)

    # ── neuroshard config ─────────────────────────────────────────────────
    sub_config = subparsers.add_parser(
        "config", help="View or change configuration",
        description="Manage saved configuration."
    )
    config_sub = sub_config.add_subparsers(dest="config_action")

    config_show = config_sub.add_parser("show", help="Show current config")
    config_show.set_defaults(func=cmd_config, config_action="show")

    config_set = config_sub.add_parser("set", help="Set a config value")
    config_set.add_argument("key", help="Config key to set")
    config_set.add_argument("value", help="Value to set")
    config_set.set_defaults(func=cmd_config, config_action="set")

    config_path = config_sub.add_parser("path", help="Print config file path")
    config_path.set_defaults(func=cmd_config, config_action="path")

    config_reset = config_sub.add_parser("reset", help="Reset config to defaults")
    config_reset.set_defaults(func=cmd_config, config_action="reset")

    sub_config.set_defaults(func=cmd_config, config_action="show")

    # ── neuroshard _supervisor (internal — launched by cmd_start) ─────────
    sub_supervisor = subparsers.add_parser("_supervisor", help=argparse.SUPPRESS)
    sub_supervisor.add_argument("child_cmd", nargs=argparse.REMAINDER,
                                help=argparse.SUPPRESS)
    sub_supervisor.set_defaults(func=_cmd_supervisor)

    # ── neuroshard run ────────────────────────────────────────────────────
    sub_run = subparsers.add_parser(
        "run", help="Run node in foreground (for debugging)",
        description="Run the node in the foreground. Use Ctrl+C to stop."
    )
    _add_node_args(sub_run)
    sub_run.add_argument("--no-browser", action="store_true",
                         help="Don't auto-open dashboard in browser")
    sub_run.add_argument("--headless", action="store_true",
                         help="Same as --no-browser")
    sub_run.add_argument("--from-daemon", action="store_true",
                         help=argparse.SUPPRESS)  # Internal flag
    sub_run.set_defaults(func=cmd_run)

    # Parse and dispatch
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        _show_quick_help()


# ═══════════════════════════════════════════════════════════════════════════════
#  Quick help (no args)
# ═══════════════════════════════════════════════════════════════════════════════

def _show_quick_help():
    """Show a friendly overview when no arguments given."""
    print_banner(compact=True)

    pid = get_running_pid()
    if pid:
        uptime = _get_uptime(pid)
        config = load_config()
        port = config.get("port", 8000)
        print(f"  {GREEN}Node is running{RESET} (PID {pid}, uptime {uptime})")
        print(f"  Dashboard: http://localhost:{port}/")
    else:
        print(f"  {DIM}Node is not running.{RESET}")
    print()

    print(f"  {BOLD}Commands:{RESET}")
    print()
    if not os.path.exists(CONFIG_FILE):
        print(f"    {CYAN}neuroshard setup{RESET}           One-time interactive setup")
    print(f"    {CYAN}neuroshard start{RESET}           Start node in background")
    print(f"    {CYAN}neuroshard stop{RESET}            Stop the node")
    print(f"    {CYAN}neuroshard restart{RESET}         Restart the node")
    print(f"    {CYAN}neuroshard status{RESET}          Check node status")
    print(f"    {CYAN}neuroshard logs -f{RESET}         Follow live logs")
    print(f"    {CYAN}neuroshard config show{RESET}     View configuration")
    print(f"    {CYAN}neuroshard config set{RESET}      Change a setting")
    print(f"    {CYAN}neuroshard run{RESET}             Run in foreground (debug)")
    print()
    print(f"  {DIM}Get your token at: https://neuroshard.com/wallet{RESET}")
    print(f"  {DIM}Version {__version__}{RESET}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Legacy flag handling (backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

def _handle_legacy_args():
    """Handle old-style --flag-only invocations for backward compatibility."""
    parser = argparse.ArgumentParser(
        prog="neuroshard",
        description="NeuroShard Node (legacy mode)",
        add_help=True,
    )

    # All the old flags
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--tracker", type=str, default="https://neuroshard.com/api/tracker")
    parser.add_argument("--seed-peers", type=str, default=None)
    parser.add_argument("--announce-ip", type=str, default=None)
    parser.add_argument("--announce-port", type=int, default=None)
    parser.add_argument("--no-training", action="store_true")
    parser.add_argument("--observer", action="store_true")
    parser.add_argument("--diloco-steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--memory", type=int, default=None)
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--max-storage", type=int, default=100)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--daemon", "-d", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--logs", action="store_true")
    parser.add_argument("--log-lines", type=int, default=50)
    parser.add_argument("--version", action="version",
                        version=f"NeuroShard {__version__}")

    args = parser.parse_args()

    # Dispatch legacy flags to new subcommands
    if args.stop:
        cmd_stop(args)
        return

    if args.status:
        cmd_status(args)
        return

    if args.logs:
        args.lines = args.log_lines
        args.follow = False
        cmd_logs(args)
        return

    # --daemon → start in background
    if args.daemon:
        cmd_start(args)
        return

    # Default: run in foreground (the old behavior)
    args.from_daemon = False
    cmd_run(args)


if __name__ == "__main__":
    main()
