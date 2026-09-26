"""Committed, bounded execution and independent replay of the A1 reference.

This is a single-operator experiment runner, not a permissionless verifier.
Published artifact hashes are checked once per preparation. File identities are
checked around every reply. Partial or failed runs are never silently resumed.
"""

import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from neuroshard.evolution.modular_reference import assess, load_plan, route_estimate, score_reply


ROOT = Path(__file__).resolve().parents[3]
PLAN = "config/experiments/modular-reference-a1.json"
AMENDMENT = "config/experiments/modular-reference-a1-execution.json"
ARTIFACTS = "config/experiments/modular-reference-a1-artifacts.json"
SCRIPT = "scripts/run_modular_reference.py"
PROFILES = {
    "reference": (PLAN, AMENDMENT),
    "tool-interface": ("config/experiments/modular-tool-interface-diagnostic.json",
                       "config/experiments/modular-tool-interface-execution.json"),
    "fresh-reference": ("config/experiments/modular-reference-fresh.json",
                        "config/experiments/modular-reference-fresh-execution.json"),
    "fresh-reference-recovery": ("config/experiments/modular-reference-fresh.json",
                                 "config/experiments/modular-reference-fresh-recovery-execution.json"),
}
FRESH_PROFILES = ("fresh-reference", "fresh-reference-recovery")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2) + "\n"
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def committed_sources(root=ROOT, profile="reference"):
    """Bind source without requiring the allocator to have the worker's CPU."""
    plan_path, amendment_path = PROFILES[profile]
    amendment = read(root / amendment_path)
    if sha256(root / plan_path) != amendment["plan_sha256"]:
        raise ValueError("original task/quality plan changed")
    if sha256(root / ARTIFACTS) != amendment["artifacts_sha256"]:
        raise ValueError("artifact inventory changed")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    sources = {}
    for name in amendment["sources"]:
        committed = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root)
        if (root / name).read_bytes() != committed:
            raise ValueError(f"uncommitted execution source: {name}")
        sources[name] = hashlib.sha256(committed).hexdigest()
    return {"commit": commit, "sources": sources}


def configure_runtime(profile):
    """Set the explicit research profile before importing the numerical stack."""
    if profile not in FRESH_PROFILES:
        return
    if "torch" in sys.modules:
        raise ValueError("configure the numerical profile before importing torch")
    for key, value in read(ROOT / PROFILES[profile][1])["environment"].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def freeze(root=ROOT, profile="reference"):
    """Reject uncommitted execution bytes and an undeclared numerical runtime."""
    _, amendment_path = PROFILES[profile]
    amendment = read(root / amendment_path)
    source = committed_sources(root, profile)
    packages = {name: importlib.metadata.version(name) for name in amendment["packages"]}
    if packages != amendment["packages"] or platform.python_version() != amendment["python"]:
        raise ValueError("numerical runtime differs from execution amendment")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("this execution amendment requires Linux x86_64")
    environment = {name: os.environ.get(name) for name in amendment.get("environment", {})}
    if environment != amendment.get("environment", {}):
        raise ValueError("numerical startup environment differs from amendment")
    cpu = Path("/proc/cpuinfo").read_text()
    if any(flag not in cpu.split() for flag in amendment.get("required_cpu_flags", [])):
        raise ValueError("worker CPU lacks the frozen instruction set")
    cpu_features = sorted(set(line for line in cpu.splitlines()
                              if line.startswith(("model name", "flags"))))
    return {**source, "profile": profile, "packages": packages, "environment": environment,
            "python": platform.python_version(), "cpu_features": cpu_features,
            "plan_sha256": amendment["plan_sha256"], "amendment_sha256": sha256(root / amendment_path),
            "artifacts_sha256": amendment["artifacts_sha256"]}


def wait_for_ci(home, commit, seconds=3600):
    """Let a background controller wait for this exact push, without inference."""
    home = Path(home)
    save(home / "ci-request.json", {"commit": commit, "seconds": seconds}, exclusive=True)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        payload = subprocess.check_output(
            ["gh", "api", f"repos/neuroshard-ai/neuroshard/actions/runs?head_sha={commit}&event=push&per_page=100"],
            cwd=ROOT, text=True, timeout=min(30, max(1, deadline - time.monotonic())))
        runs = [row for row in json.loads(payload)["workflow_runs"]
                if row["head_sha"] == commit and row["event"] == "push"
                and row["name"] == "Native release checks"]
        latest = max(runs, key=lambda row: row["id"]) if runs else None
        if latest and latest["status"] == "completed":
            save(home / "ci.json", latest, exclusive=True)
            if latest["conclusion"] != "success":
                raise ValueError("committed freeze did not pass CI")
            return latest
        save(home / "status.json", {"state": "waiting-ci", "commit": commit,
                                    "run_id": latest["id"] if latest else None})
        time.sleep(min(30, max(0, deadline - time.monotonic())))
    raise TimeoutError("CI waiting budget exhausted; no inference started")


def file_state(model_dir, inventory):
    state = {}
    for name in inventory["files"]:
        path = Path(model_dir) / name
        stat = path.stat()
        state[name] = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    # AutoTokenizer can discover optional files, so reject unpinned inputs too.
    actual = {p.name for p in Path(model_dir).iterdir()
              if p.suffix in (".json", ".jinja", ".txt", ".model", ".safetensors")}
    if actual != set(inventory["files"]):
        raise ValueError("model directory differs from pinned artifact inventory")
    return state


def verify_artifacts(model_dir, inventory, download=False):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    needed = sum(spec["bytes"] for name, spec in inventory["files"].items()
                 if not (model_dir / name).exists())
    if needed and shutil.disk_usage(model_dir).free < needed + 5 * 1024 ** 3:
        raise ValueError("insufficient disk for missing artifacts plus 5 GiB reserve")
    verified_state = {}
    for name, spec in inventory["files"].items():
        if Path(name).name != name:
            raise ValueError("artifact path must be a single filename")
        path = model_dir / name
        if not path.exists() and download:
            from huggingface_hub import hf_hub_download
            hf_hub_download(inventory["repo"], name, revision=inventory["revision"], local_dir=model_dir)
        stat = path.stat()
        if stat.st_size != spec["bytes"]:
            raise ValueError(f"artifact size mismatch: {name}")
        if spec["algorithm"] == "sha256":
            digest = hashlib.sha256()
        elif spec["algorithm"] == "git-blob-sha1":
            digest = hashlib.sha1(f"blob {stat.st_size}\0".encode())
        else:
            raise ValueError("unknown artifact digest")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != spec["digest"]:
            raise ValueError(f"artifact hash mismatch: {name}")
        after = path.stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError(f"artifact changed while hashing: {name}")
        verified_state[name] = [after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns]
    index = read(model_dir / "model.safetensors.index.json")
    shards = {name for name in inventory["files"] if name.endswith(".safetensors")}
    if set(index["weight_map"].values()) != shards:
        raise ValueError("weight index does not match the pinned shards")
    if file_state(model_dir, inventory) != verified_state:
        raise ValueError("artifact changed during inventory verification")
    return verified_state


def checked(plan, row, binding, which, phase, task):
    """A matching task ID alone is never enough to reuse a reply."""
    expected = {"binding": identity(binding), "model": which, "phase": phase,
                "id": task["id"], "category": task["category"], "task_sha256": identity(task)}
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError("reply provenance mismatch")
    if row.get("stopped"):
        raise ValueError(f"stopped reply: {row.get('reason')}")
    if row["seconds"] > plan["limits"]["per_task_seconds"] or row["max_rss_bytes"] > plan["limits"]["max_rss_bytes"]:
        raise ValueError("reply exceeded its resource budget")
    score = score_reply(task, row["text"], row["terminated"])
    if any(score[key] != row[key] for key in ("passed", "reason")):
        raise ValueError("reply rescore mismatch")
    if binding.get("profile") in ("tool-interface", *FRESH_PROFILES) and task["kind"] == "tool":
        from neuroshard.evolution.modular_tools import validate_reply
        validation = validate_reply(row["text"], json.loads(task["messages"][0]["functions"]))
        if row.get("wire_validation") != validation:
            raise ValueError("native tool validation mismatch")
    return row


def compare_replay(primary, replay):
    fields = ("text", "terminated", "passed", "reason", "binding", "task_sha256", "model", "id")
    return {"matched": all(primary[key] == replay[key] for key in fields),
            "tokens_matched": primary["token_ids"] == replay["token_ids"]}


def charged_seconds(attempt):
    outcome = attempt / "outcome.json"
    if outcome.exists():
        return read(outcome)["seconds"]
    # A supervisor interrupted before persisting its receipt cannot be billed 0.
    return read(attempt / "request.json")["seconds"]


def spent(home, which, legacy_seconds, *, preparation=False):
    total = 0.0 if preparation else legacy_seconds
    for path in (home / "attempts").glob(f"{which}-*"):
        if not (path / "request.json").exists():
            continue
        request = read(path / "request.json")
        if (request["phase"] == "prepare") == preparation:
            total += charged_seconds(path)
    return total


def supervised(command, log_path, seconds, memory_bytes, unit):
    """A kernel cgroup limit and systemd timer also survive parent interruption."""
    if seconds <= 0:
        raise ValueError("worker has no remaining budget")
    argv = ["systemd-run", "--user", "--quiet", "--wait", "--pipe", "--unit", unit,
            "-p", f"RuntimeMaxSec={math.floor(seconds * 1000)}ms",
            "-p", "TimeoutStopSec=0", "-p", "KillMode=control-group",
            "-p", f"MemoryMax={memory_bytes}", "-p", "MemorySwapMax=0",
            "-p", "OOMPolicy=kill", "-p", f"WorkingDirectory={ROOT}",
            "-p", "UnsetEnvironment=ATEN_CPU_CAPABILITY MKL_ENABLE_INSTRUCTIONS"]
    for key, value in {"PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": "",
                       "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
                       "PYTHONUNBUFFERED": "1"}.items():
        argv += ["--setenv", f"{key}={value}"]
    argv += command
    started = time.monotonic()
    code = None
    reason = "launcher-error"
    try:
        with Path(log_path).open("x") as log:
            try:
                code = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=seconds + 5, check=False).returncode
                reason = "completed" if code == 0 else "worker-failed"
            except subprocess.TimeoutExpired:
                reason = "supervisor-timeout"
    finally:
        # Stop this exact unit, including descendants, even on Ctrl-C.
        subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True, timeout=10, check=False)
    elapsed = time.monotonic() - started
    if elapsed > seconds:
        reason = "time-limit"
    return {"seconds": elapsed, "returncode": code, "reason": reason,
            "completed": code == 0 and reason == "completed", "unit": unit,
            "cpu_seconds": None, "cpu_accounting": "see worker receipt if completed; unknown on external kill"}


def worker(request_path):
    request_path = Path(request_path)
    request = read(request_path)
    profile = request.get("profile", "reference")
    configure_runtime(profile)
    plan_path, amendment_path = PROFILES[profile]
    binding = freeze(profile=profile)
    if binding != request["freeze"]:
        raise ValueError("worker source/runtime changed")
    which, phase = request["model"], request["phase"]
    inventory = read(ROOT / ARTIFACTS)["models"][which]
    model_dir = Path(request["models"]) / which
    if phase == "prepare":
        stats = verify_artifacts(model_dir, inventory, download=read(ROOT / amendment_path).get("allow_download", True))
        result = {"binding": request["binding"], "file_state": stats, "verified": True}
    else:
        if file_state(model_dir, inventory) != request["file_state"]:
            raise ValueError("artifact identity changed before generation")
        from neuroshard.evolution.modular_reference_run import generate_task
        plan = load_plan(ROOT / plan_path)
        task = next(task for task in plan["tasks"] if task["id"] == request["task"])
        result = generate_task(plan, which, model_dir, task)
        if profile in ("tool-interface", *FRESH_PROFILES) and task["kind"] == "tool":
            from neuroshard.evolution.modular_tools import validate_reply
            result["wire_validation"] = validate_reply(result["text"], json.loads(task["messages"][0]["functions"]))
        if file_state(model_dir, inventory) != request["file_state"]:
            raise ValueError("artifact identity changed during generation")
        result.update({"binding": request["binding"], "phase": phase, "task_sha256": identity(task)})
    result["process_cpu_seconds"] = time.process_time()
    save(request_path.parent / "reply.json", result, exclusive=True)


def launch(home, models, binding, which, phase, seconds, memory_bytes, *, task=None, stats=None):
    name = f"{which}-{phase}" + (f"-{task['id']}" if task else "")
    attempt = home / "attempts" / name
    attempt.mkdir(parents=True, exist_ok=False)
    request = {"freeze": binding["freeze"], "binding": identity(binding), "model": which,
               "profile": binding.get("profile", "reference"),
               "phase": phase, "task": task["id"] if task else None, "models": str(models),
               "file_state": stats, "seconds": seconds, "started_unix": time.time()}
    save(attempt / "request.json", request, exclusive=True)
    unit = "neuroshard-a1-" + uuid.uuid4().hex[:16]
    command = [sys.executable, str(ROOT / SCRIPT), "worker", "--request", str(attempt / "request.json")]
    started = time.monotonic()
    try:
        outcome = supervised(command, attempt / "worker.log", seconds, memory_bytes, unit)
    except BaseException as error:
        save(attempt / "outcome.json", {"completed": False, "seconds": time.monotonic() - started,
                                        "reason": type(error).__name__, "unit": unit}, exclusive=True)
        raise
    save(attempt / "outcome.json", outcome, exclusive=True)
    if not outcome["completed"]:
        raise RuntimeError(f"{name}: {outcome['reason']} (see {attempt / 'worker.log'})")
    result = read(attempt / "reply.json")
    if result["binding"] != identity(binding):
        raise ValueError("worker result belongs to another execution")
    return result


def gate(plan, primary, replays):
    result = assess(plan, primary)
    expected = {(model, task["id"]) for model in ("baseline", "modular") for task in plan["tasks"]}
    replay_keys = {(row["model"], row["id"]) for row in replays}
    result["replay_complete"] = (replay_keys == expected and len(replays) == len(expected)
                                 and all(row["matched"] for row in replays))
    result["quality_ready"] = result["quality_ready"] and result["replay_complete"]
    result["route"] = route_estimate()
    result["admission_evidence"] = False
    result["milestone_complete"] = False  # Placement/deviation review still required.
    if "comparison" in plan:
        from neuroshard.evolution.modular_reference_comparison import compare
        result.update(compare(plan, primary, result["replay_complete"]))
    return result


def usable_call(row):
    return bool(row["passed"] and row.get("wire_validation", {}).get("valid"))


def diagnostic_gate(tasks, primary, replays):
    """Successful opened-case diagnostics cannot close A1 or become admission."""
    expected = {task["id"] for task in tasks}
    complete = (len(primary) == len(expected) and {row["id"] for row in primary} == expected
                and all(row["model"] == "baseline" and not row["stopped"] for row in primary))
    correct = {row["id"] for row in primary if usable_call(row)}
    replay_complete = (bool(correct) and len(replays) == len(correct)
                       and {row["id"] for row in replays} == correct
                       and all(row["model"] == "baseline" and row["matched"] for row in replays))
    return {"primary_complete": complete, "correct_calls": len(correct),
            "wire_valid_calls": sum(row.get("wire_validation", {}).get("valid", False) for row in primary),
            "replay_complete": replay_complete, "interface_confirmed": complete and replay_complete,
            "quality_ready": False, "admission_evidence": False, "milestone_complete": False,
            "opened_diagnostic_cases": True}


def run(home, models, legacy_path, profile="reference"):
    home, models, legacy_path = Path(home).resolve(), Path(models).resolve(), Path(legacy_path).resolve()
    home.mkdir(parents=True, exist_ok=True)
    with (home / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(home, models, legacy_path, profile)


def _run(home, models, legacy_path, profile="reference"):
    plan_path, amendment_path = PROFILES[profile]
    plan = load_plan(ROOT / plan_path)
    amendment = read(ROOT / amendment_path)
    frozen = freeze(profile=profile)
    diagnostic = profile == "tool-interface"
    task_ids = amendment.get("task_ids", [task["id"] for task in plan["tasks"]])
    by_id = {task["id"]: task for task in plan["tasks"]}
    if len(set(task_ids)) != len(task_ids) or any(identity not in by_id for identity in task_ids):
        raise ValueError("invalid task selection")
    tasks = [by_id[identity] for identity in task_ids]
    model_order = ["baseline"] if diagnostic else ["baseline", "modular"]
    legacy = read(legacy_path)  # Its absence means the old run has not finished.
    if legacy.get("plan_sha256") != amendment.get("legacy_plan_sha256", frozen["plan_sha256"]) or legacy.get("which") != "baseline":
        raise ValueError("legacy accounting does not match the original baseline")
    if "legacy_result_sha256" in amendment and sha256(legacy_path) != amendment["legacy_result_sha256"]:
        raise ValueError("legacy receipt changed")
    if not isinstance(legacy.get("seconds"), (int, float)) or not math.isfinite(legacy["seconds"]) or legacy["seconds"] <= 0:
        raise ValueError("legacy evaluation cost is missing")
    historical_seconds = legacy["seconds"]
    historical_preparation = {which: 0.0 for which in model_order}
    for prior in amendment.get("additional_evaluations", []):
        path = ROOT / prior["path"]
        if sha256(path) != prior["sha256"]:
            raise ValueError("historical evaluation receipt changed")
        accounting = read(path)["accounting"]
        recorded = accounting["baseline"]["evaluation_seconds"]
        if recorded < historical_seconds:
            raise ValueError("cumulative evaluation accounting went backwards")
        historical_seconds = recorded
        if prior.get("carry_preparation"):
            for which in model_order:
                seconds = accounting[which]["preparation_seconds"]
                if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
                    raise ValueError("invalid historical preparation cost")
                historical_preparation[which] += seconds
    binding = {"freeze": frozen, "profile": profile, "models": str(models), "legacy_sha256": sha256(legacy_path)}
    study_path = home / "study.json"
    if study_path.exists():
        if read(study_path)["binding"] != binding:
            raise ValueError("run directory belongs to another execution")
        if (home / "result.json").exists():
            return read(home / "result.json")
        # No interrupted or failed launch can be silently retried.
        for request in (home / "attempts").glob("*/request.json"):
            outcome = request.parent / "outcome.json"
            if not outcome.exists() or not read(outcome)["completed"]:
                raise ValueError("interrupted/failed run: retain its charged budget; a new amendment is required")
    else:
        save(study_path, {"binding": binding, "started_unix": time.time(),
                          "legacy_evaluation_seconds": historical_seconds,
                          "legacy_download_seconds": None, "gpu_launch_authorized": False}, exclusive=True)
        save(home / "legacy-baseline-result.json", legacy, exclusive=True)
    primary, replays = [], []
    limits = plan["limits"]
    result = None
    try:
        for which in model_order:
            legacy_seconds = historical_seconds if which == "baseline" else 0
            prepare_path = home / "attempts" / f"{which}-prepare" / "reply.json"
            if prepare_path.exists():
                raise ValueError("partial prepared run is not resumed automatically; preserve it and amend execution")
            remaining = (limits["fetch_seconds"] - historical_preparation[which]
                         - spent(home, which, 0, preparation=True))
            if remaining <= 0:
                raise TimeoutError("checkpoint preparation budget exhausted")
            save(home / "status.json", {"state": "running", "model": which, "phase": "prepare",
                                        "deadline_unix": time.time() + remaining})
            prepared = launch(home, models, binding, which, "prepare", remaining, limits["max_rss_bytes"])
            model_primary = []
            for phase in ("primary", "replay"):
                selected = tasks
                if diagnostic and phase == "replay":
                    successes = {row["id"] for row in model_primary if usable_call(row)}
                    selected = [task for task in tasks if task["id"] in successes]
                for task in selected:
                    remaining = limits["evaluate_seconds"] - spent(home, which, legacy_seconds)
                    if "new_evaluation_seconds" in amendment:
                        allowance = amendment.get("remaining_evaluation_seconds", {}).get(
                            which, amendment["new_evaluation_seconds"])
                        remaining = min(remaining, allowance - spent(home, which, 0))
                    seconds = min(limits["per_task_seconds"], remaining)
                    if seconds <= 0:
                        raise TimeoutError("checkpoint evaluation budget exhausted")
                    save(home / "status.json", {"state": "running", "model": which, "phase": phase,
                                                "task": task["id"], "deadline_unix": time.time() + seconds})
                    row = launch(home, models, binding, which, phase, seconds, limits["max_rss_bytes"],
                                 task=task, stats=prepared["file_state"])
                    checked(plan, row, binding, which, phase, task)
                    if phase == "primary":
                        model_primary.append(row)
                        primary.append(row)
                    else:
                        original = next(row for row in model_primary if row["id"] == task["id"])
                        comparison = {"model": which, "id": task["id"], **compare_replay(original, row)}
                        replays.append(comparison)
                        if not comparison["matched"]:
                            raise ValueError("independent replay mismatch")
                    save(home / "progress.json", {"primary": primary, "replays": replays})
                if not diagnostic and which == "baseline" and phase == "primary":
                    if "comparison" in plan:
                        from neuroshard.evolution.modular_reference_comparison import compare
                        acceptable = compare(plan, primary)["baseline_gate"]
                    else:
                        acceptable = assess(plan, primary)["baseline_gate"]
                    if not acceptable:
                        raise ValueError("baseline quality gate failed; stop before modular download")
            save(home / f"{which}-result.json", {"binding": binding, "rows": model_primary,
                                                 "replays": [row for row in replays if row["model"] == which]})
        decision = diagnostic_gate(tasks, primary, replays) if diagnostic else gate(plan, primary, replays)
        result = {"execution_completed": True, **decision}
    except Exception as error:
        result = {"execution_completed": False, "quality_ready": False, "milestone_complete": False,
                  "admission_evidence": False, "error": str(error)}
    finally:
        accounting = {which: {"evaluation_seconds": spent(home, which, historical_seconds if which == "baseline" else 0),
                              "preparation_seconds": historical_preparation[which] + spent(home, which, 0, preparation=True),
                              "historical_preparation_seconds": historical_preparation[which],
                              "new_preparation_seconds": spent(home, which, 0, preparation=True)}
                      for which in model_order}
        if "new_evaluation_seconds" in amendment:
            for which in model_order:
                accounting[which].update(new_evaluation_seconds=spent(home, which, 0),
                    historical_evaluation_seconds=historical_seconds if which == "baseline" else 0)
        if result is not None:
            result.update({"binding": binding, "accounting": accounting, "primary": primary, "replays": replays})
            save(home / "result.json", result, exclusive=True)
            save(home / "status.json", {"state": "finished" if result["execution_completed"] else "stopped",
                                        "result": str(home / "result.json"), "error": result.get("error")})
    return result
