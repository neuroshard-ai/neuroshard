"""Execution amendment for staged answering; the original study is read-only."""
import copy
import hashlib
import json
import os
import random
import resource
import shutil
import subprocess
import sys
import time

import torch

from neuroshard.evolution import staged_answering as method
from neuroshard.evolution.reference_data import identity, save, sha256, tokenizer_identity
from neuroshard.evolution.seed import verify
from neuroshard.evolution.staged_answer_format import evaluate
from neuroshard.evolution.staged_integration import read_checkpoint, tensor_identity, write_checkpoint
from neuroshard.evolution.staged_integration_run import (
    batches, environment, install, load_parent, optimize_step, optimizer_for,
    peak_rss_bytes, probe, read_receipt, train_control,
)


FORMAT = "neuroshard-staged-answering-recovery-v1"
PLAN = "config/experiments/staged-answering-recovery.json"
FREEZE = "config/experiments/staged-answering-recovery-freeze.json"
RECORD = "config/experiments/staged-answering-timeout-record.json"
SCRIPT = "scripts/run_staged_recovery.py"
CONTRACT = "b35201eebf3d70d65b7671be3d25279136632ffcea8aa833e61f51e7325e8ec4"
ARMS = ("restart-gate", "control-train", "expansion-evaluate", "control-evaluate")
SOURCES = (*method.SOURCES, method.FREEZE, PLAN, RECORD, SCRIPT,
           "src/neuroshard/evolution/staged_recovery.py", "docs/STAGED_ANSWERING_RECOVERY.md")


def load_plan():
    plan = json.loads((method.root() / PLAN).read_text())
    if identity(plan) != CONTRACT:
        raise ValueError("Recovery amendment changed")
    spec = method.load_spec()
    if (plan["method_contract"] != identity(spec) or plan["method_freeze"] != method.bind_freeze()
            or plan["data"] != identity(method.load_data(spec))
            or plan["gpu_launch_authorized"] or plan["method_changed"]):
        raise ValueError("Recovery must preserve the original CPU method")
    return plan


def inventory():
    plan = load_plan()
    return {"format": FORMAT + "/freeze", "amendment": identity(plan),
            "files": {name: sha256(method.root() / name) for name in SOURCES},
            "gpu_launch_authorized": False, "admission_evidence": False}


def bind_freeze(*, committed=False):
    saved = json.loads((method.root() / FREEZE).read_text())
    if saved != inventory():
        raise ValueError("Recovery source differs from the execution freeze")
    if committed:
        for name in (*SOURCES, FREEZE):
            try:
                data = subprocess.check_output(["git", "show", "HEAD:" + name], cwd=method.root(),
                                               stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as error:
                raise ValueError("Commit the recovery amendment before running: " + name) from error
            if data != (method.root() / name).read_bytes():
                raise ValueError("Commit the recovery amendment before running: " + name)
    return identity(saved)


def validate_inputs(previous, plan):
    record = json.loads((method.root() / RECORD).read_text())
    if identity(record) != plan["timeout_record"]:
        raise ValueError("Timeout record changed")
    for name, expected in plan["inputs"].items():
        path = previous / name
        if not path.resolve().is_relative_to(previous.resolve()) or sha256(path) != expected:
            raise ValueError("Recovery input changed: " + name)
    baseline = json.loads((previous / "baseline.json").read_text())
    protected = json.loads((previous / "protected-before-training.json").read_text())
    if (baseline["binding"] != plan["old_binding"] or protected["baseline"] != identity(baseline)
            or protected["ids"] != plan["protected_ids"]
            or [r["id"] for r in baseline["retention"] if r["passed"]] != plan["protected_ids"]):
        raise ValueError("Original protected answers changed")
    manifest = json.loads((previous / "checkpoints/expert/manifest.json").read_text())
    if (identity(manifest) != plan["expert_checkpoint"]["manifest"] or manifest["phase"] != "expert"
            or manifest["step"] != plan["expert_checkpoint"]["step"]):
        raise ValueError("Wrong expert boundary")
    return record


def restore_expert(module, directory, binding, step):
    original = tensor_identity(module.incumbent)
    manifest = read_checkpoint(directory, module, binding=binding, phase="expert")
    if manifest["step"] != step or tensor_identity(module.incumbent) != original:
        raise ValueError("Restored expert changed the incumbent or boundary")
    if any(bool(torch.count_nonzero(p)) for p in module.router.parameters()):
        raise ValueError("Gate must restart from the declared zero initialization")
    # read_checkpoint verified the state file hash before this restricted load.
    state = torch.load(directory / "training.pt", map_location="cpu", weights_only=True)
    return manifest, state


def verify_prefix(current, old):
    fields = ("step", "loss", "gradient_norm", "input_tokens", "answer_tokens", "padded_tokens", "routes")
    if any(current[key] != old[key] for key in fields):
        raise ValueError("Recovered gate differs from the interrupted numerical prefix")


def restart_gate(model, tokenizer, rows, spec, home, previous, old_binding, binding):
    module = install(model, expansion=True)
    model.config.use_cache = False
    incumbent = tensor_identity(module.incumbent)
    initial_added = tensor_identity(module.added)
    module.set_phase("expert")
    prepared_probe = batches(rows["expert_new"][:spec["training"]["probe_documents"]], tokenizer, spec)
    before = probe(model, prepared_probe)
    manifest, state = restore_expert(module, previous / "checkpoints/expert", old_binding,
                                     spec["training"]["expert_steps"])
    after = probe(model, prepared_probe)
    added = tensor_identity(module.added)
    # Diagnostic probes are reconstructed, then their RNG effects are discarded.
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    module.set_phase("gate")
    optimizer = optimizer_for(model, spec)
    if optimizer.state:
        raise ValueError("Gate optimizer must be fresh")
    save(home / "gate-start.json", {
        "restored_expert": identity(manifest), "router": tensor_identity(module.router),
        "torch_rng": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        "python_rng": identity(random.getstate()), "fresh_optimizer": True, "binding": binding,
    })
    prepared = batches(rows["gate_new"] + rows["gate_replay"], tokenizer, spec)
    old = json.loads((previous / "expansion-training.json").read_text())
    prefix = [row for row in old if row["phase"] == "gate"]
    history = []
    for step in range(spec["training"]["gate_steps"]):
        receipt = {"phase": "gate", "step": step, **optimize_step(
            model, prepared[step % len(prepared)], optimizer, spec)}
        history.append(receipt)
        save(home / "gate-training.json", history)
        if step < len(prefix):
            verify_prefix(receipt, prefix[step])
    checkpoint = write_checkpoint(home / "checkpoints/gate", module, optimizer, binding=binding,
                                  phase="gate", step=len(history))
    changed = added != initial_added
    return {"steps": len(history), "effective_expert_steps": spec["training"]["expert_steps"],
            "verified_prefix_steps": len(prefix), "checkpoints": {"gate": checkpoint},
            "incumbent_unchanged": tensor_identity(module.incumbent) == incumbent,
            "added_unchanged_during_gate": tensor_identity(module.added) == added,
            "added_changed": changed, "expert_training_signal": changed and after["loss"] <= before["loss"]
            * (1 - spec["training"]["minimum_relative_probe_loss_reduction"]),
            "probe_before": before, "probe_after": after, "probe_reconstructed": True,
            "probe_cpu_seconds": before["cpu_seconds"] + after["cpu_seconds"],
            "training_cpu_seconds": sum(row["cpu_seconds"] for row in history),
            "training_input_tokens": sum(row["input_tokens"] for row in history),
            "training_answer_tokens": sum(row["answer_tokens"] for row in history)}


def cost_record(record, gate, launch):
    history = record["artifacts"]["expansion-training.json"]
    useful = sum(r["cpu_seconds"] for r in history if r["phase"] == "expert")
    discarded = sum(r["cpu_seconds"] for r in history if r["phase"] == "gate")
    prior = record["artifacts"]["expansion-train-launch.json"]["cpu_seconds"]
    if launch["outcome"] != "completed" or min(prior, launch["cpu_seconds"]) <= 0:
        raise ValueError("A complete recovery process bill is required")
    return {"comparison_cpu_seconds": prior + launch["cpu_seconds"],
            "prior_worker_cpu_seconds": prior, "recovery_worker_cpu_seconds": launch["cpu_seconds"],
            "useful_expert_optimization_cpu_seconds": useful,
            "discarded_gate_optimization_cpu_seconds": discarded,
            "prior_setup_probes_interruption_unattributed_cpu_seconds": prior - useful - discarded,
            "repeated_gate_optimization_cpu_seconds": gate["training_cpu_seconds"],
            "recovery_probe_cpu_seconds": gate["probe_cpu_seconds"],
            "recovery_other_cpu_seconds": launch["cpu_seconds"] - gate["training_cpu_seconds"] - gate["probe_cpu_seconds"],
            "effective_updates": 128, "discarded_recorded_updates": 63,
            "interrupted_update_cost_separately_identifiable": False,
            "control_target_scope": "full failed training process plus full gate recovery process"}


def validate_runtime(plan):
    actual = environment()
    expected = plan["runtime"]
    if (actual["packages"] != expected["packages"] or actual["python"].split()[0] != expected["python"]
            or actual["platform"] != expected["platform"] or actual["torch_threads"] != 1):
        raise ValueError("Recovery numerical runtime differs from the previous study")


def worker(arm, seed, previous, home):
    plan, freeze = load_plan(), bind_freeze(committed=True)
    spec = method.load_spec()
    record = validate_inputs(previous, plan)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    validate_runtime(plan)
    torch.manual_seed(spec["training"]["seed"])
    random.seed(spec["training"]["seed"])
    limit = plan["budget"]["arms"][arm]["cpu_seconds"]
    resource.setrlimit(resource.RLIMIT_CPU, (limit, limit + 1))
    started = time.monotonic()
    model, tokenizer = load_parent(seed)
    data = method.load_data(spec)
    binding = {"contract": identity(spec), "execution_amendment": identity(plan), "freeze": freeze,
               "data": identity(data), "tokenizer": tokenizer_identity(tokenizer)}
    if binding["tokenizer"] != plan["old_binding"]["tokenizer"]:
        raise ValueError("Tokenizer changed")
    if arm == "restart-gate":
        result = restart_gate(model, tokenizer, data["roles"], spec, home, previous,
                              plan["old_binding"], binding)
    elif arm == "control-train":
        gate = read_receipt(home, "restart-gate", binding)
        launch = json.loads((home / "restart-gate-launch.json").read_text())
        cost = cost_record(record, gate, launch)
        if json.loads((home / "training-cost.json").read_text()) != cost:
            raise ValueError("Control budget differs from the complete retry bill")
        effective = copy.deepcopy(spec)
        effective["budget"]["worker_cpu_seconds"] = limit
        result = train_control(model, tokenizer, data["roles"], effective, home, binding,
                               cost["comparison_cpu_seconds"])
    else:
        expanded = arm == "expansion-evaluate"
        module = install(model, expansion=expanded)
        original = tensor_identity(module.incumbent) if expanded else None
        phase = "gate" if expanded else "control"
        manifest = read_checkpoint(home / "checkpoints" / phase, module, binding=binding, phase=phase)
        trained = read_receipt(home, "restart-gate" if expanded else "control-train", binding)
        expected = trained["checkpoints"][phase] if expanded else trained["checkpoint"]
        if identity(manifest) != expected or expanded and tensor_identity(module.incumbent) != original:
            raise ValueError("Serving checkpoint differs from the trained candidate")
        result = evaluate(model, tokenizer, data["roles"], spec, home, arm)
    result.update({"format": FORMAT + "/arm", "arm": arm, "binding": binding,
                   "process_cpu_seconds": time.process_time(), "wall_seconds": time.monotonic() - started,
                   "peak_rss_bytes": peak_rss_bytes(), "environment": environment(),
                   "admission_evidence": False, "gpu_launch_authorized": False})
    save(home / (arm + ".json"), result)
    return result


def run_isolated(arm, seed, previous, home, seconds):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONPATH=str(method.root() / "src"))
    command = [sys.executable, str(method.root() / SCRIPT), "--worker", arm, "--seed", str(seed.resolve()),
               "--previous", str(previous.resolve()), "--home", str(home.resolve())]
    before, started = resource.getrusage(resource.RUSAGE_CHILDREN), time.monotonic()
    outcome = "failed"
    try:
        with (home / (arm + ".log")).open("x") as log:
            subprocess.run(command, env=env, cwd=method.root(), check=True, timeout=seconds,
                           stdout=log, stderr=subprocess.STDOUT)
        outcome = "completed"
    finally:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        save(home / (arm + "-launch.json"), {
            "arm": arm, "outcome": outcome, "wall_seconds": time.monotonic() - started,
            "cpu_seconds": after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
            "scope": "complete child process including imports, restoration, checkpointing and exit"})
    return json.loads((home / (arm + ".json")).read_text())


def total_spend(record, home, cpu_started):
    launches = {arm: json.loads((home / (arm + "-launch.json")).read_text())
                for arm in ARMS if (home / (arm + "-launch.json")).exists()}
    prior = record["measured_process_cpu_seconds"]
    coordinator = time.process_time() - cpu_started
    return {"launches": launches, "prior_attempt_process_cpu_seconds": prior,
            "coordinator_process_cpu_seconds": coordinator,
            "total_process_cpu_seconds": prior + coordinator + sum(r["cpu_seconds"] for r in launches.values())}


def run_study(seed, previous, home):
    started, cpu_started = time.monotonic(), time.process_time()
    plan, freeze = load_plan(), bind_freeze(committed=True)
    spec = method.load_spec()
    record = validate_inputs(previous, plan)
    verify(seed)
    if home.resolve().is_relative_to(previous.resolve()):
        raise ValueError("The timed-out study must remain read-only")
    home.mkdir(parents=True, exist_ok=False)
    # Snapshot every pinned recovery input before work, then verify the copies.
    inputs = home / "inputs"
    for name in plan["inputs"]:
        target = inputs / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(previous / name, target)
    validate_inputs(inputs, plan)
    save(home / "study.json", {"amendment": identity(plan), "freeze": freeze,
                               "method_contract": identity(spec), "gpu_launch_authorized": False,
                               "admission_evidence": False, "item4_complete": False})
    receipts = {}
    try:
        for arm in ARMS:
            remaining = plan["budget"]["total_wall_seconds"] - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Recovery wall budget exhausted")
            receipts[arm] = run_isolated(arm, seed, inputs, home,
                                         min(remaining, plan["budget"]["arms"][arm]["wall_seconds"]))
            if receipts[arm]["peak_rss_bytes"] > spec["gates"]["maximum_peak_rss_bytes"]:
                raise ValueError("Process exceeded the original memory gate")
            if arm == "restart-gate":
                if receipts[arm]["verified_prefix_steps"] != plan["gate"]["verify_recorded_prefix_steps"]:
                    raise ValueError("Incomplete interrupted-prefix verification")
                launch = json.loads((home / "restart-gate-launch.json").read_text())
                save(home / "training-cost.json", cost_record(record, receipts[arm], launch))
        cost = json.loads((home / "training-cost.json").read_text())
        training = {**receipts["restart-gate"], "comparison_cpu_seconds": cost["comparison_cpu_seconds"]}
        result = method.score(spec, record["artifacts"]["baseline.json"], receipts["expansion-evaluate"],
                              receipts["control-evaluate"], training, receipts["control-train"])
    except (subprocess.SubprocessError, TimeoutError, ValueError) as error:
        save(home / "failure.json", {"error": str(error), "completed_arms": list(receipts),
                                     "wall_seconds": time.monotonic() - started,
                                     **total_spend(record, home, cpu_started), "passed": False,
                                     "gpu_launch_authorized": False, "admission_evidence": False})
        raise
    result.update({"execution_amendment": identity(plan), "execution_freeze": freeze,
                   "binding": receipts["restart-gate"]["binding"], "training_cost": cost,
                   "receipts": {arm: identity(receipt) for arm, receipt in receipts.items()},
                   "recovery_wall_seconds": time.monotonic() - started,
                   "prior_attempt_wall_seconds": record["artifacts"]["failure.json"]["wall_seconds"],
                   **total_spend(record, home, cpu_started)})
    save(home / "result.json", result)
    return result
