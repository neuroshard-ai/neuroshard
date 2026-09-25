"""Isolated workers and durable progress for the observable reasoning experiment."""
import copy
import json
import os
import random
import resource
import signal
import subprocess
import sys
import time

import torch

from neuroshard.evolution import block_expert_run as shared
from neuroshard.evolution import observable_reasoning as study
from neuroshard.evolution.reference_data import identity, save, sha256
from neuroshard.evolution.seed import verify
from neuroshard.evolution.staged_integration_run import environment, load_parent, peak_rss_bytes


SCRIPT = "scripts/run_observable_reasoning.py"
ARMS = shared.ARMS


def bind_runner():
    shared.study = study
    shared.SCRIPT = SCRIPT


def generate(model, tokenizer, messages, plan):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if ids.shape[1] > plan["training"]["max_length"]:
        raise ValueError("Prompt exceeds context budget")
    with torch.inference_mode():
        output = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                max_new_tokens=plan["training"]["generation_tokens"],
                                pad_token_id=tokenizer.eos_token_id)
    generated = output[0, ids.shape[1]:]
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [tokenizer.eos_token_id])
    return {"text": tokenizer.decode(generated, skip_special_tokens=True),
            "terminated": int(generated[-1]) in eos_ids, "input_tokens": ids.shape[1],
            "generated_tokens": len(generated), "generation_calls": 1}


def evaluate(model, tokenizer, data, plan, home, arm, bound):
    model.eval()
    model.config.use_cache = True
    selector = json.loads((home / "selector.json").read_text())
    expected = study.fit_selector(data["roles"]["train_new"] + data["roles"]["train_replay"])
    if selector != expected:
        raise ValueError("Selector differs from training-only fit")
    # Copy the parent's last two blocks before loading the in-place control.
    # Both paths share frozen prefix/head tensors; neither computes a second
    # answer to decide which answer to return.
    original = list(model.model.layers)
    parent_layers = torch.nn.ModuleList(original)
    candidate_layers = None
    if arm != "baseline":
        if arm == "control-evaluate":
            count = plan["architecture"]["blocks"]
            parent_layers = torch.nn.ModuleList(original[:-count] + copy.deepcopy(original[-count:]))
        training_arm = arm.replace("-evaluate", "-train")
        trained = shared.read_receipt(home, training_arm, bound)
        blocks = study.install_blocks(model, plan["architecture"]["blocks"], expansion=arm == "expert-evaluate")
        shared.restore(home / "checkpoints" / training_arm, blocks, bound, training_arm, trained["checkpoint"])
        candidate_layers = torch.nn.ModuleList(list(model.model.layers))
    result = {}
    completed = 0
    total = sum(len(data["roles"][role]) for role in study.EVAL_ROLES)
    for role in study.EVAL_ROLES:
        result[role] = []
        if arm != "baseline":
            result[role + "_forced"] = []
        for row in data["roles"][role]:
            # Explicit serving observation boundary. Neither choose nor generate
            # receives the row, its answer, family, ID, or the protected list.
            messages = row["messages"]
            started = time.monotonic()
            route, margin = ("parent", 0.0) if arm == "baseline" else study.choose(messages[-1]["content"], selector)
            model.model.layers = candidate_layers if route == "candidate" else parent_layers
            model.config.num_hidden_layers = len(model.model.layers)
            reply = generate(model, tokenizer, messages, plan)
            reply.update({"seconds": time.monotonic() - started, "route": route, "route_margin": margin})
            # Record the raw decision/output before any evaluation labels join it.
            result[role].append({"id": row["id"], **reply})
            save(home / (arm + "-answers.json"), result)
            if arm != "baseline":
                if route == "candidate":
                    forced = {**reply, "diagnostic_extra_generation": False}
                else:
                    model.model.layers = candidate_layers
                    model.config.num_hidden_layers = len(candidate_layers)
                    diagnostic_started = time.monotonic()
                    forced = {**generate(model, tokenizer, messages, plan),
                              "seconds": time.monotonic() - diagnostic_started,
                              "diagnostic_extra_generation": True}
                result[role + "_forced"].append({"id": row["id"], **forced})
            completed += 1
            save(home / (arm + "-answers.json"), result)
            save(home / "worker-progress.json", {"arm": arm, "completed": completed, "total": total,
                                                   "updated_unix": time.time()})
    return result


def worker(arm, seed, home):
    previous_study, previous_script = shared.study, shared.SCRIPT
    bind_runner()
    try:
        return _worker(arm, seed, home)
    finally:
        shared.study, shared.SCRIPT = previous_study, previous_script


def _worker(arm, seed, home):
    plan = study.load_plan()
    study.bind_freeze(committed=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(plan["training"]["seed"])
    random.seed(plan["training"]["seed"])
    cap = plan["budget"]["arms"][arm]
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 1))
    started = time.monotonic()
    runtime = environment()
    if runtime["packages"] != plan["runtime"]["packages"]:
        raise ValueError("CPU numerical runtime differs from the committed plan")
    model, tokenizer = load_parent(seed)
    bound = shared.binding(plan, tokenizer)
    data = study.load_data(plan)
    if arm == "prepare":
        training_data = {"roles": {role: [{**row, "answer": study.training_answer(row)} for row in data["roles"][role]]
                                   for role in ("train_new", "train_replay")}}
        result = shared.prepare(model, tokenizer, training_data, plan, home, bound)
    elif arm.endswith("-train"):
        result = shared.train(model, plan, home, bound, arm)
    else:
        result = evaluate(model, tokenizer, data, plan, home, arm, bound)
    result.update({"format": study.FORMAT + "/arm", "arm": arm, "binding": bound,
                   "environment": runtime, "process_cpu_seconds": time.process_time(),
                   "wall_seconds": time.monotonic() - started, "peak_rss_bytes": peak_rss_bytes(),
                   "admission_evidence": False, "gpu_launch_authorized": False})
    save(home / (arm + ".json"), result)
    return result


def isolated(arm, seed, home, seconds):
    """Bound the entire child, terminate its process group, and bill failed work."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", ATEN_CPU_CAPABILITY="default", MKL_ENABLE_INSTRUCTIONS="SSE4_2",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONPATH=str(study.root() / "src"))
    command = [sys.executable, str(study.root() / SCRIPT), "--worker", arm,
               "--seed", str(seed.resolve()), "--home", str(home.resolve())]
    before, started = resource.getrusage(resource.RUSAGE_CHILDREN), time.monotonic()
    outcome, child = "failed", None
    try:
        with (home / (arm + ".log")).open("x") as log:
            child = subprocess.Popen(command, env=env, cwd=study.root(), start_new_session=True,
                                     stdout=log, stderr=subprocess.STDOUT)
            save(home / "active-worker.json", {"arm": arm, "pid": child.pid, "deadline_unix": time.time() + seconds})
            code = child.wait(timeout=seconds)
            if code:
                raise subprocess.CalledProcessError(code, command)
        outcome = "completed"
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        save(home / (arm + "-launch.json"), {"arm": arm, "outcome": outcome,
             "wall_seconds": time.monotonic() - started,
             "cpu_seconds": after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
             "scope": "entire child, including imports, IO, diagnostics and checkpointing"})
    return json.loads((home / (arm + ".json")).read_text())


def run(seed, home):
    plan, freeze = study.load_plan(), study.bind_freeze(committed=True)
    verify(seed)
    home.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.monotonic(), time.process_time()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=study.root()).decode().strip()
    save(home / "study.json", {"plan": identity(plan), "freeze": freeze, "commit": commit,
                               "started_unix": time.time(), "pid": os.getpid(),
                               "source": str(study.root()), "gpu_launch_authorized": False})
    data = study.load_data(plan)
    selector = study.fit_selector(data["roles"]["train_new"] + data["roles"]["train_replay"])
    save(home / "selector.json", selector)
    receipts = {}
    result = None
    arm = None
    try:
        for arm in ARMS:
            remaining = plan["budget"]["total_wall_seconds"] - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Total wall budget exhausted")
            save(home / "status.json", {"state": "running", "arm": arm, "completed_arms": list(receipts),
                                       "started_unix": time.time() - (time.monotonic() - started),
                                       "deadline_unix": time.time() + remaining})
            receipts[arm] = isolated(arm, seed, home, min(remaining, plan["budget"]["arms"][arm]))
            if receipts[arm]["peak_rss_bytes"] > plan["budget"]["maximum_peak_rss_bytes"]:
                raise ValueError("Worker exceeded peak RSS budget")
            if arm == "baseline":
                protected = [row["id"] for role in study.EVAL_ROLES
                             for row in study.checked(receipts[arm][role], data["roles"][role]) if row["passed"]]
                save(home / "protected-before-training.json", {"ids": protected, "baseline": identity(receipts[arm]),
                                                                "supplied_to_selector": False})
            if arm == "control-train" and not receipts[arm]["matched_budget"]:
                raise ValueError("Control did not spend the complete matched training budget")
        target = (json.loads((home / "expert-train-launch.json").read_text())["cpu_seconds"]
                  + receipts["prepare"]["expert_extension_cpu_seconds"])
        result = study.score(plan, data, receipts["baseline"], receipts["expert-evaluate"],
                             receipts["control-evaluate"], receipts["expert-train"],
                             receipts["control-train"], target, selector)
        result["execution_completed"] = True
    except Exception as error:
        result = {"passed": False, "execution_completed": False, "error": str(error), "failed_arm": arm,
                  "next": "stop-execution-no-quality-conclusion", "completed_arms": list(receipts),
                  "admission_evidence": False, "gpu_launch_authorized": False, "upgrade_public_0_4_0": False,
                  "item4_complete": False, "settlement_authorized": False}
    launches = {name: json.loads((home / (name + "-launch.json")).read_text())
                for name in ARMS if (home / (name + "-launch.json")).exists()}
    result.update({"contract": identity(plan), "freeze": freeze, "commit": commit,
                   "wall_seconds": time.monotonic() - started,
                   "child_cpu_seconds": sum(row["cpu_seconds"] for row in launches.values()),
                   "orchestrator_cpu_seconds": time.process_time() - cpu_started,
                   "receipts": {name: sha256(home / (name + ".json")) for name in receipts},
                   "launches": launches, "selector_sha256": sha256(home / "selector.json"),
                   "protected_sha256": sha256(home / "protected-before-training.json") if "baseline" in receipts else None})
    save(home / "result.json", result)
    save(home / "status.json", {"state": "finished" if result["execution_completed"] else "failed-execution",
                               "passed": result["passed"], "result": str(home / "result.json"),
                               "completed_unix": time.time()})
    return result
