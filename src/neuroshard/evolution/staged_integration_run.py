"""Isolated CPU execution for the separately frozen staged-integration candidate."""
import importlib.metadata
import json
import os
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch

from neuroshard.evolution.reference_data import conversation, identity, save, tokenizer_identity
from neuroshard.evolution.seed import verify
from neuroshard.evolution.staged_integration import (
    EVAL_ROLES, FORMAT, StagedMixture, bind_freeze, load_data, load_spec,
    read_checkpoint, root, score, tensor_identity, write_checkpoint,
)


ARMS = ("baseline", "expansion-train", "control-train", "expansion-evaluate", "control-evaluate")


def peak_rss_bytes():
    # VmHWM belongs to this post-exec address space. getrusage().ru_maxrss can
    # retain the launcher's pre-exec high-water mark even after exec, so it is
    # unsuitable for comparing small isolated arms from a large parent.
    if sys.platform != "linux":
        raise ValueError("This execution profile measures Linux VmHWM")
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Linux did not report the process RSS high-water mark")


def environment():
    return {"python": sys.version, "platform": sys.platform,
            "packages": {name: importlib.metadata.version(name)
                         for name in ("torch", "transformers", "tokenizers", "safetensors")},
            "torch_threads": torch.get_num_threads(), "pid": os.getpid()}


def last_mlp(model):
    return model.model.layers[-1].mlp


def install(model, *, expansion):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if expansion:
        model.model.layers[-1].mlp = StagedMixture(last_mlp(model))
    return last_mlp(model)


def load_parent(seed):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    verify(seed)
    tokenizer = AutoTokenizer.from_pretrained(seed, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(seed, local_files_only=True, dtype=torch.float32)
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("135M CPU only")
    return model, tokenizer


def batches(rows, tokenizer, spec):
    config = spec["training"]
    encoded = [conversation(tokenizer, row["messages"] + [{"role": "assistant", "content": row["answer"]}],
                            config["max_length"]) for row in rows]
    order = list(range(len(encoded)))
    random.Random(config["seed"]).shuffle(order)
    result = []
    for start in range(0, len(order), config["batch_documents"]):
        selected = [encoded[index] for index in order[start:start + config["batch_documents"]]]
        length = max(len(row["input_ids"]) for row in selected)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        result.append({
            "input_ids": torch.tensor([row["input_ids"] + [pad] * (length - len(row["input_ids"]))
                                       for row in selected]),
            "labels": torch.tensor([row["labels"] + [-100] * (length - len(row["labels"]))
                                    for row in selected]),
            "attention_mask": torch.tensor([[1] * len(row["input_ids"]) + [0] * (length - len(row["input_ids"]))
                                            for row in selected]),
        })
    return result


def optimizer_for(model, spec):
    return torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                            lr=spec["training"]["learning_rate"],
                            weight_decay=spec["training"]["weight_decay"])


def optimize_step(model, batch, optimizer, spec):
    before_cpu, before_wall = time.process_time(), time.monotonic()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model(**batch).loss
    if loss is None or not torch.isfinite(loss):
        raise ValueError("Non-finite training loss")
    loss.backward()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, spec["training"]["clip_norm"], error_if_nonfinite=True)
    optimizer.step()
    module = last_mlp(model)
    return {"loss": float(loss.detach()), "gradient_norm": float(norm),
            "cpu_seconds": time.process_time() - before_cpu,
            "wall_seconds": time.monotonic() - before_wall,
            "input_tokens": int(batch["attention_mask"].sum()),
            "answer_tokens": int((batch["labels"] != -100).sum()),
            "padded_tokens": int(batch["input_ids"].numel()),
            "routes": module.training_routes if isinstance(module, StagedMixture)
            else {"mode": "dense-training", "expert_evaluations_per_token": 1}}


def probe(model, prepared):
    """Teacher-forced training-only diagnostic; never a generated-answer score."""
    model.train()
    before = time.process_time()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for batch in prepared:
            loss = model(**batch).loss
            count = int((batch["labels"][:, 1:] != -100).sum())
            total += float(loss) * count
            tokens += count
    if not tokens or not torch.isfinite(torch.tensor(total)):
        raise ValueError("Invalid training probe")
    return {"loss": total / tokens, "cpu_seconds": time.process_time() - before, "answer_tokens": tokens}


def write_history(path, history):
    save(path, history)


def train_expansion(model, tokenizer, rows, spec, home, binding):
    module = install(model, expansion=True)
    incumbent_before = tensor_identity(module.incumbent)
    added_before = tensor_identity(module.added)
    model.config.use_cache = False
    module.set_phase("expert")
    new_probe = batches(rows["expert_new"][:spec["training"]["probe_documents"]], tokenizer, spec)
    before = probe(model, new_probe)
    history, checkpoints = [], {}
    for phase in ("expert", "gate"):
        module.set_phase(phase)
        optimizer = optimizer_for(model, spec)
        prepared = batches(rows[phase + "_new"] + rows[phase + "_replay"], tokenizer, spec)
        for step in range(spec["training"][phase + "_steps"]):
            receipt = optimize_step(model, prepared[step % len(prepared)], optimizer, spec)
            history.append({"phase": phase, "step": step, **receipt})
            # Persist progress; a timeout must not erase all evidence of spend.
            write_history(home / "expansion-training.json", history)
        if phase == "expert":
            after = probe(model, new_probe)
            added_after_expert = tensor_identity(module.added)
        checkpoints[phase] = write_checkpoint(home / "checkpoints" / phase, module, optimizer,
                                              binding=binding, phase=phase, step=step + 1)
    changed = added_before != added_after_expert
    trained = (changed and after["loss"] <= before["loss"]
               * (1 - spec["training"]["minimum_relative_probe_loss_reduction"]))
    train_cpu = sum(row["cpu_seconds"] for row in history)
    probe_cpu = before["cpu_seconds"] + after["cpu_seconds"]
    return {"training_cpu_seconds": train_cpu, "probe_cpu_seconds": probe_cpu,
            "comparison_cpu_seconds": train_cpu + probe_cpu,
            "training_wall_seconds": sum(row["wall_seconds"] for row in history),
            "training_input_tokens": sum(row["input_tokens"] for row in history),
            "training_answer_tokens": sum(row["answer_tokens"] for row in history),
            "steps": len(history), "probe_before": before, "probe_after": after,
            "expert_training_signal": trained, "added_changed": changed,
            "incumbent_unchanged": incumbent_before == tensor_identity(module.incumbent),
            "added_unchanged_during_gate": added_after_expert == tensor_identity(module.added),
            "checkpoints": checkpoints}


def control_stream(rows, tokenizer, spec):
    result = []
    for phase in ("expert", "gate"):
        prepared = batches(rows[phase + "_new"] + rows[phase + "_replay"], tokenizer, spec)
        result.extend(prepared[step % len(prepared)] for step in range(spec["training"][phase + "_steps"]))
    return result


def train_control(model, tokenizer, rows, spec, home, binding, target_cpu):
    if not 0 < target_cpu < spec["budget"]["worker_cpu_seconds"]:
        raise ValueError("Invalid measured control training budget")
    module = install(model, expansion=False)
    for parameter in module.parameters():
        parameter.requires_grad_(True)
    model.config.use_cache = False
    optimizer = optimizer_for(model, spec)
    prepared = control_stream(rows, tokenizer, spec)
    history, spent = [], 0.0
    for step in range(spec["training"]["maximum_control_steps"]):
        receipt = optimize_step(model, prepared[step % len(prepared)], optimizer, spec)
        history.append({"step": step, **receipt})
        spent += receipt["cpu_seconds"]
        write_history(home / "control-training.json", history)
        if spent >= target_cpu:
            break
    checkpoint = write_checkpoint(home / "checkpoints" / "control", module, optimizer,
                                  binding=binding, phase="control", step=len(history))
    return {"training_cpu_seconds": spent, "target_cpu_seconds": target_cpu,
            "overshoot_cpu_seconds": max(0, spent - target_cpu), "matched_budget": spent >= target_cpu,
            "training_wall_seconds": sum(row["wall_seconds"] for row in history),
            "training_input_tokens": sum(row["input_tokens"] for row in history),
            "training_answer_tokens": sum(row["answer_tokens"] for row in history),
            "steps": len(history), "checkpoint": checkpoint}


def evaluate(model, tokenizer, rows, spec, home, arm):
    module = last_mlp(model)
    if isinstance(module, StagedMixture):
        module.set_phase("serve")
        module.record_routes = True
    model.eval()
    model.config.use_cache = True
    result = {}
    for role in EVAL_ROLES:
        result[role] = []
        for row in rows[role]:
            ids = tokenizer.apply_chat_template(row["messages"], tokenize=True,
                                                add_generation_prompt=True, return_tensors="pt")
            if ids.shape[1] > spec["training"]["max_length"]:
                raise ValueError("Evaluation prompt exceeds the frozen context budget")
            if isinstance(module, StagedMixture):
                module.trace = []
            started = time.monotonic()
            with torch.inference_mode():
                output = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                        max_new_tokens=spec["training"]["generation_tokens"],
                                        pad_token_id=tokenizer.eos_token_id)
            seconds = time.monotonic() - started
            text = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
            trace = module.trace if isinstance(module, StagedMixture) else []
            added = sum(sum(call["answer_choices"]) for call in trace)
            result[role].append({"id": row["id"], "text": text, "answer": row["answer"],
                                 "passed": text.strip() == row["answer"], "seconds": seconds,
                                 "generated_tokens": int(output.shape[1] - ids.shape[1]),
                                 "added_answer_tokens": added, "automatic": True,
                                 "routes": trace})
            save(home / (arm + "-answers.json"), result)
    return result


def read_receipt(home, arm, binding):
    receipt = json.loads((Path(home) / (arm + ".json")).read_text())
    if receipt.get("binding") != binding or receipt.get("arm") != arm:
        raise ValueError("Arm receipt belongs to another experiment")
    return receipt


def worker(arm, seed, home):
    if arm not in ARMS:
        raise ValueError("Unknown isolated arm")
    spec = load_spec()
    freeze = bind_freeze(committed=True)
    data = load_data(spec)
    if sys.platform != "linux":
        raise ValueError("The frozen process-memory profile requires Linux")
    torch.set_num_threads(spec["budget"]["threads"])
    torch.set_num_interop_threads(1)
    torch.manual_seed(spec["training"]["seed"])
    random.seed(spec["training"]["seed"])
    resource.setrlimit(resource.RLIMIT_CPU, (spec["budget"]["worker_cpu_seconds"],
                                           spec["budget"]["worker_cpu_seconds"] + 1))
    started = time.monotonic()
    home = Path(home)
    model, tokenizer = load_parent(seed)
    binding = {"contract": identity(spec), "freeze": freeze, "data": identity(data),
               "tokenizer": tokenizer_identity(tokenizer)}
    if arm != "baseline":
        baseline = read_receipt(home, "baseline", binding)
        protected = [row["id"] for row in baseline["retention"] if row["passed"]]
        if len(protected) < spec["gates"]["minimum_parent_retention_correct"]:
            raise ValueError("Baseline has too few correct protected answers; do not train")
    if arm == "expansion-train":
        result = train_expansion(model, tokenizer, data["roles"], spec, home, binding)
    elif arm == "control-train":
        candidate = read_receipt(home, "expansion-train", binding)
        result = train_control(model, tokenizer, data["roles"], spec, home, binding,
                               candidate["comparison_cpu_seconds"])
    else:
        if arm != "baseline":
            expanded = arm == "expansion-evaluate"
            module = install(model, expansion=expanded)
            old_root = tensor_identity(module.incumbent) if expanded else None
            phase = "gate" if expanded else "control"
            manifest = read_checkpoint(home / "checkpoints" / phase, module, binding=binding, phase=phase)
            training = read_receipt(home, "expansion-train" if expanded else "control-train", binding)
            expected = training["checkpoints"][phase] if expanded else training["checkpoint"]
            if identity(manifest) != expected:
                raise ValueError("Evaluation checkpoint differs from the recorded trained module")
            if expanded and old_root != tensor_identity(module.incumbent):
                raise ValueError("Accepted MLP was modified")
        result = evaluate(model, tokenizer, data["roles"], spec, home, arm)
    result.update({"format": FORMAT + "/arm", "arm": arm, "binding": binding,
                   "process_cpu_seconds": time.process_time(),
                   "workload_wall_seconds": time.monotonic() - started,
                   "peak_rss_bytes": peak_rss_bytes(), "environment": environment(),
                   "gpu_launch_authorized": False, "admission_evidence": False})
    save(home / (arm + ".json"), result)
    return result


def run_isolated(arm, seed, home, seconds):
    """Each arm gets a new interpreter, allocator and RSS high-water mark."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONPATH=str(root() / "src"))
    command = [sys.executable, str(root() / "scripts/run_staged_integration.py"), "--worker", arm,
               "--seed", str(Path(seed).resolve()), "--home", str(Path(home).resolve())]
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    outcome = "failed"
    try:
        with (Path(home) / (arm + ".log")).open("x") as log:
            subprocess.run(command, env=env, cwd=root(), check=True, timeout=seconds,
                           stdout=log, stderr=subprocess.STDOUT)
        outcome = "completed"
    finally:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        save(Path(home) / (arm + "-launch.json"), {
            "arm": arm, "outcome": outcome, "wall_seconds": time.monotonic() - started,
            "cpu_seconds": after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
            "scope": "complete isolated child process, including imports, loading and artifact writes",
        })
    return json.loads((Path(home) / (arm + ".json")).read_text())


def run_study(seed, home):
    spec = load_spec()
    freeze = bind_freeze(committed=True)
    verify(seed)
    home = Path(home)
    home.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    save(home / "study.json", {"contract": identity(spec), "freeze": freeze,
                               "gpu_launch_authorized": False, "admission_evidence": False})
    receipts = {}
    try:
        for arm in ARMS:
            remaining = spec["budget"]["total_wall_seconds"] - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("CPU study wall budget exhausted")
            receipts[arm] = run_isolated(arm, seed, home, min(remaining, spec["budget"]["worker_wall_seconds"]))
            if receipts[arm]["peak_rss_bytes"] > spec["gates"]["maximum_peak_rss_bytes"]:
                raise ValueError("Process exceeded the declared peak memory gate")
            if arm == "baseline":
                protected = [row["id"] for row in receipts[arm]["retention"] if row["passed"]]
                save(home / "protected-before-training.json", {"ids": protected, "baseline": identity(receipts[arm])})
                if len(protected) < spec["gates"]["minimum_parent_retention_correct"]:
                    result = {"passed": False, "next": "stop-baseline-uninformative",
                              "protected": protected, "admission_evidence": False,
                              "gpu_launch_authorized": False, "confirmation_opened": False}
                    save(home / "result.json", result)
                    return result
    except (subprocess.SubprocessError, TimeoutError, ValueError) as error:
        save(home / "failure.json", {"error": str(error), "completed_arms": list(receipts),
                                     "elapsed_wall_seconds": time.monotonic() - started,
                                     "passed": False, "admission_evidence": False,
                                     "gpu_launch_authorized": False})
        raise
    result = score(spec, receipts["baseline"], receipts["expansion-evaluate"], receipts["control-evaluate"],
                   receipts["expansion-train"], receipts["control-train"])
    result["binding"] = receipts["baseline"]["binding"]
    result["receipts"] = {arm: identity(receipt) for arm, receipt in receipts.items()}
    result["launches"] = {arm: json.loads((home / (arm + "-launch.json")).read_text()) for arm in ARMS}
    result["total_process_cpu_seconds"] = sum(row["cpu_seconds"] for row in result["launches"].values())
    result["total_wall_seconds"] = time.monotonic() - started
    save(home / "result.json", result)
    return result
