"""Isolated execution and complete accounting for the block competence study."""
import json
import os
import random
import resource
import subprocess
import sys
import time

import torch
from safetensors.torch import load_file, save_file

from neuroshard.evolution import block_expert as study
from neuroshard.evolution.reference_data import identity, save, sha256, tokenizer_identity
from neuroshard.evolution.seed import verify
from neuroshard.evolution.staged_answer_format import answer_value
from neuroshard.evolution.staged_integration_run import batches, environment, load_parent, peak_rss_bytes


SCRIPT = "scripts/run_block_expert.py"
ARMS = ("baseline", "prepare", "expert-train", "control-train", "expert-evaluate", "control-evaluate")


def binding(plan, tokenizer):
    return {"contract": identity(plan), "freeze": study.bind_freeze(committed=True),
            "data": identity(study.load_data(plan)), "tokenizer": tokenizer_identity(tokenizer)}


def read_receipt(home, arm, expected):
    receipt = json.loads((home / (arm + ".json")).read_text())
    if receipt["binding"] != expected or receipt["arm"] != arm:
        raise ValueError("Receipt belongs to another experiment or arm")
    return receipt


def verify_cached_execution(model, batch, prefixes, count, tolerance):
    """Check both training paths against full forward before optimizing anything."""
    original, depth = model.model.layers, model.config.num_hidden_layers
    model.eval()
    with torch.no_grad():
        parent, _ = study.answer_logits(model, batch)
        errors = {}
        for name in ("control", "expert"):
            blocks = study.install_blocks(model, count, expansion=name == "expert")
            full, _ = study.answer_logits(model, batch)
            if not torch.equal(parent, full):
                raise ValueError("Initial blocks changed parent logits")
            model.model.layers = blocks
            model.config.num_hidden_layers = count
            cached, _ = study.answer_logits(model, batch, hidden=prefixes[name])
            errors[name] = float((full - cached).abs().max())
            if not torch.allclose(full, cached, atol=tolerance, rtol=0):
                raise ValueError("Cached training differs from full forward")
            # install_blocks extends the ModuleList in place; remove appended blocks.
            model.model.layers = torch.nn.ModuleList(list(original[:depth]))
            original = model.model.layers
            model.config.num_hidden_layers = depth
    model.requires_grad_(False)
    return {"initial_parent_logits_exact": True, "maximum_logit_error": errors}


def prepare(model, tokenizer, data, plan, home, bound):
    model.eval()
    model.requires_grad_(False)
    prepared = batches(data["roles"]["train_new"] + data["roles"]["train_replay"], tokenizer, plan)
    cache, checks = [], None
    for index, batch in enumerate(prepared):
        prefixes = study.capture_prefixes(model, batch, plan["architecture"]["blocks"])
        if index == 0:
            checks = verify_cached_execution(model, batch, prefixes, plan["architecture"]["blocks"],
                                             plan["training"]["cache_logit_atol"])
        cache.append({**batch, **prefixes})
        save(home / "preparation-progress.json", {"completed_batches": index + 1,
                                                  "total_batches": len(prepared)})
    torch.save(cache, home / "training-cache.pt")
    return {"cache_sha256": sha256(home / "training-cache.pt"), "batches": len(cache),
            "examples": sum(len(item["input_ids"]) for item in cache), "checks": checks,
            "expert_extension_cpu_seconds": sum(item["expert_extension_cpu_seconds"] for item in cache),
            "cache_binding": bound, "cache_bytes": (home / "training-cache.pt").stat().st_size,
            "cache_scope": "training roles only; both arms share one billed preparation"}


def checkpoint(directory, blocks, optimizer, bound, arm, steps):
    directory.mkdir(parents=True, exist_ok=False)
    save_file({name: value.detach().contiguous().clone() for name, value in blocks.state_dict().items()},
              str(directory / "weights.safetensors"))
    torch.save({"optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate()}, directory / "training.pt")
    manifest = {"format": study.FORMAT + "/checkpoint", "binding": bound, "arm": arm, "steps": steps,
                "module": study.parameters_identity(blocks.named_parameters()),
                "files": {name: sha256(directory / name) for name in ("weights.safetensors", "training.pt")}}
    save(directory / "manifest.json", manifest)
    return identity(manifest)


def restore(directory, blocks, bound, arm, expected):
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest["binding"] != bound or manifest["arm"] != arm or identity(manifest) != expected
            or set(manifest["files"]) != {"weights.safetensors", "training.pt"}):
        raise ValueError("Checkpoint identity, arm or binding differs")
    for name, digest in manifest["files"].items():
        if sha256(directory / name) != digest:
            raise ValueError("Checkpoint bytes changed")
    blocks.load_state_dict(load_file(str(directory / "weights.safetensors")), strict=True)
    if study.parameters_identity(blocks.named_parameters()) != manifest["module"]:
        raise ValueError("Restored tensors differ")


def train(model, plan, home, bound, arm):
    preparation = read_receipt(home, "prepare", bound)
    if sha256(home / "training-cache.pt") != preparation["cache_sha256"]:
        raise ValueError("Frozen-prefix cache changed")
    cache = torch.load(home / "training-cache.pt", map_location="cpu", weights_only=True)
    expanded = arm == "expert-train"
    blocks = study.install_blocks(model, plan["architecture"]["blocks"], expansion=expanded)
    frozen = [(name, parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad]
    frozen_before = study.parameters_identity(frozen)
    trainable_before = study.parameters_identity(blocks.named_parameters())
    # Frozen prefix activations replace exactly that prefix. Its parameters stay
    # referenced for the post-training preservation check.
    model.model.layers = blocks
    model.config.num_hidden_layers = len(blocks)
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(blocks.parameters(), lr=plan["training"]["learning_rate"],
                                  weight_decay=plan["training"]["weight_decay"])
    target = None
    if not expanded:
        read_receipt(home, "expert-train", bound)
        launch = json.loads((home / "expert-train-launch.json").read_text())
        if launch["outcome"] != "completed" or launch["cpu_seconds"] <= 0:
            raise ValueError("Control requires the complete expert training bill")
        target = launch["cpu_seconds"] + preparation["expert_extension_cpu_seconds"]
    steps = plan["training"]["steps"] if expanded else plan["training"]["maximum_control_steps"]
    history, spent = [], 0.0
    model.train()
    for step in range(steps):
        batch = cache[step % len(cache)]
        rate = plan["training"]["learning_rate"] * min(1.0, (step + 1) / plan["training"]["warmup_steps"])
        for group in optimizer.param_groups:
            group["lr"] = rate
        started = time.process_time()
        optimizer.zero_grad(set_to_none=True)
        value = study.loss(model, batch, hidden=batch["expert" if expanded else "control"])
        if not torch.isfinite(value):
            raise ValueError("Non-finite training loss")
        value.backward()
        norm = torch.nn.utils.clip_grad_norm_(blocks.parameters(), plan["training"]["clip_norm"],
                                             error_if_nonfinite=True)
        optimizer.step()
        cpu = time.process_time() - started
        spent += cpu
        history.append({"step": step, "batch": step % len(cache), "loss": float(value.detach()),
                        "learning_rate": rate, "gradient_norm": float(norm), "cpu_seconds": cpu,
                        "input_tokens": int(batch["attention_mask"].sum()),
                        "answer_tokens": int((batch["labels"][:, 1:] != -100).sum())})
        save(home / (arm + "-history.json"), history)
        if peak_rss_bytes() > plan["budget"]["maximum_peak_rss_bytes"]:
            raise ValueError("Training exceeded the memory budget")
        if target is not None and spent >= target:
            break
    root = checkpoint(home / "checkpoints" / arm, blocks, optimizer, bound, arm, len(history))
    return {"steps": len(history), "optimization_cpu_seconds": spent, "target_cpu_seconds": target,
            "matched_budget": target is None or spent >= target,
            "overshoot_cpu_seconds": max(0, spent - target) if target is not None else 0,
            "frozen_unchanged": frozen_before == study.parameters_identity(frozen),
            "frozen_parameters_root": frozen_before,
            "expert_changed": trainable_before != study.parameters_identity(blocks.named_parameters()),
            "trainable_parameters": sum(p.numel() for p in blocks.parameters()), "checkpoint": root,
            "input_tokens": sum(r["input_tokens"] for r in history),
            "answer_tokens": sum(r["answer_tokens"] for r in history)}


def evaluate(model, tokenizer, data, plan, home, arm):
    model.eval()
    model.config.use_cache = True
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [tokenizer.eos_token_id])
    result = {}
    for role in study.EVAL_ROLES:
        result[role] = []
        for row in data["roles"][role]:
            ids = tokenizer.apply_chat_template(row["messages"], tokenize=True,
                                                add_generation_prompt=True, return_tensors="pt")
            if ids.shape[1] > plan["training"]["max_length"]:
                raise ValueError("Prompt exceeds context budget")
            started = time.monotonic()
            with torch.inference_mode():
                output = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                        max_new_tokens=plan["training"]["generation_tokens"],
                                        pad_token_id=tokenizer.eos_token_id)
            elapsed = time.monotonic() - started
            generated = output[0, ids.shape[1]:]
            terminated = int(generated[-1]) in eos_ids
            text = tokenizer.decode(generated, skip_special_tokens=True)
            parsed = answer_value(text, row, terminated=terminated)
            result[role].append({"id": row["id"], "text": text, "answer": row["answer"],
                                 "parsed_answer": parsed, "terminated": terminated,
                                 "passed": parsed == row["answer"], "seconds": elapsed,
                                 "generated_tokens": len(generated), "selection": "declared-arm"})
            save(home / (arm + "-answers.json"), result)
    return result


def worker(arm, seed, home):
    plan = study.load_plan()
    study.bind_freeze(committed=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(plan["training"]["seed"])
    random.seed(plan["training"]["seed"])
    limit = plan["budget"]["arms"][arm]
    resource.setrlimit(resource.RLIMIT_CPU, (limit, limit + 1))
    started = time.monotonic()
    model, tokenizer = load_parent(seed)
    bound, data = binding(plan, tokenizer), study.load_data(plan)
    runtime = environment()
    if runtime["packages"] != plan["runtime"]["packages"] or runtime["torch_threads"] != 1:
        raise ValueError("CPU numerical runtime differs from the declared environment")
    if arm == "prepare":
        result = prepare(model, tokenizer, data, plan, home, bound)
    elif arm.endswith("-train"):
        result = train(model, plan, home, bound, arm)
    else:
        if arm != "baseline":
            training_arm = arm.replace("-evaluate", "-train")
            trained = read_receipt(home, training_arm, bound)
            blocks = study.install_blocks(model, plan["architecture"]["blocks"],
                                          expansion=arm == "expert-evaluate")
            restore(home / "checkpoints" / training_arm, blocks, bound, training_arm, trained["checkpoint"])
        result = evaluate(model, tokenizer, data, plan, home, arm)
    result.update({"format": study.FORMAT + "/arm", "arm": arm, "binding": bound,
                   "environment": runtime, "process_cpu_seconds": time.process_time(),
                   "wall_seconds": time.monotonic() - started, "peak_rss_bytes": peak_rss_bytes(),
                   "admission_evidence": False, "gpu_launch_authorized": False})
    save(home / (arm + ".json"), result)
    return result


def isolated(arm, seed, home, seconds):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", ATEN_CPU_CAPABILITY="default", MKL_ENABLE_INSTRUCTIONS="SSE4_2",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONPATH=str(study.root() / "src"))
    command = [sys.executable, str(study.root() / SCRIPT), "--worker", arm,
               "--seed", str(seed.resolve()), "--home", str(home.resolve())]
    before, started = resource.getrusage(resource.RUSAGE_CHILDREN), time.monotonic()
    outcome = "failed"
    try:
        with (home / (arm + ".log")).open("x") as log:
            subprocess.run(command, env=env, cwd=study.root(), timeout=seconds, check=True,
                           stdout=log, stderr=subprocess.STDOUT)
        outcome = "completed"
    finally:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        save(home / (arm + "-launch.json"), {"arm": arm, "outcome": outcome,
             "wall_seconds": time.monotonic() - started,
             "cpu_seconds": after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
             "scope": "entire child process, including imports, cache IO and checkpointing"})
    return json.loads((home / (arm + ".json")).read_text())


def run(seed, home):
    plan, freeze = study.load_plan(), study.bind_freeze(committed=True)
    verify(seed)
    home.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.monotonic(), time.process_time()
    save(home / "study.json", {"plan": identity(plan), "freeze": freeze,
         "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=study.root()).decode().strip(),
         "admission_evidence": False, "gpu_launch_authorized": False})
    receipts = {}
    try:
        for arm in ARMS:
            remaining = plan["budget"]["total_wall_seconds"] - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Total study budget exhausted")
            receipts[arm] = isolated(arm, seed, home, min(remaining, plan["budget"]["arms"][arm]))
            if receipts[arm]["peak_rss_bytes"] > plan["budget"]["maximum_peak_rss_bytes"]:
                raise ValueError("Worker exceeded memory budget")
            if arm == "baseline":
                protected = [row["id"] for row in receipts[arm]["retention"] if row["passed"]]
                save(home / "protected-before-training.json", {"ids": protected,
                                                              "baseline": identity(receipts[arm])})
                if len(protected) < plan["gates"]["minimum_protected"]:
                    raise ValueError("Baseline has too few protected answers; no training authorized")
            if arm == "control-train" and not receipts[arm]["matched_budget"]:
                raise ValueError("Control failed to spend the matched optimization budget")
        target = (json.loads((home / "expert-train-launch.json").read_text())["cpu_seconds"]
                  + receipts["prepare"]["expert_extension_cpu_seconds"])
        result = study.score(plan, study.load_data(plan), receipts["baseline"], receipts["expert-evaluate"],
                             receipts["control-evaluate"], receipts["expert-train"],
                             receipts["control-train"], target)
    except (ValueError, TimeoutError, subprocess.SubprocessError) as error:
        result = {"passed": False, "error": str(error), "next": "stop-execution-no-quality-conclusion",
                  "completed_arms": list(receipts), "admission_evidence": False,
                  "gpu_launch_authorized": False, "selector_training_authorized": False}
    launches = {arm: json.loads((home / (arm + "-launch.json")).read_text())
                for arm in ARMS if (home / (arm + "-launch.json")).exists()}
    result.update({"contract": identity(plan), "freeze": freeze,
                   "receipts": {arm: identity(value) for arm, value in receipts.items()},
                   "launches": launches, "wall_seconds": time.monotonic() - started,
                   "coordinator_cpu_seconds": time.process_time() - cpu_started,
                   "child_cpu_seconds": sum(launch["cpu_seconds"] for launch in launches.values()),
                   "preparation_accounting": "shared prefix charged once; expert-only prefix extension also credited to control optimization target; no evaluation cache"})
    save(home / "result.json", result)
    return result
