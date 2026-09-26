"""Bounded A1 qualification using upstream generation and published adapters.

This experiment deliberately does not implement another decoder or train a router.
Explicit reference invocation is not evidence of automatic skill composition.
"""

import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
import time

from neuroshard.evolution.modular_reference_execution import (
    ROOT, file_state, identity, launch, read, save, sha256, verify_artifacts,
)

PLAN = "config/experiments/granite-reference.json"
EXECUTION = "config/experiments/granite-reference-execution.json"
ARTIFACTS = "config/experiments/granite-reference-artifacts.json"
SCRIPT = "scripts/run_granite_reference.py"
CATEGORIES = ("conversation", "instruction", "tool-use")


def committed_sources(root=ROOT):
    contract = read(root / EXECUTION)
    for name, expected in contract["contracts"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"changed execution contract: {name}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    sources = {}
    for name in contract["sources"]:
        data = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root)
        if data != (root / name).read_bytes():
            raise ValueError(f"uncommitted execution source: {name}")
        sources[name] = sha256(root / name)
    return {"commit": commit, "sources": sources}


def configure():
    if "torch" in sys.modules:
        raise ValueError("configure the research runtime before importing torch")
    for key, value in read(ROOT / EXECUTION)["environment"].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def upstream_path():
    path = Path(os.environ.get("NEUROSHARD_GRANITE_SOURCE", ROOT / ".upstream/granite-switch")).resolve()
    upstream = read(ROOT / ARTIFACTS)["upstream"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    actual = {str(p.relative_to(path)) for p in (path / "src/granite_switch").rglob("*.py")}
    if commit != upstream["commit"] or actual != set(upstream["sources"]):
        raise ValueError("upstream revision or source inventory differs")
    for name, expected in upstream["sources"].items():
        if sha256(path / name) != expected:
            raise ValueError(f"upstream source changed: {name}")
    return path


def freeze():
    binding = committed_sources()
    execution = read(ROOT / EXECUTION)
    packages = {p: importlib.metadata.version(p) for p in execution["packages"]}
    if packages != execution["packages"] or platform.python_version() != execution["python"]:
        raise ValueError("research runtime differs from freeze")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("research runtime requires Linux x86_64")
    cpu = Path("/proc/cpuinfo").read_text()
    if any(flag not in cpu.split() for flag in execution["required_cpu_flags"]):
        raise ValueError("CPU lacks frozen instructions")
    if any(os.environ.get(k) != v for k, v in execution["environment"].items()):
        raise ValueError("research numerical environment differs")
    upstream_path()
    return {**binding, "packages": packages, "python": platform.python_version(),
            "upstream": read(ROOT / ARTIFACTS)["upstream"]["commit"],
            "cpu_models": sorted({line.split(":", 1)[1].strip() for line in cpu.splitlines()
                                  if line.startswith("model name")})}


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("nonfinite JSON constant")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def same_value(left, right):
    """JSON booleans cannot masquerade as integers; integral floats are numbers."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(left) and math.isfinite(right) and left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_value(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(same_value(a, b) for a, b in zip(left, right))
    return left == right


def score(task, text, terminated):
    if not terminated:
        return False
    text = text.strip().replace("\r\n", "\n")
    if task["kind"] == "exact":
        return text in task["accept"]
    if task["kind"] == "tool":
        match = re.fullmatch(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
        if not match:
            return False
        text = match.group(1)
    try:
        return same_value(strict_json(text), task["expected"])
    except (ValueError, TypeError, OverflowError):
        return False


def baseline_gate(plan, rows):
    expected = {t["id"]: t for t in plan["tasks"]}
    if len(rows) != len(expected) or {r["id"] for r in rows} != set(expected):
        return False
    return (sum(r["passed"] for r in rows) >= plan["quality"]["baseline_minimum"]
            and all(sum(r["passed"] for r in rows if expected[r["id"]]["category"] == c)
                    >= plan["quality"]["minimum_per_category"] for c in CATEGORIES))


def assess(plan, baseline, modular):
    tasks = {t["id"]: t for t in plan["tasks"] + plan["reference_tasks"]}
    groups = {}
    for name, rows in (("baseline", baseline), ("modular", modular)):
        if len({r["id"] for r in rows}) != len(rows) or any(r["id"] not in tasks for r in rows):
            raise ValueError("duplicate or unknown reply")
        for row in rows:
            if row["passed"] != score(tasks[row["id"]], row["text"], row["terminated"]):
                raise ValueError("reply rescore differs")
        groups[name] = {r["id"]: r for r in rows}
    complete = all(set(g) == set(tasks) for g in groups.values())
    protected = [t["id"] for t in plan["tasks"] if groups["baseline"].get(t["id"], {}).get("passed")]
    lost = [i for i in protected if not groups["modular"].get(i, {}).get("passed")]
    reference = plan["reference_tasks"]
    successes = {name: sum(g.get(t["id"], {}).get("passed", False) for t in reference)
                 for name, g in groups.items()}
    ref_losses = [t["id"] for t in reference
                  if groups["baseline"].get(t["id"], {}).get("passed")
                  and not groups["modular"].get(t["id"], {}).get("passed")]
    reference_gate = (successes["modular"] >= plan["quality"]["reference_minimum"]
                      and len(ref_losses) <= plan["quality"]["reference_maximum_losses_vs_prompted_base"]
                      and all(sum(groups["modular"].get(t["id"], {}).get("passed", False)
                                  for t in reference if t["expected"]["score"] == label)
                              >= plan["quality"]["reference_minimum_per_label"] for label in ("yes", "no")))
    durations = sorted(r["seconds"] for r in baseline + modular)
    p95 = durations[math.ceil(.95 * len(durations)) - 1] if durations else None
    base_ok = baseline_gate(plan, [r for r in baseline if r["id"] in {t["id"] for t in plan["tasks"]}])
    return {"baseline_gate": base_ok, "complete": complete, "protected_ids": protected,
            "lost_ids": lost, "reference_correct": successes, "reference_lost_ids": ref_losses,
            "reference_gate": reference_gate, "p95_seconds": p95,
            "quality_gate": bool(complete and base_ok and not lost and reference_gate
                                 and p95 <= plan["quality"]["p95_seconds"]),
            "automatic_composition_proven": False, "checklist_credit": False}


def load_model(model_dir, which):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if which == "modular":
        sys.path.insert(0, str(upstream_path() / "src"))
        import granite_switch.hf  # noqa: F401 -- registers the pinned architecture
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    model, info = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True,
        trust_remote_code=False, output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"checkpoint loading mismatch: {info}")
    expected = read(ROOT / ARTIFACTS)["models"][which]["parameters"]
    if len(model.model.layers) != 40 or sum(p.numel() for p in model.parameters()) != expected:
        raise ValueError("checkpoint architecture/parameter inventory mismatch")
    return model.eval(), tokenizer


def generate(model, tokenizer, plan, task, which):
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

    kwargs = {"tools": task.get("tools"), "add_generation_prompt": True, "tokenize": False}
    adapter = task.get("adapter") if which == "modular" else None
    if adapter:
        kwargs["adapter_name"] = adapter
    prompt = tokenizer.apply_chat_template(task["messages"], **kwargs)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    if inputs.input_ids.shape[-1] > plan["generation"]["max_input_tokens"]:
        raise ValueError("prompt exceeds frozen input cap")
    allowed_adapter = model.config.adapter_names.index(adapter) + 1 if adapter else 0
    route_counts = {}
    route_trace = []

    def route_hook(module, _args, output):
        routes = output[0]
        route_trace.append(routes.tolist())
        for value, count in zip(*torch.unique(routes, return_counts=True)):
            route = int(value.item())
            if route not in (0, allowed_adapter):
                raise ValueError("unexpected adapter activation")
            route_counts[str(route)] = route_counts.get(str(route), 0) + int(count.item())

    handle = model.model.switch.register_forward_hook(route_hook) if which == "modular" else None
    started = time.monotonic()
    first_token = None

    class Observe(StoppingCriteria):
        def __call__(self, input_ids, scores, **unused):
            nonlocal first_token
            if first_token is None:
                first_token = time.monotonic() - started
            return False

    class CheckLogits(LogitsProcessor):
        def __call__(self, input_ids, scores):
            if torch.isnan(scores).any() or torch.isposinf(scores).any() or not torch.isfinite(scores).any():
                raise ValueError("nonfinite model logits")
            return scores

    cap = plan["generation"]["reference_max_new_tokens" if task["category"] == "reference" else "max_new_tokens"]
    try:
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, num_beams=1, use_cache=True,
                                    max_new_tokens=cap, stopping_criteria=StoppingCriteriaList([Observe()]),
                                    logits_processor=LogitsProcessorList([CheckLogits()]))
    finally:
        if handle:
            handle.remove()
    token_ids = output[0, inputs.input_ids.shape[-1]:].tolist()
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    terminated = bool(token_ids and token_ids[-1] in eos)
    # Strip only a terminal EOS, retaining every other control token for strict scoring.
    text = tokenizer.decode(token_ids[:-1] if terminated else token_ids, skip_special_tokens=False)
    if adapter and str(allowed_adapter) not in route_counts:
        raise ValueError("requested adapter never activated")
    return {"id": task["id"], "category": task["category"], "model": which,
            "task_sha256": identity(task), "prompt_sha256": identity(prompt),
            "input_token_ids": inputs.input_ids[0].tolist(), "token_ids": token_ids,
            "text": text, "terminated": terminated, "passed": score(task, text, terminated),
            "seconds": time.monotonic() - started, "first_token_seconds": first_token,
            "route_counts": route_counts, "route_trace": route_trace,
            "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}


def worker(request_path):
    configure()
    request_path = Path(request_path)
    request = read(request_path)
    binding = freeze()
    if binding != request["freeze"]:
        raise ValueError("worker source/runtime differs")
    import torch
    torch.set_num_threads(read(ROOT / PLAN)["resources"]["threads"])
    torch.set_num_interop_threads(1)
    plan = read(ROOT / PLAN)
    results = {"baseline": [], "modular": []}
    replays = []
    checkpoint_states = {}
    reply = {"binding": request["binding"], "execution_completed": False, "checklist_credit": False}
    started = time.monotonic()
    try:
        for which in ("baseline", "modular"):
            inventory = read(ROOT / ARTIFACTS)["models"][which]
            model_dir = Path(request["models"]) / which
            state = verify_artifacts(model_dir, inventory, download=True)
            checkpoint_states[which] = state
            model, tokenizer = load_model(model_dir, which)
            for task in plan["tasks"] + plan["reference_tasks"]:
                if task["category"] == "reference" and not baseline_gate(plan, results[which][:len(plan["tasks"])]):
                    break
                row = generate(model, tokenizer, plan, task, which)
                results[which].append(row)
                save(request_path.parent / f"{which}-{task['id']}.json", row, exclusive=True)
                save(request_path.parents[2] / "status.json", {"state": "generating", "model": which,
                     "completed": len(results[which]), "last_id": task["id"]})
            if file_state(model_dir, inventory) != state:
                raise ValueError("model artifacts changed during generation")
            del model, tokenizer
            import gc
            gc.collect()
            if which == "baseline" and not baseline_gate(plan, results[which][:len(plan["tasks"])]):
                reply["stop_reason"] = "baseline quality failed; modular checkpoint not downloaded"
                break
        report = assess(plan, results["baseline"], results["modular"])
        # No repeated runs to rescue a failed gate. Independent model reload on success only.
        if report["quality_gate"]:
            for which in ("baseline", "modular"):
                model_dir = Path(request["models"]) / which
                model, tokenizer = load_model(model_dir, which)
                for task in plan["tasks"] + plan["reference_tasks"]:
                    if task["id"] not in plan["replay_ids"]:
                        continue
                    row = generate(model, tokenizer, plan, task, which)
                    original = next(r for r in results[which] if r["id"] == task["id"])
                    row["matches"] = all(row[k] == original[k] for k in (
                        "prompt_sha256", "token_ids", "text", "terminated", "passed", "route_trace"))
                    replays.append(row)
                    save(request_path.parent / f"replay-{which}-{task['id']}.json", row, exclusive=True)
                    if not row["matches"]:
                        raise ValueError("independent reload did not reproduce generation")
                if file_state(model_dir, read(ROOT / ARTIFACTS)["models"][which]) != checkpoint_states[which]:
                    raise ValueError("model artifacts changed during replay")
                del model, tokenizer
                gc.collect()
        reply.update(execution_completed=True, report=report, replays=replays,
                     reference_passed=bool(report["quality_gate"] and len(replays) == 6))
    except Exception as error:
        reply["error"] = str(error)
    finally:
        reply.update(results=results, wall_seconds=time.monotonic() - started,
                     process_cpu_seconds=time.process_time(),
                     max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        save(request_path.parent / "reply.json", reply, exclusive=True)


def run(home, models):
    configure()
    source = freeze()
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    binding = {"freeze": source, "profile": "granite-reference", "plan_sha256": sha256(ROOT / PLAN)}
    save(home / "binding.json", binding, exclusive=True)
    plan = read(ROOT / PLAN)
    result = {"execution_completed": False, "binding": binding, "checklist_credit": False}
    try:
        result.update(launch(home, models, binding, "pair", "comparison",
                             plan["resources"]["worker_seconds"], plan["resources"]["memory_bytes"],
                             worker_script=SCRIPT))
    except Exception as error:
        result["error"] = str(error)
    save(home / "result.json", result, exclusive=True)
    return result
