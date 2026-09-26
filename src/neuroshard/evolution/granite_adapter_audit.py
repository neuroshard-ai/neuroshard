"""Opened-case audit of published aLoRA versus its composed Switch checkpoint.

This is an implementation comparison, never an admission or a quality rerun.
"""

import gc
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

from neuroshard.evolution import granite_reference as reference
from neuroshard.evolution.modular_reference_execution import (
    ROOT, file_state, identity, launch, read, save, sha256, verify_artifacts,
)

PLAN = "config/experiments/granite-adapter-audit.json"
EXECUTION = "config/experiments/granite-adapter-audit-execution.json"
SCRIPT = "scripts/run_granite_adapter_audit.py"
configure = reference.configure


def committed_sources(root=ROOT):
    execution = read(root / EXECUTION)
    for name, digest in execution["contracts"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"changed audit contract: {name}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    for name in execution["sources"]:
        if subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root) != (root / name).read_bytes():
            raise ValueError(f"uncommitted audit source: {name}")
    return {"commit": commit, "sources": {name: sha256(root / name) for name in execution["sources"]}}


def freeze():
    binding = committed_sources()
    execution = read(ROOT / EXECUTION)
    packages = {name: importlib.metadata.version(name) for name in execution["packages"]}
    if packages != execution["packages"] or platform.python_version() != execution["python"]:
        raise ValueError("audit runtime differs")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("audit requires Linux x86_64")
    cpu = Path("/proc/cpuinfo").read_text()
    if any(flag not in cpu.split() for flag in execution["required_cpu_flags"]):
        raise ValueError("audit CPU lacks frozen instructions")
    if any(os.environ.get(key) != value for key, value in execution["environment"].items()):
        raise ValueError("audit numerical environment differs")
    reference.upstream_path()
    return {**binding, "packages": packages, "python": platform.python_version(),
            "cpu_models": sorted({line.split(":", 1)[1].strip() for line in cpu.splitlines()
                                  if line.startswith("model name")})}


def validate_adapter(config):
    if (config["r"] != 16 or config["lora_alpha"] != 64
            or config["alora_invocation_tokens"] != [27, 71226, 29]
            or set(config["target_modules"]) != {"q_proj", "k_proj", "v_proj", "o_proj"}
            or config["base_model_name_or_path"] != "ibm-granite/granite-4.1-3b"
            or config["rank_pattern"] or config["alpha_pattern"] or config["modules_to_save"]):
        raise ValueError("wrong published adapter configuration")


def verify_auxiliary(home, artifacts):
    """Download only hash-pinned auxiliary files; no executable remote code."""
    from huggingface_hub import hf_hub_download
    home = Path(home)
    for spec in artifacts:
        path = home / spec["filename"]
        if not path.exists():
            hf_hub_download(spec["repo"], spec["filename"], revision=spec["revision"], local_dir=home)
        if path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
            raise ValueError(f"auxiliary artifact mismatch: {spec['filename']}")


def invocation_audit(base_tokenizer, switch_tokenizer, tasks, config, old_rows):
    import torch
    from peft import LoraConfig
    from peft.tuners.lora.variants import calculate_alora_offsets
    validate_adapter(config)
    peft_config = LoraConfig(**config)
    old = {row["id"]: row for row in old_rows}
    rows = []
    for task in tasks:
        kwargs = dict(add_generation_prompt=True, tokenize=True)
        base = base_tokenizer.apply_chat_template(task["messages"], **kwargs)
        switched = switch_tokenizer.apply_chat_template(task["messages"], adapter_name=task["adapter"], **kwargs)
        # Transformers 5 may return a BatchEncoding rather than a bare list.
        base = base if isinstance(base, list) else base["input_ids"]
        switched = switched if isinstance(switched, list) else switched["input_ids"]
        positions = [i for i, token in enumerate(switched) if token == 100362]
        effective = [27 if token == 100362 else token for token in switched]
        default_offset = calculate_alora_offsets({"default": peft_config}, "default", torch.tensor([base]))[0]
        if (len(positions) != 1 or effective != base
                or switched != old[task["id"]]["input_token_ids"]):
            raise ValueError("published invocation does not match recorded effective input")
        start = positions[0]
        expected = [[0] * start + [11] * (len(base) - start)]
        trace = old[task["id"]]["route_trace"]
        if not trace or trace[0] != expected or any(step != [[11]] for step in trace[1:]):
            raise ValueError("recorded adapter activation boundary differs")
        rows.append({"id": task["id"], "effective_input_sha256": identity(base),
                     "prompt_tokens": len(base), "activation_start": start, "alora_offset": len(base) - start,
                     "peft_default_offset": default_offset,
                     "peft_default_activation_matches": default_offset == len(base) - start,
                     "effective_input_exact": True, "recorded_activation_exact": True})
    return rows


def tensor_digest(tensor):
    import torch
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def mapped_base_digests(state, mappings):
    import torch
    sources = [name for row in mappings for name in row["source"]]
    if set(sources) != set(state) or len(sources) != len(set(sources)):
        raise ValueError("published mapping does not cover the base exactly once")
    result = {}
    for row in mappings:
        tensors = [state[name] for name in row["source"]]
        if row["type"] == "direct" and len(tensors) == 1:
            tensor = tensors[0]
        elif row["type"] in ("fused_qkv_proj", "fused_shared_input_linear"):
            tensor = torch.cat(tensors, dim=0)
        else:
            raise ValueError("unknown base projection mapping")
        result[row["target"]] = tensor_digest(tensor)
    if len(result) != len(mappings):
        raise ValueError("duplicate base target mapping")
    return result


def compare_adapter_weights(state, raw, layers=40):
    """Compare published BF16 padding/scaling, not just parameter counts."""
    import torch
    expected_keys, rows, targets = set(), [], set()
    for layer in range(layers):
        prefix = f"model.layers.{layer}.self_attn."
        for projection in ("q", "k", "v", "o"):
            for letter in ("A", "B"):
                key = f"base_model.model.{prefix}{projection}_proj.lora_{letter}.weight"
                expected_keys.add(key)
                target = (f"{prefix}o_proj.lora_{letter}" if projection == "o" else
                          f"{prefix}qkv_proj.lora_{letter}_slices.{('q', 'k', 'v').index(projection)}")
                targets.add(target)
                actual = state[target][10, 0]
                source = raw[key].to(torch.bfloat16)
                expected = torch.zeros_like(actual)
                if letter == "A":
                    expected[:16, :] = source
                else:
                    expected[:, :16] = source * 4
                if not torch.equal(actual, expected):
                    raise ValueError(f"embedded adapter differs from standalone: {target}")
                rows.append({"source": key, "target": target, "sha256": tensor_digest(actual)})
    if set(raw) != expected_keys:
        raise ValueError("standalone adapter tensor inventory differs")
    zero_targets = []
    for name, tensor in state.items():
        if ".lora_" in name and name not in targets:
            if torch.count_nonzero(tensor[10, 0]).item():
                raise ValueError(f"unexpected requirement-check weights: {name}")
            zero_targets.append(name)
    return {"matched": rows, "zero_targets": zero_targets}


def standalone_generate(model, tokenizer, plan, task, start, offset):
    trace = []

    def hook(_module, args, kwargs, _output):
        offsets = kwargs.get("alora_offsets")
        length = args[0].shape[-2]
        if not offsets or len(offsets) != 1 or offsets[0] is None:
            raise ValueError("standalone aLoRA did not activate")
        trace.append({"tokens": length, "offset": offsets[0],
                      "active_from": max(0, length - offsets[0])})

    handle = model.base_model.model.model.layers[0].self_attn.q_proj.register_forward_hook(hook, with_kwargs=True)
    try:
        row = reference.generate(model, tokenizer, plan, task, "standalone",
                                 generation_kwargs={"alora_offsets": [offset]})
    finally:
        handle.remove()
    if not trace or trace[0]["active_from"] != start or any(r["active_from"] for r in trace[1:]):
        raise ValueError("standalone activation boundary differs")
    row["alora_trace"] = trace
    return row


def summarize(tasks, standalone, modular, old):
    expected = {task["id"]: task for task in tasks}
    groups = []
    for rows in (standalone, modular, old):
        group = {row["id"]: row for row in rows}
        if len(rows) != len(group) or set(group) != set(expected):
            raise ValueError("audit rows missing, duplicated, or outside opened scope")
        for key, row in group.items():
            if row["passed"] != reference.score(expected[key], row["text"], row["terminated"]):
                raise ValueError("audit rescore differs")
        groups.append(group)
    standalone, modular, old = groups
    def equal(a, b):
        return all(a[k] == b[k] for k in ("text", "token_ids", "terminated", "passed"))
    replay_mismatches = [key for key in expected if not equal(modular[key], old[key])
                         or modular[key]["route_trace"] != old[key]["route_trace"]
                         or modular[key]["input_token_ids"] != old[key]["input_token_ids"]]
    pair_mismatches = [key for key in expected if not equal(standalone[key], modular[key])]
    decision = ("prior-switch-output-not-reproduced" if replay_mismatches else
                "standalone-and-switch-differ" if pair_mismatches else "same-published-adapter-errors")
    return {"decision": decision, "pair_mismatch_ids": pair_mismatches,
            "replay_mismatch_ids": replay_mismatches,
            "correct": {"standalone": sum(r["passed"] for r in standalone.values()),
                        "modular": sum(r["passed"] for r in modular.values())},
            "admission_evidence": False, "checklist_credit": False, "old_failure_unchanged": True}


def worker(request_path):
    configure()
    request_path = Path(request_path)
    request = read(request_path)
    if freeze() != request["freeze"]:
        raise ValueError("worker audit binding differs")
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    plan = read(ROOT / PLAN)
    reference_plan = read(ROOT / reference.PLAN)
    tasks = reference_plan["reference_tasks"]
    if [task["id"] for task in tasks] != plan["task_ids"]:
        raise ValueError("audit tasks differ from declared opened cases")
    old = [row for row in read(ROOT / plan["old_result"])["results"]["modular"]
           if row["category"] == "reference"]
    models = Path(request["models"])
    reply = {"binding": request["binding"], "execution_completed": False,
             "admission_evidence": False, "checklist_credit": False, "results": {}}
    started = time.monotonic()
    try:
        inventory = read(ROOT / reference.ARTIFACTS)["models"]
        states = {which: verify_artifacts(models / which, inventory[which], download=True)
                  for which in ("baseline", "modular")}
        aux = models / "audit"
        verify_auxiliary(aux, plan["auxiliary_artifacts"])
        adapter_dir = aux / plan["adapter_subfolder"]
        config = read(adapter_dir / "adapter_config.json")
        validate_adapter(config)
        model, tokenizer = reference.load_model(models / "baseline", "baseline")
        switch_tokenizer = AutoTokenizer.from_pretrained(models / "modular", local_files_only=True)
        invocation = invocation_audit(tokenizer, switch_tokenizer, tasks, config, old)
        if invocation != read(ROOT / "config/experiments/granite-adapter-audit-invocation.json")["rows"]:
            raise ValueError("invocation differs from committed local audit")
        reply["invocation"] = invocation
        positions = {row["id"]: row for row in invocation}
        base_digests = mapped_base_digests(model.state_dict(), read(aux / "compose_report.json")["base_model_mapping"])
        raw = load_file(adapter_dir / "adapter_model.safetensors")
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False, autocast_adapter_dtype=False).eval()
        # Raw adapter is F32; compare in the published composed checkpoint's BF16 profile.
        model.to(dtype=torch.bfloat16)
        from peft import get_peft_model_state_dict
        loaded = get_peft_model_state_dict(model)
        if set(loaded) != set(raw) or any(not torch.equal(loaded[k], raw[k].to(torch.bfloat16)) for k in raw):
            raise ValueError("standalone loader omitted or changed adapter tensors")
        del loaded
        if any(p.requires_grad or p.dtype != torch.bfloat16 for p in model.parameters()):
            raise ValueError("standalone has trainable or non-BF16 parameters")
        for which in ("standalone", "modular"):
            if which == "modular":
                model, tokenizer = reference.load_model(models / "modular", "modular")
                state = model.state_dict()
                for name, digest in base_digests.items():
                    if tensor_digest(state[name]) != digest:
                        raise ValueError(f"composed backbone differs: {name}")
                reply["base_tensor_matches"] = len(base_digests)
                reply["weights"] = compare_adapter_weights(state, raw)
                del state
            rows = reply["results"][which] = []
            for task in tasks:
                row = (standalone_generate(model, tokenizer, reference_plan, task,
                                          positions[task["id"]]["activation_start"],
                                          positions[task["id"]]["alora_offset"])
                       if which == "standalone" else reference.generate(model, tokenizer, reference_plan, task, which))
                rows.append(row)
                save(request_path.parent / f"{which}-{task['id']}.json", row, exclusive=True)
                save(request_path.parents[2] / "status.json", {"state": "auditing", "model": which,
                     "completed": len(rows), "last_id": task["id"]})
            del model, tokenizer
            gc.collect()
        if any(file_state(models / which, inventory[which]) != states[which] for which in states):
            raise ValueError("published artifacts changed during audit")
        verify_auxiliary(aux, plan["auxiliary_artifacts"])
        reply.update(execution_completed=True, report=summarize(tasks, reply["results"]["standalone"],
                     reply["results"]["modular"], old))
    except Exception as error:
        reply["error"] = str(error)
    finally:
        reply.update(wall_seconds=time.monotonic() - started, process_cpu_seconds=time.process_time(),
                     cumulative_process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        save(request_path.parent / "reply.json", reply, exclusive=True)


def run(home, models):
    configure()
    binding = {"freeze": freeze(), "profile": "granite-adapter-audit", "plan_sha256": sha256(ROOT / PLAN)}
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    save(home / "binding.json", binding, exclusive=True)
    result = {"execution_completed": False, "binding": binding, "checklist_credit": False}
    try:
        limits = read(ROOT / PLAN)["resources"]
        result.update(launch(home, models, binding, "adapter", "audit", limits["worker_seconds"],
                             limits["memory_bytes"], worker_script=SCRIPT))
    except Exception as error:
        result["error"] = str(error)
    save(home / "result.json", result, exclusive=True)
    return result
