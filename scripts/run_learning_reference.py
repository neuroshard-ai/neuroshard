#!/usr/bin/env python3
"""Prepare, train and inspect the operated GPU learning reference.

No chain, token issuance, serving promotion or cloud provisioning is performed.
Final test scoring requires a Git-committed candidate selection after training.
"""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from neuroshard.dataflow.store import canonical
from neuroshard.evolution import reference as engine
from neuroshard.evolution import reference_data as data


ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "vocab.json", "merges.txt", "README.md", "model.safetensors"}


def emit(event, **values):
    print(json.dumps({"event": event, **values}), flush=True)


def implementation():
    files = ["scripts/run_learning_reference.py", "src/neuroshard/evolution/reference.py",
             "src/neuroshard/evolution/reference_data.py", "src/neuroshard/evolution/data.py",
             "src/neuroshard/dataflow/collect.py", "src/neuroshard/dataflow/store.py",
             "docs/learning-reference-requirements.txt"]
    return data.identity({name: data.sha256(ROOT / name) for name in files})


def snapshot(directory, expected):
    receipt = json.loads((directory / "snapshot.json").read_bytes())
    if receipt["model"] != expected:
        raise ValueError("Model snapshot identity differs from the plan")
    for name, digest in receipt["files"].items():
        if name not in MODEL_FILES or data.sha256(directory / name) != digest:
            raise ValueError("Pinned model snapshot checksum mismatch")
    required = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}
    if not required.issubset(receipt["files"]):
        raise ValueError("Incomplete model snapshot")
    loader_files = {path.name for path in directory.iterdir()
                    if path.is_file() and path.suffix in (".json", ".safetensors", ".bin", ".py")}
    if not loader_files.issubset(set(receipt["files"]) | {"snapshot.json"}):
        raise ValueError("Unexpected model loading file outside the snapshot")
    return receipt


def fetch_model(args, plan):
    from huggingface_hub import HfApi, hf_hub_download
    destination = args.model_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "snapshot.json").exists():
        snapshot(destination, plan["model"])
        emit("model_verified", directory=str(destination))
        return
    info = HfApi().model_info(plan["model"]["repo"], revision=plan["model"]["revision"], files_metadata=True)
    if info.sha != plan["model"]["revision"] or info.card_data.get("license") != "apache-2.0":
        raise ValueError("Upstream model revision or declared license differs")
    files = {}
    for item in info.siblings:
        if item.rfilename not in MODEL_FILES:
            continue
        path = Path(hf_hub_download(info.id, item.rfilename, revision=info.sha, local_dir=destination))
        digest = data.sha256(path)
        if item.lfs:
            if digest != item.lfs.sha256 or path.stat().st_size != item.size:
                raise ValueError("Downloaded model LFS object differs")
        else:
            raw = path.read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            if blob != item.blob_id:
                raise ValueError("Downloaded model Git object differs")
        files[item.rfilename] = digest
        emit("model_file_verified", file=item.rfilename, bytes=path.stat().st_size)
    data.save(destination / "snapshot.json", {"model": plan["model"], "files": files})
    snapshot(destination, plan["model"])


def prepare(args, plan):
    from transformers import AutoTokenizer
    from neuroshard.dataflow.collect import upstream_rows
    home = args.home.resolve()
    home.mkdir(parents=True, exist_ok=True)
    if (home / "prepared.json").exists() or (home / "run.json").exists():
        raise ValueError("Use a fresh preparation home; existing inputs remain immutable")
    model_snapshot = snapshot(args.model_dir.resolve(), plan["model"])
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    index = data.ExclusionIndex()
    inputs = home / "inputs"
    inputs.mkdir(exist_ok=True)
    if args.upstream_cache:
        (home / "upstream").symlink_to(args.upstream_cache.resolve(), target_is_directory=True)
    roles = {}
    for role in data.ROLES:
        spec = plan["roles"][role]
        source = {**plan["data"], "split": spec["split"]}
        rows = upstream_rows(source, home, spec["start"], spec["scan"])
        try:
            records, report = data.prepare_role(rows, tokenizer, spec, index, plan["max_length"], source)
        finally:
            rows.close()
        path = inputs / f"{role}.jsonl"
        with path.open("xb") as output:
            for record in records:
                output.write(canonical(record) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        roles[role] = {"sha256": data.sha256(path), "ids": [r["id"] for r in records], **report}
        emit("role_prepared", role=role, **report)
    prepared = {"format": data.FORMAT, "plan": plan, "implementation": implementation(),
                "model_snapshot": model_snapshot, "tokenizer": data.tokenizer_identity(tokenizer), "roles": roles}
    data.save(home / "prepared.json", prepared)
    emit("prepared", sha256=data.sha256(home / "prepared.json"))


def prepared_inputs(args, plan):
    prepared = json.loads((args.home / "prepared.json").read_bytes())
    if prepared["plan"] != plan or prepared["implementation"] != implementation():
        raise ValueError("Plan or implementation changed after data preparation")
    if snapshot(args.model_dir.resolve(), plan["model"]) != prepared["model_snapshot"]:
        raise ValueError("Model bytes changed after preparation")
    return prepared


def partition(home, prepared, role, *, final_test=False):
    if role == "test" and not final_test:
        raise ValueError("Training/development cannot load the final test partition")
    records = data.read_records(home / "inputs" / f"{role}.jsonl", prepared["roles"][role]["sha256"])
    if [record["id"] for record in records] != prepared["roles"][role]["ids"]:
        raise ValueError("Prepared document identities differ")
    return records


def tokenizer_for(args, prepared):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    if data.tokenizer_identity(tokenizer) != prepared["tokenizer"]:
        raise ValueError("Tokenizer runtime or serialization changed")
    return tokenizer


def recover(home, binding):
    """Recover a durable renamed checkpoint even if latest.json was not written."""
    candidates = []
    for directory in sorted(home.glob("checkpoint-[0-9][0-9][0-9][0-9][0-9][0-9]")):
        receipt = json.loads((directory / "checkpoint.json").read_bytes())
        pointer = {"directory": directory.name, "receipt": data.identity(receipt)}
        engine.verify_checkpoint(home, pointer, binding)
        if receipt["step"] != len(receipt["records"]) or directory.name != f"checkpoint-{receipt['step']:06d}":
            raise ValueError("Checkpoint step and journal disagree")
        candidates.append((receipt["step"], pointer, receipt))
    if not candidates:
        if (home / "latest.json").exists():
            raise ValueError("Checkpoint pointer exists but its data is missing")
        return None, None
    _, pointer, receipt = max(candidates, key=lambda item: item[0])
    data.save(home / "latest.json", pointer)
    return pointer, receipt


def evaluate_development(model, tokenizer, home, prepared, plan, device, budget, label):
    report_path = home / f"{label}-development.json"
    basis = {"prepared": data.identity(prepared), "label": label}
    if label == "candidate":
        basis["checkpoint"] = json.loads((home / "latest.json").read_bytes())
    if report_path.exists():
        cached = json.loads(report_path.read_bytes())
        if cached.get("basis") != basis:
            raise ValueError("Cached evaluation belongs to different inputs or checkpoint")
        return cached
    result = {"basis": basis}
    for role in ("dev", "retention"):
        records = partition(home, prepared, role)
        result[role] = engine.score(model, records, device, budget.check)
        emit("partition_scored", model=label, role=role, documents=len(records))
    records = partition(home, prepared, "dev")
    result["generations"] = engine.generate(model, tokenizer, records, device, plan["generation_tokens"],
                                            plan["generation_documents"], budget.check)
    data.save(report_path, result)
    return result


def run(args, plan):
    import torch
    home = args.home.resolve()
    prepared = prepared_inputs(args, plan)
    runtime = engine.configure(args.device, args.threads)
    binding = data.identity({"prepared": data.identity(prepared), "runtime": runtime})
    state_path = home / "run.json"
    if state_path.exists():
        state = json.loads(state_path.read_bytes())
        if state["binding"] != binding:
            raise ValueError("Resume requires the original inputs and recorded runtime")
        if state.get("completed"):
            emit("already_completed", result=str(home / "result.json"))
            return
    else:
        state = {"binding": binding, "started": time.time(), "runtime": runtime}
        data.save(state_path, state)
    budget = engine.Budget(home, state["started"], plan["budget"])
    budget.check(force_disk=True)
    tokenizer = tokenizer_for(args, prepared)
    torch.manual_seed(plan["training"]["seed"])
    pointer, receipt = recover(home, binding)
    # Baseline is always scored before any update; it is never reconstructed
    # from a resumed candidate when baseline evidence is missing.
    if not (home / "baseline-development.json").exists():
        model = engine.load_model(args.model_dir, args.device, plan["model"]["parameters"])
        evaluate_development(model, tokenizer, home, prepared, plan, args.device, budget, "baseline")
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
    baseline = evaluate_development(None, tokenizer, home, prepared, plan, args.device, budget, "baseline")
    torch.manual_seed(plan["training"]["seed"])
    directory = home / pointer["directory"] if pointer else args.model_dir
    model = engine.load_model(directory, args.device, plan["model"]["parameters"])
    optimizer = engine.optimizer_for(model, plan["training"])
    if pointer:
        engine.restore_optimizer(directory, optimizer, args.device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    records = receipt["records"] if receipt else []
    train = partition(home, prepared, "train")
    recipe = plan["training"]
    batches = engine.schedule(len(train), recipe["steps"], recipe["batch_documents"], recipe["seed"])
    for step in range(len(records), recipe["steps"]):
        started = time.monotonic()
        record = engine.train_step(model, optimizer, [train[i] for i in batches[step]], args.device,
                                   recipe, step, budget.check)
        record["seconds"] = time.monotonic() - started
        records.append(record)
        emit("trained", **{k: v for k, v in record.items() if k != "documents"})
        if (step + 1) % recipe["checkpoint_steps"] == 0 or step + 1 == recipe["steps"]:
            budget.check(force_disk=True)
            pointer = engine.checkpoint(home, model, tokenizer, optimizer, step + 1, binding, records)
            budget.check(force_disk=True)
            emit("checkpoint", **pointer)
    del optimizer
    model.gradient_checkpointing_disable()
    candidate = evaluate_development(model, tokenizer, home, prepared, plan, args.device, budget, "candidate")
    result = {"format": data.FORMAT, "purpose": "development", "binding": binding,
              "candidate": pointer, "runtime": runtime, "steps": records,
              "comparison": {role: engine.paired_summary(baseline[role], candidate[role])
                             for role in ("dev", "retention")},
              "trained_unique_documents": len({i for record in records for i in record["documents"]}),
              "seconds": time.time() - state["started"], "peak_artifact_bytes": budget.peak_bytes,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
              "test_scored": False, "serving_approved": False, "tokens_issued": 0}
    data.save(home / "result.json", result)
    data.save(home / "selection.json", {"prepared": data.identity(prepared), "binding": binding,
                                       "candidate": pointer, "result": data.identity(result)})
    data.save(state_path, {**state, "completed": time.time()})
    emit("development_complete", comparison=result["comparison"], candidate=pointer)


def committed_selection(path, expected):
    path = path.resolve()
    relative = path.relative_to(ROOT).as_posix()
    raw = path.read_bytes()
    committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    if raw != committed or json.loads(raw) != expected:
        raise ValueError("Commit this exact candidate selection before scoring the final test")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def score_test(args, plan):
    import torch
    home = args.home.resolve()
    prepared = prepared_inputs(args, plan)
    selection = json.loads((home / "selection.json").read_bytes())
    result = json.loads((home / "result.json").read_bytes())
    if selection["result"] != data.identity(result) or selection["prepared"] != data.identity(prepared):
        raise ValueError("Selection differs from completed development evidence")
    commit = committed_selection(args.selection, selection)
    runtime = engine.configure(args.device, args.threads)
    binding = data.identity({"prepared": data.identity(prepared), "runtime": runtime})
    if binding != selection["binding"]:
        raise ValueError("Final evaluation must use the recorded numerical runtime")
    if (home / "test-result.json").exists():
        cached = json.loads((home / "test-result.json").read_bytes())
        if cached["selection"] != selection:
            raise ValueError("Existing test result belongs to a different candidate")
        emit("test_already_scored", result=str(home / "test-result.json"))
        return
    marker = home / "test-started.json"
    if marker.exists():
        previous = json.loads(marker.read_bytes())
        if previous["selection"] != data.identity(selection):
            raise ValueError("Final test has already been opened for a different candidate")
        started = previous["started"]
    else:
        started = time.time()
        data.save(marker, {"started": started, "selection": data.identity(selection), "commit": commit})
    budget = engine.Budget(home, started, plan["budget"])
    directory, _ = engine.verify_checkpoint(home, selection["candidate"], binding)
    tokenizer = tokenizer_for(args, prepared)
    records = partition(home, prepared, "test", final_test=True)
    measurements = {}
    for label, model_dir in (("baseline", args.model_dir), ("candidate", directory)):
        path = home / f"{label}-test.json"
        if path.exists():
            measurements[label] = json.loads(path.read_bytes())
            continue
        model = engine.load_model(model_dir, args.device, plan["model"]["parameters"])
        measurements[label] = {"scores": engine.score(model, records, args.device, budget.check),
                               "generations": engine.generate(model, tokenizer, records, args.device,
                                   plan["generation_tokens"], plan["generation_documents"], budget.check)}
        data.save(path, measurements[label])
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
    report = {"selection": selection, "commit": commit,
              "comparison": engine.paired_summary(measurements["baseline"]["scores"], measurements["candidate"]["scores"]),
              "serving_approved": False, "tokens_issued": 0,
              "scope": "A public committed development holdout; not a broad assistant capability certificate"}
    data.save(home / "test-result.json", report)
    emit("test_complete", comparison=report["comparison"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch-model", "prepare", "run", "score-test"))
    parser.add_argument("--plan", type=Path, default=ROOT / "config/experiments/learning-reference.json")
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--upstream-cache", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    plan = data.validate_plan(json.loads(args.plan.read_bytes()))
    args.home = args.home.resolve()
    args.home.mkdir(parents=True, exist_ok=True)
    if args.command == "score-test" and args.selection is None:
        parser.error("score-test requires --selection pointing to a Git-committed selection.json copy")
    with (args.home / "reference.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {"fetch-model": fetch_model, "prepare": prepare, "run": run, "score-test": score_test}[args.command](args, plan)


if __name__ == "__main__":
    main()
