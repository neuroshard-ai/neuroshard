"""Opened-case CPU diagnostic of exact evidence selection versus free copying.

The backbone is unchanged. No published adapter or new weights are evaluated,
and no result of this diagnostic can establish A1/A2 or become admission evidence.
"""

import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

from neuroshard.evolution import evidence_selection as evidence
from neuroshard.evolution import granite_context_reference as context
from neuroshard.evolution import granite_reference as reference
from neuroshard.evolution.modular_reference_execution import (
    ROOT, file_state, identity, launch, read, save, sha256, verify_artifacts,
)

PLAN = "config/experiments/granite-evidence-diagnostic.json"
EXECUTION = "config/experiments/granite-evidence-diagnostic-execution.json"
SCRIPT = "scripts/run_granite_evidence_diagnostic.py"
ARMS = ("direct", "selection")
configure = reference.configure


def committed_sources(root=ROOT):
    execution = read(root / EXECUTION)
    for name, digest in execution["contracts"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"changed evidence diagnostic contract: {name}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    sources = {}
    for name in execution["sources"]:
        data = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root)
        if data != (root / name).read_bytes():
            raise ValueError(f"uncommitted evidence diagnostic source: {name}")
        sources[name] = sha256(root / name)
    return {"commit": commit, "sources": sources}


def freeze():
    binding = committed_sources()
    execution = read(ROOT / EXECUTION)
    packages = {key: importlib.metadata.version(key) for key in execution["packages"]}
    if packages != execution["packages"] or platform.python_version() != execution["python"]:
        raise ValueError("evidence runtime differs")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("evidence runtime requires Linux x86_64")
    cpu = Path("/proc/cpuinfo").read_text()
    if any(flag not in cpu.split() for flag in execution["required_cpu_flags"]):
        raise ValueError("evidence CPU lacks required instructions")
    if any(os.environ.get(key) != value for key, value in execution["environment"].items()):
        raise ValueError("evidence numerical environment differs")
    reference.upstream_path()
    return {**binding, "packages": packages, "python": platform.python_version(),
            "upstream": read(ROOT / reference.ARTIFACTS)["upstream"]["commit"]}


def prepare_request(task, retriever):
    messages = context.public_input(task)
    query = "\n".join(message["content"] for message in messages)
    documents = retriever.search(query)
    sources = [evidence.record_value_source({key: document[key] for key in ("id", "text")})
               for document in documents]
    return messages, sources, evidence.build_menu(messages, sources)


def model_messages(plan, messages, menu, arm):
    if arm not in ARMS:
        raise ValueError("unknown evidence diagnostic arm")
    payload = {"conversation": messages, "evidence": evidence.public_choices(menu)}
    return [{"role": "system", "content": plan["system_instruction"]},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)
             + "\nTask:\n" + plan["instructions"][arm]}]


def score(task, row):
    if not row["completed"]:
        return False
    present = {option["source_id"] for option in row["menu"]["choices"]}
    return (reference.score({"kind": "json", "expected": task["expected"]}, row["text"], True)
            and all(source in present for source in task["expected"]["sources"]))


def execute(model, tokenizer, plan, retriever, task, arm):
    started = time.monotonic()
    messages, sources, menu = prepare_request(task, retriever)
    prompt = context.stage_task(model_messages(plan, messages, menu, arm), "answer")
    generation_plan = {"generation": {**plan["generation"],
                                      "max_new_tokens": 1 if arm == "selection" else plan["generation"]["max_new_tokens"]}}
    token_ids = evidence.label_token_ids(tokenizer, menu) if arm == "selection" else {}
    kwargs = {}
    if arm == "selection":
        allowed = list(token_ids.values())
        kwargs["prefix_allowed_tokens_fn"] = lambda batch, prefix: allowed
    generated = reference.generate(model, tokenizer, generation_plan, prompt, "baseline",
                                   generation_kwargs=kwargs)
    receipt = None
    if arm == "selection":
        by_token = {value: label for label, value in token_ids.items()}
        if (len(generated["token_ids"]) != 1 or generated["token_ids"][0] not in by_token
                or generated["text"] != by_token[generated["token_ids"][0]]):
            raise ValueError("one-step decision violated its declared choice space")
        decision = {"invocation_root": menu["invocation_root"], "choice": generated["text"]}
        receipt = evidence.resolve(messages, sources, decision)
        evidence.verify_receipt(messages, sources, receipt)
        text = json.dumps(receipt["answer"], ensure_ascii=False, sort_keys=True)
        completed = True  # One constrained decision token completes this operation; EOS is not required.
    else:
        text, completed = generated["text"], generated["terminated"]
    row = {"id": task["id"], "arm": arm, "task_sha256": identity(task), "menu": menu,
           "generated": generated, "receipt": receipt, "text": text, "completed": completed,
           "seconds": time.monotonic() - started, "input_tokens": len(generated["input_token_ids"]),
           "output_tokens": len(generated["token_ids"]), "generation_calls": 1}
    row["passed"] = score(task, row)
    return row


def assess(plan, rows, anchors):
    tasks = {task["id"]: task for task in read(ROOT / plan["opened_plan"])["tasks"]}
    old_plan = read(ROOT / plan["opened_plan"])
    retriever = context.Retriever(old_plan["corpus"], old_plan["retrieval"])
    groups = {}
    for arm in ARMS:
        if len({row["id"] for row in rows[arm]}) != len(rows[arm]):
            raise ValueError("duplicate evidence diagnostic result")
        groups[arm] = {}
        for row in rows[arm]:
            task = tasks[row["id"]]
            messages, sources, menu = prepare_request(task, retriever)
            if row["arm"] != arm or row["task_sha256"] != identity(task) or row["menu"] != menu:
                raise ValueError("evidence diagnostic binding differs")
            if arm == "selection":
                evidence.verify_receipt(messages, sources, row["receipt"])
                if (row["generated"]["text"] != row["receipt"]["choice"]
                        or len(row["generated"]["token_ids"]) != 1 or not row["completed"]
                        or row["text"] != json.dumps(row["receipt"]["answer"], ensure_ascii=False, sort_keys=True)):
                    raise ValueError("selected output differs from its receipt")
            elif (row["text"] != row["generated"]["text"]
                  or row["completed"] != row["generated"]["terminated"] or row["receipt"] is not None):
                raise ValueError("direct output differs from its generation")
            if row["passed"] != score(task, row):
                raise ValueError("evidence diagnostic rescore differs")
            groups[arm][row["id"]] = row
    original = read(ROOT / reference.PLAN)
    anchor_tasks = {task["id"]: task for task in original["tasks"]}
    if len({row["id"] for row in anchors}) != len(anchors):
        raise ValueError("duplicate retention result")
    for row in anchors:
        if row["passed"] != reference.score(anchor_tasks[row["id"]], row["text"], row["terminated"]):
            raise ValueError("retention rescore differs")
    by_id = {row["id"]: row for row in anchors}
    retention = reference.baseline_gate(original, anchors) and all(
        by_id.get(key, {}).get("passed", False) for key in plan["protected_ids"])
    complete = all(set(group) == set(tasks) for group in groups.values())
    report = {"complete": complete, "retention": retention, "diagnostic_gate": False,
              "admission_evidence": False, "checklist_credit": False, "new_capability_proven": False}
    if not complete:
        return report
    report["correct"] = {arm: sum(row["passed"] for row in group.values()) for arm, group in groups.items()}
    selected = groups["selection"]
    report["selected_by_category"] = {
        category: sum(selected[t["id"]]["passed"] for t in tasks.values() if t["category"] == category)
        for category in context.CATEGORIES}
    report["lost_direct_ids"] = [key for key in tasks if groups["direct"][key]["passed"] and not selected[key]["passed"]]
    report["p95_seconds"] = {arm: context.percentile([row["seconds"] for row in group.values()], .95)
                             for arm, group in groups.items()}
    report["work"] = {arm: {key: sum(row[key] for row in group.values())
                            for key in ("input_tokens", "output_tokens", "generation_calls")}
                      for arm, group in groups.items()}
    q = plan["diagnostic_thresholds"]
    report["diagnostic_gate"] = bool(
        retention and report["correct"]["selection"] >= q["minimum_correct"]
        and all(report["selected_by_category"][c] >= q["minimum_per_category"][c] for c in context.CATEGORIES)
        and len(report["lost_direct_ids"]) <= q["maximum_lost_direct_answers"]
        and report["p95_seconds"]["selection"] <= q["p95_seconds"]
        and report["p95_seconds"]["selection"] <= q["p95_ratio_vs_direct"] * report["p95_seconds"]["direct"])
    return report


def replay_matches(original, replay):
    return (all(original[key] == replay[key] for key in ("menu", "receipt", "text", "completed", "passed"))
            and all(original["generated"][key] == replay["generated"][key]
                    for key in ("prompt_sha256", "token_ids", "text", "terminated", "route_trace")))


def worker(request_path):
    configure()
    request_path = Path(request_path)
    request = read(request_path)
    if freeze() != request["freeze"]:
        raise ValueError("evidence worker differs from freeze")
    import torch
    plan = read(ROOT / PLAN)
    torch.set_num_threads(plan["resources"]["threads"])
    torch.set_num_interop_threads(1)
    opened = read(ROOT / plan["opened_plan"])
    retriever = context.Retriever(opened["corpus"], opened["retrieval"])
    rows, anchors, replays = {arm: [] for arm in ARMS}, [], []
    reply = {"binding": request["binding"], "execution_completed": False, "diagnostic_passed": False,
             "admission_evidence": False, "checklist_credit": False}
    started = time.monotonic()
    try:
        inventory = read(ROOT / reference.ARTIFACTS)["models"]["baseline"]
        directory = Path(request["models"]) / "baseline"
        state = verify_artifacts(directory, inventory, download=True)
        model, tokenizer = reference.load_model(directory, "baseline")
        anchor_plan = read(ROOT / reference.PLAN)
        for task in anchor_plan["tasks"]:
            row = reference.generate(model, tokenizer, anchor_plan, task, "baseline")
            anchors.append(row)
            save(request_path.parent / f"anchor-{task['id']}.json", row, exclusive=True)
        if not assess(plan, rows, anchors)["retention"]:
            reply["stop_reason"] = "parent failed protected assistant retention"
        else:
            for index, task in enumerate(opened["tasks"]):
                for arm in (ARMS if index % 2 == 0 else ARMS[::-1]):
                    row = execute(model, tokenizer, plan, retriever, task, arm)
                    rows[arm].append(row)
                    save(request_path.parent / f"{arm}-{task['id']}.json", row, exclusive=True)
                    save(request_path.parents[2] / "status.json", {"state": "generating", "arm": arm,
                         "completed": len(rows[arm]), "last_id": task["id"]})
        if file_state(directory, inventory) != state:
            raise ValueError("evidence checkpoint changed")
        del model, tokenizer
        gc.collect()
        report = assess(plan, rows, anchors)
        if report["diagnostic_gate"]:
            model, tokenizer = reference.load_model(directory, "baseline")
            for task in opened["tasks"]:
                if task["id"] not in plan["replay_ids"]:
                    continue
                for arm in ARMS:
                    row = execute(model, tokenizer, plan, retriever, task, arm)
                    original = next(r for r in rows[arm] if r["id"] == task["id"])
                    row["matches"] = replay_matches(original, row)
                    replays.append(row)
                    save(request_path.parent / f"replay-{arm}-{task['id']}.json", row, exclusive=True)
                    if not row["matches"]:
                        raise ValueError("evidence replay differs")
            if file_state(directory, inventory) != state:
                raise ValueError("evidence replay checkpoint changed")
        reply.update(execution_completed=True, report=report,
                     diagnostic_passed=bool(report["diagnostic_gate"] and len(replays) == 6))
    except Exception as error:
        reply["error"] = str(error)
    finally:
        reply.update(results=rows, anchors=anchors, replays=replays,
                     wall_seconds=time.monotonic() - started, process_cpu_seconds=time.process_time(),
                     cumulative_process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        save(request_path.parent / "reply.json", reply, exclusive=True)


def run(home, models):
    configure()
    source = freeze()
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    binding = {"freeze": source, "profile": "granite-evidence-diagnostic", "plan_sha256": sha256(ROOT / PLAN)}
    save(home / "binding.json", binding, exclusive=True)
    plan = read(ROOT / PLAN)
    result = {"execution_completed": False, "binding": binding, "checklist_credit": False}
    try:
        result.update(launch(home, models, binding, "parent", "evidence",
                             plan["resources"]["worker_seconds"], plan["resources"]["memory_bytes"],
                             worker_script=SCRIPT))
    except Exception as error:
        result["error"] = str(error)
    save(home / "result.json", result, exclusive=True)
    return result
