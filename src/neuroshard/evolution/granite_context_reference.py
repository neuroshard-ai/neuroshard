"""Prospective document-grounded assistant comparison; no training or admission.

The only neural intervention is a published query-rewriting adapter. Evaluation
labels never enter rewriting, retrieval, or answer generation. Each generation
starts a fresh cache, including the answer after an adapter invocation.
"""

import gc
import importlib.metadata
import math
import os
from pathlib import Path
import platform
import random
import re
import resource
import subprocess
import time

from neuroshard.evolution import granite_reference as reference
from neuroshard.evolution.modular_reference_execution import (
    ROOT, file_state, identity, launch, read, save, sha256, verify_artifacts,
)

PLAN = "config/experiments/granite-context-reference.json"
EXECUTION = "config/experiments/granite-context-reference-execution.json"
SCRIPT = "scripts/run_granite_context_reference.py"
ARMS = ("history", "parent-rewrite", "module-rewrite")
CATEGORIES = ("contextual", "standalone", "unanswerable")
configure = reference.configure


def committed_sources(root=ROOT):
    execution = read(root / EXECUTION)
    for name, digest in execution["contracts"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"changed context contract: {name}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    sources = {}
    for name in execution["sources"]:
        committed = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root)
        if committed != (root / name).read_bytes():
            raise ValueError(f"uncommitted context source: {name}")
        sources[name] = sha256(root / name)
    return {"commit": commit, "sources": sources}


def freeze():
    binding = committed_sources()
    execution = read(ROOT / EXECUTION)
    packages = {name: importlib.metadata.version(name) for name in execution["packages"]}
    if packages != execution["packages"] or platform.python_version() != execution["python"]:
        raise ValueError("context runtime differs")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("context runtime requires Linux x86_64")
    cpu = Path("/proc/cpuinfo").read_text()
    if any(flag not in cpu.split() for flag in execution["required_cpu_flags"]):
        raise ValueError("context CPU lacks required instructions")
    if any(os.environ.get(key) != value for key, value in execution["environment"].items()):
        raise ValueError("context numerical environment differs")
    reference.upstream_path()
    return {**binding, "packages": packages, "python": platform.python_version(),
            "upstream": read(ROOT / reference.ARTIFACTS)["upstream"]["commit"]}


def public_input(task):
    """A capability receives conversation only, never IDs, categories or answers."""
    return [{"role": row["role"], "content": row["content"]} for row in task["messages"]]


def terms(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class Retriever:
    """Fixed BM25 over public document text, deterministic document-ID tie breaks."""

    def __init__(self, corpus, settings):
        from collections import Counter
        self.documents = {row["id"]: {"id": row["id"], "text": row["text"]} for row in corpus}
        if len(self.documents) != len(corpus) or not corpus:
            raise ValueError("empty corpus or duplicate documents")
        self.counts = {key: Counter(terms(row["text"])) for key, row in self.documents.items()}
        self.lengths = {key: sum(count.values()) for key, count in self.counts.items()}
        self.average = sum(self.lengths.values()) / len(corpus)
        self.frequency = Counter(term for count in self.counts.values() for term in count)
        self.settings = settings

    def search(self, query):
        k1, b = self.settings["k1"], self.settings["b"]
        result = []
        for key, count in self.counts.items():
            score = 0.0
            for term in sorted(set(terms(query))):
                frequency = count.get(term, 0)
                if not frequency:
                    continue
                idf = math.log(1 + (len(self.documents) - self.frequency[term] + .5)
                               / (self.frequency[term] + .5))
                score += idf * frequency * (k1 + 1) / (
                    frequency + k1 * (1 - b + b * self.lengths[key] / self.average))
            result.append({**self.documents[key], "score": score})
        return sorted(result, key=lambda row: (-row["score"], row["id"]))[:self.settings["top_k"]]


def parse_rewrite(text, terminated, maximum):
    if not terminated:
        return None
    try:
        value = reference.strict_json(text.strip())
    except (ValueError, TypeError, OverflowError):
        return None
    if (not isinstance(value, dict) or set(value) != {"rewritten_question"}
            or not isinstance(value["rewritten_question"], str)):
        return None
    query = value["rewritten_question"].strip()
    if not query or len(query) > maximum or "<|" in query:
        return None
    return query


def stage_task(messages, stage, adapter=None):
    # No hidden expected answer is even supplied to the generation helper.
    task = {"id": stage, "category": "reference" if stage == "rewrite" else "answer",
            "messages": messages, "kind": "exact", "accept": []}
    if adapter:
        task["adapter"] = adapter
    return task


def answer_messages(plan, messages, documents):
    payload = [{"id": row["id"], "text": row["text"]} for row in documents]
    import json
    return [{"role": "system", "content": plan["answer_instruction"] + "\nDocuments:\n"
             + json.dumps(payload, ensure_ascii=False, sort_keys=True)}] + messages


def answer_score(task, row):
    if row.get("error") or not row.get("answer"):
        return False
    answer = row["answer"]
    expected = {"kind": "json", "expected": task["expected"]}
    if not reference.score(expected, answer["text"], answer["terminated"]):
        return False
    # A correct-looking citation that was not actually retrieved is not evidence.
    available = {document["id"] for document in row["retrieved"]}
    return all(source in available for source in task["expected"]["sources"])


def failure_stage(task, row):
    if answer_score(task, row):
        return "passed"
    if row.get("error"):
        return "rewrite-format"
    if any(source not in {document["id"] for document in row["retrieved"]}
           for source in task["expected"]["sources"]):
        return "retrieval"
    return "abstention" if task["expected"]["answer"] is None else "grounded-answer"


def pipeline(model, tokenizer, plan, retriever, task, arm):
    if arm not in ARMS:
        raise ValueError("unknown comparison arm")
    messages = public_input(task)
    which = "modular" if arm == "module-rewrite" else "baseline"
    started = time.monotonic()
    row = {"id": task["id"], "arm": arm, "task_sha256": identity(task),
           "rewrite": None, "answer": None, "retrieved": [], "error": None}
    if arm == "history":
        query = "\n".join(message["content"] for message in messages)
    else:
        rewrite_task = stage_task(
            [{"role": "system", "content": plan["rewrite_instruction"]}] + messages,
            "rewrite", "query_rewrite" if which == "modular" else None)
        row["rewrite"] = reference.generate(model, tokenizer, plan, rewrite_task, which)
        query = parse_rewrite(row["rewrite"]["text"], row["rewrite"]["terminated"],
                              plan["retrieval"]["maximum_query_characters"])
    row["query"] = query
    if query is None:
        row["error"] = "invalid-rewrite; no repair or hidden fallback"
        row["retrieval_seconds"] = 0.0
    else:
        retrieval_started = time.monotonic()
        row["retrieved"] = retriever.search(query)
        row["retrieval_seconds"] = time.monotonic() - retrieval_started
        final_task = stage_task(answer_messages(plan, messages, row["retrieved"]), "answer")
        row["answer"] = reference.generate(model, tokenizer, plan, final_task, which)
    row["seconds"] = time.monotonic() - started
    stages = [row[key] for key in ("rewrite", "answer") if row[key]]
    row["input_tokens"] = sum(len(stage["input_token_ids"]) for stage in stages)
    row["output_tokens"] = sum(len(stage["token_ids"]) for stage in stages)
    row["generation_calls"] = len(stages)
    row["passed"] = answer_score(task, row)
    return row


def percentile(values, fraction):
    return sorted(values)[math.ceil(fraction * len(values)) - 1] if values else None


def paired_gain(plan, tasks, candidate, control):
    blocks = sorted({task["block"] for task in tasks})
    by_block = [[int(candidate[t["id"]]["passed"]) - int(control[t["id"]]["passed"])
                 for t in tasks if t["block"] == block] for block in blocks]
    randomizer = random.Random(plan["quality"]["bootstrap_seed"])
    samples = []
    for _ in range(plan["quality"]["bootstrap_samples"]):
        selected = [by_block[randomizer.randrange(len(blocks))] for _ in blocks]
        samples.append(sum(sum(block) for block in selected) / sum(map(len, selected)))
    gains = [t["id"] for t in tasks if candidate[t["id"]]["passed"] and not control[t["id"]]["passed"]]
    losses = [t["id"] for t in tasks if control[t["id"]]["passed"] and not candidate[t["id"]]["passed"]]
    return {"gained_ids": gains, "lost_ids": losses, "net": len(gains) - len(losses),
            "block_bootstrap_lower_95": percentile(samples, .025), "blocks": len(blocks)}


def assess(plan, results, anchors):
    tasks = {task["id"]: task for task in plan["tasks"]}
    groups = {}
    for arm in ARMS:
        rows = results[arm]
        if len({r["id"] for r in rows}) != len(rows) or any(r["id"] not in tasks for r in rows):
            raise ValueError("duplicate or unknown context result")
        for row in rows:
            if row["task_sha256"] != identity(tasks[row["id"]]):
                raise ValueError("context task binding differs")
            if row["passed"] != answer_score(tasks[row["id"]], row):
                raise ValueError("context rescore differs")
        groups[arm] = {row["id"]: row for row in rows}
    complete = all(set(rows) == set(tasks) for rows in groups.values())
    anchor_tasks = {t["id"]: t for t in read(ROOT / reference.PLAN)["tasks"]}
    protected = plan["retention"]["protected_ids"]
    anchor_ok = {}
    for which in ("baseline", "modular"):
        rows = anchors[which]
        if len({r["id"] for r in rows}) != len(rows) or any(r["id"] not in anchor_tasks for r in rows):
            raise ValueError("duplicate or unknown retention result")
        for row in rows:
            if row["passed"] != reference.score(anchor_tasks[row["id"]], row["text"], row["terminated"]):
                raise ValueError("retention rescore differs")
        by_id = {row["id"]: row for row in rows}
        anchor_ok[which] = (set(by_id) == set(anchor_tasks)
                           and all(by_id[key]["passed"] for key in protected)
                           and reference.baseline_gate(read(ROOT / reference.PLAN), rows))
    report = {"complete": complete, "retention": anchor_ok, "comparisons": {},
              "quality_gate": False, "checklist_credit": False, "admission_evidence": False,
              "training_authorized": False, "automatic_composition_proven": False}
    if not complete:
        return report
    report["correct"] = {arm: sum(row["passed"] for row in rows.values()) for arm, rows in groups.items()}
    report["failures"] = {
        arm: {stage: [key for key, row in rows.items() if failure_stage(tasks[key], row) == stage]
              for stage in ("rewrite-format", "retrieval", "grounded-answer", "abstention")}
        for arm, rows in groups.items()}
    report["work"] = {
        arm: {key: sum(row[key] for row in rows.values())
              for key in ("input_tokens", "output_tokens", "generation_calls")}
        for arm, rows in groups.items()}
    candidate = groups["module-rewrite"]
    report["candidate_by_category"] = {
        category: sum(candidate[t["id"]]["passed"] for t in tasks.values() if t["category"] == category)
        for category in CATEGORIES}
    report["p95_seconds"] = {arm: percentile([row["seconds"] for row in rows.values()], .95)
                             for arm, rows in groups.items()}
    for control in ("history", "parent-rewrite"):
        report["comparisons"][control] = paired_gain(plan, list(tasks.values()), candidate, groups[control])
    quality = plan["quality"]
    report["quality_gate"] = bool(
        all(anchor_ok.values()) and report["correct"]["module-rewrite"] >= quality["minimum_correct"]
        and all(report["candidate_by_category"][c] >= quality["minimum_per_category"][c] for c in CATEGORIES)
        and all(comparison["net"] >= quality["minimum_net_gain"]
                and comparison["block_bootstrap_lower_95"] > 0
                for comparison in report["comparisons"].values())
        and len(report["comparisons"]["parent-rewrite"]["lost_ids"])
        <= quality["maximum_losses_vs_parent_rewrite"]
        and report["p95_seconds"]["module-rewrite"] <= quality["p95_seconds"]
        and report["p95_seconds"]["module-rewrite"] <= quality["p95_ratio_vs_parent_rewrite"]
        * report["p95_seconds"]["parent-rewrite"])
    return report


def replay_matches(first, second):
    keys = ("query", "retrieved", "error", "passed", "generation_calls", "input_tokens", "output_tokens")
    if any(first[key] != second[key] for key in keys):
        return False
    for stage in ("rewrite", "answer"):
        if bool(first[stage]) != bool(second[stage]):
            return False
        if first[stage] and any(first[stage][key] != second[stage][key] for key in (
                "prompt_sha256", "token_ids", "text", "terminated", "route_trace")):
            return False
    return True


def worker(request_path):
    configure()
    request_path = Path(request_path)
    request = read(request_path)
    binding = freeze()
    if binding != request["freeze"]:
        raise ValueError("context worker differs from freeze")
    import torch
    plan = read(ROOT / PLAN)
    torch.set_num_threads(plan["resources"]["threads"])
    torch.set_num_interop_threads(1)
    retriever = Retriever(plan["corpus"], plan["retrieval"])
    results = {arm: [] for arm in ARMS}
    anchors = {which: [] for which in ("baseline", "modular")}
    reply = {"binding": request["binding"], "execution_completed": False,
             "reference_passed": False, "checklist_credit": False, "replays": []}
    started = time.monotonic()
    states = {}
    try:
        for which in ("baseline", "modular"):
            inventory = read(ROOT / reference.ARTIFACTS)["models"][which]
            directory = Path(request["models"]) / which
            states[which] = verify_artifacts(directory, inventory, download=True)
            model, tokenizer = reference.load_model(directory, which)
            anchor_plan = read(ROOT / reference.PLAN)
            for task in anchor_plan["tasks"]:
                row = reference.generate(model, tokenizer, anchor_plan, task, which)
                anchors[which].append(row)
                save(request_path.parent / f"anchor-{which}-{task['id']}.json", row, exclusive=True)
            current = {row["id"]: row for row in anchors[which]}
            if (not reference.baseline_gate(anchor_plan, anchors[which])
                    or any(not current[key]["passed"] for key in plan["retention"]["protected_ids"])):
                reply["stop_reason"] = f"{which} failed existing assistant retention"
                break
            arms = ARMS[:2] if which == "baseline" else ARMS[2:]
            # Alternate the control order by task to reduce fixed-order timing bias.
            for index, task in enumerate(plan["tasks"]):
                for arm in (arms if index % 2 == 0 else arms[::-1]):
                    row = pipeline(model, tokenizer, plan, retriever, task, arm)
                    results[arm].append(row)
                    save(request_path.parent / f"{arm}-{task['id']}.json", row, exclusive=True)
                    save(request_path.parents[2] / "status.json", {
                        "state": "generating", "arm": arm, "completed": len(results[arm]), "last_id": task["id"]})
            if file_state(directory, inventory) != states[which]:
                raise ValueError("context checkpoint changed during generation")
            del model, tokenizer
            gc.collect()
        report = assess(plan, results, anchors)
        if report["quality_gate"]:
            for which in ("baseline", "modular"):
                directory = Path(request["models"]) / which
                model, tokenizer = reference.load_model(directory, which)
                for task in plan["tasks"]:
                    if task["id"] not in plan["replay_ids"]:
                        continue
                    for arm in (ARMS[:2] if which == "baseline" else ARMS[2:]):
                        row = pipeline(model, tokenizer, plan, retriever, task, arm)
                        original = next(r for r in results[arm] if r["id"] == task["id"])
                        row["matches"] = replay_matches(original, row)
                        reply["replays"].append(row)
                        save(request_path.parent / f"replay-{arm}-{task['id']}.json", row, exclusive=True)
                        if not row["matches"]:
                            raise ValueError("context reload replay differs")
                if file_state(directory, read(ROOT / reference.ARTIFACTS)["models"][which]) != states[which]:
                    raise ValueError("context checkpoint changed during replay")
                del model, tokenizer
                gc.collect()
        reply.update(execution_completed=True, report=report,
                     reference_passed=bool(report["quality_gate"] and len(reply["replays"]) == 9))
    except Exception as error:
        reply["error"] = str(error)
    finally:
        reply.update(results=results, anchors=anchors, wall_seconds=time.monotonic() - started,
                     process_cpu_seconds=time.process_time(),
                     cumulative_process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        save(request_path.parent / "reply.json", reply, exclusive=True)


def run(home, models):
    configure()
    source = freeze()
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    binding = {"freeze": source, "profile": "granite-context-reference", "plan_sha256": sha256(ROOT / PLAN)}
    save(home / "binding.json", binding, exclusive=True)
    plan = read(ROOT / PLAN)
    result = {"execution_completed": False, "binding": binding, "checklist_credit": False}
    try:
        result.update(launch(home, models, binding, "assistant", "comparison",
                             plan["resources"]["worker_seconds"], plan["resources"]["memory_bytes"],
                             worker_script=SCRIPT))
    except Exception as error:
        result["error"] = str(error)
    save(home / "result.json", result, exclusive=True)
    return result
