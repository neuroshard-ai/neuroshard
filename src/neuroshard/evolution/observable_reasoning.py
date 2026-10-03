"""A fresh CPU experiment: worked examples and question-only sparse routing.

Historical failed experiments are inputs to exclusion, never admission evidence.
The selector has no answer, correctness flag, sample identity, or protection list.
"""
import ast
from collections import Counter
import json
import math
from pathlib import Path
import re
import subprocess

from neuroshard.evolution.block_expert import (
    answer_logits, capture_prefixes, install_blocks, loss, paired_gain, parameters_identity,
)
from neuroshard.evolution.reference_data import identity, sha256
from neuroshard.evolution.seed import FILES, MODEL_REPO, MODEL_REVISION
from neuroshard.evolution.staged_integration import p95


FORMAT = "neuroshard-observable-reasoning-v1"
PLAN = "config/experiments/observable-reasoning.json"
DATA = "config/experiments/observable-reasoning-data.json"
FREEZE = "config/experiments/observable-reasoning-freeze.json"
CONTRACT = "3913e6e7c729917da6ad0cf140dfa38f30c2eb315e0e833bf909f19de497078a"
ROLES = ("train_new", "train_replay", "development", "retention")
EVAL_ROLES = ROLES[2:]
EXCLUDED = (
    "config/experiments/staged-integration-data.json",
    "config/experiments/staged-answering-data.json",
    "config/experiments/staged-answer-calibration-system.json",
    "config/experiments/block-expert-data.json",
    "config/experiments/block-expert-measure-data.json",
    "config/experiments/append-only-execution-data.json",
)


def root():
    for directory in Path(__file__).resolve().parents:
        if (directory / PLAN).is_file():
            return directory
    raise FileNotFoundError(PLAN)


def load_plan():
    plan = json.loads((root() / PLAN).read_text())
    if identity(plan) != CONTRACT:
        raise ValueError("Declare a new contract instead of changing this experiment")
    if (plan["model"] != {"repo": MODEL_REPO, "revision": MODEL_REVISION, "files": FILES}
            or plan["host"] != "cpu" or plan["gpu_launch_authorized"]
            or plan["admission_evidence"] or plan["upgrade_public_0_4_0"]):
        raise ValueError("Only the committed CPU mechanism experiment is authorized")
    return plan


def excluded_pairs():
    result = set()
    for name in EXCLUDED:
        data = json.loads((root() / name).read_text())
        for rows in data["roles"].values():
            result.update(tuple(sorted((row["a"], row["b"]))) for row in rows)
    return result


def make_data(plan):
    excluded = excluded_pairs()
    limit = plan["data"]["operand_limit_exclusive"]
    pairs = [(a, b) for a in range(limit) for b in range(a, limit) if (a, b) not in excluded]
    pairs.sort(key=lambda pair: identity({"namespace": FORMAT, "seed": plan["data"]["seed"], "pair": pair}))
    roles, offset = {}, 0
    for role in ROLES:
        count = plan["data"]["counts"][role]
        selected = pairs[offset:offset + count]
        if len(selected) != count:
            raise ValueError("Fresh pair pool exhausted")
        offset += count
        family = "modular-addition" if role in ("train_new", "development") else "addition"
        split = "train" if role.startswith("train_") else "evaluation"
        templates = plan["data"]["prompts"][split][family]
        rows = []
        for index, (a, b) in enumerate(selected):
            task = {"family": family, "a": a, "b": b}
            rows.append({"id": identity({"namespace": FORMAT, **task}), **task,
                         "messages": [{"role": "system", "content": plan["data"]["system"]},
                                      {"role": "user", "content": templates[index % len(templates)].format(a=a, b=b)}],
                         "answer": str((a + b) % 7 if family == "modular-addition" else a + b)})
        roles[role] = rows
    return {"format": FORMAT + "/data", "roles": roles, "excluded_pairs": len(excluded),
            "source": "generated-arithmetic", "license": "Apache-2.0", "confirmation": None}


def load_data(plan):
    data = json.loads((root() / DATA).read_text())
    if data != make_data(plan):
        raise ValueError("Data differ from the frozen generator")
    return data


def training_answer(row):
    if row["family"] == "addition":
        return row["answer"]
    a, b = row["a"] % 7, row["b"] % 7
    return f"First remainder: {a}\nSecond remainder: {b}\nSum: {a + b}\nAnswer: {(a + b) % 7}"


def parse_answer(text, *, terminated):
    """Parse the entire reply without looking at operands or the expected answer."""
    if type(terminated) is not bool or not terminated:
        return None
    value = text.strip()
    number = r"(0|[1-9][0-9]*)"
    direct = re.fullmatch(number + r"\.?", value)
    if direct:
        return direct.group(1)
    trace = re.fullmatch(r"First remainder: " + number + r"\nSecond remainder: " + number
                         + r"\nSum: " + number + r"\nAnswer: " + number + r"\.?", value)
    return trace.group(4) if trace else None


def words(question):
    if not isinstance(question, str):
        raise TypeError("The selector accepts question text only")
    return re.findall(r"[a-z]+|<number>", re.sub(r"\d+", "<number>", question.lower()))


def fit_selector(training_rows):
    """Balanced multinomial NB; supervised task labels from training only."""
    counts = {"parent": Counter(), "candidate": Counter()}
    for row in training_rows:
        label = "candidate" if row["family"] == "modular-addition" else "parent"
        counts[label].update(words(row["messages"][-1]["content"]))
    vocabulary = sorted(set(counts["parent"]) | set(counts["candidate"]))
    size = len(vocabulary)
    weights = {}
    totals = {label: sum(values.values()) + size for label, values in counts.items()}
    for token in vocabulary:
        weights[token] = (math.log((counts["candidate"][token] + 1) / totals["candidate"])
                          - math.log((counts["parent"][token] + 1) / totals["parent"]))
    return {"format": FORMAT + "/selector", "weights": weights,
            "threshold": math.log(9), "prior_log_odds": 0.0,
            "scope": "training questions and task labels only; OOV tokens ignored"}


def choose(question, selector):
    margin = sum(selector["weights"].get(token, 0.0) for token in words(question))
    return ("candidate" if margin > selector["threshold"] else "parent"), margin


def local_source_closure(entries):
    """Bind transitive repository Python imports, including function-local imports."""
    found, pending = set(), list(entries)
    while pending:
        name = pending.pop()
        if name in found:
            continue
        found.add(name)
        path = root() / name
        if path.suffix != ".py":
            continue
        package = name.removeprefix("src/").removesuffix(".py").split("/")[:-1]
        modules = []
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package) - node.level + 1] if node.level else []
                module = ".".join([*prefix, *([] if node.module is None else node.module.split("."))])
                modules.append(module)
                modules.extend(module + "." + alias.name for alias in node.names if alias.name != "*")
        for module in modules:
            if not module.startswith("neuroshard"):
                continue
            pieces = module.split(".")
            for depth in range(1, len(pieces) + 1):
                base = "src/" + "/".join(pieces[:depth])
                for candidate in (base + ".py", base + "/__init__.py"):
                    if (root() / candidate).is_file() and candidate not in found:
                        pending.append(candidate)
    return sorted(found)


def inventory():
    plan = load_plan()
    load_data(plan)
    entries = [PLAN, DATA, *EXCLUDED, "config/experiments/staged-integration.json",
               "docs/OBSERVABLE_REASONING.md", "docs/llm-requirements.txt",
               "src/neuroshard/evolution/observable_reasoning.py",
               "src/neuroshard/evolution/observable_reasoning_run.py",
               "scripts/run_observable_reasoning.py", "tests/evolution/test_observable_reasoning.py"]
    return {"format": FORMAT + "/freeze", "contract": identity(plan),
            "files": {name: sha256(root() / name) for name in local_source_closure(entries)},
            "gpu_launch_authorized": False, "admission_evidence": False}


def bind_freeze(*, committed=False):
    saved = json.loads((root() / FREEZE).read_text())
    if saved != inventory():
        raise ValueError("Execution differs from its frozen sources")
    if committed:
        names = [*saved["files"], FREEZE]
        subprocess.run(["git", "ls-files", "--error-unmatch", "--", *names], cwd=root(),
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *names], cwd=root(), check=True)
    return identity(saved)


def checked(rows, truths):
    if [row["id"] for row in rows] != [row["id"] for row in truths]:
        raise ValueError("Evaluation identities changed")
    return [{**row, "parsed_answer": parse_answer(row["text"], terminated=row["terminated"]),
             "passed": parse_answer(row["text"], terminated=row["terminated"]) == truth["answer"]}
            for row, truth in zip(rows, truths)]


def score(plan, data, parent, expert, control, training, control_training, target, selector):
    scored = {}
    for name, receipt in (("parent", parent), ("expert", expert), ("control", control)):
        scored[name] = {}
        for role in EVAL_ROLES:
            scored[name][role] = checked(receipt[role], data["roles"][role])
            if name != "parent":
                scored[name][role + "_forced"] = checked(receipt[role + "_forced"], data["roles"][role])
                for row, truth in zip(receipt[role], data["roles"][role]):
                    route, margin = choose(truth["messages"][-1]["content"], selector)
                    if row["route"] != route or row["route_margin"] != margin or row["generation_calls"] != 1:
                        raise ValueError("Serving route differs from the question-only policy")
    mode = Counter(row["answer"] for row in data["roles"]["train_new"]).most_common(1)[0][0]
    constant = [{"passed": row["answer"] == mode} for row in data["roles"]["development"]]
    gains = {name: paired_gain(scored["expert"]["development"], other, plan) for name, other in (
        ("parent", scored["parent"]["development"]), ("control", scored["control"]["development"]),
        ("training_constant", constant))}
    isolated_gain = paired_gain(scored["expert"]["development_forced"],
                                scored["control"]["development_forced"], plan)
    protected = [row["id"] for role in EVAL_ROLES for row in scored["parent"][role] if row["passed"]]
    protected_set = set(protected)
    lost = [row["id"] for role in EVAL_ROLES for row in scored["expert"][role]
            if row["id"] in protected_set and not row["passed"]]
    latency = {name: p95([row["seconds"] for role in EVAL_ROLES for row in scored[name][role]])
               for name in scored}
    qualifies = lambda gain: gain["net"] >= plan["gates"]["minimum_gain"] and gain["lower_95"] > 0
    gates = {
        "new_competence": sum(row["passed"] for row in scored["expert"]["development"]) >= plan["gates"]["minimum_new_correct"],
        "beats_parent_control_and_constant": all(qualifies(gain) for gain in gains.values()),
        "forced_expert_beats_forced_control": qualifies(isolated_gain),
        "protected_set_nonempty": sum(row["passed"] for row in scored["parent"]["retention"]) >= plan["gates"]["minimum_protected"],
        "all_protected_kept": not lost,
        "frozen_parameters_unchanged": training["frozen_unchanged"] and control_training["frozen_unchanged"],
        "matched_actual_training_cpu": control_training["matched_budget"] and control_training["optimization_cpu_seconds"] >= target,
        "complete_response_latency": latency["expert"] <= min(plan["gates"]["maximum_p95_seconds"], plan["gates"]["maximum_latency_ratio"] * latency["control"]),
        "isolated_memory": expert["peak_rss_bytes"] <= min(plan["budget"]["maximum_peak_rss_bytes"], plan["gates"]["maximum_memory_ratio"] * control["peak_rss_bytes"]),
    }
    counts = {name: {role: sum(row["passed"] for row in rows) for role, rows in values.items()}
              for name, values in scored.items()}
    return {"format": FORMAT + "/result", "passed": all(gates.values()), "gates": gates,
            "counts": counts, "served_gains": gains, "forced_gain_over_control": isolated_gain,
            "constant_training_label": mode, "constant_correct": sum(row["passed"] for row in constant),
            "protected": protected, "protected_lost": lost, "p95_complete_response_seconds": latency,
            "answers": scored, "next": "review-mechanism-only" if all(gates.values()) else "stop-this-candidate",
            "admission_evidence": False, "item4_complete": False, "gpu_launch_authorized": False,
            "upgrade_public_0_4_0": False, "settlement_authorized": False}
