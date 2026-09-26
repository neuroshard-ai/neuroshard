"""Scoring and placement estimates for the A1 modular-reference contract.

The published checkpoints are executed by ``modular_reference_run``. This module
does not load weights and does not decide that a milestone is complete by itself.
"""

import ast
import hashlib
import json
from pathlib import Path


CATEGORIES = ("conversation", "instruction", "tool-use")
PLAN_FORMAT = "neuroshard-modular-reference-a1/1"

# OLMo-2 7B / BAR shapes. Dense experts=1 has no router; the released MoE does.
HIDDEN = 4096
INTERMEDIATE = 11008
LAYERS = 32
VOCAB = 100278
HEADS = 32
HEAD_DIM = 128


def load_plan(path):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = validate_plan(plan)
    if problems:
        raise ValueError("; ".join(problems))
    return plan


def plan_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_plan(plan):
    problems = []
    if plan.get("format") != PLAN_FORMAT:
        problems.append("unexpected plan format")
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return problems + ["tasks missing"]
    seen = set()
    counts = {category: 0 for category in CATEGORIES}
    for task in tasks:
        identity = task.get("id")
        category = task.get("category")
        if not identity or identity in seen:
            problems.append(f"bad task id {identity}")
        seen.add(identity)
        if category not in counts:
            problems.append(f"{identity} has unknown category")
        else:
            counts[category] += 1
        if task.get("kind") == "exact":
            accept = task.get("accept")
            if not accept or not all(isinstance(item, str) and item for item in accept):
                problems.append(f"{identity} needs a nonempty exact accept list")
        elif task.get("kind") == "tool":
            expect = task.get("expect")
            if not isinstance(expect, list) or len(expect) != 1:
                problems.append(f"{identity} needs exactly one expected call")
            else:
                call = expect[0]
                if not call.get("name") or not isinstance(call.get("arguments"), dict):
                    problems.append(f"{identity} has an incomplete expected call")
                rendered = json.dumps(task.get("messages"), ensure_ascii=False)
                if canonical_call(call) in rendered:
                    problems.append(f"{identity} prompt contains the expected call")
        else:
            problems.append(f"{identity} has an unknown kind")
    for category, count in counts.items():
        if count < 1:
            problems.append(f"{category} has no tasks")
    minimum = plan.get("scoring", {}).get("minimum_baseline_successes_per_category")
    if minimum != 1:
        problems.append("baseline minimum must stay 1")
    return problems


def normalize_reply(text):
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def canonical_call(call):
    arguments = ", ".join(f"{key}={python_literal(value)}" for key, value in call["arguments"].items())
    return f"{call['name']}({arguments})"


def python_literal(value):
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return json.dumps(value).replace("true", "True").replace("false", "False").replace("null", "None")
    raise TypeError(f"unsupported argument value {value!r}")


def score_reply(task, text, terminated):
    """Return a pass only for a finished reply that meets the task's content rule."""
    if not terminated:
        return {"passed": False, "reason": "unterminated"}
    if task["kind"] == "exact":
        normalized = normalize_reply(text)
        if normalized in task["accept"]:
            return {"passed": True, "reason": "exact"}
        return {"passed": False, "reason": "exact-mismatch"}
    try:
        calls = parse_function_calls(text)
    except ValueError as error:
        return {"passed": False, "reason": str(error)}
    expected = task["expect"]
    if calls == expected:
        return {"passed": True, "reason": "tool-call"}
    return {"passed": False, "reason": "tool-mismatch"}


def parse_function_calls(text):
    opener = "<function_calls>"
    closer = "</function_calls>"
    start = text.find(opener)
    if start < 0 or text.find(opener, start + 1) >= 0:
        raise ValueError("function-call-spans")
    end = text.find(closer, start)
    if end < 0 or text.find(closer, end + 1) >= 0:
        raise ValueError("function-call-spans")
    body = text[start + len(opener):end].strip()
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if not lines:
        raise ValueError("function-call-empty")
    return [parse_call(line) for line in lines]


def parse_call(line):
    rewritten = rewrite_json_literals(line)
    try:
        expression = ast.parse(rewritten, mode="eval").body
    except SyntaxError as error:
        raise ValueError("function-call-parse") from error
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        raise ValueError("function-call-parse")
    if expression.args or any(keyword.arg is None for keyword in expression.keywords):
        raise ValueError("positional-arguments")
    arguments = {}
    for keyword in expression.keywords:
        try:
            arguments[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, SyntaxError) as error:
            raise ValueError("function-call-parse") from error
        if not isinstance(arguments[keyword.arg], (str, int, float, bool)) and arguments[keyword.arg] is not None:
            raise ValueError("function-call-parse")
    return {"name": expression.func.id, "arguments": arguments}


def rewrite_json_literals(text):
    replacements = {"true": "True", "false": "False", "null": "None"}
    output = []
    index = 0
    quote = ""
    while index < len(text):
        character = text[index]
        if quote:
            output.append(character)
            if character == "\\":
                if index + 1 < len(text):
                    output.append(text[index + 1])
                    index += 2
                    continue
            elif character == quote:
                quote = ""
            index += 1
            continue
        if character in ("'", '"'):
            quote = character
            output.append(character)
            index += 1
            continue
        matched = False
        for word, replacement in replacements.items():
            if not text.startswith(word, index):
                continue
            end = index + len(word)
            boundary_before = index == 0 or not (text[index - 1].isalnum() or text[index - 1] == "_")
            boundary_after = end >= len(text) or not (text[end].isalnum() or text[end] == "_")
            if boundary_before and boundary_after:
                output.append(replacement)
                index = end
                matched = True
                break
        if not matched:
            output.append(character)
            index += 1
    return "".join(output)


def assess(plan, rows):
    """Combine baseline and modular rows. An empty category cannot pass."""
    by_model = {}
    for row in rows:
        by_model.setdefault(row["model"], {})[row["id"]] = row
    task_ids = [task["id"] for task in plan["tasks"]]
    protected = {category: [] for category in CATEGORIES}
    baseline = by_model.get("baseline", {})
    baseline_complete = list(baseline) == task_ids or set(baseline) == set(task_ids) and len(baseline) == len(task_ids)
    for task in plan["tasks"]:
        row = baseline.get(task["id"])
        if row and row.get("passed") and not row.get("stopped"):
            protected[task["category"]].append(task["id"])
    minimum = plan["scoring"]["minimum_baseline_successes_per_category"]
    baseline_gate = baseline_complete and all(len(ids) >= minimum for ids in protected.values())
    baseline_gate = baseline_gate and all(not baseline[task_id].get("stopped") for task_id in task_ids)
    modular = by_model.get("modular", {})
    modular_complete = (
        set(modular) == set(task_ids)
        and len(modular) == len(task_ids)
        and all(not modular[task_id].get("stopped") for task_id in task_ids)
        and all("text" in modular[task_id] for task_id in task_ids)
    )
    return {
        "baseline_gate": baseline_gate,
        "modular_complete": modular_complete,
        "protected": protected,
        "quality_ready": baseline_gate and modular_complete,
    }


def parameter_layout(experts):
    if experts < 1:
        raise ValueError("experts must be positive")
    attention = 4 * HIDDEN * HIDDEN + 4 * HIDDEN
    feedforward = 3 * HIDDEN * INTERMEDIATE
    router = HIDDEN * experts if experts > 1 else 0
    per_layer = attention + feedforward * experts + router
    embeddings = VOCAB * HIDDEN
    shared_tail = embeddings * 2 + HIDDEN
    parameters = per_layer * LAYERS + shared_tail
    return {
        "experts": experts,
        "parameters": parameters,
        "per_layer_parameters": per_layer,
        "shared_tail_parameters": shared_tail,
        "bytes_bf16": parameters * 2,
        "per_layer_bytes_bf16": per_layer * 2,
        "shared_tail_bytes_bf16": shared_tail * 2,
        "adam_bytes": parameters * 16,
        "per_layer_adam_bytes": per_layer * 16,
    }


def route_estimate():
    """Layer placement for BAR-5x7B. All experts in a layer stay on one worker."""
    layout = parameter_layout(5)
    worker_bytes = 16 * 1024 ** 3
    layers_per_worker = 8
    slice_bytes = layers_per_worker * layout["per_layer_bytes_bf16"]
    first_worker = slice_bytes + layout["shared_tail_bytes_bf16"]
    sequence = 2048
    kv_bytes = 2 * LAYERS * sequence * HEADS * HEAD_DIM * 2
    kv_slice = kv_bytes // (LAYERS // layers_per_worker)
    boundary = HIDDEN * 2
    pipeline_workers = LAYERS // layers_per_worker
    return {
        "layout": layout,
        "inference": {
            "dtype": "bfloat16",
            "workers": pipeline_workers,
            "layers_per_worker": layers_per_worker,
            "experts_colocated_per_layer": True,
            "active_experts_per_token": 5,
            "largest_worker_weight_bytes": first_worker,
            "worker_budget_bytes": worker_bytes,
            "fits_weight_budget": first_worker + kv_slice < worker_bytes,
            "kv_bytes_at_2048": kv_bytes,
            "bytes_per_token_between_workers": boundary * (pipeline_workers - 1),
        },
        "training": {
            "dtype": "fp32 weights, gradients, and both Adam moments",
            "bytes_per_parameter": 16,
            "layers_per_16gib_worker": 1,
            "layer_workers": LAYERS,
            "embedding_worker_adam_bytes": layout["shared_tail_parameters"] * 16,
            "fits_one_layer_in_16gib": layout["per_layer_adam_bytes"] < worker_bytes,
        },
    }
