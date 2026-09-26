import json
from pathlib import Path

import torch
from jinja2 import Environment
from transformers import Olmo2Config
from transformers.models.olmo2.modeling_olmo2 import Olmo2DecoderLayer

from neuroshard.evolution.modular_reference import (
    assess, load_plan, parameter_layout, route_estimate, score_reply, validate_plan)
from neuroshard.evolution.modular_reference_run import assign_weights, classify_keys


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "config/experiments/modular-reference-a1.json"
TEMPLATE = (ROOT / "config/experiments/modular-reference-chat-template.jinja").read_text(encoding="utf-8")


def task(plan, identity):
    return next(item for item in plan["tasks"] if item["id"] == identity)


def test_plan_pins_both_published_checkpoints_and_three_nonempty_categories():
    plan = load_plan(PLAN)
    assert plan["decision"] == "BAR"
    assert plan["models"]["baseline"]["parameters"] == 7_298_011_136
    assert plan["models"]["modular"]["parameters"] == 24_612_753_408
    assert plan["models"]["modular"]["experts_per_token"] == 5
    assert plan["limits"]["gpu"] is False
    assert plan["limits"]["max_new_tokens"] == 32
    counts = {}
    for item in plan["tasks"]:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
    assert counts == {"conversation": 3, "instruction": 3, "tool-use": 3}
    assert validate_plan(plan) == []


def test_layout_matches_the_published_parameter_counts():
    assert parameter_layout(1)["parameters"] == 7_298_011_136
    assert parameter_layout(5)["parameters"] == 24_612_753_408
    route = route_estimate()
    assert route["inference"]["workers"] == 4
    assert route["inference"]["active_experts_per_token"] == 5
    assert route["inference"]["fits_weight_budget"] is True
    assert route["training"]["fits_one_layer_in_16gib"] is True
    assert route["inference"]["largest_worker_weight_bytes"] < route["inference"]["worker_budget_bytes"]


def test_exact_score_rejects_prose_and_unfinished_replies():
    plan = load_plan(PLAN)
    apples = task(plan, "conversation-apples")
    assert score_reply(apples, "3", terminated=True)["passed"] is True
    assert score_reply(apples, " 3\r\n", terminated=True)["passed"] is True
    assert score_reply(apples, "The answer is 3", terminated=True)["passed"] is False
    assert score_reply(apples, "3", terminated=False)["reason"] == "unterminated"
    words = task(plan, "instruction-three-words")
    assert score_reply(words, "blue glass river", terminated=True)["passed"] is True
    assert score_reply(words, "blue glass river.", terminated=True)["passed"] is False


def test_tool_score_requires_the_selected_call_and_keyword_arguments():
    plan = load_plan(PLAN)
    weather = task(plan, "tool-weather")
    reply = 'The forecast follows.\n<function_calls>\nget_weather(city="Paris")\n</function_calls>'
    assert score_reply(weather, reply, terminated=True)["passed"] is True
    assert score_reply(weather, '<function_calls>get_weather(city="Lyon")</function_calls>', terminated=True)["passed"] is False
    assert score_reply(weather, '<function_calls>get_time(city="Paris")</function_calls>', terminated=True)["passed"] is False
    assert score_reply(weather, '<function_calls>get_weather("Paris")</function_calls>', terminated=True)["reason"] == "positional-arguments"
    extra = '<function_calls>\nget_weather(city="Paris")\nget_time(city="Paris")\n</function_calls>'
    assert score_reply(weather, extra, terminated=True)["passed"] is False
    addition = task(plan, "tool-add")
    assert score_reply(addition, "<function_calls>add(a=12, b=30)</function_calls>", terminated=True)["passed"] is True
    assert score_reply(addition, "<function_calls>add(b=30, a=12)</function_calls>", terminated=True)["passed"] is True
    assert score_reply(addition, "42", terminated=True)["reason"] == "function-call-spans"


def test_json_literals_inside_strings_are_not_rewritten():
    plan = load_plan(PLAN)
    balance = task(plan, "tool-balance")
    reply = '<function_calls>lookup_balance(account="true", currency="USD")</function_calls>'
    scored = score_reply(balance, reply, terminated=True)
    assert scored["passed"] is False
    reply = '<function_calls>lookup_balance(account="cedar-47", currency="USD")</function_calls>'
    assert score_reply(balance, reply, terminated=True)["passed"] is True


def recorded(item, model, passed=True, stopped=False):
    return {"id": item["id"], "model": model, "passed": passed, "stopped": stopped, "text": "x"}


def test_empty_or_partial_baseline_cannot_open_the_quality_gate():
    plan = load_plan(PLAN)
    assert assess(plan, [])["quality_ready"] is False
    partial = [recorded(item, "baseline") for item in plan["tasks"] if item["category"] != "tool-use"]
    decision = assess(plan, partial)
    assert decision["baseline_gate"] is False
    assert decision["protected"]["tool-use"] == []
    complete = [recorded(item, "baseline") for item in plan["tasks"]]
    complete += [recorded(item, "modular", passed=False) for item in plan["tasks"]]
    ready = assess(plan, complete)
    assert ready["baseline_gate"] is True
    assert ready["protected"]["conversation"]
    assert ready["modular_complete"] is True
    assert ready["quality_ready"] is True
    broken = [row for row in complete if not (row["model"] == "modular" and row["id"] == "tool-weather")]
    broken.append(recorded(task(plan, "tool-weather"), "modular", passed=False, stopped=True))
    assert assess(plan, broken)["quality_ready"] is False


def test_template_keeps_functions_and_does_not_embed_the_expected_call():
    plan = load_plan(PLAN)
    rendered = Environment().from_string(TEMPLATE).render(
        messages=task(plan, "tool-weather")["messages"], add_generation_prompt=True, eos_token="<|endoftext|>")
    assert "<functions>" in rendered
    assert 'get_weather(city="Paris")' not in rendered
    assert rendered.endswith("<|im_start|>assistant\n")


def test_checkpoint_keys_and_meta_assignment_round_trip():
    keys = {
        "model.embed_tokens.weight": "a",
        "lm_head.weight": "c",
        "model.norm.weight": "a",
        "model.layers.0.self_attn.q_proj.weight": "a",
        "model.layers.31.mlp.experts.4.down_proj.weight": "b",
    }
    classified = classify_keys(keys, 32)
    assert classified["unclassified"] == []
    assert classified["counts"] == {"embed": 1, "head": 1, "norm": 1, "layer": 2}
    assert classify_keys({"optimizer.step": "a"}, 32)["unclassified"] == ["optimizer.step"]
    config = Olmo2Config(
        hidden_size=32, intermediate_size=64, num_attention_heads=4, num_key_value_heads=4,
        num_hidden_layers=1, vocab_size=128, head_dim=8)
    with torch.device("meta"):
        layer = Olmo2DecoderLayer(config, 0)
    state = {name: torch.zeros(tuple(param.shape), dtype=torch.bfloat16) for name, param in layer.named_parameters()}
    assign_weights(layer, state)
    assert layer.self_attn.q_proj.weight.dtype == torch.bfloat16
    assert layer.self_attn.q_proj.weight.device.type == "cpu"


def test_vendored_template_matches_the_plan_hash():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    import hashlib
    digest = hashlib.sha256(TEMPLATE.encode("utf-8")).hexdigest()
    assert digest == plan["models"]["baseline"]["files"]["chat_template.jinja"]
    assert digest == plan["models"]["modular"]["files"]["chat_template.jinja"]
