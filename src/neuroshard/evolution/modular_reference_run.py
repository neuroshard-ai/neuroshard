"""Stream published BAR checkpoints one layer at a time on CPU.

Scoring lives in ``modular_reference`` and runs only after a reply is recorded.
This process keeps embeddings, one decoder layer, and the cache, not the full
weight set.
"""

import gc
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask

from neuroshard.evolution.modular_reference import score_reply


def prepare_runtime(threads=1):
    torch.set_num_threads(threads)
    torch.set_grad_enabled(False)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_small_files(plan, which, model_dir):
    expected = plan["models"][which]["files"]
    found = {}
    for name, digest in expected.items():
        actual = file_sha256(Path(model_dir) / name)
        if actual != digest:
            raise RuntimeError(f"{name} hash {actual} does not match the plan")
        found[name] = actual
    return found


class Checkpoint:
    def __init__(self, model_dir):
        index_path = Path(model_dir) / "model.safetensors.index.json"
        self.weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        self.model_dir = Path(model_dir)
        self.handles = {}
        self.layers = {}
        for key in self.weight_map:
            parts = key.split(".")
            if len(parts) > 3 and parts[0] == "model" and parts[1] == "layers" and parts[2].isdigit():
                self.layers.setdefault(int(parts[2]), []).append(key)

    def tensor(self, name):
        filename = self.weight_map[name]
        if filename not in self.handles:
            self.handles[filename] = safe_open(self.model_dir / filename, framework="pt", device="cpu")
        return self.handles[filename].get_tensor(name)

    def layer_weights(self, index):
        prefix = f"model.layers.{index}."
        return {key[len(prefix):]: self.tensor(key) for key in self.layers[index]}


def classify_keys(weight_map, layers):
    unclassified = []
    counts = {"embed": 0, "head": 0, "norm": 0, "layer": 0}
    allowed = {
        "model.embed_tokens.weight": "embed",
        "lm_head.weight": "head",
        "model.norm.weight": "norm",
    }
    for key in weight_map:
        if key in allowed:
            counts[allowed[key]] += 1
            continue
        parts = key.split(".")
        if len(parts) > 3 and parts[0] == "model" and parts[1] == "layers" and parts[2].isdigit():
            if int(parts[2]) < layers:
                counts["layer"] += 1
                continue
        unclassified.append(key)
    return {"counts": counts, "unclassified": unclassified}


def layer_classes(config):
    if config.model_type == "olmo2":
        from transformers.models.olmo2.modeling_olmo2 import (
            Olmo2DecoderLayer, Olmo2RMSNorm, Olmo2RotaryEmbedding)
        return Olmo2DecoderLayer, Olmo2RMSNorm, Olmo2RotaryEmbedding
    if config.model_type == "flex_olmo":
        from transformers.models.flex_olmo.modeling_flex_olmo import (
            FlexOlmoDecoderLayer, FlexOlmoRMSNorm, FlexOlmoRotaryEmbedding)
        return FlexOlmoDecoderLayer, FlexOlmoRMSNorm, FlexOlmoRotaryEmbedding
    raise RuntimeError(f"unsupported model type {config.model_type}")


def assign_weights(module, weights):
    missing = [name for name, _ in module.named_parameters() if name not in weights]
    extra = [name for name in weights if name not in dict(module.named_parameters())]
    if missing or extra:
        raise RuntimeError(f"weight mismatch missing={missing} extra={extra}")
    result = module.load_state_dict(weights, strict=True, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("assigned state dict was rejected")
    return module


def generate_task(plan, which, model_dir, task):
    prepare_runtime(plan["limits"]["threads"])
    started = time.monotonic()
    verify_small_files(plan, which, model_dir)
    config = AutoConfig.from_pretrained(model_dir)
    config._attn_implementation = plan["limits"]["attention"]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    rendered = tokenizer.apply_chat_template(
        task["messages"], tokenize=False, add_generation_prompt=True)
    if task["kind"] == "tool" and "<functions>" not in rendered:
        raise RuntimeError("chat template dropped the function list")
    prompt = tokenizer.apply_chat_template(
        task["messages"], tokenize=True, add_generation_prompt=True, return_tensors="pt")
    checkpoint = Checkpoint(model_dir)
    classified = classify_keys(checkpoint.weight_map, config.num_hidden_layers)
    if classified["unclassified"] or classified["counts"]["embed"] != 1:
        raise RuntimeError(f"unexpected checkpoint layout {classified}")
    layer_cls, norm_cls, rotary_cls = layer_classes(config)
    embed = checkpoint.tensor("model.embed_tokens.weight")
    head = checkpoint.tensor("lm_head.weight")
    with torch.device("meta"):
        norm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
    norm = assign_weights(norm, {"weight": checkpoint.tensor("model.norm.weight")})
    rotary = rotary_cls(config)
    cache = DynamicCache(config=config)
    eos_ids = set(json.loads((Path(model_dir) / "generation_config.json").read_text())["eos_token_id"])
    generated = []
    first_token_seconds = None
    budget_exhausted = False
    seen = 0
    tokens = prompt
    deadline = started + plan["limits"]["per_task_seconds"]
    while len(generated) < plan["limits"]["max_new_tokens"]:
        if time.monotonic() > deadline:
            budget_exhausted = True
            break
        logits = forward_logits(
            config, layer_cls, checkpoint, embed, head, norm, rotary, cache, tokens, seen)
        if time.monotonic() > deadline:
            budget_exhausted = True
            del logits
            break
        seen += tokens.shape[1]
        choice = int(torch.argmax(logits[0, -1]).item())
        generated.append(choice)
        if first_token_seconds is None:
            first_token_seconds = time.monotonic() - started
        del logits
        if choice in eos_ids:
            break
        tokens = torch.tensor([[choice]], dtype=torch.long)
        gc.collect()
    terminated = bool(generated) and generated[-1] in eos_ids
    reply_ids = generated[:-1] if terminated else generated
    text = tokenizer.decode(reply_ids, skip_special_tokens=True)
    scored = score_reply(task, text, terminated)
    if budget_exhausted:
        scored = {"passed": False, "reason": "time-limit"}
    return {
        "id": task["id"],
        "model": which,
        "category": task["category"],
        "text": text,
        "token_ids": generated,
        "terminated": terminated,
        "passed": scored["passed"],
        "reason": scored["reason"],
        "prompt_tokens": int(prompt.shape[1]),
        "generated_tokens": len(generated),
        "seconds": round(time.monotonic() - started, 3),
        "first_token_seconds": first_token_seconds,
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "rendered_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "stopped": budget_exhausted,
    }


def forward_logits(config, layer_cls, checkpoint, embed, head, norm, rotary, cache, tokens, seen):
    hidden = F.embedding(tokens, embed)
    cache_position = torch.arange(seen, seen + tokens.shape[1])
    position_ids = cache_position.unsqueeze(0)
    attention_mask = torch.ones((1, seen + tokens.shape[1]), dtype=torch.long)
    causal = create_causal_mask(
        config=config, input_embeds=hidden, attention_mask=attention_mask,
        cache_position=cache_position, past_key_values=cache, position_ids=position_ids)
    position_embeddings = rotary(hidden, position_ids)
    for index in range(config.num_hidden_layers):
        with torch.device("meta"):
            layer = layer_cls(config, index)
        layer = assign_weights(layer, checkpoint.layer_weights(index))
        hidden = layer(
            hidden, attention_mask=causal, position_ids=position_ids,
            past_key_values=cache, cache_position=cache_position,
            position_embeddings=position_embeddings, use_cache=True)
        del layer
        gc.collect()
    return F.linear(norm(hidden), head)
