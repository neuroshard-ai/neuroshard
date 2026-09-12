#!/usr/bin/env python3
"""Inspect a pinned seed on four public development prompts; never train it.

This is a diagnostic with exact-format checks, not a held-out quality benchmark.
It shares the learning reference's numerical backend and publishes every answer.
"""
import argparse
import json
import resource
import time
from pathlib import Path

from neuroshard.evolution import reference as engine
from neuroshard.evolution import reference_data as data
from run_learning_reference import snapshot


ROOT = Path(__file__).resolve().parents[1]


def check_answer(text, rule):
    if rule["kind"] == "manual":
        return None
    if rule["kind"] == "exact":
        return text.strip().casefold() == rule["expected"].casefold()
    if rule["kind"] == "json":
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate JSON key")
                result[key] = value
            return result
        try:
            value = json.loads(text, object_pairs_hook=unique_object)
        except ValueError:
            return False
        return (value == rule["expected"] and isinstance(value, dict)
                and all(type(value[key]) is type(expected) for key, expected in rule["expected"].items()))
    raise ValueError("Unsupported diagnostic rule")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    plan = data.validate_plan(json.loads(args.plan.read_bytes()))
    pinned = snapshot(args.model_dir, plan["model"])
    prompts_path = ROOT / "config/experiments/assistant-probes.json"
    prompts = json.loads(prompts_path.read_bytes())
    args.home.mkdir(parents=True, exist_ok=True)
    if (args.home / "result.json").exists():
        raise ValueError("Preserve the completed diagnostic; use a separate output home")
    runtime = engine.configure(args.device, threads=2)
    budget = engine.Budget(args.home, time.time(), {"seconds": 1200, "disk_gib": 1})
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    model = engine.load_model(args.model_dir, args.device, plan["model"]["parameters"])
    records = [{"id": prompt["id"], "messages": prompt["messages"] + [{"role": "assistant", "content": ""}]}
               for prompt in prompts]
    results = []
    for record, prompt in zip(records, prompts):
        generated = engine.generate(model, tokenizer, [record], args.device, 128, 1, budget.check)[0]
        generated["check"] = prompt["check"]
        generated["passes_exact_check"] = check_answer(generated["text"], prompt["check"])
        results.append(generated)
        data.save(args.home / "progress.json", {"responses": results})
        print(json.dumps(generated), flush=True)
    data.save(args.home / "result.json", {
        "scope": "four public development probes; no training or general capability claim",
        "model": pinned, "runtime": runtime, "prompts_sha256": data.sha256(prompts_path),
        "driver_sha256": data.sha256(Path(__file__)), "max_new_tokens": 128, "responses": results,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024})


if __name__ == "__main__":
    main()
