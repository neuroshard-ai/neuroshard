"""Bounded decoder diagnosis on opened outputs; never a quality admission run."""

import hashlib
import importlib.util
from pathlib import Path
import resource
import time

from neuroshard.evolution import modular_reference_execution as execution
from neuroshard.evolution.modular_reference import load_plan, score_reply


PROFILE = "decoder-parity"
SCRIPT = "scripts/run_modular_decoder_parity.py"


def verify_upstream(amendment):
    package = Path(importlib.util.find_spec("transformers").origin).parent
    for name, digest in amendment["upstream_sources"].items():
        if execution.sha256(package / name) != digest:
            raise ValueError(f"upstream implementation changed: {name}")


def inputs():
    plan_path, amendment_path = execution.PROFILES[PROFILE]
    amendment = execution.read(execution.ROOT / amendment_path)
    verify_upstream(amendment)
    plan = load_plan(execution.ROOT / plan_path)
    prior_path = execution.ROOT / amendment["prior_result"]["path"]
    if execution.sha256(prior_path) != amendment["prior_result"]["sha256"]:
        raise ValueError("prior diagnostic outputs changed")
    prior = execution.read(prior_path)
    tasks = {task["id"]: task for task in plan["tasks"]}
    rows = {row["id"]: row for row in prior["primary"]}
    if (len(rows) != len(prior["primary"]) or set(rows) != set(tasks)
            or amendment["generation_ids"] != [task["id"] for task in plan["tasks"]]
            or not set(amendment["logit_ids"]) <= set(tasks)):
        raise ValueError("incomplete or changed diagnostic selection")
    for row in rows.values():
        execution.checked(plan, row, prior["binding"], "baseline", "primary", tasks[row["id"]])
    return plan, amendment, prior, tasks, rows


def logit_comparison(actual, expected):
    import torch

    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("decoder logit shape or dtype differs")
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())

    def digest(tensor):
        # Hash only the next-token vocabulary vector; compare all positions.
        data = tensor[0, -1].detach().contiguous().view(torch.uint8).numpy().tobytes()
        return hashlib.sha256(data).hexdigest()

    return {"shape": list(actual.shape), "dtype": str(actual.dtype), "finite": finite,
            "equal": finite and torch.equal(actual, expected),
            "argmax_equal": finite and torch.equal(actual.argmax(-1), expected.argmax(-1)),
            "maximum_absolute_difference": float((actual.float() - expected.float()).abs().max()) if finite else None,
            "streamed_next_logits_sha256": digest(actual), "upstream_next_logits_sha256": digest(expected)}


def generation_comparison(task, text, token_ids, terminated, rendered_sha256, prior):
    score = score_reply(task, text, terminated)
    return {"id": task["id"], "category": task["category"], "text": text,
            "token_ids": token_ids, "terminated": terminated, "rendered_sha256": rendered_sha256,
            **score,
            "matched": (token_ids == prior["token_ids"] and text == prior["text"]
                        and terminated == prior["terminated"] and rendered_sha256 == prior["rendered_sha256"]
                        and all(score[key] == prior[key] for key in ("passed", "reason")))}


def diagnose(model_dir, plan, amendment, tasks, prior_rows, progress_path):
    """Keep the upstream model independent of the streamed module/weight loader."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache
    from neuroshard.evolution.modular_reference_run import (
        Checkpoint, assign_weights, forward_logits, generate_task, layer_classes, prepare_runtime)

    prepare_runtime(plan["limits"]["threads"])
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    load_seconds = time.monotonic() - started
    config = model.config
    checkpoint = Checkpoint(model_dir)
    layer_class, norm_class, rotary_class = layer_classes(config)
    with torch.device("meta"):
        norm = norm_class(config.hidden_size, eps=config.rms_norm_eps)
    norm = assign_weights(norm, {"weight": checkpoint.tensor("model.norm.weight")})
    rotary = rotary_class(config)
    embed, head = checkpoint.tensor("model.embed_tokens.weight"), checkpoint.tensor("lm_head.weight")
    result = {"execution_completed": True, "decoder_agreement": False,
              "logit_checks": [], "generations": [], "upstream_load_seconds": load_seconds,
              "quality_ready": False, "admission_evidence": False, "milestone_complete": False,
              "opened_diagnostic_cases": True, "stop_reason": None}

    def record():
        result["seconds"] = time.monotonic() - started
        result["max_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        execution.save(progress_path, result)

    with torch.inference_mode():
        for identity in amendment["logit_ids"]:
            task = tasks[identity]
            tokens = tokenizer.apply_chat_template(task["messages"], tokenize=True,
                                                  add_generation_prompt=True, return_tensors="pt")
            upstream_cache, streamed_cache = DynamicCache(config=config), DynamicCache(config=config)
            seen = 0
            for index, recorded_token in enumerate(prior_rows[identity]["token_ids"]):
                tick = time.monotonic()
                expected = model(input_ids=tokens,
                    attention_mask=torch.ones((1, seen + tokens.shape[1]), dtype=torch.long),
                    past_key_values=upstream_cache, use_cache=True, logits_to_keep=0).logits
                upstream_seconds = time.monotonic() - tick
                tick = time.monotonic()
                actual = forward_logits(config, layer_class, checkpoint, embed, head, norm,
                                        rotary, streamed_cache, tokens, seen)
                comparison = {"id": identity, "prediction_index": index, "seen": seen,
                              "streamed_seconds": time.monotonic() - tick,
                              "upstream_seconds": upstream_seconds, **logit_comparison(actual, expected)}
                result["logit_checks"].append(comparison)
                del actual, expected
                if not comparison["equal"]:
                    result["stop_reason"] = "decoder-logit-disagreement"
                    record()
                    return result
                seen += tokens.shape[1]
                # This is an opened-output diagnostic, not an answering policy.
                tokens = torch.tensor([[recorded_token]], dtype=torch.long)
                record()
            del upstream_cache, streamed_cache

        for identity in amendment["generation_ids"]:
            task = tasks[identity]
            rendered = tokenizer.apply_chat_template(task["messages"], tokenize=False, add_generation_prompt=True)
            prompt = tokenizer.apply_chat_template(task["messages"], tokenize=True,
                                                  add_generation_prompt=True, return_tensors="pt")
            tick = time.monotonic()
            # Use the standard greedy loop. Matching projection shape avoids
            # confusing last-logit-only GEMM rounding with a decoder defect.
            output = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                max_new_tokens=plan["limits"]["max_new_tokens"], do_sample=False, num_beams=1,
                use_cache=True, logits_to_keep=0, return_dict_in_generate=False)
            seconds = time.monotonic() - tick
            generated = output[0, prompt.shape[1]:].tolist()
            eos = model.generation_config.eos_token_id
            eos = [eos] if isinstance(eos, int) else eos
            terminated = bool(generated) and generated[-1] in eos
            text = tokenizer.decode(generated[:-1] if terminated else generated, skip_special_tokens=True)
            comparison = generation_comparison(task, text, generated, terminated,
                hashlib.sha256(rendered.encode()).hexdigest(), prior_rows[identity])
            comparison["seconds"] = seconds
            result["generations"].append(comparison)
            if not comparison["matched"]:
                result["stop_reason"] = "upstream-generation-disagreement"
                record()
                # One predeclared same-host control distinguishes decoder
                # disagreement from differences with the earlier machine/run.
                streamed = generate_task(plan, "baseline", model_dir, task)
                result["same_host_streamed"] = streamed
                if streamed["stopped"]:
                    result.update(execution_completed=False, error="same-host control exceeded its reply budget")
                    record()
                    return result
                result["same_host_streamed_matches_upstream"] = generation_comparison(
                    task, streamed["text"], streamed["token_ids"], streamed["terminated"],
                    streamed["rendered_sha256"], comparison)["matched"]
                result["same_host_streamed_matches_recorded"] = generation_comparison(
                    task, streamed["text"], streamed["token_ids"], streamed["terminated"],
                    streamed["rendered_sha256"], prior_rows[identity])["matched"]
                record()
                return result
            record()
    result["decoder_agreement"] = True
    record()
    return result


def validate_completion(result, amendment, prior_rows):
    if not result.get("decoder_agreement"):
        return
    expected_steps = [(identity, index) for identity in amendment["logit_ids"]
                      for index in range(len(prior_rows[identity]["token_ids"]))]
    if (not result.get("execution_completed") or result.get("stop_reason") is not None
            or [(row["id"], row["prediction_index"]) for row in result["logit_checks"]] != expected_steps
            or not all(row["equal"] and row["argmax_equal"] and row["finite"] for row in result["logit_checks"])
            or [row["id"] for row in result["generations"]] != amendment["generation_ids"]
            or not all(row["matched"] for row in result["generations"])):
        raise ValueError("incomplete or disagreeing decoder audit cannot pass")


def worker(request_path):
    request_path = Path(request_path)
    request = execution.read(request_path)
    execution.configure_runtime(PROFILE)
    frozen = execution.freeze(profile=PROFILE)
    if request["profile"] != PROFILE or request["freeze"] != frozen or request["phase"] != "parity":
        raise ValueError("parity worker binding changed")
    plan, amendment, _, tasks, prior_rows = inputs()
    inventory = execution.read(execution.ROOT / execution.ARTIFACTS)["models"]["baseline"]
    model_dir = Path(request["models"]) / "baseline"
    if execution.file_state(model_dir, inventory) != request["file_state"]:
        raise ValueError("checkpoint changed before parity")
    result = diagnose(model_dir, plan, amendment, tasks, prior_rows, request_path.parent / "progress.json")
    validate_completion(result, amendment, prior_rows)
    if execution.file_state(model_dir, inventory) != request["file_state"]:
        raise ValueError("checkpoint changed during parity")
    if result["max_rss_bytes"] > plan["limits"]["max_rss_bytes"]:
        raise ValueError("parity worker exceeded recorded RSS budget")
    result.update(binding=request["binding"], process_cpu_seconds=time.process_time())
    execution.save(request_path.parent / "reply.json", result, exclusive=True)


def run(home, models):
    home, models = Path(home).resolve(), Path(models).resolve()
    execution.configure_runtime(PROFILE)
    frozen = execution.freeze(profile=PROFILE)
    plan, amendment, prior, _, prior_rows = inputs()
    binding = {"freeze": frozen, "profile": PROFILE, "models": str(models),
               "prior_result_sha256": amendment["prior_result"]["sha256"]}
    execution.save(home / "study.json", {"binding": binding, "started_unix": time.time()}, exclusive=True)
    result = None
    try:
        execution.save(home / "status.json", {"state": "running", "phase": "prepare"})
        prepared = execution.launch(home, models, binding, "baseline", "prepare",
            amendment["preparation_seconds"], plan["limits"]["max_rss_bytes"])
        execution.save(home / "status.json", {"state": "running", "phase": "decoder-parity"})
        result = execution.launch(home, models, binding, "baseline", "parity",
            amendment["worker_seconds"], plan["limits"]["max_rss_bytes"],
            stats=prepared["file_state"], worker_script=SCRIPT)
        validate_completion(result, amendment, prior_rows)
    except Exception as error:
        result = {"execution_completed": False, "decoder_agreement": False, "error": str(error)}
    finally:
        if result is not None:
            result.update(binding=binding, quality_ready=False, admission_evidence=False, milestone_complete=False,
                accounting={"historical_baseline_evaluation_seconds": prior["accounting"]["baseline"]["evaluation_seconds"],
                            "new_parity_worker_seconds": execution.spent(home, "baseline", 0),
                            "new_preparation_seconds": execution.spent(home, "baseline", 0, preparation=True)})
            execution.save(home / "result.json", result, exclusive=True)
            execution.save(home / "status.json", {"state": "finished" if result["execution_completed"] else "stopped",
                "decoder_agreement": result["decoder_agreement"], "error": result.get("error"),
                "stop_reason": result.get("stop_reason")})
    return result
