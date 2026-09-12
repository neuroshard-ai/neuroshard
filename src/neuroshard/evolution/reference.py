"""Operated full-model learning reference, outside native consensus.

FP32 parameters/AdamW states are retained on both devices. CUDA uses BF16
autocast for execution. Reproducibility is scoped to a recorded environment;
this implementation makes no claim of cross-device byte-identical execution.
"""
import contextlib
import math
import os
import platform
import random
import time
import uuid
from pathlib import Path

from .reference_data import identity, save, sha256


def configure(device, threads=2):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    if device not in ("cpu", "cuda"):
        raise ValueError("Choose one CPU or CUDA device explicitly")
    if device == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise ValueError("CUDA reference requires a BF16-capable GPU")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from importlib.metadata import version
    return {"device": device, "python": platform.python_version(), "machine": platform.machine(),
            "host": platform.node(), "torch_build": identity(torch.__config__.show()),
            "torch": version("torch"), "transformers": version("transformers"),
            "tokenizers": version("tokenizers"), "safetensors": version("safetensors"),
            "threads": threads, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "parameters": "float32", "optimizer": "float32-adamw",
            "autocast": "bfloat16" if device == "cuda" else None,
            "attention": "sdpa", "deterministic_algorithms": True,
            "cross_device_exact": False}


def autocast(device):
    import torch
    return torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext()


def load_model(directory, device, parameters):
    import torch
    from transformers import AutoModelForCausalLM
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        # Four FP32 arrays: parameter, gradient and two Adam moments. Reserve
        # additional space for activations and kernels; this is a preflight
        # estimate, not a guarantee that an arbitrary batch will fit.
        estimate = parameters * 16 + 4 * 1024**3
        if free < estimate:
            raise ValueError(f"Insufficient free GPU memory: {free} bytes; estimate {estimate}")
    model = AutoModelForCausalLM.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="sdpa")
    actual = sum(parameter.numel() for parameter in model.parameters())
    if actual != parameters or any(not parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Seed parameter count differs or some parameters are frozen")
    model.config.use_cache = False
    return model.to(device)


def optimizer_for(model, recipe):
    import torch
    decay, other = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else other).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": recipe["weight_decay"]},
         {"params": other, "weight_decay": 0.0}],
        lr=recipe["learning_rate"], betas=(.9, .95), eps=1e-8, foreach=False)


def learning_rate(recipe, step):
    warmup = recipe["warmup_steps"]
    if step < warmup:
        return recipe["learning_rate"] * (step + 1) / warmup
    progress = (step - warmup) / max(1, recipe["steps"] - warmup - 1)
    return recipe["learning_rate"] * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))


def schedule(size, steps, batch_documents, seed):
    if min(size, steps, batch_documents) <= 0:
        raise ValueError("A schedule requires documents, steps and a positive batch size")
    generator = random.Random(seed)
    indices = []
    required = steps * batch_documents
    while len(indices) < required:
        epoch = list(range(size))
        generator.shuffle(epoch)
        indices.extend(epoch)
    return [indices[start:start + batch_documents]
            for start in range(0, required, batch_documents)]


def response_loss(model, record, device):
    import torch
    import torch.nn.functional as functional
    tokens = torch.tensor([record["input_ids"]], dtype=torch.long, device=device)
    labels = torch.tensor(record["labels"][1:], dtype=torch.long, device=device)
    with autocast(device):
        logits = model(input_ids=tokens, use_cache=False).logits[0, :-1]
        # Compute CE in FP32; prompt/control/padding labels remain ignored.
        loss = functional.cross_entropy(logits.float(), labels, ignore_index=-100, reduction="sum")
    if not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite assistant loss")
    return loss


def train_step(model, optimizer, records, device, recipe, step, check=lambda: None):
    import torch
    total_targets = sum(record["targets"] for record in records)
    if not total_targets:
        raise ValueError("Training batch has no assistant targets")
    rate = learning_rate(recipe, step)
    for group in optimizer.param_groups:
        group["lr"] = rate
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for record in records:
        check()
        loss = response_loss(model, record, device)
        loss_sum += float(loss.detach())
        # Weight accumulation by actual answer tokens, not equally by each
        # differently sized microbatch. This equals a single padded batch loss.
        (loss / total_targets).backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["clip_norm"], error_if_nonfinite=True)
    check()
    optimizer.step()
    return {"step": step + 1, "loss": loss_sum / total_targets, "targets": total_targets,
            "learning_rate": rate, "gradient_norm": float(norm),
            "documents": [record["id"] for record in records]}


def score(model, records, device, check=lambda: None):
    import torch
    model.eval()
    result = []
    with torch.inference_mode():
        for record in records:
            check()
            loss = float(response_loss(model, record, device))
            result.append({"id": record["id"], "targets": record["targets"],
                           "loss_sum": loss, "loss": loss / record["targets"]})
    return result


def generate(model, tokenizer, records, device, max_new_tokens, count, check=lambda: None):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    class BudgetStop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            check()
            return False
    model.eval()
    result = []
    with torch.inference_mode():
        for record in records[:count]:
            check()
            messages = record["messages"][:-1]
            tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            if len(tokens) + max_new_tokens > model.config.max_position_embeddings:
                raise ValueError("Generation exceeds model context; no silent prompt truncation")
            inputs = torch.tensor([tokens], dtype=torch.long, device=device)
            if device == "cuda":
                torch.cuda.synchronize()
            started = time.monotonic()
            with autocast(device):
                output = model.generate(
                    inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
                    max_new_tokens=max_new_tokens, use_cache=True,
                    pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([BudgetStop()]))
            ids = output[0, len(tokens):].tolist()
            elapsed = time.monotonic() - started
            result.append({"id": record["id"], "messages": messages, "input_ids": tokens,
                           "output_ids": ids, "text": tokenizer.decode(ids, skip_special_tokens=True),
                           "seconds": elapsed, "tokens_per_second": len(ids) / elapsed,
                           "ended_with_eos": bool(ids and ids[-1] == tokenizer.eos_token_id),
                           "hit_token_limit": len(ids) == max_new_tokens and ids[-1] != tokenizer.eos_token_id})
    return result


def paired_summary(baseline, candidate):
    import statistics
    if ([row["id"] for row in baseline] != [row["id"] for row in candidate]
            or len(baseline) < 2):
        raise ValueError("Paired evaluation requires identical document IDs and at least two documents")
    if any(before.get("targets") != after.get("targets") for before, after in zip(baseline, candidate)):
        raise ValueError("Paired target counts differ")
    changes = [after["loss"] - before["loss"] for before, after in zip(baseline, candidate)]
    if not all(math.isfinite(value) for value in changes):
        raise ValueError("Nonfinite paired evaluation")
    mean = statistics.mean(changes)
    error = statistics.stdev(changes) / math.sqrt(len(changes))
    return {"documents": len(changes), "mean_change": mean, "standard_error": error,
            "normal_99pct_upper": mean + 2.576 * error,
            "improved": sum(value < 0 for value in changes),
            "worsened": sum(value > 0 for value in changes),
            "scope": "descriptive development comparison; no serving approval"}


class Budget:
    def __init__(self, home, started, limits, clock=time.time):
        self.home, self.started, self.limits, self.clock = Path(home), started, limits, clock
        self.last_disk_check = -math.inf
        self.peak_bytes = 0

    def check(self, force_disk=False):
        now = self.clock()
        if now >= self.started + self.limits["seconds"]:
            raise TimeoutError("Original experiment wall-clock budget exhausted")
        if force_disk or now - self.last_disk_check >= 30:
            size = sum(path.stat().st_size for path in self.home.rglob("*") if path.is_file())
            self.peak_bytes = max(self.peak_bytes, size)
            self.last_disk_check = now
            if size > self.limits["disk_gib"] * 1024**3:
                raise ValueError("Experiment artifact budget exhausted")


def checkpoint(home, model, tokenizer, optimizer, step, binding, records):
    import torch
    directory = Path(home) / f"checkpoint-{step:06d}"
    if directory.exists():
        raise ValueError("Checkpoint already exists; resume from its durable pointer")
    temporary = Path(home) / f".checkpoint-{step:06d}-{uuid.uuid4().hex}"
    temporary.mkdir()
    model.save_pretrained(temporary, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(temporary)
    torch.save({"optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else []},
               temporary / "optimizer.pt")
    files = {path.name: sha256(path) for path in temporary.iterdir() if path.is_file()}
    receipt = {"step": step, "binding": binding, "files": files, "records": records}
    save(temporary / "checkpoint.json", receipt)
    # Files and directory are durable before publishing the pointer.
    for path in temporary.iterdir():
        with path.open("rb") as source:
            os.fsync(source.fileno())
    os.replace(temporary, directory)
    pointer = {"directory": directory.name, "receipt": identity(receipt)}
    save(Path(home) / "latest.json", pointer)
    return pointer


def verify_checkpoint(home, pointer, binding):
    import json
    import re
    if not re.fullmatch(r"checkpoint-[0-9]{6}", pointer["directory"]):
        raise ValueError("Invalid checkpoint directory")
    directory = Path(home) / pointer["directory"]
    receipt = json.loads((directory / "checkpoint.json").read_bytes())
    if receipt["binding"] != binding or identity(receipt) != pointer["receipt"]:
        raise ValueError("Checkpoint belongs to different inputs or runtime")
    actual_files = {path.name for path in directory.iterdir()}
    if actual_files != set(receipt["files"]) | {"checkpoint.json"}:
        raise ValueError("Checkpoint contains missing or unexpected files")
    for name, expected in receipt["files"].items():
        if Path(name).name != name or sha256(directory / name) != expected:
            raise ValueError("Checkpoint file checksum mismatch")
    return directory, receipt


def restore_optimizer(directory, optimizer, device):
    import torch
    # Only our locally created, hash-verified checkpoints reach this function.
    state = torch.load(Path(directory) / "optimizer.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["cpu_rng"])
    if device == "cuda":
        torch.cuda.set_rng_state_all(state["cuda_rng"])
