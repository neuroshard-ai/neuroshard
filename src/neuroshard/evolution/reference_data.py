"""Complete-conversation inputs for the operated CPU/GPU learning reference.

This module has no neural runtime or ledger imports. These inputs are research
artifacts; they are not a new native execution profile.
"""
import hashlib
import json
import math
import os
import re
from pathlib import Path

from neuroshard.dataflow.store import canonical
from .data import fingerprint, normalized


ROLES = ("test", "retention", "dev", "train")
FORMAT = "neuroshard-learning-reference-v1"


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("wb") as output:
        output.write(canonical(value))
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def integer(value, minimum, maximum, name):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"Invalid {name}: expected {minimum}..{maximum}")
    return value


def validate_plan(plan):
    if plan.get("format") != FORMAT or plan.get("purpose") != "development":
        raise ValueError("This reference supports development experiments only")
    for source in (plan["model"], plan["data"]):
        if (not re.fullmatch(r"[\w.-]+/[\w.-]+", source["repo"])
                or not re.fullmatch("[0-9a-f]{40}", source["revision"])
                or source["license"] != "Apache-2.0"):
            raise ValueError("Pin an Apache-2.0 source and full revision")
    integer(plan["model"]["parameters"], 1, 2_000_000_000, "parameter count")
    integer(plan["max_length"], 32, 4096, "complete conversation length")
    integer(plan["generation_tokens"], 1, 512, "generation length")
    integer(plan["generation_documents"], 1, 64, "generation count")
    if set(plan["roles"]) != set(ROLES):
        raise ValueError("Train, development, retention and test roles are required")
    ranges = []
    for role in ROLES:
        spec = plan["roles"][role]
        if spec["split"] not in ("train", "test"):
            raise ValueError("Use an explicit upstream split")
        start = integer(spec["start"], 0, 10_000_000, "scan start")
        count = integer(spec["scan"], 1, 100_000, "scan size")
        integer(spec["documents"], 1, count, "document quota")
        if any(split == spec["split"] and start < end and before < start + count
               for split, before, end in ranges):
            raise ValueError("Source scan ranges must not overlap")
        ranges.append((spec["split"], start, start + count))
    train = plan["training"]
    integer(train["steps"], 1, 2048, "training steps")
    integer(train["batch_documents"], 1, 128, "effective batch size")
    integer(train["checkpoint_steps"], 1, train["steps"], "checkpoint interval")
    integer(train["warmup_steps"], 0, train["steps"] - 1, "warmup steps")
    integer(train["seed"], 0, 2**32 - 1, "seed")
    for name, minimum, maximum in (("learning_rate", 0, .001), ("weight_decay", 0, 1),
                                    ("clip_norm", 0, 10)):
        value = train[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"Invalid optimizer {name}")
        if name != "weight_decay" and value == 0:
            raise ValueError(f"Optimizer {name} must be positive")
    integer(plan["budget"]["seconds"], 1, 43200, "wall-clock budget")
    integer(plan["budget"]["disk_gib"], 1, 160, "artifact budget")
    return plan


def tokenizer_identity(tokenizer):
    if not tokenizer.is_fast or not tokenizer.chat_template or tokenizer.eos_token_id is None:
        raise ValueError("A fast tokenizer with a chat template and EOS is required")
    return identity({"backend": json.loads(tokenizer.backend_tokenizer.to_str()),
                     "template": tokenizer.chat_template,
                     "special_tokens": tokenizer.special_tokens_map,
                     "eos": tokenizer.eos_token_id})


def conversation(tokenizer, messages, max_length):
    """Mask every assistant answer, preserving its complete context and real EOS.

    No truncation or packing is performed. A document that exceeds the declared
    length is rejected, with that exclusion counted in the preparation report.
    """
    if not isinstance(messages, list) or not 2 <= len(messages) <= 128:
        raise ValueError("Invalid conversation")
    expected = "user"
    size = 0
    for index, message in enumerate(messages):
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                or not isinstance(message["content"], str) or not message["content"].strip()):
            raise ValueError("Invalid message")
        role = message["role"]
        if index == 0 and role == "system":
            pass
        elif role == expected:
            expected = "assistant" if role == "user" else "user"
        else:
            raise ValueError("Conversation roles must alternate")
        size += len(message["content"].encode())
        if size > 256 * 1024 or any(t in message["content"] for t in tokenizer.all_special_tokens):
            raise ValueError("Oversized message or embedded chat control token")
    if messages[-1]["role"] != "assistant":
        raise ValueError("Training conversations must end in an assistant answer")
    def render(items, generation=False):
        return tokenizer.apply_chat_template(items, tokenize=True, add_generation_prompt=generation)
    tokens = render(messages)
    if len(tokens) > max_length:
        raise OverflowError("Complete conversation exceeds maximum length")
    labels = [-100] * len(tokens)
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = render(messages[:index], True)
        complete = render(messages[:index + 1])
        if len(prefix) < 2 or complete[:len(prefix)] != prefix or tokens[:len(complete)] != complete:
            raise ValueError("Template has an unstable assistant prefix")
        answer = complete[len(prefix):]
        if tokenizer.eos_token_id not in answer:
            raise ValueError("Template must terminate each answer with EOS")
        end = len(prefix) + answer.index(tokenizer.eos_token_id) + 1
        labels[len(prefix):end] = tokens[len(prefix):end]
    targets = sum(label != -100 for label in labels[1:])
    if not targets:
        raise ValueError("Conversation has no assistant targets")
    return {"input_ids": tokens, "labels": labels, "targets": targets}


class ExclusionIndex:
    """Exact prompt/document exclusion plus the existing SimHash heuristic."""
    def __init__(self):
        self.exact = set()
        self.bands = {}

    def add(self, messages):
        text = "\n".join(message["content"] for message in messages)
        keys = {identity(normalized(text))}
        keys.update(identity(normalized(message["content"])) for message in messages
                    if message["role"] == "user")
        signature = fingerprint(text)
        bands = [(index, (signature >> (16 * index)) & 65535) for index in range(4)]
        nearby = set().union(*(self.bands.get(band, set()) for band in bands))
        if self.exact.intersection(keys) or any((signature ^ old).bit_count() <= 3 for old in nearby):
            return False
        self.exact.update(keys)
        for band in bands:
            self.bands.setdefault(band, set()).add(signature)
        return True


def prepare_role(rows, tokenizer, spec, index, max_length, source):
    records = []
    rejected = {"invalid": 0, "too_long": 0, "duplicate": 0}
    scanned = 0
    for offset, row in enumerate(rows):
        if offset >= spec["scan"]:
            break
        scanned += 1
        messages = row.get("messages") if isinstance(row, dict) else None
        try:
            encoded = conversation(tokenizer, messages, max_length)
        except OverflowError:
            rejected["too_long"] += 1
            continue
        except (ValueError, TypeError, UnicodeError):
            rejected["invalid"] += 1
            continue
        if not index.add(messages):
            rejected["duplicate"] += 1
            continue
        record = {"messages": messages, "source": source, "row": spec["start"] + offset,
                  **encoded}
        records.append({"id": identity(record), **record})
        if len(records) == spec["documents"]:
            break
    if len(records) != spec["documents"]:
        raise ValueError(f"Source cannot fill quota: {len(records)}/{spec['documents']}; {rejected}")
    return records, {"scanned": scanned, "rejected": rejected,
                     "documents": len(records), "targets": sum(r["targets"] for r in records)}


def read_records(path, expected):
    path = Path(path)
    if sha256(path) != expected:
        raise ValueError("Prepared dataset checksum mismatch")
    return [json.loads(line) for line in path.read_text().splitlines()]
