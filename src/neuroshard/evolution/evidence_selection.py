"""Bounded evidence selection with exact, request-bound source copying.

This is an experimental serving primitive, not a ledger transition. A valid
receipt proves how an answer was copied from supplied bytes. It does not prove
that the source is true, the selection is relevant, or a neural worker ran.
"""

import hashlib
import re

from neuroshard.dataflow.store import canonical

FORMAT = "neuroshard-evidence-selection/1"
LABELS = "ABCDEFGHIJKLMNOP"
ABSTAIN = "Z"
MAX_SOURCE_BYTES = 65536
MAX_REQUEST_BYTES = 32768


def root(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def fields(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("invalid evidence object fields")


def bounded_text(value, maximum):
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > maximum:
        raise ValueError("invalid or oversized evidence text")
    return value


def build_menu(messages, sources):
    """Only public conversation and byte spans enter the neural choice space.

    Span inventories are supplied by the data adapter, never by evaluation
    answers. There is no implicit truncation, fuzzy matching, or source repair.
    """
    if not isinstance(messages, list) or not 1 <= len(messages) <= 32:
        raise ValueError("require a bounded conversation")
    for message in messages:
        fields(message, {"role", "content"})
        if message["role"] not in ("user", "assistant"):
            raise ValueError("evidence requests contain user/assistant turns only")
        bounded_text(message["content"], MAX_REQUEST_BYTES)
    if messages[-1]["role"] != "user" or len(canonical(messages)) > MAX_REQUEST_BYTES:
        raise ValueError("require a bounded request ending with the user")
    if not isinstance(sources, list) or len(sources) > len(LABELS):
        raise ValueError("too many evidence sources")
    identifiers = set()
    candidates = []
    total_bytes = 0
    for source in sources:
        fields(source, {"id", "text", "spans"})
        identifier = bounded_text(source["id"], 256)
        if identifier in identifiers:
            raise ValueError("duplicate source identifier")
        identifiers.add(identifier)
        text = bounded_text(source["text"], MAX_SOURCE_BYTES)
        raw = text.encode("utf-8")
        total_bytes += len(raw)
        spans = source["spans"]
        if not isinstance(spans, list) or not 1 <= len(spans) <= len(LABELS):
            raise ValueError("require a bounded nonempty span inventory")
        seen = set()
        for span in spans:
            fields(span, {"start", "end"})
            start, end = span["start"], span["end"]
            if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(raw)
                    or (start, end) in seen):
                raise ValueError("invalid or duplicate evidence span")
            seen.add((start, end))
            try:
                value = raw[start:end].decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("span splits a UTF-8 character") from error
            candidates.append({"source_id": identifier, "source_sha256": hashlib.sha256(raw).hexdigest(),
                               "start": start, "end": end, "value": value, "text": text})
    if total_bytes > MAX_SOURCE_BYTES or len(candidates) > len(LABELS):
        raise ValueError("evidence exceeds the request budget")
    request_root = root(messages)
    # Bind source IDs, full bytes and the complete span inventory, including spans
    # not selected. Canonicalize transport ordering before assigning local labels.
    normalized = [{**source, "spans": sorted(source["spans"], key=lambda s: (s["start"], s["end"]))}
                  for source in sorted(sources, key=lambda s: s["id"])]
    source_root = root(normalized)
    candidates.sort(key=lambda item: root({"request": request_root, "candidate": item}))
    choices = [{"choice": LABELS[index], **item} for index, item in enumerate(candidates)]
    payload = {"format": FORMAT, "request_root": request_root, "source_root": source_root,
               "choices": choices, "abstain": ABSTAIN}
    return {**payload, "invocation_root": root(payload)}


def public_choices(menu):
    """Same source information can be shown to pointer and free-text controls."""
    return [{key: item[key] for key in ("choice", "source_id", "text", "value")}
            for item in menu["choices"]]


def resolve(messages, sources, decision):
    """Reconstruct the expected invocation; reject stale or relabeled decisions."""
    fields(decision, {"invocation_root", "choice"})
    menu = build_menu(messages, sources)
    if decision["invocation_root"] != menu["invocation_root"]:
        raise ValueError("decision belongs to another request or source snapshot")
    choice = decision["choice"]
    if not isinstance(choice, str):
        raise ValueError("choice must be an exact local label")
    selected = next((item for item in menu["choices"] if item["choice"] == choice), None)
    if selected is None and choice != ABSTAIN:
        raise ValueError("choice is outside this invocation")
    answer = {"answer": selected["value"] if selected else None,
              "sources": [selected["source_id"]] if selected else []}
    evidence = ({key: selected[key] for key in ("source_id", "source_sha256", "start", "end")}
                if selected else None)
    return {"format": FORMAT, "invocation_root": menu["invocation_root"], "choice": choice,
            "answer": answer, "evidence": evidence}


def verify_receipt(messages, sources, receipt):
    fields(receipt, {"format", "invocation_root", "choice", "answer", "evidence"})
    expected = resolve(messages, sources, {key: receipt[key] for key in ("invocation_root", "choice")})
    # Canonical bytes preserve JSON types, unlike Python's True == 1.
    if canonical(expected) != canonical(receipt):
        raise ValueError("receipt differs from exact source resolution")
    return True


def label_token_ids(tokenizer, menu):
    """The one-step decision requires exact one-token labels, without aliases."""
    labels = [item["choice"] for item in menu["choices"]] + [ABSTAIN]
    result = {}
    for label in labels:
        tokens = tokenizer(label, add_special_tokens=False)["input_ids"]
        if (len(tokens) != 1 or type(tokens[0]) is not int
                or tokenizer.decode(tokens, skip_special_tokens=False) != label
                or tokens[0] in tokenizer.all_special_ids):
            raise ValueError("tokenizer does not support exact one-token choices")
        result[label] = tokens[0]
    if len(set(result.values())) != len(result):
        raise ValueError("choice labels share a token")
    return result


def record_value_source(document):
    """Adapter for the declared '... is VALUE.' fixture format, not general NLP.

    Process every public record using the same syntax. This adapter accepts no
    task, question, expected answer, category, or oracle retrieval query.
    """
    fields(document, {"id", "text"})
    text = document["text"]
    bounded_text(text, MAX_SOURCE_BYTES)
    match = re.fullmatch(r"(.+ is )(.+)\.", text, flags=re.DOTALL)
    if match is None:
        raise ValueError("record does not match the declared value format")
    start = len(match[1].encode("utf-8"))
    return {**document, "spans": [{"start": start, "end": start + len(match[2].encode("utf-8"))}]}
