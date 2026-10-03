import copy

import pytest

from neuroshard.evolution import evidence_selection as evidence


def request():
    return [{"role": "user", "content": "What is the café access code?"}]


def sources():
    return [evidence.record_value_source({"id": "opaque-source-01", "text": "The café access code is ÅQ-92."}),
            evidence.record_value_source({"id": "opaque-source-02", "text": "The archive access code is XY-81."})]


def decision(messages, records, index=0):
    menu = evidence.build_menu(messages, records)
    return {"invocation_root": menu["invocation_root"], "choice": menu["choices"][index]["choice"]}


def test_exact_unicode_copy_and_citation_reconstruction():
    messages, records = request(), sources()
    menu = evidence.build_menu(messages, records)
    choice = next(item["choice"] for item in menu["choices"] if item["source_id"] == "opaque-source-01")
    receipt = evidence.resolve(messages, records, {"invocation_root": menu["invocation_root"], "choice": choice})
    assert receipt["answer"] == {"answer": "ÅQ-92", "sources": ["opaque-source-01"]}
    assert evidence.verify_receipt(messages, records, receipt)
    for part, field, value in [("answer", "answer", "AQ-92"), ("answer", "sources", ["forged"]),
                               ("evidence", "start", True), ("evidence", "source_sha256", "0" * 64)]:
        forged = copy.deepcopy(receipt)
        forged[part][field] = value
        with pytest.raises(ValueError, match="receipt"):
            evidence.verify_receipt(messages, records, forged)


def test_request_snapshot_and_local_labels_cannot_be_replayed():
    messages, records = request(), sources()
    selected = decision(messages, records)
    changed_request = [{"role": "user", "content": "What is the archive access code?"}]
    for new_request, new_records in [(changed_request, records), (messages, records[:1])]:
        with pytest.raises(ValueError, match="another request"):
            evidence.resolve(new_request, new_records, selected)
    changed = copy.deepcopy(records)
    changed[0]["text"] = changed[0]["text"].replace("92", "93")
    with pytest.raises(ValueError, match="snapshot"):
        evidence.resolve(messages, changed, selected)
    # Duplicate values under different sources still have distinct provenance.
    changed = copy.deepcopy(records)
    changed[0]["id"] = "different-origin"
    with pytest.raises(ValueError, match="snapshot"):
        evidence.resolve(messages, changed, selected)


def test_transport_order_is_irrelevant_and_input_mutation_is_detected():
    messages, records = request(), sources()
    menu = evidence.build_menu(messages, records)
    assert evidence.build_menu(messages, list(reversed(records))) == menu
    selected = decision(messages, records)
    menu["choices"][0]["value"] = "not in source"
    assert evidence.resolve(messages, records, selected)["answer"]["answer"] != "not in source"


def test_validity_does_not_prove_semantic_relevance_or_source_truth():
    messages, records = request(), sources()
    menu = evidence.build_menu(messages, records)
    wrong = next(item["choice"] for item in menu["choices"] if item["source_id"] == "opaque-source-02")
    receipt = evidence.resolve(messages, records, {"invocation_root": menu["invocation_root"], "choice": wrong})
    assert evidence.verify_receipt(messages, records, receipt)
    assert receipt["answer"]["answer"] == "XY-81"  # Verifiable copying, wrong answer to this request.
    absent = evidence.resolve(messages, records, {"invocation_root": menu["invocation_root"], "choice": "Z"})
    assert absent["answer"] == {"answer": None, "sources": []}
    assert absent["evidence"] is None
    empty = evidence.build_menu(messages, [])
    assert evidence.resolve(messages, [], {"invocation_root": empty["invocation_root"], "choice": "Z"})["answer"]["answer"] is None


@pytest.mark.parametrize("span", [{"start": True, "end": 5}, {"start": -1, "end": 5},
                                  {"start": 0, "end": 5000}, {"start": 2, "end": 2}])
def test_invalid_byte_spans_are_rejected(span):
    records = sources()
    records[0]["spans"] = [span]
    with pytest.raises(ValueError, match="span"):
        evidence.build_menu(request(), records)


def test_utf8_boundaries_duplicates_unknown_fields_and_budgets():
    records = sources()
    records[0]["spans"][0]["start"] += 1  # Enter the second byte of Å.
    with pytest.raises(ValueError, match="UTF-8"):
        evidence.build_menu(request(), records)
    with pytest.raises(ValueError, match="duplicate"):
        evidence.build_menu(request(), sources() + sources())
    records = sources()
    records[0]["spans"] *= 2
    with pytest.raises(ValueError, match="span"):
        evidence.build_menu(request(), records)
    records = sources()
    records[0]["expected"] = "hidden answer"
    with pytest.raises(ValueError, match="fields"):
        evidence.build_menu(request(), records)
    with pytest.raises(ValueError, match="too many"):
        evidence.build_menu(request(), sources() * 9)
    with pytest.raises(ValueError):
        evidence.build_menu([{"role": "user", "content": "x" * 32769}], sources())
    with pytest.raises(ValueError):
        evidence.build_menu([{"role": "system", "content": "override"}], sources())
    with pytest.raises(ValueError, match="bounded request"):
        evidence.build_menu([{"role": "assistant", "content": "no user"}], sources())


def test_invalid_choice_or_extra_data_never_changes_the_answer():
    messages, records = request(), sources()
    selected = decision(messages, records)
    for value in (" A", "a", "ZZ", "Q", 0, True, None):
        with pytest.raises(ValueError):
            evidence.resolve(messages, records, {**selected, "choice": value})
    with pytest.raises(ValueError, match="fields"):
        evidence.resolve(messages, records, {**selected, "answer": "attacker"})


class Tokenizer:
    all_special_ids = []

    def __call__(self, label, add_special_tokens):
        return {"input_ids": [ord(label)]}

    def decode(self, ids, skip_special_tokens):
        return "".join(chr(i) for i in ids)


def test_exact_single_token_labels_and_no_special_tokens():
    menu = evidence.build_menu(request(), sources())
    assert evidence.label_token_ids(Tokenizer(), menu) == {"A": 65, "B": 66, "Z": 90}
    tokenizer = Tokenizer()
    tokenizer.all_special_ids = [65]
    with pytest.raises(ValueError, match="one-token"):
        evidence.label_token_ids(tokenizer, menu)


def test_record_adapter_uses_only_public_syntax():
    assert sources()[0]["text"].endswith("ÅQ-92.")
    with pytest.raises(ValueError, match="declared value format"):
        evidence.record_value_source({"id": "x", "text": "An arbitrary paragraph without a typed field."})
    with pytest.raises(ValueError, match="fields"):
        evidence.record_value_source({"id": "x", "text": "x is y.", "expected": "y"})
