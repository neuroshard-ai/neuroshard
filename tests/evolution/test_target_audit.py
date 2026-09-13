from neuroshard.evolution.target_audit import audit_answer, audit_conversation


def test_repeated_word_is_not_satisfied_by_related_words_or_one_occurrence():
    instruction = 'In your response, the word "happy" should appear at least 3 times.'
    failed = audit_answer(instruction, "Happiness matters; feel happy and happily continue.")[0]
    assert failed["observed"] == 1 and not failed["passes"]
    assert audit_answer(instruction, "Happy, happy! Be HAPPY.")[0]["passes"]


def test_keyword_check_does_not_accept_healthier_for_health():
    instruction = "Include keywords [health, fitness, lifestyle] in the response."
    failed = audit_answer(instruction, "A healthier lifestyle.")[0]
    assert failed["observed"] == {"health": 0, "fitness": 0, "lifestyle": 1}
    assert not failed["passes"]


def test_exclusive_word_limit_and_exact_ending_are_separate_obligations():
    instruction = 'Your response should contain less than 4 words. Finish your response with this exact phrase "All done."'
    assert all(check["passes"] for check in audit_answer(instruction, "Yes. All done.\n"))
    checks = audit_answer(instruction, "Now yes. All done!")
    assert not checks[0]["passes"] and not checks[1]["passes"]


def test_bullets_count_markdown_markers_without_counting_prose_asterisks():
    instruction = "The response must contain exactly 3 bullet points."
    assert audit_answer(instruction, "* one\n- two\n  + three\nThis *word* is emphasized.")[0]["passes"]
    assert not audit_answer(instruction, "* one\n* two")[0]["passes"]


def test_each_assistant_turn_uses_its_own_user_request_and_records_system_origin():
    messages = [
        {"role": "system", "content": "Your response should contain less than 10 words."},
        {"role": "user", "content": 'In your response, the word "happy" should appear at least 3 times.'},
        {"role": "assistant", "content": "Happy happy happy."},
        {"role": "user", "content": "Say hello."},
        {"role": "assistant", "content": "Hello."},
    ]
    checks = audit_conversation(messages)
    assert [(c["assistant_turn"], c["instruction_origin"], c["kind"]) for c in checks] == [
        (2, "system", "word_limit"), (2, "user", "word_minimum"), (4, "system", "word_limit")]


def test_unknown_instructions_and_ambiguous_bracketed_endings_remain_unjudged():
    assert audit_answer("Be accurate. Finish with [this exact phrase].", "Unknown") == []


def test_missing_assistant_answer_is_not_counted_as_a_checked_target():
    assert audit_conversation([{"role": "user", "content": "Your response should contain less than 3 words."}]) == []


def test_keyword_template_variables_are_not_misread_as_required_literals():
    instruction = "Include keywords [keywords] in the response. [keywords] are: cat, sunshine"
    assert audit_answer(instruction, "A cat enjoys sunshine.") == []


def test_cli_preserves_input_and_previous_report(tmp_path):
    import hashlib
    import json
    import subprocess
    import sys
    from pathlib import Path

    source, output = tmp_path / "records.jsonl", tmp_path / "report.json"
    record = {"id": "example", "messages": [
        {"role": "user", "content": 'In your response, the word "happy" should appear at least 3 times.'},
        {"role": "assistant", "content": "Happy."}]}
    raw = (json.dumps(record) + "\n").encode()
    source.write_bytes(raw)
    command = [sys.executable, str(Path(__file__).resolve().parents[2] / "scripts/audit_instruction_targets.py"),
               str(source), "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True)
    saved = output.read_bytes()
    report = json.loads(saved)
    assert report["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert report["counts"]["records_with_failed_checks"] == 1
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert output.read_bytes() == saved and source.read_bytes() == raw
