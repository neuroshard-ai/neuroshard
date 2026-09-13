import copy
import json

import pytest

from neuroshard.evolution import cooperative_report as report
from neuroshard.evolution import grounded_tasks as tasks
from neuroshard.evolution import reference_data as data


def test_exact_paired_accuracy_counts_regressions_and_discordant_ties():
    result = report.paired_accuracy([False] * 10 + [True] * 2, [True] * 10 + [False] * 2)
    assert result["gained"] == 10 and result["lost"] == 2
    assert result["accuracy_gain"] == 8 / 12
    # Under the null: 12 choose 10, 11 or 12 out of 2**12 equiprobable patterns.
    assert result["exact_one_sided_p"] == 79 / 4096
    assert report.paired_accuracy([True, False], [True, False])["exact_one_sided_p"] == 1


def test_paired_accuracy_rejects_missing_or_nonboolean_outcomes():
    with pytest.raises(ValueError):
        report.paired_accuracy([True], [])
    with pytest.raises(ValueError):
        report.paired_accuracy([1], [True])


def test_serving_comparison_rejects_different_answers_and_failed_requests():
    def phase(tokens, success=True):
        return {"model_digest": "model", "requests_per_second": 2, "concurrency": 2, "requests": 1,
                "results": [{"success": success, "task_id": "task", "attempts": [{}],
                             "answer": {"cached": False, "generation": {"output_ids": tokens}}}]}
    reports = {name: phase([1, 2]) for name in ("single", "pair", "failure")}
    assert report.serving_report(reports, "model")["identical_output_tokens"]
    reports["pair"] = phase([1, 3])
    with pytest.raises(ValueError, match="different tasks or answer tokens"):
        report.serving_report(reports, "model")
    reports["pair"] = phase([1, 2], False)
    with pytest.raises(ValueError, match="failed requests"):
        report.serving_report(reports, "model")
    reports["pair"] = phase([1, 2])
    reports["pair"]["results"][0]["answer"]["cached"] = True
    with pytest.raises(ValueError, match="fresh inference"):
        report.serving_report(reports, "model")


def test_learning_contract_rechecks_answers_and_requires_retention():
    cases = [tasks.make_case(42, "test", index) for index in range(16)]
    records = [{"id": tasks.task_identity(case), "task": case,
                "messages": [{"role": "user", "content": tasks.prompt(case)}, {"role": "assistant", "content": "target"}]}
               for case in cases]
    prepared = {
        "roles": {"test": {"ids": [row["id"] for row in records]}, "retention": {"ids": ["a", "b"]}},
        "plan": {"cooperation": {"maximum_accuracy_drop_vs_single": .02},
                 "quality_contract": {"minimum_accuracy_gain": .05, "maximum_one_sided_p": .01,
                                       "retention_99pct_upper_max": .02, "scope": "fixture"}},
    }
    selection = {"prepared": data.identity(prepared), "candidates": {arm: {"name": arm} for arm in report.ARMS}}
    evaluations = {}
    for arm in ("seed", *report.ARMS):
        texts = ["invalid" if arm == "seed" else json.dumps(tasks.expected(case)) for case in cases]
        checks = [{"id": row["id"], "family": row["task"]["family"], "variant": row["task"]["variant"],
                   **tasks.check_answer(row["task"], text)} for row, text in zip(records, texts)]
        evaluations[arm] = {
            "arm": arm, "role": "test", "prepared": data.identity(prepared),
            "candidate": None if arm == "seed" else selection["candidates"][arm],
            "checks": checks, "documents": 16, "correct": sum(row["correct"] for row in checks),
            "generations": [{"id": row["id"], "messages": row["messages"][:-1], "text": text}
                            for row, text in zip(records, texts)],
            "retention": [{"id": name, "targets": 1, "loss": 1.0} for name in ("a", "b")],
        }
    result = report.learning_report(prepared, selection, records, evaluations)
    assert result["narrow_learning_contract_passed"] and not result["serving_approved"]
    wrong = copy.deepcopy(evaluations)
    wrong["clean-single"]["generations"][0]["text"] = "invalid"
    with pytest.raises(ValueError, match="Retained answer"):
        report.learning_report(prepared, selection, records, wrong)
    for row in evaluations["clean-single"]["retention"]:
        row["loss"] = 1.1
    result = report.learning_report(prepared, selection, records, evaluations)
    assert not result["narrow_learning_contract_passed"]
    assert not result["primary_conditions"]["retention_99pct_upper_max"]


def test_training_report_rejects_a_different_schedule_or_disagreeing_ranks():
    prepared = {"plan": {"training": {"steps": 1, "batch_documents": 2, "seed": 42}},
                "roles": {"train": {"ids": ["a", "b"]}}}
    results = {}
    for arm in report.ARMS:
        world = 2 if arm == "clean-pair" else 1
        binding = data.identity({"prepared": data.identity(prepared), "profile": {}, "arm": arm, "world": world})
        results[arm] = [{
            "rank": rank, "world": world, "arm": arm, "binding": binding, "runtime": {},
            "parameter_digest": "agreed", "steps": [{"documents": ["b", "a"], "seconds": 1}],
            "seconds": 2, "network_start": {}, "network_end": {}, "peak_cuda_allocated_bytes": 1,
        } for rank in range(world)]
    assert report.training_report(prepared, results)["pair_update_speedup"] == 1
    changed = copy.deepcopy(results)
    changed["clean-single"][0]["steps"][0]["documents"] = ["a", "b"]
    with pytest.raises(ValueError, match="fixed global schedule"):
        report.training_report(prepared, changed)
    results["clean-pair"][1]["parameter_digest"] = "different"
    with pytest.raises(ValueError, match="disagree on final parameters"):
        report.training_report(prepared, results)
