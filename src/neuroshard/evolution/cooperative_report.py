"""Recompute the frozen experiment's claims from retained public evidence."""
import statistics

from . import grounded_tasks as tasks
from . import cooperative as group
from . import reference as engine
from . import reference_data as data


ARMS = ("clean-single", "damaged-single", "clean-pair")


def paired_accuracy(baseline, candidate):
    """Exact one-sided McNemar test, conditional on discordant pairs."""
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("Accuracy comparison requires equal, nonempty pairs")
    if any(type(value) is not bool for value in [*baseline, *candidate]):
        raise ValueError("Accuracy outcomes must be booleans")
    paired = tasks.paired_accuracy(
        [{"id": index, "correct": value} for index, value in enumerate(baseline)],
        [{"id": index, "correct": value} for index, value in enumerate(candidate)],
    )
    return {
        "documents": len(baseline), "baseline_correct": sum(baseline),
        "candidate_correct": sum(candidate), "gained": paired["wins"], "lost": paired["losses"],
        "accuracy_gain": paired["accuracy_change"],
        "exact_one_sided_p": paired["one_sided_p"],
    }


def checked_outcomes(report, records, prepared, arm, selection):
    if (report["arm"] != arm or report["role"] != "test"
            or report["prepared"] != data.identity(prepared)):
        raise ValueError("Evaluation identity differs from the final test")
    expected_candidate = None if arm == "seed" else selection["candidates"][arm]
    if report["candidate"] != expected_candidate:
        raise ValueError("Evaluation differs from the selected candidate")
    ids = [record["id"] for record in records]
    if ([row["id"] for row in report["generations"]] != ids
            or [row["id"] for row in report["checks"]] != ids
            or len(set(ids)) != len(ids)):
        raise ValueError("Final-test answers are missing, duplicated or reordered")
    outcomes = []
    for record, generation, recorded in zip(records, report["generations"], report["checks"]):
        expected = {
            "id": record["id"], "family": record["task"]["family"],
            "variant": record["task"]["variant"],
            **tasks.check_answer(record["task"], generation["text"]),
        }
        if expected != recorded or generation["messages"] != record["messages"][:-1]:
            raise ValueError("Retained answer, prompt or strict check differs")
        outcomes.append(expected["correct"])
    if report["correct"] != sum(outcomes) or report["documents"] != len(outcomes):
        raise ValueError("Evaluation totals differ from checked answers")
    return outcomes


def learning_report(prepared, selection, records, evaluations):
    if selection["prepared"] != data.identity(prepared) or set(selection["candidates"]) != set(ARMS):
        raise ValueError("Selection must bind all three candidates to the preparation")
    if [row["id"] for row in records] != prepared["roles"]["test"]["ids"]:
        raise ValueError("Test records differ from the committed partition")
    outcomes = {
        arm: checked_outcomes(evaluations[arm], records, prepared, arm, selection)
        for arm in ("seed", *ARMS)
    }
    comparisons = {}
    for arm in ARMS:
        comparisons[arm] = {
            **paired_accuracy(outcomes["seed"], outcomes[arm]),
            "retention": engine.paired_summary(evaluations["seed"]["retention"], evaluations[arm]["retention"]),
        }
    for report in evaluations.values():
        if [row["id"] for row in report["retention"]] != prepared["roles"]["retention"]["ids"]:
            raise ValueError("Retention rows differ from the committed probes")
    primary = comparisons["clean-single"]
    contract = prepared["plan"]["quality_contract"]
    conditions = {
        "minimum_accuracy_gain": primary["accuracy_gain"] >= contract["minimum_accuracy_gain"],
        "maximum_one_sided_p": primary["exact_one_sided_p"] <= contract["maximum_one_sided_p"],
        "retention_99pct_upper_max": primary["retention"]["normal_99pct_upper"] <= contract["retention_99pct_upper_max"],
    }
    breakdowns = {}
    for field in ("family", "variant"):
        breakdowns[field] = {}
        for value in sorted({record["task"][field] for record in records}):
            indices = [i for i, record in enumerate(records) if record["task"][field] == value]
            breakdowns[field][str(value)] = {
                "documents": len(indices),
                **{arm: sum(outcomes[arm][i] for i in indices) for arm in outcomes},
            }
    pair = paired_accuracy(outcomes["clean-single"], outcomes["clean-pair"])
    return {
        "prepared": data.identity(prepared), "selection": data.identity(selection),
        "primary_arm": "clean-single", "primary_conditions": conditions,
        "narrow_learning_contract_passed": all(conditions.values()),
        "comparisons_to_seed": comparisons, "descriptive_breakdowns": breakdowns,
        "clean_vs_damaged": paired_accuracy(outcomes["damaged-single"], outcomes["clean-single"]),
        "pair_vs_single": pair,
        "descriptive_pair_accuracy_within_margin": pair["accuracy_gain"] >= -prepared["plan"]["cooperation"]["maximum_accuracy_drop_vs_single"],
        "scope": contract["scope"], "serving_approved": False, "tokens_issued": 0,
    }


def training_report(prepared, results):
    recipe = prepared["plan"]["training"]
    train_ids = prepared["roles"]["train"]["ids"]
    expected_schedule = [
        [train_ids[index] for index in batch]
        for batch in engine.schedule(len(train_ids), recipe["steps"], recipe["batch_documents"], recipe["seed"])
    ]
    summaries = {}
    for arm in ARMS:
        ranks = results[arm]
        world = 2 if arm == "clean-pair" else 1
        if [row["rank"] for row in ranks] != list(range(world)):
            raise ValueError("Missing or duplicate training ranks")
        if len({row["parameter_digest"] for row in ranks}) != 1:
            raise ValueError("Ranks disagree on final parameters")
        first = ranks[0]
        schedule = [step["documents"] for step in first["steps"]]
        if schedule != expected_schedule:
            raise ValueError("Experiment arm differs from the fixed global schedule")
        if len(schedule) != prepared["plan"]["training"]["steps"]:
            raise ValueError("Incomplete training schedule")
        for rank in ranks:
            binding = data.identity({"prepared": data.identity(prepared),
                                     "profile": group.runtime_profile(rank["runtime"]), "arm": arm, "world": world})
            if (rank["arm"] != arm or rank["world"] != world
                    or rank["binding"] != binding
                    or [row["documents"] for row in rank["steps"]] != schedule):
                raise ValueError("Training rank, binding or schedule differs")
        network = []
        for rank in ranks:
            network.append({
                name: {direction: values[direction] - rank["network_start"][name][direction]
                       for direction in ("rx", "tx")}
                for name, values in rank["network_end"].items()
            })
        step_seconds = [step["seconds"] for step in first["steps"]]
        summaries[arm] = {
            "world": world, "parameter_digest": first["parameter_digest"],
            "updates": len(schedule), "update_seconds": sum(step_seconds),
            "median_update_seconds": statistics.median(step_seconds),
            "training_seconds_including_checkpoints": max(row["seconds"] for row in ranks),
            "allocated_gpu_seconds_including_checkpoints": sum(row["seconds"] for row in ranks),
            "peak_cuda_allocated_bytes": [row["peak_cuda_allocated_bytes"] for row in ranks],
            "host_interface_byte_deltas": network,
            "network_scope": "Whole host interface during measured training; not isolated NCCL bytes",
        }
    summaries["pair_update_speedup"] = summaries["clean-single"]["update_seconds"] / summaries["clean-pair"]["update_seconds"]
    summaries["pair_training_loop_speedup"] = summaries["clean-single"]["training_seconds_including_checkpoints"] / summaries["clean-pair"]["training_seconds_including_checkpoints"]
    summaries["timer_scope"] = "Training loop through final digest, including checkpoints; excludes seed verification, model loading and process-group initialization. Retain process resource logs and whole-instance lifetime separately."
    return summaries


def serving_report(reports, expected_digest):
    if set(reports) != {"single", "pair", "failure"}:
        raise ValueError("Retain all three serving phases")
    baseline = None
    concurrency = reports["single"]["concurrency"]
    for report in reports.values():
        if report["model_digest"] != expected_digest:
            raise ValueError("Serving used a different model")
        if any(not result["success"] for result in report["results"]):
            raise ValueError("Serving phase has failed requests; inspect raw evidence")
        if any(result["answer"].get("cached") is not False for result in report["results"]):
            raise ValueError("Serving throughput requires fresh inference, without retry-cache hits")
        if report["concurrency"] != concurrency or report["requests"] != len(report["results"]):
            raise ValueError("Serving concurrency or request count differs")
        outputs = [(row["task_id"], row["answer"]["generation"]["output_ids"]) for row in report["results"]]
        if baseline is not None and outputs != baseline:
            raise ValueError("Serving phases produced different tasks or answer tokens")
        baseline = outputs
    return {
        "identical_output_tokens": True,
        "pair_throughput_ratio": reports["pair"]["requests_per_second"] / reports["single"]["requests_per_second"],
        "failure_retries": sum(len(row["attempts"]) - 1 for row in reports["failure"]["results"]),
        "phases": {name: {key: value for key, value in report.items() if key != "results"}
                   for name, report in reports.items()},
        "scope": "Fixed task workload, equal client concurrency; operated replicas and volatile retry cache",
    }
