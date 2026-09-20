#!/usr/bin/env python3
"""One frozen CPU candidate, including producer, auditor and modeled traffic costs."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
import statistics
import subprocess
import time

import numpy as np

import hybrid_shard_verifier as hybrid
import neural_work_reference as reference


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "config/experiments/hybrid-shard-verification.json"
FROZEN_PATHS = (
    "config/experiments/hybrid-shard-verification.json",
    "scripts/neural_work_reference.py",
    "scripts/hybrid_shard_verifier.py",
    "scripts/study_hybrid_verification.py",
    "tests/test_hybrid_shard_verifier.py",
)


def timed(operation):
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    result = operation()
    return result, {"cpu_seconds": time.process_time() - cpu_start,
                    "wall_seconds": time.perf_counter() - wall_start}


def fixture(shape: list[int], seed: int) -> reference.Job:
    batch, width, outputs = shape
    rng = np.random.default_rng(seed)
    inputs = rng.integers(-64, 65, (batch, width), dtype=np.int64)
    weights = rng.integers(-16, 17, (width, outputs), dtype=np.int64)
    teacher = rng.integers(-16, 17, (width, outputs), dtype=np.int64)
    targets = reference.rounded_divide(inputs @ teacher, 64)
    return reference.Job(inputs, weights, targets)


def sample(job: reference.Job, trace: dict, *, reverse: bool) -> dict:
    (job_payload, identity), preparation = timed(lambda: (hybrid.encode_job(job), job.work_id()))
    measured = {"input_preparation": preparation}
    claims = {}
    for label in (("hybrid", "boundary") if reverse else ("boundary", "hybrid")):
        claims[label], measured[label + "_production"] = timed(
            lambda label=label: hybrid.produce(job, trace, identity, hybrid=label == "hybrid"))
    def verify_with_fresh_seed():
        seed = secrets.token_bytes(32)  # both payloads already fixed and committed
        hybrid.verify(job_payload, identity, claims["hybrid"], seed=seed,
                      committed_root=claims["hybrid"].commitment)
        return seed.hex()
    operations = [
        ("replay_optimized", lambda: hybrid.replay(job_payload, identity, claims["boundary"],
                                                  optimized=True)),
        ("hybrid_verification", verify_with_fresh_seed),
        ("replay_reference", lambda: hybrid.replay(job_payload, identity, claims["boundary"],
                                                  optimized=False)),
    ]
    if reverse:
        operations.reverse()
    seed_hex = None
    for name, operation in operations:
        value, measured[name] = timed(operation)
        if name == "hybrid_verification":
            seed_hex = value
    for name, operation in (
        ("ordinary_training_optimized", lambda: hybrid.train_optimized(job)),
        ("ordinary_training_reference", lambda: reference.train(job, backend="float64")),
    ):
        _, measured[name] = timed(operation)
    return {"timings": measured, "verifier_seed": seed_hex,
            "commitments": {name: claim.commitment for name, claim in claims.items()},
            "bytes": {"job": len(job_payload),
                      "boundary_payload": len(claims["boundary"].payload),
                      "hybrid_payload": len(claims["hybrid"].payload)}}


def account(sample_record: dict, scenario: dict) -> dict:
    timings = sample_record["timings"]
    lengths = sample_record["bytes"]
    # Both messages include the 32-byte work identity and the 32-byte commitment.
    # The hybrid exchange additionally delivers the 32-byte verifier challenge.
    input_bytes = 0 if scenario["inputs_cached"] else lengths["job"]
    boundary_bytes = input_bytes + lengths["boundary_payload"] + 64
    hybrid_bytes = input_bytes + lengths["hybrid_payload"] + 96
    cpu_rate = scenario["cpu_usd_per_hour"] / 3600
    byte_rate = scenario["transfer_usd_per_decimal_gb"] / 1e9
    def bill(production, verification, traffic, challenge):
        phases = (timings["input_preparation"], timings[production], timings[verification])
        cpu = sum(phase["cpu_seconds"] for phase in phases)
        wall = sum(phase["wall_seconds"] for phase in phases)
        network = traffic * 8 / scenario["link_bits_per_second"]
        network += scenario["challenge_rtt_seconds"] if challenge else 0
        return {"cpu_seconds": cpu, "local_wall_seconds": wall, "bytes": traffic,
                "cost_usd": cpu * cpu_rate + traffic * byte_rate,
                "modeled_serialized_seconds": wall + network}
    candidate = bill("hybrid_production", "hybrid_verification", hybrid_bytes, True)
    controls = {name: bill("boundary_production", name, boundary_bytes, False)
                for name in ("replay_optimized", "replay_reference")}
    cheapest = min(controls, key=lambda name: controls[name]["cost_usd"])
    fastest = min(controls, key=lambda name: controls[name]["modeled_serialized_seconds"])
    cost_control = controls[cheapest]
    latency_control = controls[fastest]
    ordinary_cpu = min(timings[name]["cpu_seconds"] for name in
                       ("ordinary_training_optimized", "ordinary_training_reference"))
    ordinary_cost = ordinary_cpu * cpu_rate
    cheapest_cpu = min(control["cpu_seconds"] for control in controls.values())
    budget_for_bytes = (0.5 * cheapest_cpu - candidate["cpu_seconds"]) * cpu_rate
    extra_bytes = hybrid_bytes - 0.5 * boundary_bytes
    return {
        "candidate": candidate, "controls": controls,
        "cheapest_control": cheapest, "fastest_latency_control": fastest,
        "audit_cost_ratio": candidate["cost_usd"] / cost_control["cost_usd"],
        "modeled_latency_ratio": (candidate["modeled_serialized_seconds"]
                                  / latency_control["modeled_serialized_seconds"]),
        "ordinary_training_cpu_seconds": ordinary_cpu,
        "ordinary_work_plus_audit_cost_ratio": ((ordinary_cost + candidate["cost_usd"])
                                               / (ordinary_cost + cost_control["cost_usd"])),
        "candidate_traffic_cost_floor_over_replay_cost": (
            hybrid_bytes * byte_rate / cost_control["cost_usd"]),
        "maximum_transfer_usd_per_gb_for_half_audit_cost": (
            budget_for_bytes / extra_bytes * 1e9 if budget_for_bytes >= 0 else None),
        "pricing_is_hypothetical": True, "network_latency_is_modeled": True,
    }


def attacks(job: reference.Job, trace: dict) -> dict:
    identity = job.work_id()
    job_payload = hybrid.encode_job(job)
    honest = hybrid.produce(job, trace, identity, hybrid=True)
    cases = {}
    fake = {name: value.copy() for name, value in trace.items()}
    fake["weight_gradient"][:] = 0
    fake["after"][:] = job.weights
    cases["coherent_fabricated_gradient"] = (
        job_payload, hybrid.produce(job, fake, identity, hybrid=True))
    fake_upstream = {name: value.copy() for name, value in trace.items()}
    fake_upstream["input_gradient"][0, 0] += 1
    cases["forged_upstream_gradient"] = (
        job_payload, hybrid.produce(job, fake_upstream, identity, hybrid=True))
    for name, trace_case in (("coherent_fabricated_gradient", fake),
                             ("forged_upstream_gradient", fake_upstream)):
        boundary = hybrid.produce(job, trace_case, identity, hybrid=False)
        try:
            hybrid.replay(job_payload, identity, boundary, optimized=True)
        except reference.Rejected:
            pass
        else:
            raise AssertionError("dense oracle accepted numerical attack: " + name)
    for label, payload in (("truncated", honest.payload[:-1]),
                           ("extended", honest.payload + b"\0")):
        cases[label] = (job_payload, hybrid.Claim(payload, hybrid.commitment(identity, payload)))
    changed = bytearray(honest.payload)
    changed[-1] ^= 1
    cases["post_commit_change"] = (job_payload, replace(honest, payload=bytes(changed)))
    changed = bytearray(honest.payload)
    changed[8:12] = (2**31 - 1).to_bytes(4, "little")
    changed = bytes(changed)
    cases["out_of_range_weight"] = (
        job_payload, hybrid.Claim(changed, hybrid.commitment(identity, changed)))
    changed = bytearray(honest.payload)
    start = 8 + 4 * (job.weights.size + job.inputs.size)
    width = hybrid.remainder_bytes(job.scale * len(job.inputs) * job.learning_rate_denominator)
    changed[start:start + width] = bytes([255]) * width
    changed = bytes(changed)
    cases["noncanonical_remainder"] = (
        job_payload, hybrid.Claim(changed, hybrid.commitment(identity, changed)))
    changed_job = bytearray(job_payload)
    changed_job[hybrid.JOB_HEADER.size] ^= 1
    cases["changed_job"] = (bytes(changed_job), honest)
    result = {}
    for name, (input_bytes, claim) in cases.items():
        seed = secrets.token_bytes(32)
        def reject():
            try:
                hybrid.verify(input_bytes, identity, claim, seed=seed,
                              committed_root=claim.commitment)
            except reference.Rejected as error:
                return str(error)
            raise AssertionError("hybrid verifier accepted attack: " + name)
        reason, timing = timed(reject)
        result[name] = {"rejected": True, "reason": reason, "timing": timing,
                        "commitment": claim.commitment, "seed": seed.hex()}
    return result


def run_shape(shape: list[int], plan: dict) -> dict:
    job = fixture(shape, plan["fixture_seed"])
    integer_oracle = reference.train(job, backend="int64")
    for candidate in (hybrid.train_optimized(job), reference.train(job, backend="float64")):
        for name in reference.TRACE_NAMES:
            if not np.array_equal(candidate[name], integer_oracle[name]):
                raise AssertionError("integer oracle disagreement: " + name)
    attack_results = attacks(job, integer_oracle)
    for _ in range(plan["warmup_repetitions"]):
        sample(job, integer_oracle, reverse=False)
    samples = []
    for index in range(plan["timing_repetitions"]):
        record = sample(job, integer_oracle, reverse=bool(index % 2))
        record["scenarios"] = {scenario["name"]: account(record, scenario)
                               for scenario in plan["scenarios"]}
        samples.append(record)
    summaries = {}
    for scenario in plan["scenarios"]:
        name = scenario["name"]
        values = [record["scenarios"][name] for record in samples]
        summaries[name] = {
            "median_audit_cost_ratio": statistics.median(value["audit_cost_ratio"] for value in values),
            "worst_audit_cost_ratio": max(value["audit_cost_ratio"] for value in values),
            "median_modeled_latency_ratio": statistics.median(value["modeled_latency_ratio"] for value in values),
            "worst_modeled_latency_ratio": max(value["modeled_latency_ratio"] for value in values),
            "median_work_plus_audit_cost_ratio": statistics.median(
                value["ordinary_work_plus_audit_cost_ratio"] for value in values),
            "all_samples_meet_half_cost_and_latency": all(
                value["audit_cost_ratio"] <= plan["maximum_cost_ratio"]
                and value["modeled_latency_ratio"] <= plan["maximum_cost_ratio"]
                for value in values),
        }
    return {"shape": shape, "work_id": job.work_id(), "integer_oracle_agrees": True,
            "attacks": attack_results, "samples": samples, "summary": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("preserve existing evidence; choose a new output path")
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(variable) != "1":
            raise SystemExit("set " + variable + "=1 before starting Python")
    hashes = {}
    for path in FROZEN_PATHS:
        source = (ROOT / path).read_bytes()
        frozen = subprocess.check_output(["git", "show", "HEAD:" + path], cwd=ROOT)
        if source != frozen:
            raise SystemExit("commit the exact source before measuring: " + path)
        hashes[path] = hashlib.sha256(source).hexdigest()
    plan = json.loads(PLAN.read_text())
    result = {
        "schema": "neuroshard-hybrid-shard-verification-result-v1",
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "running",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "source_sha256": hashes, "plan": plan,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform(), "machine": platform.machine(),
                        "cpu_threads_requested": 1,
                        "cpu_model": next((line.split(":", 1)[1].strip()
                                           for line in Path("/proc/cpuinfo").read_text().splitlines()
                                           if line.startswith("model name")), "unknown"),
                        "host_load_at_start": list(os.getloadavg()),
                        "numpy_build": np.show_config(mode="dicts")},
        "shapes": [], "aws_spend_usd": 0, "neuro_issued": 0,
        "new_infrastructure": False, "network_times_are_measured": False,
        "transformer_verification_established": False,
        "public_audit_economy_established": False,
    }
    left = np.array([[2**24, 1]], dtype=np.float32)
    right = np.array([[1, -1], [1, 0]], dtype=np.float32)
    vector = np.ones((2, 1), dtype=np.float32)
    result["float32_counterexample"] = {
        "honest_product_then_projection": ((left @ right) @ vector).item(),
        "reassociated_projection": (left @ (right @ vector)).item(),
    }
    if result["float32_counterexample"] != {
        "honest_product_then_projection": 0.0, "reassociated_projection": 1.0
    }:
        raise SystemExit("unexpected floating-point counterexample behavior")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.output)
    save()
    try:
        for shape in plan["shapes"]:
            row = run_shape(shape, plan)
            result["shapes"].append(row)
            save()
            print(json.dumps({"shape": shape, "summary": row["summary"]}), flush=True)
        primary = next(row for row in result["shapes"] if row["shape"] == plan["primary_shape"])
        passed = primary["summary"][plan["primary_scenario"]]["all_samples_meet_half_cost_and_latency"]
        result["verdict"] = "conditional-linear-profile-pass" if passed else "cost-target-failed"
        result["next_action"] = "stop; no GPU allocation or native integration"
        result["status"] = "completed"
        save()
    except Exception as error:
        result["status"] = "failed"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        save()
        raise
    print(json.dumps({"output": str(args.output), "verdict": result["verdict"]}), flush=True)


if __name__ == "__main__":
    main()
