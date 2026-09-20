#!/usr/bin/env python3
"""Run the frozen CPU-only neural-work binding study. No network or cloud calls."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np

import neural_work_reference as nw


ROOT = Path(__file__).resolve().parents[1]


def measure(operation, repetitions: int) -> dict:
    operation()  # one explicitly unreported warmup
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return {"samples_seconds": samples, "median_seconds": statistics.median(samples)}


def fixture(shape: list[int], seed: int) -> nw.Job:
    batch, width, outputs = shape
    rng = np.random.default_rng(seed)
    inputs = rng.integers(-64, 65, (batch, width), dtype=np.int64)
    weights = rng.integers(-16, 17, (width, outputs), dtype=np.int64)
    teacher = rng.integers(-16, 17, (width, outputs), dtype=np.int64)
    targets = nw.rounded_divide(inputs @ teacher, 64)
    return nw.Job(inputs, weights, targets)


def rejected(operation) -> bool:
    try:
        operation()
    except nw.Rejected:
        return True
    return False


def run_shape(shape: list[int], plan: dict) -> dict:
    job = fixture(shape, plan["fixture_seed"])
    trace = nw.train(job)
    root = nw.trace_root(job, trace)
    boundary = nw.boundary_root(job, trace["after"], trace["input_gradient"])
    book = nw.AdmissionBook()
    work_id = book.admit(job, "assigned-worker")
    book.commit(work_id, "assigned-worker", root)
    seed = book.challenge(work_id, "assigned-worker")
    book.settle(job, trace, "assigned-worker")

    timings = {}
    for backend in ("int64", "float64"):
        timings["training_" + backend] = measure(
            lambda: nw.train(job, backend=backend), plan["timing_repetitions"])
        timings["replay_" + backend] = measure(
            lambda: nw.full_replay(job, trace, committed_root=root, backend=backend),
            plan["timing_repetitions"])
        timings["boundary_replay_" + backend] = measure(
            lambda: nw.replay_boundary(job, trace["after"], trace["input_gradient"],
                                       committed_root=boundary, backend=backend),
            plan["timing_repetitions"])
    timings["producer_commitment"] = measure(
        lambda: nw.trace_root(job, trace), plan["timing_repetitions"])
    timings["boundary_commitment"] = measure(
        lambda: nw.boundary_root(job, trace["after"], trace["input_gradient"]),
        plan["timing_repetitions"])
    timings["projected_verification"] = measure(
        lambda: nw.verify(job, trace, committed_root=root, seed=seed),
        plan["timing_repetitions"])
    timings["modular_verification"] = measure(
        lambda: nw.verify(job, trace, committed_root=root, seed=seed, method="modular"),
        plan["timing_repetitions"])
    timings["witness_serialization"] = measure(
        lambda: b"".join(np.asarray(trace[name], dtype="<i8", order="C").tobytes()
                         for name in nw.TRACE_NAMES), plan["timing_repetitions"])

    attacks = {}
    for name in nw.TRACE_NAMES:
        forged = {key: value.copy() for key, value in trace.items()}
        forged[name][0, 0] += 1
        forged_root = nw.trace_root(job, forged)
        # Distinct roots are fixed before each newly sampled verifier seed.
        verifier_seed = os.urandom(32)
        attacks["forged_" + name] = {
            "commitment": forged_root, "challenge_seed": verifier_seed.hex(),
            "projected_rejects": rejected(lambda: nw.verify(
                job, forged, committed_root=forged_root, seed=verifier_seed)),
            "replay_rejects": rejected(lambda: nw.full_replay(
                job, forged, committed_root=forged_root)),
        }
    forged = {key: value.copy() for key, value in trace.items()}
    forged["weight_gradient"][:] = 0
    forged["after"][:] = job.weights
    forged_root = nw.trace_root(job, forged)
    verifier_seed = os.urandom(32)
    attacks["coherent_fake_gradient"] = {
        "commitment": forged_root, "challenge_seed": verifier_seed.hex(),
        "projected_rejects": rejected(lambda: nw.verify(
            job, forged, committed_root=forged_root, seed=verifier_seed)),
        "replay_rejects": rejected(lambda: nw.full_replay(
            job, forged, committed_root=forged_root)),
    }
    if not all(a["projected_rejects"] and a["replay_rejects"] for a in attacks.values()):
        raise RuntimeError("an adversarial numerical case survived")
    cached_seed = os.urandom(32)
    nw.verify(job, trace, committed_root=root, seed=cached_seed)
    replay = min(timings["replay_int64"]["median_seconds"],
                 timings["replay_float64"]["median_seconds"])
    audit = timings["projected_verification"]["median_seconds"]
    boundary_replay = min(timings["boundary_replay_int64"]["median_seconds"],
                          timings["boundary_replay_float64"]["median_seconds"])
    witness_bytes = sum(value.nbytes for value in trace.values())
    input_bytes = sum(value.nbytes for value in (job.inputs, job.weights, job.targets))
    return {
        "shape_batch_input_output": shape, "work_id": work_id, "commitment": root,
        "challenge_seed": seed.hex(), "cached_result_fresh_seed": cached_seed.hex(),
        "honest_checks_passed": True, "accepted_result": book.accepted_results[work_id],
        "timings": timings, "verification_over_fastest_exact_replay": audit / replay,
        "verification_over_fastest_boundary_replay": audit / boundary_replay,
        "producer_commitment_over_fastest_training": timings["producer_commitment"]["median_seconds"]
        / min(timings["training_int64"]["median_seconds"],
              timings["training_float64"]["median_seconds"]),
        "witness_tensor_bytes": witness_bytes, "uncached_job_tensor_bytes": input_bytes,
        "boundary_replay_tensor_bytes": trace["after"].nbytes + trace["input_gradient"].nbytes,
        "boundary_replay_commitment": boundary,
        "ideal_witness_transfer_seconds_at_100_mbit": witness_bytes * 8 / 100_000_000,
        "traffic_note": "Payload bytes and ideal transfer floor only; no network measurement or framing included.",
        "numerical_attacks": attacks,
        "duplicate_settlement_rejected": rejected(lambda: book.settle(job, trace, "assigned-worker")),
        "cached_result_passes_new_correctness_challenge_without_training": True,
        "receipt_nonce_attack": nw.receipt_nonce_attack(work_id, root),
    }


def learning_demo(plan: dict) -> dict:
    job = fixture(plan["learning_shape"], plan["fixture_seed"])
    job = replace(job, weights=np.zeros_like(job.weights))
    start = job.work_id()
    book = nw.AdmissionBook()
    losses = []
    stop_reason = "step budget reached"
    for step in range(plan["learning_steps"]):
        if job.work_id() in book.accepted_results:
            stop_reason = "quantized fixed point; identical work is not paid again"
            break
        trace = nw.train(job, backend="float64")
        residual = nw.rounded_divide(trace["forward"], job.scale) - job.targets
        losses.append(float(np.mean(residual.astype(np.float64) ** 2)))
        work_id = book.admit(job, "learner")
        book.commit(work_id, "learner", nw.trace_root(job, trace))
        book.challenge(work_id, "learner")
        book.settle(job, trace, "learner")
        job = replace(job, weights=trace["after"])
    residual = nw.rounded_divide(job.inputs @ job.weights, job.scale) - job.targets
    final_loss = float(np.mean(residual.astype(np.float64) ** 2))
    if final_loss >= losses[0]:
        raise RuntimeError("toy training did not reduce training error")
    return {"initial_work_id": start, "loss_before_each_step": losses,
            "loss_after_last_step": final_loss, "accepted_steps": len(book.accepted_results),
            "stop_reason": stop_reason,
            "final_weight_root": nw.matrix_root(job.weights),
            "interpretation": "Synthetic linear training-set fit only; no LLM or held-out quality claim."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path,
                        default=ROOT / "config/experiments/neural-work-reference.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if (plan["profile"] != nw.PROFILE or plan["moduli"] != list(nw.PRIMES)
            or plan["rounds_per_modulus"] != nw.ROUNDS
            or plan["integer_projection_rounds"] != nw.INTEGER_ROUNDS
            or plan["scale"] != 64 or plan["learning_rate_denominator"] != 8):
        raise SystemExit("plan and implementation disagree")
    if any(os.environ.get(name) != "1" for name in
           ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")):
        raise SystemExit("Set OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 before starting Python")
    paths = ["scripts/neural_work_reference.py", "scripts/study_neural_work.py",
             "config/experiments/neural-work-reference.json"]
    for path in paths:
        committed = subprocess.check_output(["git", "show", "HEAD:" + path], cwd=ROOT)
        if committed != (ROOT / path).read_bytes():
            raise SystemExit(f"commit the frozen source before measuring: {path}")
    if args.plan.resolve() != (ROOT / paths[-1]).resolve():
        raise SystemExit("this study uses the committed plan")
    if args.output.exists():
        raise SystemExit("use a new output path; preserve earlier runs")
    results = {
        "schema": "neuroshard-neural-work-reference-result-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform(), "machine": platform.machine(),
                        "blas_threads_requested": 1},
        "shapes": [],
        "decision": {
            "arithmetic_binding": "implemented for the declared linear fixed-point shard only",
            "receipt_hash_mining": "rejected: cheap nonce grinding after cached training",
            "final_output_only_low_rank_noising": "rejected: algebraic shortcut on zero useful inputs",
            "cupow_transcript_construction": "not implemented or attacked by these negative controls",
            "native_integration": "not authorized by this evidence",
            "permissionless_admission": "not implemented; inputs and assignments are stewarded",
            "proof_of_work_security": "not established", "neuro_issued": 0, "aws_spend_usd": 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.output)
    results["status"] = "running"
    save()
    try:
        for shape in plan["shapes"]:
            results["shapes"].append(run_shape(shape, plan))
            save()
        results["toy_learning"] = learning_demo(plan)
        results["output_only_noising_attack"] = nw.output_only_noise_attack()
        results["status"] = "completed"
        save()
    except Exception as error:
        results["status"] = "failed"
        results["error"] = {"type": type(error).__name__, "message": str(error)}
        save()
        raise
    print(json.dumps({"output": str(args.output), "source_commit": results["source_commit"],
                      "verification_over_fastest_boundary_replay": [r["verification_over_fastest_boundary_replay"]
                                                                     for r in results["shapes"]],
                      "decision": results["decision"]}, indent=2))


if __name__ == "__main__":
    main()
