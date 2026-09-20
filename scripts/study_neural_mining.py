#!/usr/bin/env python3
"""Frozen CPU mechanism study; never launches instances or submits transactions."""
from dataclasses import replace
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

import numpy as np

import neural_work_mining as mining
import neural_work_reference as nw
from study_neural_work import measure, rejected


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("preserve earlier records; choose a fresh output path")
    paths = ["scripts/neural_work_reference.py", "scripts/neural_work_mining.py",
             "scripts/study_neural_work.py", "scripts/study_neural_mining.py",
             "config/experiments/neural-work-mining-sketch.json"]
    for path in paths:
        if subprocess.check_output(["git", "show", "HEAD:" + path], cwd=ROOT) != (ROOT / path).read_bytes():
            raise SystemExit("commit the research source before measuring: " + path)
    if any(os.environ.get(name) != "1" for name in
           ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")):
        raise SystemExit("set the three BLAS/OpenMP thread variables to 1 before starting Python")
    plan = json.loads((ROOT / paths[-1]).read_text())
    if plan["tile_size"] != mining.TILE or plan["noise_rank"] != mining.NOISE_RANK:
        raise SystemExit("plan/kernel mismatch")
    batch, width, outputs = plan["shape"]
    rng = np.random.default_rng(plan["fixture_seed"])
    job = nw.Job(rng.integers(-64, 65, (batch, width), dtype=np.int64),
                 rng.integers(-16, 17, (width, outputs), dtype=np.int64),
                 rng.integers(-128, 129, (batch, outputs), dtype=np.int64))
    challenge, worker = "11" * 32, "22" * 32
    trace, certificates, operations = mining.train_with_tickets(
        job, challenge=challenge, worker=worker, bits=plan["difficulty_bits"])
    expected = nw.train(job, backend="float64")
    if not all(np.array_equal(trace[name], expected[name]) for name in nw.TRACE_NAMES):
        raise RuntimeError("mining changed training arithmetic")
    bundle = {name: proofs for name, _, _, _, proofs in certificates}
    root = nw.trace_root(job, trace)
    audit_seed = os.urandom(32)  # issued AFTER the numerical commitment
    options = dict(committed_root=root, audit_seed=audit_seed, challenge=challenge,
                   worker=worker, bits=plan["difficulty_bits"])
    accepted = mining.verify_bundle(job, trace, bundle, **options)
    _, left, right, context, proofs = next(c for c in certificates if c[-1])
    relabel = {}
    for field, value in (("chain_challenge", "44" * 32), ("worker", "33" * 32),
                         ("work_id", "66" * 32), ("operation", "weight_gradient")):
        relabel[field] = rejected(lambda: mining.verify_ticket(
            left, right, replace(context, **{field: value}), proofs[0]))
    if not all(relabel.values()):
        raise RuntimeError("ticket relabeling survived")
    other, _, other_ops = mining.train_with_tickets(job, challenge="55" * 32, worker=worker)
    loser, _, losing_ops = mining.train_with_tickets(
        job, challenge=challenge, worker=worker, bits=plan["losing_difficulty_bits"])
    unchanged = all(np.array_equal(trace[name], other[name]) and np.array_equal(trace[name], loser[name])
                    for name in nw.TRACE_NAMES)
    if not unchanged:
        raise RuntimeError("challenge or ticket selection changed the useful result")
    count = plan["timing_repetitions"]
    timings = {
        "ordinary_training": measure(lambda: nw.train(job, backend="float64"), count),
        "mining_training": measure(lambda: mining.train_with_tickets(
            job, challenge=challenge, worker=worker), count),
        "arithmetic_and_all_winning_tickets": measure(
            lambda: mining.verify_bundle(job, trace, bundle, **options), count),
    }
    result = {
        "schema": "neuroshard-tile-mining-sketch-result-v1", "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform(), "blas_threads_requested": 1},
        "work_id": job.work_id(), "trace_root": root, "audit_seed": audit_seed.hex(),
        "test_challenge": challenge, "test_worker": worker,
        "mined_result_equals_ordinary_training": True,
        "challenge_and_lottery_preserve_useful_result": unchanged,
        "operations": operations, "changed_challenge_operations": other_ops,
        "losing_operations": losing_ops, "accepted_ticket_ids": accepted,
        "ticket_count": len(accepted), "relabel_attempts_rejected": relabel,
        "timings": timings,
        "producer_time_over_ordinary": timings["mining_training"]["median_seconds"]
        / timings["ordinary_training"]["median_seconds"],
        "winning_ticket_witnesses": {name: [
            {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in p.items()}
            for p in proofs] for name, _, _, _, proofs in certificates},
        "decision": {"mechanism": "numerical preservation and tested bindings pass",
                     "resource_hardness": "unproved; toy scheme does not inherit cuPOW security",
                     "deployment": "not authorized; no fork choice, public admission or new native issuance",
                     "total_efficiency": "CPU sketch includes substantial encoding/hash/proof overhead",
                     "neuro_issued": 0, "new_aws_instances": 0},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "source_commit": result["source_commit"],
                      "winning_tickets": len(accepted), "producer_time_over_ordinary": result["producer_time_over_ordinary"],
                      "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
