"""Measure reproducibility and verification candidates before adopting them."""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from neuroshard.demo import work
from neuroshard.lab import matrix


REPO = Path(__file__).resolve().parents[3]


def numerical_child(args):
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = args.mkldnn
    data = work.read_data(REPO / "docs/eval/data/input.txt")
    weights = work.encode_weights(work.make_model())
    results = []
    for step in range(3):
        replay = work.replay(weights, data, step, "reproducibility-task")
        weights = replay["weights"]
        results.append({"model_root": work.digest(weights), "loss_hex": float(replay["loss"]).hex(),
                        "gradient_root": work.digest(replay["gradients"])})
    Path(args.output).write_text(json.dumps({"torch": torch.__version__, "python": platform.python_version(),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(), "threads": args.threads,
        "mkldnn": args.mkldnn, "results": results, "weights": weights}))


def reproducibility():
    profiles = [(1, False, "default"), (1, False, "default"), (1, False, "avx2"),
                (2, False, "avx2"), (4, False, "avx2"), (1, True, "avx2")]
    results, reference = [], None
    with tempfile.TemporaryDirectory(prefix="neuroshard-numerics-") as tmp:
        for index, (threads, mkldnn, capability) in enumerate(profiles):
            output = Path(tmp) / f"{index}.json"
            command = [sys.executable, "-m", "neuroshard.lab.experiments", "numeric-child",
                       "--threads", str(threads), "--output", str(output)]
            if mkldnn:
                command.append("--mkldnn")
            env = os.environ.copy()
            env.update(ATEN_CPU_CAPABILITY=capability, MKL_ENABLE_INSTRUCTIONS="SSE4_2" if capability == "default" else "AVX2",
                       PYTHONPATH=str(REPO / "src"))
            subprocess.run(command, check=True, env=env, capture_output=True)
            result = json.loads(output.read_text())
            weights = result.pop("weights")
            result["requested_capability"] = capability
            if reference is None:
                reference = weights
            result["bitwise_matches_first"] = weights == reference
            result["max_absolute_weight_difference"] = max(
                float((work.decode_tensor(weights[name], value["shape"]) - work.decode_tensor(value, value["shape"])).abs().max())
                for name, value in reference.items())
            results.append(result)
    assert results[1]["bitwise_matches_first"], "Repeated identical execution profile did not reproduce"
    return {"scope": "Separate processes and CPU dispatch/thread profiles on one x86_64 host; no independent hardware claim",
            "profiles": results}


def elapsed(fn, repeats=3):
    values = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        values.append(time.perf_counter() - start)
    return result, statistics.median(values)


def verification():
    generator = np.random.default_rng(20260910)
    timings = []
    for size in (64, 256, 512, 1024):
        a = generator.integers(-127, 128, (size, size), dtype=np.int64)
        b = generator.integers(-127, 128, (size, size), dtype=np.int64)
        c, full_seconds = elapsed(lambda: a @ b)
        accepted, verify_seconds = elapsed(lambda: matrix.verify(a, b, c, "benchmark"))
        assert accepted
        bad = c.copy()
        bad[size // 2, size // 3] += 1
        assert not matrix.verify(a, b, bad, "benchmark")
        timings.append({"dimension": size, "repetitions": matrix.ROUNDS,
                        "full_product_seconds": full_seconds, "verification_seconds_including_hash_and_bounds": verify_seconds,
                        "observed_speedup": full_seconds / verify_seconds,
                        "statement_bytes": a.nbytes + b.nbytes + c.nbytes,
                        "prover_product_still_required": True})
    a = generator.integers(-10, 11, (16, 16), dtype=np.int64)
    b = generator.integers(-10, 11, (16, 16), dtype=np.int64)
    c = a @ b
    fixed = np.ones((16, 1), dtype=np.int64)
    forged = c.copy()
    forged[0, 0] += 1
    forged[0, 1] -= 1
    assert matrix.check_with_challenges(a, b, forged, fixed)
    assert not matrix.verify(a, b, forged, "committed-output")
    attack_count, detected = 64, 0
    for _ in range(attack_count):
        corrupt = c.copy()
        i, j = generator.integers(0, 16, 2)
        corrupt[i, j] += int(generator.integers(1, 20))
        detected += not matrix.verify(a, b, corrupt, "attack-corpus")
    # Reassociation in floating arithmetic is not an exact integer-product statement.
    fa, fb = a.astype(np.float32) / 7, b.astype(np.float32) / 9
    fr = generator.integers(0, 2, (16, 128)).astype(np.float32)
    fc = fa @ fb
    fbad = fc.copy()
    fbad[0, 0] += np.float32(0.0001)
    lhs = fa @ (fb @ fr)
    assert not np.array_equal(lhs, fc @ fr)
    loose_accepts = bool(np.allclose(lhs, fbad @ fr, rtol=1e-3, atol=1e-2))
    assert loose_accepts
    return {"profile": "Bounded int8-valued inputs in int64 containers; exact int64 accumulation; binary projections",
            "timings": timings, "known_challenge_forgery_accepted": True,
            "statement_bound_challenge_rejected_same_forgery": True,
            "corrupted_products_rejected": detected, "corrupted_products_tested": attack_count,
            "honest_fp32_reassociation_max_error": float(np.max(np.abs(lhs - fc @ fr))),
            "loose_float_tolerance_accepted_modified_output": loose_accepts,
            "soundness_condition": "Independent binary challenges: at most 2^-128 per fixed false statement. Fiat-Shamir additionally needs random-oracle/grinding analysis, at most Q*2^-128 for Q attempts.",
            "limitations": ["Not a proof of floating-point NeuroLLM training", "No nonlinear, quantization, optimizer, range-proof, or full-graph proof implemented",
                            "Full matrices are read and authenticated; O(n^2) availability and bandwidth remain", "Single-host timings; no GPU or WAN speedup claim"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    child = sub.add_parser("numeric-child")
    child.add_argument("--threads", type=int, required=True)
    child.add_argument("--mkldnn", action="store_true")
    child.add_argument("--output", required=True)
    run = sub.add_parser("run")
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "numeric-child":
        numerical_child(args)
        return
    result = {"recorded_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
              "reproducibility": reproducibility()}
    print("CPU numerical profiles tested", flush=True)
    result["verification"] = verification()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"numeric_matches": [p["bitwise_matches_first"] for p in result["reproducibility"]["profiles"]],
                      "matrix_speedups": [t["observed_speedup"] for t in result["verification"]["timings"]]}, indent=2))


if __name__ == "__main__":
    main()
