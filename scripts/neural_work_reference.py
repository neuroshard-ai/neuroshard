#!/usr/bin/env python3
"""Experimental interactive verification of one fixed-point linear training shard.

Apache-2.0. This is not a PoW implementation or a production settlement service.
See docs/NEURAL_WORK_RESEARCH.md for the numerical and trust boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import secrets

import numpy as np


PROFILE = "neuroshard-linear-fixed-point-research-v1"
PRIMES = (65521, 65519)
ROUNDS = 5
INTEGER_ROUNDS = 10
MAX_DIM = 512
MAX_ENTRY = 32768
MAX_PRODUCT = (PRIMES[0] * PRIMES[1] - 1) // 2
TRACE_NAMES = ("forward", "weight_gradient", "input_gradient", "after")


class Rejected(ValueError):
    """A claim is malformed, unauthenticated, unavailable, or numerically wrong."""


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def matrix(value: np.ndarray, *, limit: int = MAX_ENTRY) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype("int64"):
        raise Rejected("expected an int64 array")
    if value.ndim != 2 or not all(0 < n <= MAX_DIM for n in value.shape):
        raise Rejected("matrix dimensions exceed the research profile")
    # Compare before taking abs: abs(INT64_MIN) overflows.
    if np.any(value < -limit) or np.any(value > limit):
        raise Rejected("matrix range exceeds the research profile")
    return value


def matrix_root(value: np.ndarray, *, limit: int = MAX_PRODUCT) -> str:
    matrix(value, limit=limit)
    header = json.dumps(list(value.shape), separators=(",", ":")).encode()
    return hashlib.sha256(
        b"neuroshard/research/int64-matrix/v1\0" + header + b"\0"
        + np.asarray(value, dtype="<i8", order="C").tobytes()
    ).hexdigest()


def rounded_divide(value: np.ndarray, denominator: int) -> np.ndarray:
    """Nearest integer, ties away from zero; callers establish int64 bounds."""
    if type(denominator) is not int or not 0 < denominator <= 1 << 30:
        raise Rejected("invalid fixed-point denominator")
    return np.sign(value) * ((np.abs(value) + denominator // 2) // denominator)


@dataclass(frozen=True)
class Job:
    inputs: np.ndarray
    weights: np.ndarray
    targets: np.ndarray
    scale: int = 64
    learning_rate_denominator: int = 8

    def validate(self) -> None:
        for value in (self.inputs, self.weights, self.targets):
            matrix(value)
        batch, width = self.inputs.shape
        if self.weights.shape[0] != width:
            raise Rejected("input/weight shape mismatch")
        if self.targets.shape != (batch, self.weights.shape[1]):
            raise Rejected("target shape mismatch")
        for value in (self.scale, self.learning_rate_denominator):
            if type(value) is not int or not 1 <= value <= 1024:
                raise Rejected("invalid numerical profile")

    def work_id(self) -> str:
        self.validate()
        # No account, sponsor, job label, challenge, or block height: relabeling
        # the same numerical work cannot create a second payable obligation.
        return digest({
            "profile": PROFILE, "scale": self.scale,
            "learning_rate_denominator": self.learning_rate_denominator,
            "inputs": matrix_root(self.inputs),
            "weights": matrix_root(self.weights),
            "targets": matrix_root(self.targets),
        })


def product_bound(left: np.ndarray, right: np.ndarray) -> int:
    matrix(left)
    matrix(right)
    if left.shape[1] != right.shape[0]:
        raise Rejected("product shape mismatch")
    bound = left.shape[1] * int(np.abs(left).max()) * int(np.abs(right).max())
    if bound > MAX_PRODUCT:
        raise Rejected("integer product cannot be uniquely recovered from both fields")
    return bound


def exact_product(left: np.ndarray, right: np.ndarray, *, backend: str) -> np.ndarray:
    product_bound(left, right)
    if backend == "int64":
        return left @ right
    if backend == "float64":
        # Every term and every partial sum has absolute bound < 2^32 < 2^53.
        # Binary64 BLAS is an exact-integer baseline for this profile only.
        return (left.astype(np.float64) @ right.astype(np.float64)).astype(np.int64)
    raise Rejected("unknown reference backend")


def train(job: Job, *, backend: str = "int64") -> dict[str, np.ndarray]:
    job.validate()
    forward = exact_product(job.inputs, job.weights, backend=backend)
    prediction = rounded_divide(forward, job.scale)
    residual = matrix(prediction - job.targets)
    weight_gradient = exact_product(job.inputs.T, residual, backend=backend)
    input_gradient = exact_product(residual, job.weights.T, backend=backend)
    denominator = job.scale * len(job.inputs) * job.learning_rate_denominator
    after = matrix(job.weights - rounded_divide(weight_gradient, denominator))
    return {"forward": forward, "weight_gradient": weight_gradient,
            "input_gradient": input_gradient, "after": after}


def trace_root(job: Job, trace: dict[str, np.ndarray]) -> str:
    if set(trace) != set(TRACE_NAMES):
        raise Rejected("incomplete or unexpected witness tensors")
    return digest({"work_id": job.work_id(), "trace": {
        name: matrix_root(trace[name]) for name in TRACE_NAMES
    }})


def projection(seed: bytes, context: str, columns: int, prime: int) -> np.ndarray:
    """SHAKE-derived field elements, with rejection sampling (no modulo bias).

    The seed must be unpredictable to the worker until its immutable commitment
    is recorded. This is an interactive verifier challenge, not Fiat-Shamir PoW.
    """
    if not isinstance(seed, bytes) or len(seed) != 32:
        raise Rejected("expected a 256-bit verifier seed")
    bound = (1 << 32) - (1 << 32) % prime
    values: list[int] = []
    counter = 0
    while len(values) < columns * ROUNDS:
        payload = (b"neuroshard/research/projection/v1\0" + seed
                   + context.encode() + prime.to_bytes(4, "big")
                   + counter.to_bytes(4, "big"))
        chunk = np.frombuffer(hashlib.shake_256(payload).digest(4096), dtype="<u4")
        values.extend(int(v) % prime for v in chunk if int(v) < bound)
        counter += 1
    return np.array(values[:columns * ROUNDS], dtype=np.int64).reshape(columns, ROUNDS)


def check_product(left: np.ndarray, right: np.ndarray, claimed: np.ndarray,
                  *, seed: bytes, context: str) -> None:
    bound = product_bound(left, right)
    matrix(claimed, limit=bound)
    if claimed.shape != (left.shape[0], right.shape[1]):
        raise Rejected("claimed product shape mismatch")
    # For each field check C R = A (B R). At most 512*(65520^2)
    # accumulates in an int64 dot product. Two moduli plus the integer bounds
    # exclude a forged C = AB + p that a single-field check would accept.
    for prime in PRIMES:
        vectors = projection(seed, context, right.shape[1], prime)
        projected_right = ((right % prime) @ vectors) % prime
        expected = ((left % prime) @ projected_right) % prime
        observed = ((claimed % prime) @ vectors) % prime
        if not np.array_equal(expected, observed):
            raise Rejected(f"incorrect product: {context}")


def check_product_integer(left: np.ndarray, right: np.ndarray, claimed: np.ndarray,
                          *, seed: bytes, context: str) -> None:
    """Freivalds over rationals with ten independent byte-valued vectors.

    Range bounds make all integer intermediates exactly representable in
    binary64. This avoids modular arithmetic without float tolerances.
    """
    bound = product_bound(left, right)
    matrix(claimed, limit=bound)
    if claimed.shape != (left.shape[0], right.shape[1]):
        raise Rejected("claimed product shape mismatch")
    if bound * right.shape[1] * 255 >= 1 << 53:
        raise Rejected("projection is outside exact binary64 integer range")
    if not isinstance(seed, bytes) or len(seed) != 32:
        raise Rejected("expected a 256-bit verifier seed")
    payload = b"neuroshard/research/integer-projection/v1\0" + seed + context.encode()
    raw = hashlib.shake_256(payload).digest(right.shape[1] * INTEGER_ROUNDS)
    vectors = np.frombuffer(raw, dtype=np.uint8).astype(np.float64).reshape(
        right.shape[1], INTEGER_ROUNDS)
    projected_right = right.astype(np.float64) @ vectors
    expected = left.astype(np.float64) @ projected_right
    observed = claimed.astype(np.float64) @ vectors
    if not np.array_equal(expected, observed):
        raise Rejected(f"incorrect product: {context}")


def verify(job: Job, trace: dict[str, np.ndarray], *, committed_root: str,
           seed: bytes, method: str = "integer") -> None:
    """Check the entire declared linear-shard transition, without dense replay."""
    if trace_root(job, trace) != committed_root:
        raise Rejected("witness changed after commitment")
    if method not in ("integer", "modular"):
        raise Rejected("unsupported verifier")
    checker = check_product_integer if method == "integer" else check_product
    context = committed_root + "/"
    checker(job.inputs, job.weights, trace["forward"], seed=seed,
            context=context + "forward")
    residual = matrix(rounded_divide(trace["forward"], job.scale) - job.targets)
    checker(job.inputs.T, residual, trace["weight_gradient"], seed=seed,
            context=context + "weight_gradient")
    checker(residual, job.weights.T, trace["input_gradient"], seed=seed,
            context=context + "input_gradient")
    denominator = job.scale * len(job.inputs) * job.learning_rate_denominator
    expected = matrix(job.weights - rounded_divide(trace["weight_gradient"], denominator))
    if not np.array_equal(expected, trace["after"]):
        raise Rejected("incorrect optimizer update")


def full_replay(job: Job, trace: dict[str, np.ndarray], *, committed_root: str,
                backend: str = "float64") -> None:
    if trace_root(job, trace) != committed_root:
        raise Rejected("witness changed after commitment")
    expected = train(job, backend=backend)
    if any(not np.array_equal(expected[name], trace[name]) for name in TRACE_NAMES):
        raise Rejected("dense replay mismatch")


class AdmissionBook:
    """In-memory authorization/payment MODEL, not a ledger or identity system.

    An external steward authenticates callers, chooses assignments and admits
    committed input data. One honest verifier controls challenges and decisions.
    No NEURO is issued. Caller strings are trusted test-harness identities.
    """

    def __init__(self) -> None:
        self.entries: dict[str, dict] = {}
        self.accepted_results: dict[str, str] = {}

    def admit(self, job: Job, worker: str) -> str:
        work_id = job.work_id()
        if work_id in self.entries:
            raise Rejected("numerical work already admitted")
        self.entries[work_id] = {"worker": worker, "commitment": None, "seed": None}
        return work_id

    def _entry(self, work_id: str, caller: str) -> dict:
        entry = self.entries.get(work_id)
        if entry is None or entry["worker"] != caller:
            raise Rejected("unassigned work or wrong authenticated caller")
        if work_id in self.accepted_results:
            raise Rejected("numerical work already settled")
        return entry

    def commit(self, work_id: str, caller: str, commitment: str) -> None:
        entry = self._entry(work_id, caller)
        if entry["commitment"] is not None:
            raise Rejected("commitment is immutable")
        if not isinstance(commitment, str) or len(commitment) != 64:
            raise Rejected("invalid commitment")
        try:
            bytes.fromhex(commitment)
        except ValueError as error:
            raise Rejected("invalid commitment") from error
        entry["commitment"] = commitment

    def challenge(self, work_id: str, caller: str) -> bytes:
        entry = self._entry(work_id, caller)
        if entry["commitment"] is None:
            raise Rejected("commit before requesting a challenge")
        if entry["seed"] is None:
            entry["seed"] = secrets.token_bytes(32)
        return entry["seed"]

    def settle(self, job: Job, trace: dict[str, np.ndarray], caller: str) -> str:
        work_id = job.work_id()
        entry = self._entry(work_id, caller)
        if entry["seed"] is None:
            raise Rejected("missing verifier challenge")
        verify(job, trace, committed_root=entry["commitment"], seed=entry["seed"])
        result = matrix_root(trace["after"])
        self.accepted_results[work_id] = result
        return result


def receipt_nonce_attack(work_id: str, commitment: str, *, bits: int = 8) -> dict:
    """Counterexample to hashing a valid training receipt as a work proof.

    It needs cached receipt bytes and fresh hash attempts, never fresh training.
    This is deliberately insecure; it is not the cited cuPOW construction.
    """
    if not 1 <= bits <= 12:
        raise ValueError("keep the negative control bounded")
    for nonce in range(1 << 20):
        ticket = digest({"work_id": work_id, "commitment": commitment, "nonce": nonce})
        if int(ticket, 16) < 1 << (256 - bits):
            return {"nonce": nonce, "hash_attempts": nonce + 1,
                    "additional_training_products": 0, "ticket": ticket}
    raise RuntimeError("bounded negative control did not find a ticket")


def output_only_noise_attack(size: int = 128, rank: int = 4) -> dict:
    """Known shortcut against FINAL-output-only low-rank noising, not cuPOW.

    For A=B=0, (EL ER)(FL FR) = EL(ER FL)FR. The paper's construction
    addresses this by challenging intermediate work, not merely the product.
    """
    rng = np.random.default_rng(20260920)
    el = rng.integers(-3, 4, (size, rank), dtype=np.int64)
    er = rng.integers(-3, 4, (rank, size), dtype=np.int64)
    fl = rng.integers(-3, 4, (size, rank), dtype=np.int64)
    fr = rng.integers(-3, 4, (rank, size), dtype=np.int64)
    dense = (el @ er) @ (fl @ fr)
    shortcut = (el @ (er @ fl)) @ fr
    return {"same_output": bool(np.array_equal(dense, shortcut)),
            "dense_product_multiplications": size ** 3,
            "shortcut_multiplications": 2 * size * rank ** 2 + size ** 2 * rank,
            "scope": "reject final-output-only shortcut; does not attack cuPOW"}
