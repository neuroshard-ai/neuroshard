#!/usr/bin/env python3
"""CPU research: forward replay and compact, checked backward witnesses.

Apache-2.0. This verifies only the existing exact linear numerical profile.
It is neither a public proof system nor a native consensus implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import struct

import numpy as np

import neural_work_reference as reference


JOB_HEADER = struct.Struct("<8s5I")
JOB_MAGIC = b"NSJOB001"
BOUNDARY_MAGIC = b"NSBND001"
HYBRID_MAGIC = b"NSHYB001"


@dataclass(frozen=True)
class Claim:
    payload: bytes
    commitment: str


def encode_job(job: reference.Job) -> bytes:
    job.validate()
    batch, width = job.inputs.shape
    outputs = job.weights.shape[1]
    header = JOB_HEADER.pack(JOB_MAGIC, batch, width, outputs, job.scale,
                             job.learning_rate_denominator)
    return header + b"".join(array.astype("<i4").tobytes(order="C")
                              for array in (job.inputs, job.weights, job.targets))


def decode_job(payload: bytes, expected_work_id: str) -> reference.Job:
    if not isinstance(payload, bytes) or len(payload) < JOB_HEADER.size:
        raise reference.Rejected("missing job bytes")
    magic, batch, width, outputs, scale, rate = JOB_HEADER.unpack_from(payload)
    if magic != JOB_MAGIC or any(not 0 < size <= reference.MAX_DIM
                                 for size in (batch, width, outputs)):
        raise reference.Rejected("invalid job encoding")
    shapes = ((batch, width), (width, outputs), (batch, outputs))
    expected_size = JOB_HEADER.size + 4 * sum(rows * cols for rows, cols in shapes)
    if len(payload) != expected_size:
        raise reference.Rejected("incorrect job length")
    arrays = []
    offset = JOB_HEADER.size
    for rows, cols in shapes:
        count = rows * cols
        array = np.frombuffer(payload, dtype="<i4", count=count, offset=offset)
        arrays.append(array.astype(np.int64).reshape(rows, cols))
        offset += 4 * count
    job = reference.Job(*arrays, scale=scale, learning_rate_denominator=rate)
    if not hmac.compare_digest(job.work_id(), expected_work_id):
        raise reference.Rejected("unadmitted job bytes")
    return job


def commitment(work_id: str, payload: bytes) -> str:
    if not isinstance(work_id, str) or len(work_id) != 64:
        raise reference.Rejected("invalid admitted work identity")
    try:
        identity = bytes.fromhex(work_id)
    except ValueError as error:
        raise reference.Rejected("invalid admitted work identity") from error
    return hashlib.sha256(b"neuroshard/research/hybrid-claim/v1\0"
                          + identity + payload).hexdigest()


def remainder_bytes(denominator: int) -> int:
    if type(denominator) is not int or not 0 < denominator <= 1 << 30:
        raise reference.Rejected("invalid rounding denominator")
    return max(1, ((denominator - 1).bit_length() + 7) // 8)


def pack_remainder(gradient: np.ndarray, quotient: np.ndarray,
                   denominator: int) -> bytes:
    width = remainder_bytes(denominator)
    if gradient.dtype != np.int64 or quotient.dtype != np.int64:
        raise reference.Rejected("remainder requires exact integers")
    reference.matrix(gradient, limit=reference.MAX_PRODUCT)
    reference.matrix(quotient, limit=2 * reference.MAX_ENTRY)
    if gradient.shape != quotient.shape:
        raise reference.Rejected("rounding shape mismatch")
    sign = np.where(quotient < 0, -1, 1)
    encoded = sign * (gradient - denominator * quotient) + denominator // 2
    if np.any(encoded < 0) or np.any(encoded >= denominator):
        raise reference.Rejected("gradient outside rounding interval")
    if not np.array_equal(reference.rounded_divide(gradient, denominator), quotient):
        raise reference.Rejected("noncanonical rounded quotient")
    words = encoded.astype("<u4").reshape(-1, 1)
    shifts = np.arange(width, dtype=np.uint32) * 8
    return ((words >> shifts) & 255).astype(np.uint8).tobytes()


def unpack_remainder(payload: bytes, quotient: np.ndarray,
                     denominator: int) -> np.ndarray:
    width = remainder_bytes(denominator)
    reference.matrix(quotient, limit=2 * reference.MAX_ENTRY)
    if len(payload) != quotient.size * width:
        raise reference.Rejected("incorrect remainder length")
    digits = np.frombuffer(payload, dtype=np.uint8).reshape(-1, width)
    shifts = np.arange(width, dtype=np.uint32) * 8
    encoded = np.sum(digits.astype(np.uint32) << shifts, axis=1,
                     dtype=np.uint32).astype(np.int64).reshape(quotient.shape)
    if np.any(encoded >= denominator):
        raise reference.Rejected("noncanonical remainder code")
    sign = np.where(quotient < 0, -1, 1)
    gradient = denominator * quotient + sign * (encoded - denominator // 2)
    reference.matrix(gradient, limit=reference.MAX_PRODUCT)
    if not np.array_equal(reference.rounded_divide(gradient, denominator), quotient):
        raise reference.Rejected("noncanonical rounded quotient")
    return gradient


def produce(job: reference.Job, trace: dict[str, np.ndarray], work_id: str,
            *, hybrid: bool) -> Claim:
    after = reference.matrix(trace["after"])
    upstream = reference.matrix(trace["input_gradient"], limit=reference.MAX_PRODUCT)
    if after.shape != job.weights.shape or upstream.shape != job.inputs.shape:
        raise reference.Rejected("boundary shape mismatch")
    payload = (HYBRID_MAGIC if hybrid else BOUNDARY_MAGIC)
    payload += after.astype("<i4").tobytes() + upstream.astype("<i4").tobytes()
    if hybrid:
        denominator = job.scale * len(job.inputs) * job.learning_rate_denominator
        payload += pack_remainder(trace["weight_gradient"], job.weights - after,
                                  denominator)
    return Claim(payload, commitment(work_id, payload))


def _read_claim(job: reference.Job, work_id: str, claim: Claim, *, hybrid: bool):
    if not isinstance(claim.payload, bytes) or not isinstance(claim.commitment, str):
        raise reference.Rejected("invalid claim")
    magic = HYBRID_MAGIC if hybrid else BOUNDARY_MAGIC
    boundary_size = 8 + 4 * (job.weights.size + job.inputs.size)
    denominator = job.scale * len(job.inputs) * job.learning_rate_denominator
    expected_size = boundary_size
    if hybrid:
        expected_size += remainder_bytes(denominator) * job.weights.size
    if len(claim.payload) != expected_size or claim.payload[:8] != magic:
        raise reference.Rejected("incorrect witness encoding or length")
    if not hmac.compare_digest(commitment(work_id, claim.payload), claim.commitment):
        raise reference.Rejected("payload changed after commitment")
    after = np.frombuffer(claim.payload, dtype="<i4", count=job.weights.size,
                          offset=8).astype(np.int64).reshape(job.weights.shape)
    upstream = np.frombuffer(claim.payload, dtype="<i4", count=job.inputs.size,
                             offset=8 + 4 * job.weights.size)
    upstream = upstream.astype(np.int64).reshape(job.inputs.shape)
    reference.matrix(after)
    reference.matrix(upstream, limit=reference.MAX_PRODUCT)
    return after, upstream, claim.payload[boundary_size:]


def _dense(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Only called after an exact-integer dot-product bound has been checked."""
    return (left @ right).astype(np.int64)


def _forward(job: reference.Job):
    reference.product_bound(job.inputs, job.weights)
    inputs_float = job.inputs.astype(np.float64)
    weights_float = job.weights.astype(np.float64)
    forward = _dense(inputs_float, weights_float)
    residual = reference.matrix(reference.rounded_divide(forward, job.scale)
                                - job.targets)
    return forward, residual, inputs_float, weights_float, residual.astype(np.float64)


def train_optimized(job: reference.Job) -> dict[str, np.ndarray]:
    """Strong exact dense control, reusing conversions across all three products."""
    job.validate()
    forward, residual, inputs_float, weights_float, residual_float = _forward(job)
    reference.product_bound(job.inputs.T, residual)
    reference.product_bound(residual, job.weights.T)
    gradient = _dense(inputs_float.T, residual_float)
    upstream = _dense(residual_float, weights_float.T)
    denominator = job.scale * len(job.inputs) * job.learning_rate_denominator
    after = reference.matrix(job.weights - reference.rounded_divide(gradient, denominator))
    return {"forward": forward, "weight_gradient": gradient,
            "input_gradient": upstream, "after": after}


def replay(job_payload: bytes, expected_work_id: str, claim: Claim,
           *, optimized: bool) -> None:
    job = decode_job(job_payload, expected_work_id)
    after, upstream, _ = _read_claim(job, expected_work_id, claim, hybrid=False)
    expected = train_optimized(job) if optimized else reference.train(job, backend="float64")
    if (not np.array_equal(after, expected["after"])
            or not np.array_equal(upstream, expected["input_gradient"])):
        raise reference.Rejected("minimal dense replay mismatch")


def _check_product(left: np.ndarray, right: np.ndarray, claimed: np.ndarray,
                   left_float: np.ndarray, right_float: np.ndarray,
                   seed: bytes, context: str) -> None:
    bound = reference.product_bound(left, right)
    reference.matrix(claimed, limit=bound)
    if claimed.shape != (left.shape[0], right.shape[1]):
        raise reference.Rejected("claimed product shape mismatch")
    if bound * right.shape[1] * 255 >= 1 << 53:
        raise reference.Rejected("projection outside exact binary64 range")
    raw = hashlib.shake_256(b"neuroshard/research/hybrid-projection/v1\0"
                           + seed + context.encode()).digest(
                               right.shape[1] * reference.INTEGER_ROUNDS)
    vectors = np.frombuffer(raw, dtype=np.uint8).astype(np.float64).reshape(
        right.shape[1], reference.INTEGER_ROUNDS)
    expected = left_float @ (right_float @ vectors)
    observed = claimed.astype(np.float64) @ vectors
    if not np.array_equal(expected, observed):
        raise reference.Rejected("incorrect backward product: " + context)


def verify(job_payload: bytes, expected_work_id: str, claim: Claim, *, seed: bytes,
           committed_root: str) -> None:
    if not isinstance(seed, bytes) or len(seed) != 32:
        raise reference.Rejected("fresh 256-bit audit seed required")
    if claim.commitment != committed_root:
        raise reference.Rejected("claim replaced after the audit challenge")
    job = decode_job(job_payload, expected_work_id)
    after, upstream, remainder = _read_claim(job, expected_work_id, claim, hybrid=True)
    _, residual, inputs_float, weights_float, residual_float = _forward(job)
    denominator = job.scale * len(job.inputs) * job.learning_rate_denominator
    gradient = unpack_remainder(remainder, job.weights - after, denominator)
    _check_product(job.inputs.T, residual, gradient, inputs_float.T, residual_float,
                   seed, claim.commitment + "/weight")
    _check_product(residual, job.weights.T, upstream, residual_float, weights_float.T,
                   seed, claim.commitment + "/input")
