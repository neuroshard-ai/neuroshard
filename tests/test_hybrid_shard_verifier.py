"""Security and numerical coverage for the frozen hybrid CPU experiment."""
from dataclasses import replace
from pathlib import Path
import secrets
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import hybrid_shard_verifier as hybrid
import neural_work_reference as reference
from study_hybrid_verification import account


@pytest.fixture
def job():
    rng = np.random.default_rng(20260920)
    return reference.Job(rng.integers(-64, 65, (16, 24), dtype=np.int64),
                         rng.integers(-16, 17, (24, 12), dtype=np.int64),
                         rng.integers(-32, 33, (16, 12), dtype=np.int64))


def checked_claim(job, trace=None):
    if trace is None:
        trace = reference.train(job)
    identity = job.work_id()
    claim = hybrid.produce(job, trace, identity, hybrid=True)
    return hybrid.encode_job(job), identity, claim, secrets.token_bytes(32)


def test_honest_rectangular_execution_matches_independent_integer_oracle(job):
    exact = reference.train(job)
    fast = hybrid.train_optimized(job)
    for name in reference.TRACE_NAMES:
        np.testing.assert_array_equal(fast[name], exact[name])
    np.testing.assert_array_equal(fast["forward"],
                                  job.inputs.astype(object) @ job.weights.astype(object))
    payload, identity, claim, seed = checked_claim(job, exact)
    hybrid.verify(payload, identity, claim, seed=seed, committed_root=claim.commitment)
    boundary = hybrid.produce(job, exact, identity, hybrid=False)
    for optimized in (False, True):
        hybrid.replay(payload, identity, boundary, optimized=optimized)


@pytest.mark.parametrize("denominator", [1, 2, 3, 4, 64, 65536, 262144, 1 << 30])
def test_compact_rounding_preserves_signed_ties_and_zero(denominator):
    half = denominator // 2
    values = np.array([[-denominator, -half - 1, -half, -half + 1, -1,
                        0, 1, half - 1, half, half + 1, denominator]], dtype=np.int64)
    quotient = reference.rounded_divide(values, denominator)
    packed = hybrid.pack_remainder(values, quotient, denominator)
    recovered = hybrid.unpack_remainder(packed, quotient, denominator)
    np.testing.assert_array_equal(values, recovered)
    assert len(packed) == values.size * hybrid.remainder_bytes(denominator)


def test_zero_quotient_cannot_claim_a_halfway_rounding_tie():
    with pytest.raises(reference.Rejected, match="rounded quotient"):
        hybrid.unpack_remainder(bytes([0]), np.zeros((1, 1), dtype=np.int64), 4)


def test_verification_replays_only_forward_not_either_backward_product(job, monkeypatch):
    payload, identity, claim, seed = checked_claim(job)
    dense = hybrid._dense
    calls = []
    def counted(left, right):
        calls.append((left.shape, right.shape))
        return dense(left, right)
    def forbidden(*args, **kwargs):
        raise AssertionError("complete replay invoked on hybrid path")
    monkeypatch.setattr(hybrid, "_dense", counted)
    monkeypatch.setattr(hybrid, "train_optimized", forbidden)
    monkeypatch.setattr(reference, "train", forbidden)
    hybrid.verify(payload, identity, claim, seed=seed, committed_root=claim.commitment)
    assert calls == [(job.inputs.shape, job.weights.shape)]


def test_optimizer_consistent_fabricated_gradient_is_rejected(job):
    trace = reference.train(job)
    trace["weight_gradient"] = np.zeros_like(job.weights)
    trace["after"] = job.weights.copy()
    payload, identity, claim, seed = checked_claim(job, trace)
    with pytest.raises(reference.Rejected, match="backward product"):
        hybrid.verify(payload, identity, claim, seed=seed, committed_root=claim.commitment)
    boundary = hybrid.produce(job, trace, identity, hybrid=False)
    with pytest.raises(reference.Rejected, match="replay mismatch"):
        hybrid.replay(payload, identity, boundary, optimized=True)


def test_upstream_gradient_forgery_is_rejected_after_commitment(job):
    trace = reference.train(job)
    trace["input_gradient"][0, 0] += 1
    payload, identity, claim, seed = checked_claim(job, trace)
    with pytest.raises(reference.Rejected, match="backward product"):
        hybrid.verify(payload, identity, claim, seed=seed, committed_root=claim.commitment)


@pytest.mark.parametrize("attack", ["truncated", "extended", "weight_range", "remainder"])
def test_malformed_witness_fails_even_with_matching_hash(job, attack):
    payload, identity, claim, _ = checked_claim(job)
    changed = bytearray(claim.payload)
    if attack == "truncated":
        changed = changed[:-1]
    elif attack == "extended":
        changed += b"\0"
    elif attack == "weight_range":
        struct.pack_into("<i", changed, 8, 2**31 - 1)
    else:
        start = 8 + 4 * (job.weights.size + job.inputs.size)
        width = hybrid.remainder_bytes(job.scale * len(job.inputs)
                                        * job.learning_rate_denominator)
        changed[start:start + width] = bytes([255]) * width
    changed = bytes(changed)
    forged = hybrid.Claim(changed, hybrid.commitment(identity, changed))
    with pytest.raises(reference.Rejected):
        hybrid.verify(payload, identity, forged, seed=secrets.token_bytes(32),
                      committed_root=forged.commitment)


def test_prechallenge_commitment_cannot_be_changed(job):
    payload, identity, claim, seed = checked_claim(job)
    changed = bytearray(claim.payload)
    changed[-1] ^= 1
    with pytest.raises(reference.Rejected, match="after commitment"):
        hybrid.verify(payload, identity, replace(claim, payload=bytes(changed)), seed=seed,
                      committed_root=claim.commitment)
    rehashed = hybrid.Claim(bytes(changed), hybrid.commitment(identity, bytes(changed)))
    with pytest.raises(reference.Rejected, match="replaced after"):
        hybrid.verify(payload, identity, rehashed, seed=seed,
                      committed_root=claim.commitment)


def test_job_binding_rejects_input_substitution_and_relabeling(job):
    payload, identity, claim, seed = checked_claim(job)
    altered_weights = job.weights.copy()
    altered_weights[0, 0] += 1
    altered = replace(job, weights=altered_weights)
    with pytest.raises(reference.Rejected, match="unadmitted"):
        hybrid.verify(hybrid.encode_job(altered), identity, claim, seed=seed,
                      committed_root=claim.commitment)
    with pytest.raises(reference.Rejected, match="after commitment"):
        hybrid.verify(hybrid.encode_job(altered), altered.work_id(), claim, seed=seed,
                      committed_root=claim.commitment)


@pytest.mark.parametrize("attack", ["truncated", "extra", "too_large", "zero_scale"])
def test_input_wire_rejects_bad_sizes_and_numerical_parameters(job, attack):
    payload, identity, claim, seed = checked_claim(job)
    changed = bytearray(payload)
    if attack == "truncated":
        changed = changed[:-1]
    elif attack == "extra":
        changed += b"\0"
    elif attack == "too_large":
        struct.pack_into("<I", changed, 8, 2**32 - 1)
    else:
        struct.pack_into("<I", changed, 20, 0)
    with pytest.raises(reference.Rejected):
        hybrid.verify(bytes(changed), identity, claim, seed=seed,
                      committed_root=claim.commitment)


def test_float32_reassociation_rejects_an_honest_float_product():
    left = np.array([[2**24, 1]], dtype=np.float32)
    right = np.array([[1, -1], [1, 0]], dtype=np.float32)
    vector = np.ones((2, 1), dtype=np.float32)
    assert ((left @ right) @ vector).item() == 0
    assert (left @ (right @ vector)).item() == 1
    job = reference.Job(left, right, np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(reference.Rejected, match="int64"):
        hybrid.encode_job(job)


def test_profile_bounds_reject_large_products_before_binary64_execution():
    job = reference.Job(np.full((1, 512), reference.MAX_ENTRY, dtype=np.int64),
                         np.full((512, 1), reference.MAX_ENTRY, dtype=np.int64),
                         np.zeros((1, 1), dtype=np.int64))
    with pytest.raises(reference.Rejected, match="uniquely recovered"):
        hybrid.train_optimized(job)


def test_wire_saving_does_not_claim_less_traffic_than_minimal_replay(job):
    trace = reference.train(job)
    identity = job.work_id()
    candidate = hybrid.produce(job, trace, identity, hybrid=True)
    boundary = hybrid.produce(job, trace, identity, hybrid=False)
    expected_extra = job.weights.size * hybrid.remainder_bytes(
        job.scale * len(job.inputs) * job.learning_rate_denominator)
    assert len(candidate.payload) - len(boundary.payload) == expected_extra
    assert len(candidate.payload) < sum(array.nbytes for array in trace.values())


def test_an_honest_receipt_can_be_rechecked_without_fresh_training(job):
    payload, identity, claim, _ = checked_claim(job)
    for _ in range(3):
        hybrid.verify(payload, identity, claim, seed=secrets.token_bytes(32),
                      committed_root=claim.commitment)


def test_multiple_shapes_and_rounding_profiles_match_int64_oracle():
    rng = np.random.default_rng(753)
    for scale in (1, 3, 64):
        for shape in ((1, 1, 1), (5, 7, 3), (16, 8, 24)):
            batch, width, outputs = shape
            job = reference.Job(
                rng.integers(-4, 5, (batch, width), dtype=np.int64),
                rng.integers(-3, 4, (width, outputs), dtype=np.int64),
                rng.integers(-4, 5, (batch, outputs), dtype=np.int64),
                scale=scale, learning_rate_denominator=3)
            exact = reference.train(job)
            fast = hybrid.train_optimized(job)
            for name in reference.TRACE_NAMES:
                np.testing.assert_array_equal(exact[name], fast[name])
            payload, identity, claim, seed = checked_claim(job, exact)
            hybrid.verify(payload, identity, claim, seed=seed, committed_root=claim.commitment)


def test_accounting_counts_both_parties_and_does_not_call_audit_savings_training_savings():
    phases = {"input_preparation": 1, "boundary_production": 1,
              "hybrid_production": 2, "hybrid_verification": 1,
              "replay_optimized": 6, "replay_reference": 8,
              "ordinary_training_optimized": 3, "ordinary_training_reference": 5}
    sample = {"timings": {key: {"cpu_seconds": value, "wall_seconds": value}
                           for key, value in phases.items()},
              "bytes": {"job": 200, "boundary_payload": 100, "hybrid_payload": 140}}
    scenario = {"inputs_cached": False, "cpu_usd_per_hour": 3600,
                "transfer_usd_per_decimal_gb": 0, "link_bits_per_second": 1000,
                "challenge_rtt_seconds": 0.1}
    result = account(sample, scenario)
    assert result["candidate"]["cost_usd"] == 4
    assert result["controls"]["replay_optimized"]["cost_usd"] == 8
    assert result["audit_cost_ratio"] == 0.5
    assert result["ordinary_work_plus_audit_cost_ratio"] == pytest.approx(7 / 11)
    assert result["candidate"]["bytes"] == 436
    assert result["controls"]["replay_optimized"]["bytes"] == 364
    assert result["modeled_latency_ratio"] > 0.5
    assert result["maximum_transfer_usd_per_gb_for_half_audit_cost"] == 0
    cached = account(sample, {**scenario, "inputs_cached": True})
    assert cached["candidate"]["cpu_seconds"] == 4
    assert cached["candidate"]["bytes"] == 236
    charged = account(sample, {**scenario, "transfer_usd_per_decimal_gb": 1e9})
    assert charged["audit_cost_ratio"] > 1
    assert charged["candidate_traffic_cost_floor_over_replay_cost"] > 0.5
