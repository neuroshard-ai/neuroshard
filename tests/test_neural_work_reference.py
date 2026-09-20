"""Adversarial checks of the research verifier, not live consensus tests."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import neural_work_reference as nw


@pytest.fixture
def job():
    rng = np.random.default_rng(19)
    return nw.Job(rng.integers(-64, 65, (16, 24), dtype=np.int64),
                  rng.integers(-16, 17, (24, 12), dtype=np.int64),
                  rng.integers(-32, 33, (16, 12), dtype=np.int64))


def challenge_trace(job, trace):
    book = nw.AdmissionBook()
    work_id = book.admit(job, "worker-a")
    root = nw.trace_root(job, trace)
    book.commit(work_id, "worker-a", root)
    seed = book.challenge(work_id, "worker-a")
    return book, root, seed


def test_honest_transition_and_exact_binary64_baseline(job):
    trace = nw.train(job)
    optimized = nw.train(job, backend="float64")
    # Independent unbounded-integer product oracle (avoids int64/BLAS bugs).
    assert np.array_equal(trace["forward"], job.inputs.astype(object) @ job.weights.astype(object))
    for name in nw.TRACE_NAMES:
        assert np.array_equal(trace[name], optimized[name])
    book, root, seed = challenge_trace(job, trace)
    nw.verify(job, trace, committed_root=root, seed=seed)
    nw.full_replay(job, trace, committed_root=root)
    assert book.settle(job, trace, "worker-a") == nw.matrix_root(trace["after"])


def test_linear_training_matches_hand_calculated_forward_backward_and_update():
    job = nw.Job(np.array([[64, 0], [0, 64]], dtype=np.int64),
                 np.array([[64, -64], [32, 64]], dtype=np.int64),
                 np.array([[0, 64], [64, 0]], dtype=np.int64))
    trace = nw.train(job)
    assert trace["forward"].tolist() == [[4096, -4096], [2048, 4096]]
    assert trace["weight_gradient"].tolist() == [[4096, -8192], [-2048, 4096]]
    assert trace["input_gradient"].tolist() == [[12288, -6144], [-6144, 3072]]
    assert trace["after"].tolist() == [[60, -56], [34, 60]]
    nw.verify(job, trace, committed_root=nw.trace_root(job, trace), seed=bytes(32))


def test_minimal_replay_checks_outputs_without_intermediate_witness(job):
    trace = nw.train(job)
    after, gradient = trace["after"], trace["input_gradient"]
    commitment = nw.boundary_root(job, after, gradient)
    nw.replay_boundary(job, after, gradient, committed_root=commitment)
    after[0, 0] += 1
    forged_commitment = nw.boundary_root(job, after, gradient)
    with pytest.raises(nw.Rejected, match="boundary replay mismatch"):
        nw.replay_boundary(job, after, gradient, committed_root=forged_commitment)


@pytest.mark.parametrize("method", ["integer", "modular"])
@pytest.mark.parametrize("tensor", nw.TRACE_NAMES)
def test_each_forged_operation_is_rejected_before_payment(job, tensor, method):
    trace = nw.train(job)
    trace[tensor][0, 0] += 1
    book, root, seed = challenge_trace(job, trace)
    with pytest.raises(nw.Rejected):
        nw.full_replay(job, trace, committed_root=root)
    with pytest.raises(nw.Rejected):
        nw.verify(job, trace, committed_root=root, seed=seed, method=method)
    with pytest.raises(nw.Rejected):
        book.settle(job, trace, "worker-a")
    assert book.accepted_results == {}


def test_coherent_fake_gradient_and_optimizer_are_rejected(job):
    trace = nw.train(job)
    trace["weight_gradient"][:] = 0
    trace["after"][:] = job.weights
    book, root, seed = challenge_trace(job, trace)
    with pytest.raises(nw.Rejected, match="incorrect product"):
        book.settle(job, trace, "worker-a")
    assert not book.accepted_results


def test_two_fields_prevent_integer_modulus_alias():
    left = np.full((2, 3), 300, dtype=np.int64)
    right = np.full((3, 2), 300, dtype=np.int64)
    honest = left @ right
    forged = honest.copy()
    forged[0, 0] -= nw.PRIMES[0]
    assert np.array_equal(honest % nw.PRIMES[0], forged % nw.PRIMES[0])
    with pytest.raises(nw.Rejected):
        nw.check_product(left, right, forged, seed=bytes(32), context="alias")


def test_challenge_requires_commit_and_cannot_be_rerolled(job):
    book = nw.AdmissionBook()
    work_id = book.admit(job, "a")
    with pytest.raises(nw.Rejected):
        book.challenge(work_id, "a")
    trace = nw.train(job)
    book.commit(work_id, "a", nw.trace_root(job, trace))
    seed = book.challenge(work_id, "a")
    assert book.challenge(work_id, "a") == seed
    with pytest.raises(nw.Rejected):
        book.commit(work_id, "a", "00" * 32)
    trace["forward"][0, 0] += 1
    with pytest.raises(nw.Rejected, match="changed after commitment"):
        book.settle(job, trace, "a")


def test_assignment_and_duplicate_work_cannot_be_bypassed_by_new_identity(job):
    trace = nw.train(job)
    book, root, seed = challenge_trace(job, trace)
    with pytest.raises(nw.Rejected):
        book.settle(job, trace, "worker-b")
    with pytest.raises(nw.Rejected):
        book.admit(replace(job), "worker-b")
    book.settle(job, trace, "worker-a")
    with pytest.raises(nw.Rejected):
        book.settle(job, trace, "worker-a")
    assert len(book.accepted_results) == 1


@pytest.mark.parametrize("field", ["weights", "inputs", "targets", "scale", "learning_rate_denominator"])
def test_unadmitted_or_stale_job_cannot_use_an_old_receipt(job, field):
    trace = nw.train(job)
    book, root, seed = challenge_trace(job, trace)
    value = getattr(job, field)
    if isinstance(value, np.ndarray):
        value = value.copy()
        value[0, 0] += 1
    else:
        value += 1
    altered = replace(job, **{field: value})
    with pytest.raises(nw.Rejected):
        book.settle(altered, trace, "worker-a")
    assert not book.accepted_results


def test_missing_witness_cannot_pay(job):
    trace = nw.train(job)
    book, root, seed = challenge_trace(job, trace)
    del trace["input_gradient"]
    with pytest.raises(nw.Rejected):
        book.settle(job, trace, "worker-a")
    assert not book.accepted_results


@pytest.mark.parametrize("bad", [np.array([[np.iinfo(np.int64).min]], dtype=np.int64),
                               np.zeros((513, 1), dtype=np.int64),
                               np.array([[float("nan")]]),
                               np.array([[0]], dtype=np.int32)])
def test_numeric_profile_fails_closed(bad):
    with pytest.raises(nw.Rejected):
        nw.matrix(bad)


def test_integer_crt_range_cannot_wrap():
    left = np.full((1, 512), nw.MAX_ENTRY, dtype=np.int64)
    with pytest.raises(nw.Rejected):
        nw.exact_product(left, left.T, backend="int64")


def test_verifier_never_invokes_training_replay(job, monkeypatch):
    trace = nw.train(job)
    book, root, seed = challenge_trace(job, trace)
    def forbidden(*args, **kwargs):
        raise AssertionError("full training replay called")
    monkeypatch.setattr(nw, "train", forbidden)
    monkeypatch.setattr(nw, "exact_product", forbidden)
    book.settle(job, trace, "worker-a")


def test_verified_cached_work_is_not_proof_of_fresh_computation(job, monkeypatch):
    trace = nw.train(job)
    root = nw.trace_root(job, trace)
    def forbidden(*args, **kwargs):
        raise AssertionError("fresh training should not be necessary for this attack")
    monkeypatch.setattr(nw, "train", forbidden)
    for challenge in range(8):
        nw.verify(job, trace, committed_root=root, seed=challenge.to_bytes(32, "big"))
    attack = nw.receipt_nonce_attack(job.work_id(), root)
    assert attack["additional_training_products"] == 0
    assert int(attack["ticket"], 16) < 1 << 248


def test_final_product_only_noise_has_a_low_rank_shortcut():
    attack = nw.output_only_noise_attack()
    assert attack["same_output"]
    assert attack["shortcut_multiplications"] < attack["dense_product_multiplications"] / 10


def test_rounding_is_explicit_at_signed_ties():
    values = np.array([[-3, -1, 0, 1, 3]], dtype=np.int64)
    assert nw.rounded_divide(values, 2).tolist() == [[-2, -1, 0, 1, 2]]


def test_prime_parameters_are_prime_and_arithmetic_cannot_overflow():
    for prime in nw.PRIMES:
        assert all(prime % divisor for divisor in range(2, int(prime ** .5) + 1))
        assert nw.MAX_DIM * (prime - 1) ** 2 < np.iinfo(np.int64).max


def test_quantized_sgd_learns_a_small_linear_task():
    rng = np.random.default_rng(20260920)
    inputs = rng.integers(-64, 65, (64, 16), dtype=np.int64)
    teacher = rng.integers(-16, 17, (16, 8), dtype=np.int64)
    targets = nw.rounded_divide(inputs @ teacher, 64)
    job = nw.Job(inputs, np.zeros_like(teacher), targets)
    losses = []
    for step in range(16):
        trace = nw.train(job, backend="float64")
        residual = nw.rounded_divide(trace["forward"], job.scale) - targets
        losses.append(float(np.mean(residual.astype(np.float64) ** 2)))
        job = replace(job, weights=trace["after"])
    assert losses[-1] < losses[0]


def test_integer_projection_near_range_limit_is_exact():
    left = np.full((16, 512), 2000, dtype=np.int64)
    right = np.full((512, 24), 2000, dtype=np.int64)
    left[0, ::2] *= -1  # exercise signed cancellation as well as large sums
    claimed = left @ right
    assert np.array_equal(claimed, left.astype(object) @ right.astype(object))
    nw.check_product_integer(left, right, claimed, seed=bytes(32), context="bounds")
    claimed[1, 0] -= 1  # must detect unit error even near the allowed maximum
    with pytest.raises(nw.Rejected):
        nw.check_product_integer(left, right, claimed, seed=bytes(32), context="bounds")


def test_integer_projection_has_no_float_tolerance_or_modular_alias():
    left = np.full((2, 3), 300, dtype=np.int64)
    right = np.full((3, 2), 300, dtype=np.int64)
    claimed = left @ right
    nw.check_product_integer(left, right, claimed, seed=bytes(32), context="honest")
    claimed[0, 0] -= nw.PRIMES[0]
    with pytest.raises(nw.Rejected):
        nw.check_product_integer(left, right, claimed, seed=bytes(32), context="alias")


def test_learning_driver_records_fixed_point_without_duplicate_payment():
    import json
    from study_neural_work import learning_demo
    plan = json.loads((Path(__file__).resolve().parents[1]
                       / "config/experiments/neural-work-reference.json").read_text())
    result = learning_demo(plan)
    assert result["accepted_steps"] < plan["learning_steps"]
    assert "fixed point" in result["stop_reason"]
    assert result["loss_after_last_step"] < result["loss_before_each_step"][0]
