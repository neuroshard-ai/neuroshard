"""Mechanism checks only: these tests do not prove computational hardness."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import neural_work_mining as mining
import neural_work_reference as nw


@pytest.fixture(scope="module")
def fixture():
    rng = np.random.default_rng(20260921)
    job = nw.Job(rng.integers(-64, 65, (16, 16), dtype=np.int64),
                 rng.integers(-16, 17, (16, 16), dtype=np.int64),
                 rng.integers(-128, 129, (16, 16), dtype=np.int64))
    challenge, worker = "11" * 32, "22" * 32
    trace, certificates, measurements = mining.train_with_tickets(
        job, challenge=challenge, worker=worker)
    return job, challenge, worker, trace, certificates, measurements


def test_mined_computation_produces_identical_training_result(fixture):
    job, challenge, worker, trace, certificates, measurements = fixture
    expected = nw.train(job)
    for name in nw.TRACE_NAMES:
        assert np.array_equal(expected[name], trace[name])
    bundle = {name: proofs for name, left, right, context, proofs in certificates}
    accepted = mining.verify_bundle(job, trace, bundle, committed_root=nw.trace_root(job, trace),
                                    audit_seed=bytes(32), challenge=challenge, worker=worker)
    assert len(accepted) == sum(len(proofs) for proofs in bundle.values()) > 0
    assert sum(m["tile_attempts"] for m in measurements) == 384


@pytest.mark.parametrize("change", ["work_id", "operation", "chain_challenge", "worker"])
def test_cached_ticket_cannot_be_relabelled(fixture, change):
    _, _, _, _, certificates, _ = fixture
    name, left, right, context, proofs = next(c for c in certificates if c[-1])
    value = "weight_gradient" if change == "operation" else "33" * 32
    with pytest.raises(nw.Rejected):
        mining.verify_ticket(left, right, replace(context, **{change: value}), proofs[0])


def test_rehashing_cached_partial_for_a_new_header_does_not_work(fixture):
    _, _, _, _, certificates, _ = fixture
    _, left, right, context, proofs = next(c for c in certificates if c[-1])
    proof = dict(proofs[0])
    changed = replace(context, chain_challenge="44" * 32)
    proof["descriptor"] = nw.digest(changed.descriptor(left, right))
    proof["ticket"] = mining.ticket_hash(proof["descriptor"], proof["prime"], proof["row"],
                                          proof["inner"], proof["column"], proof["partial"])
    with pytest.raises(nw.Rejected, match="forged intermediate product"):
        mining.verify_ticket(left, right, changed, proof)


def test_no_free_nonce_or_user_selected_difficulty(fixture):
    _, _, _, _, certificates, _ = fixture
    _, left, right, context, proofs = next(c for c in certificates if c[-1])
    for field in ("nonce", "bits"):
        forged = {**proofs[0], field: 0}
        with pytest.raises(nw.Rejected, match="unexpected proof fields"):
            mining.verify_ticket(left, right, context, forged)


def test_forged_tile_cannot_become_a_valid_ticket_by_rehashing(fixture):
    _, _, _, _, certificates, _ = fixture
    _, left, right, context, proofs = next(c for c in certificates if c[-1])
    proof = dict(proofs[0])
    proof["partial"] = proof["partial"].copy()
    proof["partial"][0, 0] = (proof["partial"][0, 0] + 1) % proof["prime"]
    proof["ticket"] = mining.ticket_hash(proof["descriptor"], proof["prime"], proof["row"],
                                          proof["inner"], proof["column"], proof["partial"])
    with pytest.raises(nw.Rejected, match="forged intermediate product"):
        mining.verify_ticket(left, right, context, proof)


def test_duplicate_tickets_and_coherent_fake_gradients_rejected(fixture):
    job, challenge, worker, trace, certificates, _ = fixture
    bundle = {name: list(proofs) for name, _, _, _, proofs in certificates}
    name = next(name for name in bundle if bundle[name])
    bundle[name].append(bundle[name][0])
    with pytest.raises(nw.Rejected, match="duplicate mining ticket"):
        mining.verify_bundle(job, trace, bundle, committed_root=nw.trace_root(job, trace),
                             audit_seed=bytes(32), challenge=challenge, worker=worker)
    bundle[name].pop()
    forged = {name: value.copy() for name, value in trace.items()}
    forged["weight_gradient"][:] = 0
    forged["after"][:] = job.weights
    with pytest.raises(nw.Rejected, match="incorrect product"):
        mining.verify_bundle(job, forged, bundle, committed_root=nw.trace_root(job, forged),
                             audit_seed=bytes(32), challenge=challenge, worker=worker)


def test_changed_header_keeps_useful_result_but_changes_transcript(fixture):
    job, _, worker, old_trace, _, old_measurements = fixture
    trace, certificates, measurements = mining.train_with_tickets(
        job, challenge="55" * 32, worker=worker)
    for name in nw.TRACE_NAMES:
        assert np.array_equal(old_trace[name], trace[name])
    assert all(a["descriptor"] != b["descriptor"] for a, b in zip(old_measurements, measurements))


def test_losing_lottery_does_not_discard_the_useful_output(fixture):
    job, challenge, worker, expected, _, _ = fixture
    trace, certificates, measurements = mining.train_with_tickets(
        job, challenge=challenge, worker=worker, bits=16)
    assert sum(m["winning_tiles"] for m in measurements) == 0
    for name in nw.TRACE_NAMES:
        assert np.array_equal(expected[name], trace[name])


def test_zero_inputs_still_generate_resource_tickets_not_learning_evidence():
    # Explicit limitation: this mechanism cannot itself establish useful jobs.
    zero = np.zeros((16, 16), dtype=np.int64)
    context = mining.Context("11" * 32, "forward", "22" * 32, "33" * 32)
    recovered, proofs, measurement = mining.mine_product(zero, zero, context)
    assert np.array_equal(recovered, zero)
    assert proofs
    for proof in proofs:
        mining.verify_ticket(zero, zero, context, proof)
