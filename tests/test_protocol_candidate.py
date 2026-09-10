"""Adversarial state transitions for the complete bounded v2 candidate."""

import copy
import hashlib
import random
from pathlib import Path

import numpy as np
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from neuroshard.demo import protocol, work
from neuroshard.lab import matrix, state, storage
from neuroshard.lab.app import Application, execution_manifest
from neuroshard.lab import abci_pb2 as pb
from neuroshard.demo import abci_pb2 as base_pb


DATA = Path(__file__).resolve().parents[1] / "docs/eval/data/input.txt"


def consensus_key(seed):
    private = Ed25519PrivateKey.from_private_bytes(hashlib.sha256(seed.encode()).digest())
    public = private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
    return public, private


def native_params():
    return pb.ConsensusParams(block=pb.BlockParams(max_bytes=65_536, max_gas=-1),
        evidence=pb.EvidenceParams(max_age_num_blocks=state.PARAMS["evidence_blocks"],
            max_age_duration=pb.Duration(seconds=state.PARAMS["evidence_seconds"]), max_bytes=16_384),
        validator=pb.ValidatorParams(pub_key_types=["ed25519"]))


@pytest.fixture
def setup():
    work.configure_cpu()
    owners = [protocol.Identity(f"candidate-owner-{i}") for i in range(4)]
    keys = [consensus_key(f"candidate-consensus-{i}") for i in range(4)]
    entries = [{"owner": owner.public_key, "consensus_key": key[0], "bond": 10 * state.PARAMS["bond_unit"],
                "liquid": 20_000_000} for owner, key in zip(owners, keys)]
    genesis = state.genesis("test-candidate", entries, {})
    return genesis, owners, keys, entries


def tx(s, identity, kind, **fields):
    return identity.sign({"chain_id": s["chain_id"], "nonce": s["accounts"].get(identity.public_key, {"nonce": 0})["nonce"],
                          "kind": kind, **fields})


def tick(s, envelope=None, evidence=(), time_ns=None, committers=None):
    if committers is None:
        committers = [state.consensus_address(key) for key in state.voting_power(s, s["height"])]
    projected, updates = state.advance(s, s["height"] + 1, s["time_ns"] + 1_000_000_000 if time_ns is None else time_ns,
                                        evidence, committers)
    if envelope:
        projected = state.transition(projected, envelope, lambda value: state.execute(value, work.read_data(DATA)))
    return projected, updates


def bond_tx(s, owner, key, amount):
    nonce = s["accounts"][owner.public_key]["nonce"]
    proof = key[1].sign(state.possession_message(s["chain_id"], owner.public_key, key[0], amount, nonce)).hex()
    return tx(s, owner, "bond", consensus_key=key[0], amount=amount, possession=proof)


def reserve_tx(s, owner, workers, kind="train", request=None, price=0):
    return tx(s, owner, "reserve", task_kind=kind, parent=s["model_root"], round=s["round"],
              workers=[worker.public_key for worker in workers], request=request or {}, price=price)


def submission(s, owner, workers):
    expected = state.execute(s, work.read_data(DATA))
    return tx(s, owner, "submit", task_id=s["lease"]["task_id"], result_root=expected["result_root"],
              receipts=[worker.sign(copy.deepcopy(receipt)) for worker, receipt in zip(workers, expected["receipts"])])


def test_seeded_transfer_sequences_conserve_supply_and_reject_replays(setup):
    s, owners, _, _ = setup
    generator = random.Random(97)
    for _ in range(80):
        a, b = generator.sample(owners, 2)
        original = copy.deepcopy(s)
        envelope = tx(s, a, "transfer", to=b.public_key, amount=generator.randint(1, 10000))
        s, _ = tick(s, envelope)
        state.invariant(s)
        assert original["accounts"][a.public_key]["nonce"] + 1 == s["accounts"][a.public_key]["nonce"]
        with pytest.raises(ValueError, match="nonce"):
            tick(s, a.sign(envelope["body"]))
    assert s["burned"] == 80 * state.PARAMS["fee"]


@pytest.mark.parametrize("amount", [True, -1, 0, 2 ** 61, 20_000_000])
def test_invalid_or_unfunded_transfers_do_not_spend_fees(setup, amount):
    s, owners, _, _ = setup
    original = copy.deepcopy(s)
    with pytest.raises(ValueError):
        tick(s, tx(s, owners[0], "transfer", to=owners[1].public_key, amount=amount))
    assert s == original


def test_training_issues_once_and_only_committers_receive_verifier_budget(setup):
    s, owners, keys, _ = setup
    workers = [protocol.Identity("worker-a"), protocol.Identity("worker-b")]
    s, _ = tick(s, reserve_tx(s, owners[0], workers))
    submit = submission(s, owners[0], workers)
    s, _ = tick(s, submit)
    assert s["issued"] == 1_000_000
    assert s["pending_verifier_reward"]["budget"] == 200_000
    assert all(s["accounts"][w.public_key]["balance"] == 400_000 for w in workers)
    before = [s["accounts"][o.public_key]["balance"] for o in owners]
    s, _ = tick(s, committers=[state.consensus_address(k[0]) for k in keys[:3]])
    after = [s["accounts"][o.public_key]["balance"] for o in owners]
    assert [b - a for a, b in zip(before, after)] == [50_000, 50_000, 50_000, 0]
    assert s["pending_verifier_reward"] is None
    assert s["burned"] == 2 * state.PARAMS["fee"] + 50_000
    with pytest.raises(ValueError, match="nonce"):
        tick(s, submit)


def test_a_minority_cannot_authorize_deferred_payment(setup):
    s, owners, keys, _ = setup
    s, _ = tick(s, reserve_tx(s, owners[0], owners[:2]))
    s, _ = tick(s, submission(s, owners[0], owners[:2]))
    with pytest.raises(ValueError, match="commit quorum"):
        tick(s, committers=[state.consensus_address(k[0]) for k in keys[:2]])


@pytest.mark.parametrize("attack", ["gradient", "boundary", "wrong_worker", "result", "task"])
def test_authenticated_wrong_work_cannot_change_the_model(setup, attack):
    s, owners, _, _ = setup
    s, _ = tick(s, reserve_tx(s, owners[0], owners[:2]))
    body = submission(s, owners[0], owners[:2])["body"]
    if attack in ("gradient", "boundary"):
        receipt = body["receipts"][0]["body"]
        receipt["gradient_root" if attack == "gradient" else "output_root"] = "0" * 64
        body["receipts"][0] = owners[0].sign(receipt)
    elif attack == "wrong_worker":
        body["receipts"][0] = owners[2].sign(body["receipts"][0]["body"])
    elif attack == "result":
        body["result_root"] = "0" * 64
    else:
        body["task_id"] = "0" * 64
    original = copy.deepcopy(s)
    with pytest.raises(ValueError):
        tick(s, owners[0].sign(body))
    assert original == s


def test_paid_inference_is_escrowed_and_does_not_mint(setup):
    s, owners, _, _ = setup
    provider = protocol.Identity("inference-provider")
    root = s["model_root"]
    s, _ = tick(s, reserve_tx(s, owners[0], [provider], "infer", {"prompt": "ROMEO:", "max_tokens": 4}, 100_000))
    assert s["lease"]["escrow"] == 2_100_000
    s, _ = tick(s, submission(s, owners[0], [provider]))
    s, _ = tick(s)
    assert s["accounts"][provider.public_key]["balance"] == 80_000
    assert s["issued"] == 0 and s["model_root"] == root
    assert s["last_inference"]["model_root"] == root
    state.invariant(s)


def test_reservation_griefing_burns_collateral_and_cannot_reserve_twice(setup):
    s, owners, _, _ = setup
    before = s["accounts"][owners[0].public_key]["balance"]
    for attempt in range(3):
        s, _ = tick(s, reserve_tx(s, owners[0], owners[:2]))
        with pytest.raises(ValueError, match="already reserved"):
            tick(s, reserve_tx(s, owners[1], owners[:2]))
        bad = submission(s, owners[0], owners[:2])
        expiry = s["lease"]["expires"]
        while s["height"] <= expiry:
            s, _ = tick(s)
        assert s["lease"] is None
        with pytest.raises(ValueError, match="not reserved"):
            tick(s, bad)
    assert before - s["accounts"][owners[0].public_key]["balance"] == 3 * (2_000_000 + state.PARAMS["fee"])


def test_bond_power_is_delayed_and_matches_exact_stake_units(setup):
    s, owners, _, _ = setup
    key = consensus_key("new-validator")
    s, _ = tick(s, bond_tx(s, owners[0], key, 1_000_000))
    emit = s["validators"][key[0]]["emit_at"]
    while s["height"] < emit:
        s, updates = tick(s)
    assert updates == {key[0]: 4}
    assert key[0] not in state.voting_power(s, emit + 1)
    assert state.voting_power(s, emit + 2)[key[0]] == 4


def test_splitting_stake_among_keys_does_not_create_power(setup):
    original, owners, _, _ = setup
    totals = []
    for pieces in (1, 2, 4):
        s = copy.deepcopy(original)
        for index in range(pieces):
            key = consensus_key(f"split-{pieces}-{index}")
            s, _ = tick(s, bond_tx(s, owners[0], key, 1_000_000 // pieces))
        for _ in range(20):
            s, _ = tick(s)
        totals.append(sum(state.voting_power(s, s["height"]).values()))
    assert totals == [44, 44, 44]


def test_copied_consensus_possession_proof_cannot_steal_or_duplicate_a_key(setup):
    s, owners, _, _ = setup
    key = consensus_key("bound-proof")
    envelope = bond_tx(s, owners[0], key, 250_000)
    with pytest.raises(ValueError, match="possession"):
        tick(s, owners[1].sign(envelope["body"]))
    s, _ = tick(s, envelope)
    with pytest.raises(ValueError, match="previously used"):
        tick(s, bond_tx(s, owners[1], key, 250_000))


def test_both_unbond_windows_and_post_removal_slashing(setup):
    s, owners, keys, _ = setup
    s, _ = tick(s)
    offense_height = s["height"]
    s, _ = tick(s, tx(s, owners[0], "unbond", consensus_key=keys[0][0]))
    removal = s["validators"][keys[0][0]]["emit_at"] + 2
    while s["height"] < removal:
        s, _ = tick(s, time_ns=s["time_ns"] + 1)
    v = s["validators"][keys[0][0]]
    original_amount = v["amount"]
    evidence = [{"kind": 1, "address": state.consensus_address(keys[0][0]), "height": offense_height}]
    s, _ = tick(s, evidence=evidence, time_ns=s["time_ns"] + 1)
    assert s["validators"][keys[0][0]]["amount"] == original_amount * 3 // 4
    penalty = s["burned"]
    s, _ = tick(s, evidence=evidence, time_ns=s["time_ns"] + 1)
    assert s["burned"] == penalty
    while s["height"] < v["release_height"]:
        s, _ = tick(s, time_ns=s["time_ns"] + 1)
    with pytest.raises(ValueError, match="evidence window"):
        tick(s, tx(s, owners[0], "withdraw", consensus_key=keys[0][0]), time_ns=v["release_time_ns"])
    s, _ = tick(s, tx(s, owners[0], "withdraw", consensus_key=keys[0][0]), time_ns=v["release_time_ns"] + 1)
    assert s["validators"][keys[0][0]]["amount"] == 0
    assert s["validators"][keys[0][0]]["status"] == "withdrawn"


def test_time_alone_does_not_release_bond(setup):
    s, owners, keys, _ = setup
    s, _ = tick(s, tx(s, owners[0], "unbond", consensus_key=keys[0][0]))
    removal = s["validators"][keys[0][0]]["emit_at"] + 2
    while s["height"] < removal:
        s, _ = tick(s)
    with pytest.raises(ValueError, match="evidence window"):
        tick(s, tx(s, owners[0], "withdraw", consensus_key=keys[0][0]), time_ns=s["time_ns"] + 100_000_000_000)


def test_last_validator_cannot_exit_for_a_replacement_not_yet_activated(setup):
    _, owners, keys, entries = setup
    s = state.genesis("test-candidate", entries[:1], {})
    new = consensus_key("replacement")
    s, _ = tick(s, bond_tx(s, owners[0], new, 250_000))
    with pytest.raises(ValueError, match="replacement"):
        tick(s, tx(s, owners[0], "unbond", consensus_key=keys[0][0]))


def test_genesis_engine_powers_must_match_collateral(setup, tmp_path):
    _, _, _, entries = setup
    app = Application(tmp_path / "app.sqlite", DATA)
    request = pb.RequestInitChain(chain_id="candidate", consensus_params=native_params(), app_state_bytes=work.canonical({"manifest": app.spec, "validators": entries}),
        validators=[pb.ValidatorUpdate(pub_key=pb.PublicKey(ed25519=bytes.fromhex(v["consensus_key"])), power=10) for v in entries])
    request.validators[0].power = 1000
    with pytest.raises(ValueError, match="voting power"):
        app.InitChain(request, None)
    request.validators[0].power = 10
    app.InitChain(request, None)
    assert state.voting_power(app.state, 1) == {v["consensus_key"]: 10 for v in entries}
    app.db.close()


def test_numerical_conformance_mismatch_is_rejected_at_genesis(setup, tmp_path):
    _, _, _, entries = setup
    app = Application(tmp_path / "app.sqlite", DATA)
    wrong = copy.deepcopy(app.spec)
    wrong["numerical_conformance"]["three_step_vectors"][0]["gradient_root"] = "0" * 64
    request = pb.RequestInitChain(chain_id="candidate", consensus_params=native_params(), app_state_bytes=work.canonical({"manifest": wrong, "validators": entries}),
        validators=[pb.ValidatorUpdate(pub_key=pb.PublicKey(ed25519=bytes.fromhex(v["consensus_key"])), power=10) for v in entries])
    with pytest.raises(ValueError, match="genesis"):
        app.InitChain(request, None)
    assert app.state is None
    app.db.close()


@pytest.mark.parametrize("field", ["evidence_height", "evidence_time", "evidence_size", "block_size", "key_types", "vote_extensions"])
def test_native_genesis_must_enforce_the_same_evidence_and_execution_bounds(setup, tmp_path, field):
    _, _, _, entries = setup
    app = Application(tmp_path / "app.sqlite", DATA)
    params = native_params()
    if field == "evidence_height":
        params.evidence.max_age_num_blocks += 1
    elif field == "evidence_time":
        params.evidence.max_age_duration.nanos = 1
    elif field == "evidence_size":
        params.evidence.max_bytes = 0
    elif field == "block_size":
        params.block.max_bytes = 10_000_000
    elif field == "key_types":
        params.validator.pub_key_types.append("secp256k1")
    else:
        params.abci.vote_extensions_enable_height = 1
    request = pb.RequestInitChain(chain_id="candidate", consensus_params=params,
        app_state_bytes=work.canonical({"manifest": app.spec, "validators": entries}),
        validators=[pb.ValidatorUpdate(pub_key=pb.PublicKey(ed25519=bytes.fromhex(v["consensus_key"])), power=10) for v in entries])
    with pytest.raises(ValueError, match="consensus parameters"):
        app.InitChain(request, None)
    assert app.state is None
    app.db.close()


def test_crash_replays_validator_update_and_application_hash_identically(setup, tmp_path):
    _, owners, _, entries = setup
    path = tmp_path / "app.sqlite"
    app = Application(path, DATA)
    init = pb.RequestInitChain(chain_id="candidate", consensus_params=native_params(), app_state_bytes=work.canonical({"manifest": app.spec, "validators": entries}),
        validators=[pb.ValidatorUpdate(pub_key=pb.PublicKey(ed25519=bytes.fromhex(v["consensus_key"])), power=10) for v in entries])
    app.InitChain(init, None)
    key = consensus_key("crash-validator")
    envelope = bond_tx(app.state, owners[0], key, 250_000)
    app.FinalizeBlock(pb.RequestFinalizeBlock(height=1, time=pb.Time(seconds=1), txs=[work.canonical(envelope)]), None)
    app.Commit(base_pb.RequestCommit(), None)
    emit = app.state["validators"][key[0]]["emit_at"]
    for height in range(2, emit):
        app.FinalizeBlock(pb.RequestFinalizeBlock(height=height, time=pb.Time(seconds=height)), None)
        app.Commit(base_pb.RequestCommit(), None)
    request = pb.RequestFinalizeBlock(height=emit, time=pb.Time(seconds=emit))
    first = app.FinalizeBlock(request, None)
    assert first.validator_updates[0].pub_key.ed25519.hex() == key[0]
    assert first.validator_updates[0].power == 1
    app.db.close()
    restarted = Application(path, DATA)
    assert restarted.state["height"] == emit - 1
    second = restarted.FinalizeBlock(request, None)
    assert first.SerializeToString() == second.SerializeToString()
    restarted.Commit(base_pb.RequestCommit(), None)
    restarted.db.close()


def test_training_budget_exhaustion_does_not_disable_transfers_or_exits(setup, monkeypatch):
    s, owners, _, _ = setup
    monkeypatch.setitem(state.PARAMS, "max_training_tasks", 0)
    with pytest.raises(ValueError, match="exhausted"):
        tick(s, reserve_tx(s, owners[0], owners[:2]))
    s, _ = tick(s, tx(s, owners[0], "transfer", to=owners[1].public_key, amount=100))
    state.invariant(s)


def test_integer_check_matches_independent_python_arithmetic_and_binds_output():
    rng = np.random.default_rng(51)
    a, b = rng.integers(-127, 128, (4, 7), dtype=np.int64), rng.integers(-127, 128, (7, 5), dtype=np.int64)
    independent = np.array([[sum(int(a[i, k]) * int(b[k, j]) for k in range(7)) for j in range(5)] for i in range(4)], dtype=np.int64)
    assert np.array_equal(a @ b, independent)
    assert matrix.verify(a, b, independent, "job/operation")
    forged = independent.copy()
    forged[0, 1] += 1
    assert matrix.statement(a, b, independent, "job/operation") != matrix.statement(a, b, forged, "job/operation")
    assert not matrix.verify(a, b, forged, "job/operation")


def test_predictable_projection_can_be_fooled():
    a = np.ones((4, 4), dtype=np.int64)
    b = a.copy()
    c = a @ b
    c[0, 0] += 1
    c[0, 1] -= 1
    assert matrix.check_with_challenges(a, b, c, np.ones((4, 1), dtype=np.int64))
    assert not matrix.verify(a, b, c, "full-statement")


@pytest.mark.parametrize("attack", ["overflow", "float", "negative_overflow", "shape", "bad_challenge"])
def test_integer_profile_rejects_ambiguous_or_unsafe_inputs(attack):
    a = np.ones((4, 4), dtype=np.int64)
    b, c = a.copy(), a @ a
    if attack == "overflow":
        a[0, 0] = 2 ** 62
    elif attack == "negative_overflow":
        c[0, 0] = -(2 ** 63)
    elif attack == "float":
        a = a.astype(np.float32)
    elif attack == "shape":
        c = c[:, :2]
    with pytest.raises(ValueError):
        if attack == "bad_challenge":
            matrix.check_with_challenges(a, b, c, np.full((4, 2), 2, dtype=np.int64))
        else:
            matrix.verify(a, b, c, "bounded")


def test_chunk_manifest_binds_length_and_content(tmp_path):
    data = b"available checkpoint data" * 1000
    spec = storage.publish(tmp_path, data)
    combined = b"".join((tmp_path / digest).read_bytes() for digest in spec["chunks"])
    assert combined == data and storage.manifest(combined) == spec
    with pytest.raises(ValueError):
        storage.manifest(b"x" * (storage.MAX_OBJECT + 1))
