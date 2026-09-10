"""Critical invariants of the native-chain reference, without network sockets."""

import copy
from pathlib import Path

import pytest
import torch

from neuroshard.demo import abci_pb2 as pb, client, protocol, work
from neuroshard.demo.app import Application


DATA = Path(__file__).resolve().parents[1] / "docs/eval/data/input.txt"


@pytest.fixture
def task():
    work.configure_cpu()
    data = work.read_data(DATA)
    miner = protocol.Identity("test-miner")
    stages = [protocol.Identity("test-stage-0"), protocol.Identity("test-stage-1")]
    genesis = protocol.genesis_state("test-chain", work.manifest(data))
    claim = miner.sign(client.claim_body(genesis, [s.public_key for s in stages]))
    state = protocol.transition(genesis, claim, 1, None)
    expected = work.replay(state["weights"], data, 0, state["lease"]["task_id"])
    body = {"kind": "train", "chain_id": state["chain_id"], "round": 0,
            "parent": state["model_root"], "task_id": state["lease"]["task_id"],
            "result_root": work.digest(expected["weights"]),
            "receipts": [s.sign(copy.deepcopy(r)) for s, r in zip(stages, expected["receipts"])]}
    return data, miner, stages, genesis, claim, state, expected, body


def test_two_stage_training_matches_full_model_across_updates(task):
    data, _, _, genesis, *_ = task
    weights = genesis["weights"]
    first, second = work.Stage(0), work.Stage(1)
    assert set(dict(first.named_parameters())).isdisjoint(dict(second.named_parameters()))
    for round_number in range(3):
        common = {"task_id": f"task-{round_number}", "input_ids": work.batch(data, round_number).tolist()}
        request = {**common, "weights": work.stage_weights(weights, 0), "operation": "forward"}
        activation = first.compute(request)["activation"]
        backward1 = second.compute({**common, "weights": work.stage_weights(weights, 1),
                                    "operation": "backward", "activation": activation})
        request.update(operation="backward", adjoint=backward1["adjoint"],
                       loss_hex=backward1["receipt"]["loss_hex"])
        backward0 = first.compute(request)
        expected = work.replay(weights, data, round_number, common["task_id"])
        gradients = {**backward0["gradients"], **backward1["gradients"]}
        assert gradients == expected["gradients"]
        assert [backward0["receipt"], backward1["receipt"]] == expected["receipts"]
        weights = work.apply_gradients(weights, gradients)
        assert weights == expected["weights"]


def test_atomic_reward_and_content_based_replay_prevention(task):
    _, miner, stages, _, _, state, expected, body = task
    original = copy.deepcopy(state)
    result = protocol.transition(state, miner.sign(body), 2, lambda _: expected)
    assert state == original
    assert result["round"] == 1 and result["lease"] is None
    assert result["model_root"] == work.digest(expected["weights"])
    assert result["total_issued"] == sum(result["balances"].values()) == work.REWARD
    assert result["balances"] == {s.address: work.REWARD // 2 for s in stages}
    with pytest.raises(ValueError, match="already applied"):
        protocol.transition(result, miner.sign(body), 3, lambda _: expected)


@pytest.mark.parametrize("field,value", [("result_root", "0" * 64), ("chain_id", "other"),
    ("parent", "0" * 64), ("round", True), ("task_id", "other")])
def test_invalid_claims_never_mutate_state(task, field, value):
    _, miner, _, _, _, state, expected, body = task
    body[field] = value
    original = copy.deepcopy(state)
    with pytest.raises(ValueError):
        protocol.transition(state, miner.sign(body), 2, lambda _: expected)
    assert state == original and state["total_issued"] == 0


@pytest.mark.parametrize("attack", ["gradient", "boundary", "stage_type", "wrong_worker", "signature", "missing"])
def test_stage_receipts_bind_computation_and_identity(task, attack):
    _, miner, stages, _, _, state, expected, body = task
    if attack == "missing":
        body["receipts"].pop()
    elif attack == "signature":
        body["receipts"][0]["signature"] = "00"
    elif attack == "wrong_worker":
        body["receipts"][0] = stages[1].sign(body["receipts"][0]["body"])
    else:
        receipt = body["receipts"][0]["body"]
        field = {"gradient": "gradient_root", "boundary": "output_root", "stage_type": "stage"}[attack]
        receipt[field] = False if attack == "stage_type" else "f" * 64
        body["receipts"][0] = stages[0].sign(receipt)
    with pytest.raises(ValueError):
        protocol.transition(state, miner.sign(body), 2, lambda _: expected)
    assert state["round"] == state["total_issued"] == 0


def test_expired_lease_can_be_reassigned_but_old_work_cannot_be_paid(task):
    _, miner, stages, _, _, state, expected, body = task
    deadline = state["lease"]["expires"]
    with pytest.raises(ValueError, match="expired"):
        protocol.transition(state, miner.sign(body), deadline + 1, lambda _: expected)
    replacement = miner.sign(client.claim_body(state, [s.public_key for s in stages]))
    with pytest.raises(ValueError, match="already assigned"):
        protocol.transition(state, replacement, deadline, None)
    replaced = protocol.transition(state, replacement, deadline + 1, None)
    with pytest.raises(ValueError, match="not assigned"):
        protocol.transition(replaced, miner.sign(body), deadline + 2, lambda _: expected)


def test_emission_budget_is_consensus_enforced(task):
    _, miner, stages, _, _, state, expected, body = task
    state["manifest"]["max_rewarded_tasks"] = 1
    result = protocol.transition(state, miner.sign(body), 2, lambda _: expected)
    claim = miner.sign(client.claim_body(result, [s.public_key for s in stages]))
    with pytest.raises(ValueError, match="budget exhausted"):
        protocol.transition(result, claim, 3, None)


def test_commit_is_durable_and_uncommitted_work_is_replayed_after_crash(task, tmp_path):
    _, miner, _, genesis, claim, _, _, body = task
    database = tmp_path / "state.sqlite"
    app = Application(database, DATA)
    app.InitChain(pb.RequestInitChain(chain_id=genesis["chain_id"],
                                     app_state_bytes=work.canonical(genesis["manifest"])), None)
    app.FinalizeBlock(pb.RequestFinalizeBlock(height=1, txs=[work.canonical(claim)]), None)
    app.Commit(pb.RequestCommit(), None)
    tx = work.canonical(miner.sign(body))
    assert app.ProcessProposal(pb.RequestProcessProposal(height=2, txs=[tx]), None).status == 1
    finalized = app.FinalizeBlock(pb.RequestFinalizeBlock(height=2, txs=[tx]), None)
    assert app.state["round"] == 0  # FinalizeBlock must not expose or persist speculative rewards.
    app.db.close()
    recovered = Application(database, DATA)
    assert recovered.state["round"] == 0 and recovered.state["height"] == 1
    replayed = recovered.FinalizeBlock(pb.RequestFinalizeBlock(height=2, txs=[tx]), None)
    assert replayed.app_hash == finalized.app_hash
    recovered.Commit(pb.RequestCommit(), None)
    recovered.db.close()
    committed = Application(database, DATA)
    assert committed.state["round"] == 1 and committed.state["total_issued"] == work.REWARD
    assert committed.Info(pb.RequestInfo(), None).last_block_app_hash == replayed.app_hash
    assert committed.ProcessProposal(pb.RequestProcessProposal(height=3, txs=[tx]), None).status == 2
    assert committed.ProcessProposal(pb.RequestProcessProposal(height=3, txs=[b"{}", b"{}"]), None).status == 2
    assert committed.Query(pb.RequestQuery(path="/status", prove=True), None).code == 1
    committed.db.close()


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'])
def test_ambiguous_json_rejected(raw):
    with pytest.raises(ValueError):
        protocol.parse_json(raw)


def test_nonfinite_or_misshaped_tensor_rejected():
    with pytest.raises(ValueError, match="Nonfinite"):
        work.decode_tensor(work.encode_tensor(torch.tensor([float("nan")])), (1,))
    with pytest.raises(ValueError, match="shape"):
        work.decode_tensor(work.encode_tensor(torch.ones(2)), (1,))
