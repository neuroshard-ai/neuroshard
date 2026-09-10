"""Real-process acceptance test; no consensus or worker mocks."""

import copy
import datetime
import platform
import tempfile
import time
from pathlib import Path

from neuroshard.demo import client, network, protocol, work


def eventually(predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.2)
    raise AssertionError("Condition did not become true before the deadline")


def rejected(url, envelope):
    before = client.query(url)
    try:
        client.broadcast(url, envelope)
    except client.Rejected as exc:
        after = client.query(url)
        for field in ("round", "model_root", "balances", "total_issued"):
            assert before[field] == after[field], field
        return str(exc)
    raise AssertionError("Invalid or replayed work was accepted")


def run(steps=12, base_port=28650, engine=None, output=None):
    work.configure_cpu()
    (network.REPO / ".neuroshard").mkdir(exist_ok=True)
    home = Path(tempfile.mkdtemp(prefix="check-", dir=network.REPO / ".neuroshard"))
    config = network.initialize(home, base_port, engine)
    report = {"recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "consensus": "CometBFT " + network.COMET_VERSION, "validators": 4,
              "validator_voting_power": [1, 1, 1, 1], "workers": 2,
              "host_architecture": platform.machine(), "logical_cpus": __import__("os").cpu_count(),
              "scope": "Single-host development network; fixed validators; full replay; development tokens",
              "chain_id": config["chain_id"], "home": str(home)}
    try:
        initial = network.start(config)[0]
        url, workers = network.urls(config)
        identity = protocol.Identity.load_or_create(home / "miner.key")
        report["manifest"] = initial["manifest"]
        report["parameter_counts_by_stage"] = [client.http(w + "/identity")["parameter_count"] for w in workers]
        report["initial_validation_loss"] = initial["validation_loss"]
        print("CHECK: native chain ready; mining real two-stage training tasks", flush=True)
        report["training"] = client.mine(url, workers, identity, steps)
        final = client.query(url)
        assert final["round"] == steps
        assert final["validation_loss"] < initial["validation_loss"]

        envelope = client.prepare_training(url, workers, identity)
        forged = copy.deepcopy(envelope["body"])
        forged["result_root"] = "0" * 64
        report["invalid_result_rejection"] = rejected(url, identity.sign(forged))
        forged = copy.deepcopy(envelope["body"])
        receipt_body = forged["receipts"][0]["body"]
        receipt_body["gradient_root"] = "f" * 64
        worker_id = protocol.Identity.load_or_create(home / "worker0.key")
        forged["receipts"][0] = worker_id.sign(receipt_body)
        report["signed_invalid_gradient_rejection"] = rejected(url, identity.sign(forged))
        client.broadcast(url, envelope)
        report["duplicate_rejection"] = rejected(url, envelope)
        # A fresh ECDSA signature for identical content must not bypass replay protection.
        report["resigned_duplicate_rejection"] = rejected(url, identity.sign(envelope["body"]))
        print("CHECK: invalid results, signed bad gradients, and duplicate rewards rejected", flush=True)

        network.stop_validator(config, 0)
        surviving_url, _ = network.urls(config, 1)
        before = client.query(surviving_url)
        result = client.mine(surviving_url, workers, identity, 1)[0]
        assert result["round"] == before["round"] + 1
        report["one_validator_offline_training_finalized"] = True
        network.stop_validator(config, 1)
        surviving_url, _ = network.urls(config, 2)
        # Let any already-committed in-flight block settle, then queue a valid transaction.
        time.sleep(2)
        halted = client.query(surviving_url)
        stage_keys = [client.http(w + "/identity")["public_key"] for w in workers]
        queued = identity.sign(client.claim_body(halted, stage_keys))
        client.broadcast(surviving_url, queued, wait=False)
        time.sleep(4)
        later = client.query(surviving_url)
        assert later["height"] == halted["height"]
        assert later["app_hash"] == halted["app_hash"]
        report["two_validators_offline_halt"] = {"height": halted["height"], "observation_seconds": 4}
        print("CHECK: three validators finalized training; two validators could not finalize", flush=True)

        network.start_validator(config, 0)
        network.start_validator(config, 1)
        network.wait_ready(config)
        task_id = protocol.transaction_id(queued)
        eventually(lambda: all((client.query(network.urls(config, i)[0])["lease"] or {}).get("task_id") == task_id
                               for i in range(4)))
        resumed = client.mine(url, workers, identity, 1)[0]
        report["restart_and_catchup"] = True
        target_round = resumed["round"]

        def converged():
            states = [client.query(network.urls(config, i)[0]) for i in range(4)]
            if any(s["round"] != target_round for s in states):
                return None
            for field in ("model_root", "balances", "total_issued"):
                assert all(s[field] == states[0][field] for s in states)
            return states

        states = eventually(converged)
        final = states[0]
        assert final["round"] == steps + 3
        assert final["total_issued"] == (steps + 3) * work.REWARD == sum(final["balances"].values())
        assert all(balance == final["total_issued"] // 2 for balance in final["balances"].values())
        assert len(final["balances"]) == 2
        report.update(final_round=final["round"], final_model_root=final["model_root"],
                      final_validation_loss=final["validation_loss"],
                      total_issued_atoms=final["total_issued"], worker_balances=final["balances"],
                      four_validators_agree=True)
        # Compare actual native block hashes at the same finalized height.
        height = min(s["height"] for s in states)
        block_hashes = [client.rpc(network.urls(config, i)[0], "block", {"height": str(height)})["block_id"]["hash"]
                        for i in range(4)]
        assert len(set(block_hashes)) == 1
        report["shared_block"] = {"height": height, "hash": block_hashes[0]}
        inference = [client.query(network.urls(config, i)[0], "/infer", {"prompt": "ROMEO:", "max_tokens": 12})
                     for i in range(4)]
        assert all(answer == inference[0] for answer in inference)
        report["inference_agreement"] = inference[0]
        report["passed"] = True
        print("CHECK: recovered validators agree on blocks, model, balances, and inference", flush=True)
    finally:
        network.stop(config)
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Keep ephemeral local paths out of the publication artifact.
        report.pop("home", None)
        output.write_bytes(work.canonical(report) + b"\n")
    return report
