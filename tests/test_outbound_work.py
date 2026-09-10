import copy
import json

import pytest

from neuroshard.demo import protocol, work
from neuroshard.publicnet.pool import Worker, authorized_request, coordinator_url, signed


@pytest.fixture
def assignment():
    work.configure_cpu()
    sponsor, worker = protocol.Identity("sponsor"), protocol.Identity("worker")
    weights = work.encode_weights(work.make_model())
    lease = {"task_id": "t", "owner": sponsor.public_key, "task_kind": "train",
             "workers": [worker.public_key, worker.public_key], "expires": 120}
    task = {"chain_id": "test", "weights": weights, "input_ids": [[1, 2, 3]], "lease": lease}
    summary = {"chain_id": "test", "height": 1, "lease": lease}
    request = {"task_id": "t", "operation": "forward", "weights": work.stage_weights(weights, 0),
               "input_ids": task["input_ids"]}
    envelope = signed(sponsor, "test", "assignment", worker=worker.public_key, stage=0, request=request)
    return sponsor, worker, task, summary, envelope


@pytest.mark.parametrize("attack", ["foreign_signer", "foreign_chain", "wrong_worker", "different_weights", "different_batch", "expired", "stale"])
def test_workers_check_assignments_against_their_own_finalized_lease(assignment, attack):
    sponsor, worker, task, summary, envelope = assignment
    body = copy.deepcopy(envelope["body"])
    if attack == "foreign_signer":
        sponsor = protocol.Identity("attacker")
    elif attack == "foreign_chain":
        body["chain_id"] = "other"
    elif attack == "wrong_worker":
        body["worker"] = sponsor.public_key
    elif attack == "different_weights":
        body["request"]["weights"] = {}
    elif attack == "different_batch":
        body["request"]["input_ids"] = [[9, 9]]
    elif attack == "expired":
        summary["height"] = 121
    else:
        body["time"] -= 61
    with pytest.raises(ValueError):
        authorized_request(sponsor.sign(body), worker, 0, task, summary)


def test_worker_retries_reuse_persisted_result_and_refuse_changed_operation(tmp_path, monkeypatch, assignment):
    sponsor, identity, task, summary, envelope = assignment
    (tmp_path / "node.json").write_text(json.dumps({"base_port": 26656, "chain_id": "test"}))
    worker = Worker(tmp_path, 0)
    worker.identity = identity
    calls = []
    monkeypatch.setattr("neuroshard.publicnet.pool.wire.query", lambda _, path: task if path == "/task" else summary)
    monkeypatch.setattr(worker.stage, "compute", lambda _: calls.append(1) or {"activation": [1]})
    assert worker.compute(envelope) == worker.compute(envelope) == {"activation": [1]}
    assert calls == [1]
    restored = Worker(tmp_path, 0)
    restored.identity = identity
    monkeypatch.setattr(restored.stage, "compute", lambda _: pytest.fail("Must use durable cached result"))
    assert restored.compute(envelope) == {"activation": [1]}
    worker.journal["operations"]["forward"] = {"digest": work.digest(envelope["body"]["request"])}
    worker.save()
    with pytest.raises(ValueError, match="Interrupted"):
        worker.compute(envelope)


@pytest.mark.parametrize("url", ["http://example.org", "https://user:pass@example.org", "file:///tmp/pool"])
def test_public_worker_transport_requires_https(url):
    with pytest.raises(ValueError):
        coordinator_url(url)
