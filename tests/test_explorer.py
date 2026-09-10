import base64
import copy
import json

import pytest

from neuroshard.demo import work
from neuroshard.publicnet.gateway import Gateway, client_ip
from neuroshard.publicnet.history import TrainingHistory


@pytest.mark.parametrize("address,header,expected", [
    ("127.0.0.1", "203.0.113.7", "203.0.113.7"),
    ("::1", "2001:db8::1", "2001:db8::1"),
    ("203.0.113.7", "198.51.100.2", "203.0.113.7"),
    ("127.0.0.1", "203.0.113.7, 198.51.100.2", "127.0.0.1"),
    ("127.0.0.1", None, "127.0.0.1"),
])
def test_only_loopback_proxy_can_supply_one_valid_client_address(address, header, expected):
    assert client_ip(address, header) == expected


def block(height, code=0, inference=False):
    receipts = [{"public_key": f"worker-{i}", "body": {"stage": i, "loss_hex": (5.0).hex()}} for i in range(2)]
    if inference:
        receipts = [{"public_key": "worker", "body": {"result_root": "output"}}]
    return {"height": height, "hash": f"block-{height}", "previous_hash": f"block-{height-1}",
            "time": "2026-09-10T00:00:00Z", "transactions": [{"kind": "submit", "code": code,
            "hash": f"tx-{height}", "body": {"task_id": f"task-{height}", "result_root": f"model-{height}",
                                              "receipts": receipts}}]}


def test_history_excludes_failed_submissions_and_inference_and_survives_restart(tmp_path):
    path = tmp_path / "index.sqlite"
    index = TrainingHistory(path, "genesis")
    index.append(block(1))
    index.append(block(2, code=1))
    index.append(block(3, inference=True))
    index.close()
    index = TrainingHistory(path, "genesis")
    index.append(block(4))
    page = index.page(limit=1)
    assert page["indexed_height"] == 4 and page["indexed_rounds"] == 2
    assert page["records"][0]["round"] == 2 and page["records"][0]["loss"] == 5.0
    assert index.page(before=page["next_before"])["records"][0]["height"] == 1
    index.close()


def test_history_refuses_gaps_forks_and_foreign_genesis(tmp_path):
    path = tmp_path / "index.sqlite"
    index = TrainingHistory(path, "genesis")
    index.append(block(1))
    for candidate in [block(1), block(3), {**block(2), "previous_hash": "foreign"}]:
        with pytest.raises(ValueError, match="contiguous"):
            index.append(candidate)
        assert index.cursor()[0] == 1
    index.close()
    with pytest.raises(ValueError, match="genesis"):
        TrainingHistory(path, "other")


def test_bad_receipts_do_not_advance_index(tmp_path):
    index = TrainingHistory(tmp_path / "index.sqlite", "genesis")
    broken = block(1)
    broken["transactions"][0]["body"]["receipts"][1]["body"]["loss_hex"] = (9.0).hex()
    with pytest.raises(ValueError, match="Inconsistent"):
        index.append(broken)
    assert index.page()["indexed_height"] == index.page()["indexed_rounds"] == 0
    with pytest.raises(ValueError):
        index.page(limit=101)
    index.close()


def test_checkpoint_uses_atomic_query_height_and_checks_content_root(tmp_path, monkeypatch):
    (tmp_path / "node.json").write_text(json.dumps({"base_port": 26656, "genesis_sha256": "genesis"}))
    gateway = Gateway(tmp_path)
    weights = {"tiny": {"shape": [1], "data": "AAAAAA=="}}
    task = {"weights": weights, "model_root": work.digest(weights), "chain_id": "chain", "round": 3}
    response = {"response": {"height": "72", "value": base64.b64encode(work.canonical(task)).decode()}}
    monkeypatch.setattr("neuroshard.publicnet.gateway.client.rpc", lambda *a, **k: response)
    checkpoint = gateway.checkpoint()
    assert checkpoint["state_height"] == 72 and checkpoint["round"] == 3
    assert checkpoint["model_root"] == work.digest(checkpoint["weights"])
    bad = copy.deepcopy(task)
    bad["weights"]["tiny"]["data"] = "AACAPw=="
    response["response"]["value"] = base64.b64encode(work.canonical(bad)).decode()
    gateway.cache.clear()
    with pytest.raises(ValueError, match="root"):
        gateway.checkpoint()
    gateway.history.close()
