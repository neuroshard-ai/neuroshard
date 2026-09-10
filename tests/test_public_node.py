"""Release boundaries: genesis trust, account access, and restricted public RPC."""

import base64
import hashlib
import json
from pathlib import Path

import pytest

from neuroshard.demo import protocol, work
from neuroshard.lab import state
from neuroshard.lab.app import Application, execution_manifest
from neuroshard.demo import abci_pb2 as pb
from neuroshard.publicnet import bootstrap
from neuroshard.publicnet.gateway import validate_rpc, parse_block_time


DATA = Path(__file__).resolve().parents[1] / "docs/eval/data/input.txt"


@pytest.mark.parametrize("fraction, micros", [("", 0), (".1", 100000), (".123456789", 123456)])
def test_native_nanosecond_timestamps_work_on_python_310(fraction, micros):
    parsed = parse_block_time(f"2026-09-10T20:55:47{fraction}Z")
    assert parsed.microsecond == micros and parsed.utcoffset().total_seconds() == 0


@pytest.fixture(autouse=True)
def cpu():
    work.configure_cpu()


def test_genesis_download_requires_exact_published_bytes(tmp_path):
    path = tmp_path / "genesis.json"
    raw = b'{"chain_id":"candidate"}'
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert bootstrap.load_genesis(str(path), digest)[0]["chain_id"] == "candidate"
    path.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="checksum"):
        bootstrap.load_genesis(str(path), digest)
    with pytest.raises(ValueError, match="HTTPS"):
        bootstrap.load_genesis("http://example.com/genesis.json", digest)


@pytest.mark.parametrize("method", ["unsafe_flush_mempool", "dial_peers", "net_info", "dump_consensus_state", "abci_info"])
def test_public_gateway_does_not_relay_administrative_or_unbounded_rpc(method):
    with pytest.raises(ValueError, match="not public"):
        validate_rpc({"method": method})


@pytest.mark.parametrize("params", [{"path": "/status"}, {"path": "/summary", "prove": True},
    {"path": "/summary", "height": 10}, {"path": "/summary", "unexpected": 1}])
def test_unsupported_state_or_proof_queries_are_rejected(params):
    with pytest.raises(ValueError):
        validate_rpc({"method": "abci_query", "params": params})


def test_signed_transaction_relay_checks_size_and_signature():
    identity = protocol.Identity("public-gateway")
    body = {"kind": "transfer", "chain_id": "test", "nonce": 0, "to": identity.public_key, "amount": 1}
    envelope = identity.sign(body)
    encoded = base64.b64encode(work.canonical(envelope)).decode()
    assert validate_rpc({"method": "broadcast_tx_sync", "params": {"tx": encoded}})[0] == "broadcast_tx_sync"
    envelope["body"]["amount"] = 2
    with pytest.raises(ValueError, match="Signature"):
        validate_rpc({"method": "broadcast_tx_sync", "params": {"tx": base64.b64encode(work.canonical(envelope)).decode()}})
    with pytest.raises(ValueError, match="16 KiB"):
        validate_rpc({"method": "broadcast_tx_sync", "params": {"tx": base64.b64encode(b"a" * 16385).decode()}})


def test_public_profile_uses_its_committed_parameters_without_changing_lab_defaults():
    manifest = execution_manifest(work.read_data(DATA), "testnet")
    assert manifest["params"]["evidence_seconds"] == 172800
    assert manifest["native_consensus"]["evidence_duration_ns"] == 172800 * 10 ** 9
    assert state.PARAMS["evidence_seconds"] == 6
    owner = protocol.Identity("profile-owner")
    # Public-key parsing is independent of possession checks, which happen at bond/genesis declaration.
    key = "01" * 32
    s = state.genesis("profile", [{"owner": owner.public_key, "consensus_key": key,
                                  "bond": 2500000, "liquid": 20000000}], manifest)
    projected, _ = state.advance(s, 1, 1)
    tx = owner.sign({"kind": "reserve", "chain_id": "profile", "nonce": 0, "task_kind": "train",
        "parent": s["model_root"], "round": 0, "workers": [owner.public_key, owner.public_key], "price": 0, "request": {}})
    reserved = state.transition(projected, tx, lambda _: None)
    assert reserved["lease"]["expires"] == 121


def test_summary_and_account_do_not_export_all_balances(tmp_path):
    app = Application(tmp_path / "app.sqlite", DATA)
    identity = protocol.Identity("summary-owner")
    app.state = state.genesis("summary", [{"owner": identity.public_key, "consensus_key": "01" * 32,
                                          "bond": 2500000, "liquid": 20000000}], app.spec)
    summary = app.Query(pb.RequestQuery(path="/summary"), None)
    value = json.loads(summary.value)
    assert summary.code == 0 and "accounts" not in value and "weights" not in value
    account = app.Query(pb.RequestQuery(path="/account", data=work.canonical({"public_key": identity.public_key})), None)
    assert json.loads(account.value)["balance"] == 20000000
    missing = app.Query(pb.RequestQuery(path="/account", data=work.canonical({"public_key": protocol.Identity("new").public_key})), None)
    assert json.loads(missing.value)["balance"] == 0
    assert len(app.state["accounts"]) == 1
    app.db.close()


@pytest.mark.parametrize("value", ["x@host:26656", "a" * 40 + "@host:0", "a" * 40 + "@127.0.0.1:26656"])
def test_public_peers_require_valid_routable_addresses(value):
    with pytest.raises(ValueError):
        bootstrap.peer(value)


def test_private_peers_require_an_explicit_local_network_profile():
    value = "a" * 40 + "@127.0.0.1:26656"
    assert bootstrap.peer(value, private=True) == value
