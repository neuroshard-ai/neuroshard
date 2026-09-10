"""Candidate ABCI application with delayed, bonded validator-set changes."""

import argparse
import hashlib
import json
import logging
import os
import signal
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import torch

from neuroshard.demo import abci_pb2 as pb, protocol, work
from neuroshard.demo.app import Application as ReferenceApplication, INVALID
from neuroshard.lab import abci_pb2 as lab_pb, state


PROFILES = {"lab": {}, "testnet": {"epoch_blocks": 60, "activation_blocks": 60,
    "evidence_blocks": 172800, "evidence_seconds": 172800, "lease_blocks": 120,
    "max_training_tasks": 100000}}


def native_parameters(params=None):
    params = state.PARAMS if params is None else params
    return {"block_max_bytes": 65_536, "block_max_gas": -1,
            "evidence_blocks": params["evidence_blocks"],
            "evidence_duration_ns": params["evidence_seconds"] * 1_000_000_000,
            "evidence_max_bytes": 16_384, "validator_key_types": ["ed25519"],
            "vote_extensions_enable_height": 0}


def check_native_parameters(params, expected=None):
    actual = {"block_max_bytes": params.block.max_bytes, "block_max_gas": params.block.max_gas,
              "evidence_blocks": params.evidence.max_age_num_blocks,
              "evidence_duration_ns": params.evidence.max_age_duration.seconds * 1_000_000_000
                                      + params.evidence.max_age_duration.nanos,
              "evidence_max_bytes": params.evidence.max_bytes,
              "validator_key_types": list(params.validator.pub_key_types),
              "vote_extensions_enable_height": params.abci.vote_extensions_enable_height}
    if actual != (native_parameters() if expected is None else expected):
        raise ValueError("Native consensus parameters differ from the candidate's evidence/execution bounds")


def execution_manifest(data, profile="lab"):
    from neuroshard.core.model import llm
    from neuroshard.core.crypto import ecdsa
    sources = [Path(llm.__file__), Path(ecdsa.__file__), Path(work.__file__), Path(protocol.__file__),
               *sorted(Path(__file__).parent.glob("*.py"))]
    weights = work.encode_weights(work.make_model())
    vectors = []
    for step in range(3):
        result = work.replay(weights, data, step, "neuroshard/conformance/v2")
        weights = result["weights"]
        vectors.append({"model_root": work.digest(weights), "gradient_root": work.digest(result["gradients"]),
                        "loss_hex": float(result["loss"]).hex()})
    execution = work.manifest(data)
    # Reuse only the v1 numerical definition, not its development economic parameters.
    for key in ("reward_atoms", "max_rewarded_tasks", "lease_blocks"):
        execution.pop(key)
    params = {**state.PARAMS, **PROFILES[profile]}
    return {"version": "neuroshard-candidate-v2", "profile": profile, "execution": execution, "params": params,
            "native_consensus": native_parameters(params),
            "numerical_conformance": {"aten_dispatch": torch.backends.cpu.get_cpu_capability(),
                "ATEN_CPU_CAPABILITY": os.environ.get("ATEN_CPU_CAPABILITY", "native"),
                "MKL_ENABLE_INSTRUCTIONS": os.environ.get("MKL_ENABLE_INSTRUCTIONS", "native"),
                "three_step_vectors": vectors},
            "candidate_source_hash": work.digest({str(p.resolve().relative_to(Path(__file__).resolve().parents[1])):
                        hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})}


def metadata(request):
    evidence = [{"kind": item.type, "address": item.validator.address.hex().upper(), "height": item.height}
                for item in request.misbehavior]
    for field in ("decided_last_commit", "proposed_last_commit", "local_last_commit"):
        if hasattr(request, field):
            commit = getattr(request, field)
            break
    committers = {vote.validator.address.hex().upper() for vote in commit.votes if vote.block_id_flag == 2}
    return request.height, request.time.seconds * 1_000_000_000 + request.time.nanos, evidence, committers


class Application(ReferenceApplication):
    def __init__(self, database, data_path, profile="lab"):
        work.configure_cpu()
        self.data = work.read_data(data_path)
        self.spec = execution_manifest(self.data, profile)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(database), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, value BLOB)")
        saved = self.db.execute("SELECT value FROM state WHERE id=1").fetchone()
        self.state = json.loads(saved[0]) if saved else None
        if self.state and self.state["manifest"] != self.spec:
            raise ValueError("Candidate manifest differs from saved state")
        self.pending = self.replay_cache = self.validation_cache = None

    def InitChain(self, request, context):
        with self.lock:
            genesis = protocol.parse_json(request.app_state_bytes)
            if genesis["manifest"] != self.spec or request.initial_height not in (0, 1):
                raise ValueError("Incompatible candidate genesis")
            check_native_parameters(request.consensus_params, self.spec["native_consensus"])
            expected = {v["consensus_key"]: v["bond"] // state.PARAMS["bond_unit"] for v in genesis["validators"]}
            supplied = {v.pub_key.ed25519.hex(): v.power for v in request.validators}
            if supplied != expected or len(request.validators) != len(expected):
                raise ValueError("Native genesis voting power differs from bonded application state")
            if self.state is None:
                self.state = state.genesis(request.chain_id, genesis["validators"], self.spec)
                self.persist(self.state)
            elif self.state["chain_id"] != request.chain_id:
                raise ValueError("Wrong chain")
            return pb.ResponseInitChain(app_hash=self.app_hash())

    def replay(self, candidate):
        lease = candidate["lease"]
        key = (candidate["model_root"], candidate["round"], lease["task_id"])
        if self.replay_cache is None or self.replay_cache[0] != key:
            self.replay_cache = (key, state.execute(candidate, self.data))
        return self.replay_cache[1]

    def apply_to(self, candidate, raw):
        if len(raw) > state.PARAMS["max_tx_bytes"]:
            raise ValueError("Oversized transaction")
        return state.transition(candidate, protocol.parse_json(raw), self.replay)

    def CheckTx(self, request, context):
        with self.lock:
            try:
                projected, _ = state.advance(self.state, self.state["height"] + 1, self.state["time_ns"])
                self.apply_to(projected, request.tx)
                return pb.ResponseCheckTx(gas_wanted=1)
            except INVALID as exc:
                return pb.ResponseCheckTx(code=1, log=str(exc))

    def PrepareProposal(self, request, context):
        with self.lock:
            projected, _ = state.advance(self.state, *metadata(request))
            for raw in request.txs:
                if len(raw) > request.max_tx_bytes:
                    continue
                try:
                    self.apply_to(projected, raw)
                    return pb.ResponsePrepareProposal(txs=[raw])
                except INVALID:
                    pass
            return pb.ResponsePrepareProposal()

    def ProcessProposal(self, request, context):
        with self.lock:
            try:
                if len(request.txs) > 1:
                    raise ValueError("At most one transaction per candidate block")
                projected, _ = state.advance(self.state, *metadata(request))
                for raw in request.txs:
                    self.apply_to(projected, raw)
                return pb.ResponseProcessProposal(status=1)
            except INVALID:
                return pb.ResponseProcessProposal(status=2)

    def FinalizeBlock(self, request, context):
        with self.lock:
            if len(request.txs) > 1:
                raise ValueError("Too many transactions")
            candidate, updates = state.advance(self.state, *metadata(request))
            results = []
            for raw in request.txs:
                try:
                    candidate = self.apply_to(candidate, raw)
                    results.append(lab_pb.ExecTxResult())
                except INVALID as exc:
                    results.append(lab_pb.ExecTxResult(code=1, log=str(exc)))
            self.pending = candidate
            return lab_pb.ResponseFinalizeBlock(tx_results=results,
                validator_updates=[lab_pb.ValidatorUpdate(pub_key=lab_pb.PublicKey(ed25519=bytes.fromhex(key)), power=power)
                                   for key, power in sorted(updates.items())], app_hash=self.app_hash(candidate))

    def Query(self, request, context):
        with self.lock:
            try:
                if self.state is None or request.prove or request.height not in (0, self.state["height"]):
                    raise ValueError("Only current queries without proofs are supported")
                s = self.state
                if request.path == "/status":
                    if self.validation_cache is None or self.validation_cache[0] != s["model_root"]:
                        self.validation_cache = (s["model_root"], work.evaluate(s["weights"], self.data))
                    value = {key: s[key] for key in ("chain_id", "height", "time_ns", "round", "model_root", "accounts",
                             "validators", "lease", "issued", "burned", "initial_supply", "last_inference")}
                    value.update(validation_loss=self.validation_cache[1], app_hash=self.app_hash(),
                                 powers=state.voting_power(s, s["height"]), params=state.parameters(s))
                    value["app_hash"] = value["app_hash"].hex()
                elif request.path == "/task":
                    value = {key: s[key] for key in ("chain_id", "round", "model_root", "weights", "lease")}
                    value["input_ids"] = work.batch(self.data, s["round"]).tolist()
                elif request.path == "/manifest":
                    value = self.spec
                elif request.path == "/summary":
                    value = {key: s[key] for key in ("chain_id", "height", "time_ns", "round", "model_root",
                        "issued", "burned", "initial_supply", "last_training_loss_hex")}
                    powers = state.voting_power(s, s["height"])
                    value.update(validator_count=len(powers), total_voting_power=sum(powers.values()),
                        account_count=len(s["accounts"]), params=state.parameters(s), profile=self.spec["profile"],
                        manifest_hash=work.digest(self.spec), app_hash=self.app_hash().hex(), lease=s["lease"])
                elif request.path == "/account":
                    owner = state.public_key(protocol.parse_json(request.data)["public_key"])
                    value = {"public_key": owner, **s["accounts"].get(owner, {"balance": 0, "nonce": 0})}
                elif request.path == "/validators":
                    options = protocol.parse_json(request.data) if request.data else {}
                    limit = state.integer(options.get("limit", 100), 1, 100)
                    after = options.get("after", "")
                    if not isinstance(after, str) or len(after) not in (0, 64):
                        raise ValueError("Invalid validator cursor")
                    powers = sorted(state.voting_power(s, s["height"]).items())
                    page = [(key, power) for key, power in powers if key > after][:limit]
                    value = {"validators": [{"public_key": key, "power": power, "owner": s["validators"][key]["owner"],
                              "bond": s["validators"][key]["amount"]}
                             for key, power in page], "total": len(powers),
                             "next_after": page[-1][0] if len(page) == limit else None}
                else:
                    raise ValueError("Unknown query")
                return pb.ResponseQuery(value=work.canonical(value), height=s["height"])
            except INVALID as exc:
                return pb.ResponseQuery(code=1, log=str(exc))


def register(app, server):
    """ABCI service path is upstream's; message packages do not affect the wire."""
    names = ["Echo", "Flush", "Info", "InitChain", "Query", "CheckTx", "Commit", "ListSnapshots", "OfferSnapshot",
             "LoadSnapshotChunk", "ApplySnapshotChunk", "PrepareProposal", "ProcessProposal", "ExtendVote", "VerifyVoteExtension", "FinalizeBlock"]
    handlers = {}
    for name in names:
        module = lab_pb if name in ("InitChain", "PrepareProposal", "ProcessProposal", "FinalizeBlock") else pb
        handlers[name] = grpc.unary_unary_rpc_method_handler(getattr(app, name),
            request_deserializer=getattr(module, "Request" + name).FromString,
            response_serializer=lambda message: message.SerializeToString())
    server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler("tendermint.abci.ABCI", handlers),))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="lab")
    args = parser.parse_args()
    app = Application(args.home / "candidate.sqlite", args.data, args.profile)
    server = grpc.server(ThreadPoolExecutor(max_workers=8), options=[("grpc.so_reuseport", 0),
        ("grpc.max_send_message_length", work.MAX_MESSAGE_BYTES), ("grpc.max_receive_message_length", work.MAX_MESSAGE_BYTES)])
    register(app, server)
    server.add_insecure_port(f"127.0.0.1:{args.port}")
    signal.signal(signal.SIGTERM, lambda *_: server.stop(1))
    server.start()
    print(f"Candidate ABCI on {args.port}", flush=True)
    server.wait_for_termination()
    app.db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
