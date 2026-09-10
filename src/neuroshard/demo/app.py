"""Durable training/reward state machine behind a native CometBFT validator."""

import argparse
import copy
import json
import logging
import signal
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc

from neuroshard.demo import abci_pb2 as pb, abci_pb2_grpc as rpc, protocol, work


INVALID = (ValueError, KeyError, TypeError, OverflowError, RecursionError)


class Application(rpc.ABCIServicer):
    def __init__(self, database, data_path):
        work.configure_cpu()
        self.data = work.read_data(data_path)
        self.spec = work.manifest(self.data)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(database), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, value BLOB)")
        saved = self.db.execute("SELECT value FROM state WHERE id=1").fetchone()
        self.state = protocol.parse_json(saved[0]) if saved else None
        if self.state and self.state["manifest"] != self.spec:
            raise ValueError("Saved state uses a different training manifest")
        self.pending = None
        self.replay_cache = None
        self.validation_cache = None

    def persist(self, state):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO state VALUES (1, ?)", (work.canonical(state),))

    def app_hash(self, state=None):
        value = self.state if state is None else state
        return bytes.fromhex(work.digest(value)) if value else b""

    def replay(self, state):
        key = (state["model_root"], state["round"], state["lease"]["task_id"])
        if self.replay_cache is None or self.replay_cache[0] != key:
            result = work.replay(state["weights"], self.data, state["round"], key[2])
            self.replay_cache = (key, result)
        return self.replay_cache[1]

    def apply(self, raw, height):
        if not self.state:
            raise ValueError("Chain has not initialized")
        # Training transactions contain commitments and signatures, never weights.
        if len(raw) > 16384:
            raise ValueError("Transaction exceeds 16 KiB")
        return protocol.transition(self.state, protocol.parse_json(raw), height, self.replay)

    def Echo(self, request, context):
        return pb.ResponseEcho(message=request.message)

    def Flush(self, request, context):
        return pb.ResponseFlush()

    def Info(self, request, context):
        with self.lock:
            return pb.ResponseInfo(data="NeuroShard verified-work reference", version="1",
                                   app_version=1, last_block_height=self.state["height"] if self.state else 0,
                                   last_block_app_hash=self.app_hash())

    def InitChain(self, request, context):
        with self.lock:
            if protocol.parse_json(request.app_state_bytes) != self.spec:
                raise ValueError("Genesis and local training manifests differ")
            if request.initial_height not in (0, 1):
                raise ValueError("The reference chain starts at height one")
            if self.state is None:
                self.state = protocol.genesis_state(request.chain_id, self.spec)
                self.persist(self.state)
            elif self.state["chain_id"] != request.chain_id:
                raise ValueError("Saved chain ID differs from genesis")
            return pb.ResponseInitChain(app_hash=self.app_hash())

    def Query(self, request, context):
        with self.lock:
            try:
                if not self.state:
                    raise ValueError("Chain has not initialized")
                if request.prove or request.height not in (0, self.state["height"]):
                    raise ValueError("Reference queries support current state without proofs only")
                state = self.state
                if request.path == "/status":
                    if self.validation_cache is None or self.validation_cache[0] != state["model_root"]:
                        self.validation_cache = (state["model_root"], work.evaluate(state["weights"], self.data))
                    value = {key: state[key] for key in (
                        "chain_id", "height", "round", "model_root", "lease", "balances",
                        "total_issued", "manifest", "training_loss_hex")}
                    value["validation_loss"] = self.validation_cache[1]
                    value["app_hash"] = self.app_hash().hex()
                elif request.path == "/task":
                    value = {key: state[key] for key in (
                        "chain_id", "height", "round", "model_root", "weights", "lease")}
                    value["input_ids"] = work.batch(self.data, state["round"]).tolist()
                elif request.path == "/infer":
                    args = protocol.parse_json(request.data)
                    value = work.infer(state["weights"], args["prompt"], args.get("max_tokens", 24))
                    value["round"] = state["round"]
                else:
                    raise ValueError("Unknown query path")
                return pb.ResponseQuery(value=work.canonical(value), height=state["height"])
            except INVALID as exc:
                return pb.ResponseQuery(code=1, log=str(exc))

    def CheckTx(self, request, context):
        with self.lock:
            try:
                self.apply(request.tx, self.state["height"] + 1 if self.state else 1)
                return pb.ResponseCheckTx(gas_wanted=1)
            except INVALID as exc:
                return pb.ResponseCheckTx(code=1, log=str(exc))

    def PrepareProposal(self, request, context):
        with self.lock:
            for raw in request.txs:
                if len(raw) > request.max_tx_bytes:
                    continue
                try:
                    self.apply(raw, request.height)
                    return pb.ResponsePrepareProposal(txs=[raw])
                except INVALID:
                    continue
            return pb.ResponsePrepareProposal()

    def ProcessProposal(self, request, context):
        with self.lock:
            if len(request.txs) > 1:
                return pb.ResponseProcessProposal(status=2)
            try:
                for raw in request.txs:
                    self.apply(raw, request.height)
                return pb.ResponseProcessProposal(status=1)
            except INVALID:
                return pb.ResponseProcessProposal(status=2)

    def FinalizeBlock(self, request, context):
        with self.lock:
            if request.height != self.state["height"] + 1 or len(request.txs) > 1:
                raise ValueError("Unexpected block height or transaction count")
            state, results = copy.deepcopy(self.state), []
            for raw in request.txs:
                try:
                    state = self.apply(raw, request.height)
                    results.append(pb.ExecTxResult())
                except INVALID as exc:
                    results.append(pb.ExecTxResult(code=1, log=str(exc)))
            state["height"] = request.height
            self.pending = state
            return pb.ResponseFinalizeBlock(tx_results=results, app_hash=self.app_hash(state))

    def Commit(self, request, context):
        with self.lock:
            if self.pending is not None:
                self.persist(self.pending)
                self.state, self.pending = self.pending, None
            return pb.ResponseCommit()

    def ListSnapshots(self, request, context):
        return pb.ResponseListSnapshots()

    def OfferSnapshot(self, request, context):
        return pb.ResponseOfferSnapshot(result=3)

    def LoadSnapshotChunk(self, request, context):
        return pb.ResponseLoadSnapshotChunk()

    def ApplySnapshotChunk(self, request, context):
        return pb.ResponseApplySnapshotChunk(result=2)

    def ExtendVote(self, request, context):
        return pb.ResponseExtendVote()

    def VerifyVoteExtension(self, request, context):
        return pb.ResponseVerifyVoteExtension(status=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    args.home.mkdir(parents=True, exist_ok=True)
    app = Application(args.home / "application.sqlite", args.data)
    server = grpc.server(ThreadPoolExecutor(max_workers=8), options=[
        ("grpc.max_receive_message_length", work.MAX_MESSAGE_BYTES),
        ("grpc.max_send_message_length", work.MAX_MESSAGE_BYTES)])
    rpc.add_ABCIServicer_to_server(app, server)
    if not server.add_insecure_port(f"127.0.0.1:{args.port}"):
        raise RuntimeError("Cannot bind ABCI port")
    signal.signal(signal.SIGTERM, lambda *_: server.stop(2))
    signal.signal(signal.SIGINT, lambda *_: server.stop(2))
    server.start()
    print(f"NeuroShard ABCI listening on {args.port}", flush=True)
    server.wait_for_termination()
    app.db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
