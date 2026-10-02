"""ABCI application for the bonded optimistic serving ledger behind native CometBFT validators.

Every block first advances the ledger, settling jobs whose challenge window closed,
then applies its transactions. Mempool admission and proposals evaluate a transaction
against the next block's state, so only transactions that apply reach a block. A
challenge's proof is replayed once per validator and its verdict cached.
"""

import argparse
import json
import signal
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc

from neuroshard.demo import abci_pb2 as pb, abci_pb2_grpc as rpc
from neuroshard.inference import optimistic as ledger

INVALID = (ValueError, KeyError, TypeError, OverflowError, RecursionError)
MAX_TX_BYTES = 16_384
MAX_BLOCK_TXS = 16


def parse_json(raw):
    """Strict JSON: no duplicate keys and no non-finite numbers."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result

    def bad_constant(_value):
        raise ValueError('Nonfinite JSON number')

    if len(raw) > MAX_TX_BYTES:
        raise ValueError('Transaction exceeds 16 KiB')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_constant)


def cached(check):
    """One replay per proof: the verdict depends only on the request, which binds the proof and commitment."""
    verdicts, lock = {}, threading.Lock()

    def verify(state, request):
        key = ledger.digest(request)
        with lock:
            if key in verdicts:
                return verdicts[key]
        try:
            verdict = check(state, request) is True
        except Exception:
            # A bundle that cannot be read or replayed proves nothing, on every validator alike.
            verdict = False
        with lock:
            verdicts[key] = verdict
        return verdict
    return verify


class Application(rpc.ABCIServicer):
    def __init__(self, database, check):
        self.check = cached(check)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(database), check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, value BLOB)')
        saved = self.db.execute('SELECT value FROM state WHERE id=1').fetchone()
        self.state = json.loads(saved[0]) if saved else None
        self.pending = None

    def persist(self, state):
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO state VALUES (1, ?)', (ledger.canonical(state),))

    def app_hash(self, state=None):
        value = self.state if state is None else state
        return bytes.fromhex(ledger.root(value)) if value else b''

    def apply(self, state, raw):
        return ledger.transition(state, parse_json(raw), self.check)

    def block(self, height, txs):
        """The state after advancing to ``height`` and applying ``txs``; raises on the first that does not apply."""
        if not self.state:
            raise ValueError('Chain has not initialized')
        state = ledger.advance(self.state, height)
        for raw in txs:
            state = self.apply(state, raw)
        return state

    def Echo(self, request, context):
        return pb.ResponseEcho(message=request.message)

    def Flush(self, request, context):
        return pb.ResponseFlush()

    def Info(self, request, context):
        with self.lock:
            return pb.ResponseInfo(data='NeuroShard bonded optimistic serving', version='1', app_version=1,
                                   last_block_height=self.state['height'] if self.state else 0,
                                   last_block_app_hash=self.app_hash())

    def InitChain(self, request, context):
        with self.lock:
            terms = parse_json(request.app_state_bytes)
            genesis = ledger.genesis(request.chain_id, terms['model_root'], terms['shards'], terms['allocations'],
                                     terms['params'])
            if request.initial_height not in (0, 1):
                raise ValueError('The settlement chain starts at height one')
            if self.state is None:
                self.state = genesis
                self.persist(self.state)
            elif self.state['chain_id'] != request.chain_id:
                raise ValueError('Saved chain differs from genesis')
            return pb.ResponseInitChain(app_hash=self.app_hash())

    def Query(self, request, context):
        with self.lock:
            try:
                if not self.state:
                    raise ValueError('Chain has not initialized')
                if request.prove or request.height not in (0, self.state['height']):
                    raise ValueError('Queries support current state without proofs only')
                if request.path == '/state':
                    value = {**self.state, 'root': ledger.root(self.state)}
                else:
                    raise ValueError('Unknown query path')
                return pb.ResponseQuery(value=ledger.canonical(value), height=self.state['height'])
            except INVALID as exc:
                return pb.ResponseQuery(code=1, log=str(exc))

    def CheckTx(self, request, context):
        with self.lock:
            try:
                self.block(self.state['height'] + 1 if self.state else 1, [request.tx])
                return pb.ResponseCheckTx(gas_wanted=1)
            except INVALID as exc:
                return pb.ResponseCheckTx(code=1, log=str(exc))

    def PrepareProposal(self, request, context):
        with self.lock:
            chosen = []
            for raw in request.txs:
                if len(chosen) >= MAX_BLOCK_TXS or len(raw) > MAX_TX_BYTES:
                    continue
                try:
                    self.block(request.height, chosen + [raw])
                    chosen.append(raw)
                except INVALID:
                    continue
            return pb.ResponsePrepareProposal(txs=chosen)

    def ProcessProposal(self, request, context):
        with self.lock:
            if len(request.txs) > MAX_BLOCK_TXS:
                return pb.ResponseProcessProposal(status=2)
            try:
                self.block(request.height, list(request.txs))
                return pb.ResponseProcessProposal(status=1)
            except INVALID:
                return pb.ResponseProcessProposal(status=2)

    def FinalizeBlock(self, request, context):
        with self.lock:
            if not self.state or request.height != self.state['height'] + 1:
                raise ValueError('Unexpected block height')
            state, results = ledger.advance(self.state, request.height), []
            for raw in request.txs:
                try:
                    state = self.apply(state, raw)
                    results.append(pb.ExecTxResult())
                except INVALID as exc:
                    results.append(pb.ExecTxResult(code=1, log=str(exc)))
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


def shard_checker(node):
    """The validator's real proof checker from its node configuration: shard holdings and the bundle store."""
    if not node.get('shards'):
        return lambda state, request: False
    import torch

    from neuroshard.evolution.sharded import granite, granite_audit

    torch.set_num_threads(node['threads'])
    partitions = {}
    for rank, paths in node['shards'].items():
        partition, _ = granite.load_partition(granite.load_config(paths['config']), paths['shard'], int(rank))
        partition.warm_up(**({'lengths': tuple(node['warm_up_lengths'])} if node.get('warm_up_lengths') else {}))
        partitions[int(rank)] = partition
    return granite_audit.challenge_checker(node['bundles'], partitions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    node = json.loads((args.home / 'settlement.json').read_text())
    app = Application(args.home / 'settlement.sqlite', shard_checker(node))
    server = grpc.server(ThreadPoolExecutor(max_workers=8))
    rpc.add_ABCIServicer_to_server(app, server)
    if not server.add_insecure_port(f'127.0.0.1:{args.port}'):
        raise RuntimeError('Cannot bind ABCI port')
    signal.signal(signal.SIGTERM, lambda *_: server.stop(2))
    signal.signal(signal.SIGINT, lambda *_: server.stop(2))
    server.start()
    print(f'NeuroShard settlement ABCI listening on {args.port}', flush=True)
    server.wait_for_termination()
    app.db.close()


if __name__ == '__main__':
    main()
