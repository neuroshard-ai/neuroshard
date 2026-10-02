"""ABCI application for the bonded optimistic serving ledger behind native CometBFT validators.

Every block first advances the ledger, settling jobs whose challenge window closed,
then applies its transactions. Mempool admission and proposals evaluate a transaction
against the next block's state, so only transactions that apply reach a block.
Admission checks a transaction's signature, schema, nonce, fee and target before any
proof replay; a challenge's proof is then replayed once per validator, outside the
state lock, and its verdict cached.

Holding a challenged bundle is local to each validator and never a ledger outcome. A
validator that cannot judge a proof (it does not hold the bundle, or cannot replay it)
refuses the challenge at admission with a retryable code, leaves it out of its own
proposals and rejects proposals containing it; it caches nothing, so it judges the
proof once the bundle arrives. A committed block was accepted by validators with more
than two thirds of the voting power, and an honest validator accepts a block only after
judging every challenge in it, so execution applies committed challenges without
replaying them and needs no bundle.
"""

import argparse
import faulthandler
import json
import logging
import signal
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc

from neuroshard.demo import abci_pb2 as pb, abci_pb2_grpc as rpc
from neuroshard.inference import optimistic as ledger

INVALID = (ValueError, KeyError, TypeError, OverflowError, RecursionError)
LOG = logging.getLogger('neuroshard.settlement')
MAX_TX_BYTES = 16_384
MAX_BLOCK_TXS = 16
NO_VERDICT = 3


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


class cached:
    """One replay per proof on one dedicated thread: the verdict depends only on the request, which binds the proof.

    A single worker gives every replay the same thread and so the same parallel arithmetic.
    Only verdicts are kept. ``NoVerdict`` (the bundle is not held here, or this validator
    cannot run the replay) propagates and the proof is judged afresh when asked again; so
    does a failure of the checker itself, which says nothing about the proof.
    """

    def __init__(self, check):
        self.check, self.verdicts = check, {}
        self.lock, self.worker = threading.Lock(), ThreadPoolExecutor(max_workers=1)

    def replay(self, state, request):
        started = time.monotonic()
        try:
            verdict = self.check(state, request) is True
        except ledger.NoVerdict as reason:
            LOG.info('proof %s has no verdict on this validator: %s', request['proof_root'], reason)
            raise
        except Exception as error:
            LOG.exception('proof check failed for %s', request['proof_root'])
            raise ledger.NoVerdict(f'the proof check failed on this validator ({type(error).__name__})') from error
        LOG.info('proof %s verdict %s in %.1f s', request['proof_root'], verdict, time.monotonic() - started)
        return verdict

    def known(self, request):
        """The cached verdict for ``request``, or None if this validator has not judged it."""
        with self.lock:
            return self.verdicts.get(ledger.digest(request))

    def __call__(self, state, request):
        key = ledger.digest(request)
        with self.lock:
            if key in self.verdicts:
                return self.verdicts[key]
        verdict = self.worker.submit(self.replay, state, request).result()
        with self.lock:
            self.verdicts[key] = verdict
        return verdict


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

    def apply(self, state, raw, execute=None):
        return ledger.transition(state, parse_json(raw), execute or self.check)

    def block(self, height, txs, execute=None):
        """The state after advancing to ``height`` and applying ``txs``; raises on the first that does not apply."""
        if not self.state:
            raise ValueError('Chain has not initialized')
        state = ledger.advance(self.state, height)
        for raw in txs:
            state = self.apply(state, raw, execute)
        return state

    def committed(self, previous, request):
        """The verdict execution applies to a challenge in a committed block: it verified.

        Validators with more than two thirds of the voting power accepted the block, and an
        honest validator accepts a block only after judging every challenge in it.
        """
        if self.check.known(request) is False:
            LOG.error('committed challenge of %s contradicts this validator\'s replay of %s', request['log_key'],
                      request['proof_root'])
        return True

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

    def admit(self, raw):
        """Admission's cheap stage, under the state lock: the next block's state and a challenge's request, if any."""
        envelope = parse_json(raw)
        with self.lock:
            if not self.state:
                raise ValueError('Chain has not initialized')
            state = ledger.advance(self.state, self.state['height'] + 1)
            return state, ledger.admit(state, envelope)[2]

    def CheckTx(self, request, context):
        try:
            state, challenge = self.admit(request.tx)
            if challenge is not None:
                # Only an authenticated, funded challenge of a committed log in an open window reaches a replay,
                # which runs outside the state lock so consensus never waits behind it.
                self.check(state, challenge)
            with self.lock:
                self.block(self.state['height'] + 1, [request.tx])
            return pb.ResponseCheckTx(gas_wanted=1)
        except ledger.NoVerdict as reason:
            return pb.ResponseCheckTx(code=NO_VERDICT, log=f'No verdict on this validator yet: {reason}')
        except INVALID as exc:
            return pb.ResponseCheckTx(code=1, log=str(exc))
        except Exception as exc:
            LOG.exception('CheckTx failed unexpectedly')
            return pb.ResponseCheckTx(code=2, log=f'internal error: {type(exc).__name__}')

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
                except ledger.NoVerdict as reason:
                    LOG.info('PrepareProposal left out a challenge this validator cannot judge: %s', reason)
                except Exception:
                    LOG.exception('PrepareProposal skipped a transaction that failed unexpectedly')
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
            except ledger.NoVerdict as reason:
                # Never vote for a challenge this validator has not judged itself.
                LOG.warning('ProcessProposal rejected a block at height %s it cannot judge: %s', request.height, reason)
                return pb.ResponseProcessProposal(status=2)
            except Exception:
                LOG.exception('ProcessProposal rejected a block that failed unexpectedly')
                return pb.ResponseProcessProposal(status=2)

    def FinalizeBlock(self, request, context):
        with self.lock:
            if not self.state or request.height != self.state['height'] + 1:
                raise ValueError('Unexpected block height')
            state, results = ledger.advance(self.state, request.height), []
            for raw in request.txs:
                try:
                    state = self.apply(state, raw, self.committed)
                    results.append(pb.ExecTxResult())
                except INVALID as exc:
                    results.append(pb.ExecTxResult(code=1, log=str(exc)))
                except Exception as exc:
                    LOG.exception('FinalizeBlock recorded a transaction that failed unexpectedly')
                    results.append(pb.ExecTxResult(code=2, log=f'internal error: {type(exc).__name__}'))
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
    """The validator's real proof checker from its node configuration: shard holdings and the bundle store.

    A validator holding no shard judges no proof: every challenge gets no verdict from it.
    """
    if not node.get('shards'):
        def judge_nothing(state, request):
            raise ledger.NoVerdict('this validator holds no shard')
        return judge_nothing
    if node.get('runtime') == 'granite-shard':
        # Replays must run in the owners' numerical environment, set before torch loads.
        from neuroshard.evolution import granite_shard_execution as shard

        shard.configure()
    import torch

    from neuroshard.evolution.sharded import granite, granite_audit

    torch.set_num_threads(node['threads'])
    configured = threading.local()
    partitions = {}
    for rank, paths in node['shards'].items():
        partition, _ = granite.load_partition(granite.load_config(paths['config']), paths['shard'], int(rank))
        partition.warm_up(**({'lengths': tuple(node['warm_up_lengths'])} if node.get('warm_up_lengths') else {}))
        partitions[int(rank)] = partition
    check = granite_audit.challenge_checker(node.get('bundles'), partitions)

    def replay(state, request):
        # Thread counts are per thread; the replay thread uses the same count as the loading thread.
        if not getattr(configured, 'threads', False):
            torch.set_num_threads(node['threads'])
            configured.threads = True
        return check(state, request)
    return replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    faulthandler.enable()
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
