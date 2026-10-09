"""ABCI application for the bonded optimistic serving ledger behind native CometBFT validators.

Every block first advances the ledger, settling jobs whose challenge window closed,
then applies its transactions. Mempool admission and proposals evaluate a transaction
against the next block's state, so only transactions that apply reach a block.
Admission checks a transaction's signature, schema, nonce, fee and target before any
proof replay.

Opening a challenge locks its deposit and replays nothing. After each commit, every
validator replays the proof of each open challenge once, in the background on one
dedicated thread outside the state lock, and caches its verdict; so only challenges
whose deposits are locked on chain are ever replayed, and a ``prove`` transaction is
usually judged from the cache.

Holding a challenged bundle is local to each validator and never a ledger outcome. A
validator that cannot judge a proof (it does not hold the bundle, or cannot replay it)
refuses the proof at admission with a retryable code, leaves it out of its own
proposals and rejects proposals containing it; it caches nothing, so it judges the
proof once the bundle arrives. A committed block was accepted by validators with more
than two thirds of the voting power, and an honest validator accepts a block only after
judging every proof in it, so execution applies committed proofs without replaying
them and needs no bundle.
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

    A single worker gives every replay the same thread and so the same parallel arithmetic,
    whether a proof is judged in the background or when asked. Only verdicts are kept.
    ``NoVerdict`` (the bundle is not held here, or this validator cannot run the replay)
    propagates and the proof is judged afresh when asked again; so does a failure of the
    checker itself, which says nothing about the proof.
    """

    def __init__(self, check):
        self.check, self.verdicts, self.queued = check, {}, set()
        self.lock, self.worker = threading.Lock(), ThreadPoolExecutor(max_workers=1)

    def replay(self, state, request):
        started = time.monotonic()
        key = request.get('proof_root', request.get('model_root'))
        try:
            verdict = self.check(state, request) is True
        except ledger.NoVerdict as reason:
            LOG.info('verification %s has no verdict on this validator: %s', key, reason)
            raise
        except Exception as error:
            LOG.exception('verification failed for %s', key)
            raise ledger.NoVerdict(f'the proof check failed on this validator ({type(error).__name__})') from error
        LOG.info('verification %s verdict %s in %.1f s', key, verdict, time.monotonic() - started)
        return verdict

    def known(self, request):
        """The cached verdict for ``request``, or None if this validator has not judged it."""
        with self.lock:
            return self.verdicts.get(ledger.digest(request))

    def judge(self, state, request, key):
        """On the worker: the cached verdict, or a replay's, which is then cached."""
        with self.lock:
            if key in self.verdicts:
                return self.verdicts[key]
        verdict = self.replay(state, request)
        with self.lock:
            self.verdicts[key] = verdict
        return verdict

    def __call__(self, state, request):
        key = ledger.digest(request)
        with self.lock:
            if key in self.verdicts:
                return self.verdicts[key]
        return self.worker.submit(self.judge, state, request, key).result()

    def prefetch(self, state, request):
        """Judge ``request`` in the background unless it is judged or queued; with no verdict, it is tried again later."""
        key = ledger.digest(request)
        with self.lock:
            if key in self.verdicts or key in self.queued:
                return
            self.queued.add(key)

        def run():
            try:
                self.judge(state, request, key)
            except ledger.NoVerdict:
                pass
            finally:
                with self.lock:
                    self.queued.discard(key)
        self.worker.submit(run)


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
        """The verdict execution applies to a proof in a committed block: it verified.

        Validators with more than two thirds of the voting power accepted the block, and an
        honest validator accepts a block only after judging every proof in it.
        """
        if self.check.known(request) is False:
            LOG.error('committed verification contradicts this validator: %s', ledger.digest(request))
        return True

    def prefetch(self):
        """Start judging the proof of every open challenge, so a verdict is ready before anyone submits it."""
        for job_id, job in sorted(self.state['jobs'].items()):
            for challenge_id in sorted(job['challenges']):
                request = ledger.challenge_request(self.state, job_id, job['challenges'][challenge_id])
                self.check.prefetch(self.state, request)

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
                # Only a proof of an open challenge, whose deposit is locked, reaches a replay; it is usually
                # judged already, and otherwise runs outside the state lock so consensus never waits behind it.
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
                self.prefetch()
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


def promotion_or_fraud(node, replay):
    """Stewarded testnet admission: signed by the genesis authority, with an approved quality digest."""
    def check(state, request):
        if request.get('kind') == 'model_promotion':
            approved = node.get('approved_models', {}).get(request['model_root'])
            return bool(approved and request['signer'] == node.get('promotion_authority')
                        and request['quality_root'] == approved['quality_root'])
        return replay(state, request)
    return check


def shard_checker(node):
    """The validator's real proof checker from its node configuration: shard holdings and the bundle store.

    A validator holding no shard judges no proof: every challenge gets no verdict from it.
    """
    if not node.get('shards'):
        def judge_nothing(state, request):
            raise ledger.NoVerdict('this validator holds no shard')
        return promotion_or_fraud(node, judge_nothing)
    if node.get('runtime') == 'granite-shard':
        # Replays must run in the owners' numerical environment, set before torch loads.
        from neuroshard.evolution import granite_shard_execution as shard

        shard.configure()
    import torch

    from neuroshard.evolution.sharded import granite, granite_audit, granite_serving
    from neuroshard.assistant.network import warm

    torch.set_num_threads(node['threads'])
    configured = threading.local()
    partitions, adapters, update_adapters = {}, {}, {}
    for rank, paths in node['shards'].items():
        partition, _ = granite.load_partition(granite.load_config(paths['config']), paths['shard'], int(rank))
        adapter = None
        if paths.get('arm'):
            update = granite_serving.Adapter(partition, paths['spec'], paths['arm'])
            update.set(False)
            update_adapters[int(rank)] = update
            adapter = (granite_serving.AdapterBank(partition, paths['spec'], paths['arm'], paths['additions'])
                       if paths.get('additions') else update)
        if node.get('warm_up_lengths') and adapter is None:
            partition.warm_up(lengths=tuple(node['warm_up_lengths']))
        else:
            warm(partition, adapter)
        partitions[int(rank)] = partition
        if adapter is not None:
            adapters[int(rank)] = adapter
    check = granite_audit.challenge_checker(node.get('bundles'), partitions, adapters)

    def replay(state, request):
        # Thread counts are per thread; the replay thread uses the same count as the loading thread.
        if not getattr(configured, 'threads', False):
            torch.set_num_threads(node['threads'])
            configured.threads = True
        routed = node.get('approved_models', {}).get(state['model_root'], {}).get('routed', True)
        checker = check if routed else granite_audit.challenge_checker(node.get('bundles'), partitions, update_adapters)
        return checker(state, request)
    return promotion_or_fraud(node, replay)


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
