"""Exactly-once funding of a user turn (DESIGN_ADMISSION.md §1).

A turn's job is the ledger ID of its signed ``serve_open`` envelope. Before broadcasting, the client writes a
durable operation record holding that exact envelope, and resolves the turn only from it:

1. if the CometBFT index has ``tx_hash``, the turn is funded or failed with the indexed code;
2. if not, and the account nonce has not passed the envelope's, the same envelope is broadcast again;
3. if the nonce has passed it, ``tx_hash`` is queried again; the turn is unfunded only once a *different*
   transaction that consumed that nonce has been identified, and ``tx_hash`` is still absent;
4. anything else leaves it ``unknown``. A client never signs a second ``serve_open`` for the turn.

One funding operation is in flight per account; a restarted client resolves every unresolved operation before
starting a new turn.
"""
import base64
import contextlib
import hashlib
import json
import os
import secrets
import time
from pathlib import Path

from neuroshard.inference import optimistic as ledger

PREPARED, BROADCAST, FUNDED, UNFUNDED, FAILED, SERVED, ABANDONED, UNKNOWN = (
    'prepared', 'broadcast', 'funded', 'unfunded', 'failed', 'served', 'abandoned', 'unknown')
UNRESOLVED = (PREPARED, BROADCAST, UNKNOWN)
TERMINAL = (UNFUNDED, FAILED, SERVED, ABANDONED)


class Unresolved(RuntimeError):
    """A funding outcome is not known yet; the turn must not be funded again."""


def tx_hash(envelope):
    """CometBFT's index key: the SHA-256 of the canonical envelope (distinct from the ledger's job ID)."""
    return hashlib.sha256(ledger.canonical(envelope)).hexdigest().upper()


def write_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    with open(partial, 'w') as handle:
        handle.write(json.dumps(value, sort_keys=True))
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class Operations:
    """The durable operation records of one client home: ``<home>/operations/<turn_id>.json``."""

    def __init__(self, home):
        self.directory = Path(home) / 'operations'

    def path(self, turn_id):
        return self.directory / f'{turn_id}.json'

    @contextlib.contextmanager
    def exclusive(self, account):
        """One funding operation in flight per account: a second turn (any process) waits here."""
        import fcntl

        self.directory.mkdir(parents=True, exist_ok=True)
        with open(self.directory / f'{account[:32]}.lock', 'w') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def save(self, record):
        write_atomic(self.path(record['turn_id']), record)
        return record

    def load(self, turn_id):
        return json.loads(self.path(turn_id).read_text())

    def all(self):
        if not self.directory.exists():
            return []
        return [json.loads(path.read_text()) for path in sorted(self.directory.glob('*.json'))]

    def unresolved(self, account=None):
        return [r for r in self.all() if r['state'] in UNRESOLVED
                and (account is None or r['envelope']['public_key'] == account)]

    def prepare(self, envelope, session_key, request_root, owners, reservations=None):
        """Persist the exact signed envelope before it is broadcast; refuses while another turn is in flight."""
        public = envelope['public_key']
        if self.unresolved(public):
            raise Unresolved('another funding operation on this account is unresolved; resolve it first')
        record = {'turn_id': secrets.token_hex(16), 'envelope': envelope, 'nonce': envelope['body']['nonce'],
                  'job_id': ledger.transaction_id(envelope), 'tx_hash': tx_hash(envelope),
                  'request_root': request_root, 'session_key': session_key, 'owners': list(owners),
                  'reservations': dict(reservations or {}), 'state': PREPARED, 'since_height': None,
                  'height': None, 'log': None, 'updated': time.time()}
        return self.save(record)

    def mark(self, record, state, **fields):
        record = {**record, **fields, 'state': state, 'updated': time.time()}
        return self.save(record)


def resolve(chain, operations, record):
    """One resolution step from the persisted envelope; returns the updated record.

    ``chain`` provides ``lookup(tx_hash)`` (None, or {'height', 'code', 'log'}), ``account_nonce(public)``,
    ``rebroadcast(envelope)`` and ``nonce_consumer(public, nonce, since_height)``: the tx_hash of the committed
    transaction from that account at that nonce, or None if the index cannot yet show one.
    """
    if record['state'] not in UNRESOLVED:
        return record
    envelope, key = record['envelope'], record['tx_hash']
    try:
        found = chain.lookup(key)
        if found is not None:
            return settled(operations, record, found)
        nonce = chain.account_nonce(envelope['public_key'])
        if nonce <= record['nonce']:
            chain.rebroadcast(envelope)  # idempotent: the same signed bytes
            return operations.mark(record, BROADCAST)
        found = chain.lookup(key)  # it may have committed between the two reads
        if found is not None:
            return settled(operations, record, found)
        consumer = chain.nonce_consumer(envelope['public_key'], record['nonce'], record.get('since_height'))
        if consumer == key:  # the index lags, but the committed block shows this transaction executed
            return operations.mark(record, FUNDED, log='found in a committed block before the index')
        if consumer is not None:
            if chain.lookup(key) is None:
                return operations.mark(record, UNFUNDED, log=f'nonce {record["nonce"]} was consumed by {consumer}')
            return settled(operations, record, chain.lookup(key))
        return operations.mark(record, UNKNOWN, log='the nonce advanced, but its transaction is not identified yet')
    except (OSError, ValueError, KeyError) as error:  # an unavailable index is not evidence of anything
        return operations.mark(record, UNKNOWN, log=f'resolution unavailable: {error}')


def settled(operations, record, found):
    if found.get('code', 0):
        return operations.mark(record, FAILED, height=int(found['height']), log=found.get('log'))
    return operations.mark(record, FUNDED, height=int(found['height']))


def fund(chain, operations, record, timeout=180, poll=0.5, sleep=time.sleep, clock=time.monotonic):
    """Broadcast a prepared operation and resolve it until funded, failed, unfunded, or the timeout passes.

    Raises ``Unresolved`` on timeout; the record stays durable and is resolved again before any new turn."""
    if record['state'] == PREPARED:
        record = operations.mark(record, BROADCAST, since_height=chain.height())
        admitted = chain.rebroadcast(record['envelope'])
        if admitted.get('code'):
            # Mempool refusal (CheckTx) of a fresh nonce: the transaction can still not be included.
            record = operations.mark(record, UNKNOWN, log=admitted.get('log'))
    deadline = clock() + timeout
    while True:
        record = resolve(chain, operations, record)
        if record['state'] not in UNRESOLVED:
            return record
        if clock() >= deadline:
            raise Unresolved(f'funding outcome unknown for job {record["job_id"][:16]}: {record.get("log")}')
        sleep(poll)


def recover(chain, operations, account, timeout=60, sleep=time.sleep, clock=time.monotonic):
    """Resolve every unresolved operation of ``account`` (after a restart); raises if any stays unknown."""
    outcomes = []
    for record in operations.unresolved(account):
        outcomes.append(fund(chain, operations, record, timeout=timeout, sleep=sleep, clock=clock))
    return outcomes


class LedgerIndex:
    """``resolve``'s view of a live network through ``network.Chain`` and its CometBFT RPCs."""

    def __init__(self, chain, rpc):
        self.chain, self.rpc = chain, rpc

    def height(self):
        return self.chain.state()['height']

    def lookup(self, key):
        from neuroshard.inference import optimistic_network as chain_network
        try:
            result = self.rpc(self.chain.rpc, 'tx', {'hash': base64.b64encode(bytes.fromhex(key)).decode()})
        except chain_network.Rejected as error:
            if 'not found' in str(error):
                return None
            raise ValueError(str(error))
        return {'height': result['height'], 'code': result['tx_result'].get('code', 0),
                'log': result['tx_result'].get('log', '')}

    def account_nonce(self, public):
        return self.chain.state()['accounts'].get(public, {'nonce': 0})['nonce']

    def rebroadcast(self, envelope):
        from neuroshard.inference import optimistic_network as chain_network
        try:
            admitted = self.chain.broadcast(envelope)
        except chain_network.Rejected as error:
            if 'already exists in cache' in str(error):
                return {'code': 0, 'log': ''}  # the same signed bytes are already pending
            raise ValueError(str(error))
        if admitted['code'] and 'already exists in cache' in admitted.get('log', ''):
            return {'code': 0, 'log': ''}
        return admitted

    def nonce_consumer(self, public, nonce, since_height):
        """Scan committed blocks from ``since_height`` for the account's transaction at ``nonce``."""
        if since_height is None:
            return None
        top = self.chain.state()['height']
        for height in range(max(1, since_height), top + 1):
            block = self.rpc(self.chain.rpc, 'block', {'height': str(height)})
            for index, raw in enumerate(block['block']['data'].get('txs') or []):
                data = base64.b64decode(raw)
                try:
                    envelope = json.loads(data)
                except ValueError:
                    continue
                body = envelope.get('body', {}) if isinstance(envelope, dict) else {}
                if envelope.get('public_key') == public and body.get('nonce') == nonce:
                    # Only a transaction that executed consumed the nonce: a failed one leaves state unchanged.
                    results = self.rpc(self.chain.rpc, 'block_results', {'height': str(height)})
                    if (results.get('txs_results') or [])[index].get('code', 0) == 0:
                        return hashlib.sha256(data).hexdigest().upper()
        return None
