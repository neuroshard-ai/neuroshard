"""Exactly-once funding of a user turn (DESIGN_ADMISSION.md §1).

A turn's job is the ledger ID of its signed ``serve_open`` envelope. Before broadcasting, the client writes a
durable operation record holding that exact envelope, and resolves the turn only from it and from
light-client-verified ledger state (see ``resolve``). CometBFT's transaction index and block RPCs are not
authenticated, so they never decide an outcome: a failure they report is a hint, made provable by consuming the
envelope's nonce. A client never signs a second ``serve_open`` for the turn.

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

PREPARED, BROADCAST, FUNDED, UNFUNDED, FAILED, SERVED, ABANDONED, UNKNOWN, LAPSED = (
    'prepared', 'broadcast', 'funded', 'unfunded', 'failed', 'served', 'abandoned', 'unknown', 'lapsed')
UNRESOLVED = (PREPARED, BROADCAST, UNKNOWN)
TERMINAL = (UNFUNDED, FAILED, SERVED, ABANDONED, LAPSED)


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

    Every terminal decision rests on the light-client-verified ledger state (``chain.verified_state()``), never on
    an RPC's unauthenticated transaction, block or execution answers:

    - the job is open or has a result in verified state: ``funded``;
    - the verified nonce has not passed the envelope's: it has not executed (yet), so the same envelope is
      broadcast again. An index that reports a failure is only a hint (``failure_hint``): it may lie, and a
      failed transaction can still be included later while its nonce is free;
    - the verified nonce has passed it, the job is absent, and no result recorded after ``since_height`` can have
      been evicted (results drop oldest first): it never executed, ``unfunded``;
    - the nonce has passed, the job is absent, but results after ``since_height`` may have been evicted: the job,
      if it ever existed, is closed (expired or settled) and its outcome is no longer provable, ``lapsed``.
    """
    if record['state'] not in UNRESOLVED:
        return record
    envelope, public = record['envelope'], record['envelope']['public_key']
    try:
        state = chain.verified_state()
        job_id = record['job_id']
        if job_id in state['jobs'] or job_id in state['results']:
            return operations.mark(record, FUNDED, height=state['height'], log='present in verified ledger state')
        nonce = state['accounts'].get(public, {'nonce': 0})['nonce']
        if nonce <= record['nonce']:
            hint = chain.failure_hint(record['tx_hash'])
            admitted = chain.rebroadcast(envelope)  # idempotent: the same signed bytes
            hint = hint or (admitted.get('log') if admitted.get('code') else None)
            return operations.mark(record, BROADCAST, failure_hint=hint,
                                   log=f'not executed as of verified height {state["height"]}'
                                       + (f'; reported failing: {hint}' if hint else ''))
        since = record.get('since_height')
        results = state['results']
        evicted = (since is None or len(results) >= state['params']['max_results']
                   and min(r['height'] for r in results.values()) > since)
        if not evicted:
            return operations.mark(record, UNFUNDED, height=state['height'],
                                   log=f'nonce {record["nonce"]} was consumed without opening this job '
                                       f'(verified at height {state["height"]})')
        return operations.mark(record, LAPSED, height=state['height'],
                               log='the nonce was consumed and the job is not open; whether it was ever funded '
                                   'can no longer be shown from verified state (any escrow was settled or refunded)')
    except (OSError, ValueError, KeyError) as error:  # an unavailable or unverified read is not evidence
        return operations.mark(record, UNKNOWN, log=f'resolution unavailable: {error}')


def fund(chain, operations, record, timeout=180, poll=0.5, sleep=time.sleep, clock=time.monotonic, supersede=None):
    """Broadcast a prepared operation and resolve it until funded, unfunded or lapsed, or the timeout passes.

    ``supersede(nonce)``, if given, signs and broadcasts a self-transfer at the envelope's nonce. It is used only
    when the transaction is reported failing while its nonce is still free: consuming the nonce makes the outcome
    provable from verified state (whichever of the two executes), instead of trusting the failure report.

    Raises ``Unresolved`` on timeout; the record stays durable and is resolved again before any new turn."""
    if record['state'] == PREPARED:
        since = record.get('since_height')
        record = operations.mark(record, BROADCAST, since_height=chain.height() if since is None else since)
    deadline = clock() + timeout
    while True:
        record = resolve(chain, operations, record)
        if record['state'] not in UNRESOLVED:
            return record
        if supersede is not None and record.get('failure_hint') and not record.get('superseded'):
            try:
                supersede(record['nonce'])
                record = operations.mark(record, record['state'], superseded=True)
            except (OSError, ValueError, KeyError) as error:
                record = operations.mark(record, record['state'], log=f'superseding failed: {error}')
        if clock() >= deadline:
            raise Unresolved(f'funding outcome unknown for job {record["job_id"][:16]}: {record.get("log")}')
        sleep(poll)


def recover(chain, operations, account, timeout=60, sleep=time.sleep, clock=time.monotonic, supersede=None):
    """Resolve every unresolved operation of ``account`` (after a restart); raises if any stays unknown."""
    outcomes = []
    for record in operations.unresolved(account):
        outcomes.append(fund(chain, operations, record, timeout=timeout, sleep=sleep, clock=clock,
                             supersede=supersede))
    return outcomes


class LedgerIndex:
    """``resolve``'s view of a live network through ``network.Chain`` (light-client verified state) and its RPCs."""

    def __init__(self, chain, rpc):
        self.chain, self.rpc = chain, rpc

    def height(self):
        return self.chain.state()['height']

    def verified_state(self):
        return self.chain.state()

    def account_nonce(self, public):
        return self.chain.state()['accounts'].get(public, {'nonce': 0})['nonce']

    def failure_hint(self, key):
        """The index's claim that ``key`` executed and failed, or None. Unauthenticated: a hint, never an outcome."""
        from neuroshard.inference import optimistic_network as chain_network
        try:
            result = self.rpc(self.chain.rpc, 'tx', {'hash': base64.b64encode(bytes.fromhex(key)).decode()})
            code = result['tx_result'].get('code', 0)
        except (chain_network.Rejected, OSError, ValueError, KeyError, TypeError):
            return None
        return (result['tx_result'].get('log') or f'code {code}') if code else None

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
