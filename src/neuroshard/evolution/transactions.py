"""Durable, single-account transaction outbox for candidate operators.

Unknown outcomes retain the exact signed bytes across process crashes. A
different operation cannot consume the pending nonce. Use one outbox and one
process per account; the database also rejects overlapping pending operations.
"""
import base64
import hashlib
import sqlite3
import time
from http.client import HTTPException

from neuroshard.dataflow.store import canonical
from neuroshard.demo import client as wire, protocol


class ClosedOperation(wire.Rejected):
    """The native ledger has permanently closed this operation's context."""


class Outbox:
    def __init__(self, path, url, chain_id, owner, *, rpc=wire.rpc, query=wire.query):
        self.url, self.chain_id, self.owner = url, chain_id, owner
        self.rpc, self.query = rpc, query
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS operations (id TEXT PRIMARY KEY, intent BLOB, envelope BLOB, receipt BLOB)')
        self.db.execute('CREATE TABLE IF NOT EXISTS retired (id TEXT PRIMARY KEY, evidence BLOB)')
        self.db.execute('CREATE TABLE IF NOT EXISTS identity (chain_id TEXT, owner TEXT)')
        saved = self.db.execute('SELECT chain_id, owner FROM identity').fetchone()
        expected = (chain_id, owner.public_key)
        if saved is not None and saved != expected:
            raise ValueError('Outbox belongs to a different chain or account')
        if saved is None:
            self.db.execute('INSERT INTO identity VALUES (?, ?)', expected)
            self.db.commit()

    def pending(self):
        row = self.db.execute('SELECT id FROM operations WHERE receipt IS NULL '
                              'AND id NOT IN (SELECT id FROM retired)').fetchone()
        return row[0] if row else None

    def retire_closed(self, state):
        """Recover a stale audit/vote only from trusted committed native state.

        A CheckTx error or a missing current claim alone is insufficient: the
        completed native history must identify this exact, non-reusable context.
        Retirement preserves the signed bytes and records an unknown transaction
        outcome, never a fabricated receipt, refund or successful payment.
        """
        operation = self.pending()
        if operation is None:
            return None
        if (state['chain_id'] != self.chain_id or state['height'] < 1
                or state['manifest'].get('auditing', {}).get('format') != 'neuroshard-native-quorum-audit-v1'):
            raise ValueError('Retirement requires this account network\'s committed native state')
        raw = self.db.execute('SELECT envelope FROM operations WHERE id=?', (operation,)).fetchone()[0]
        body, owner = protocol.verify(protocol.parse_json(raw))
        if owner != self.owner.public_key or body['chain_id'] != self.chain_id:
            raise ValueError('Outbox envelope belongs to another account or chain')
        kind, closed = body['kind'], None
        if kind in ('audit_commit', 'audit_verdict', 'audit_reveal'):
            closed = next((row for row in state['auditing']['history']
                           if row.get('claim_id') == body['claim_id']), None)
        elif kind == 'accept_audit':
            closed = next((row for row in state['auditing']['history'] if row['id'] == body['budget_id']), None)
        elif kind == 'vote_expert_job':
            closed = next((row for row in state.get('expert_lifecycle', {}).get('history', [])
                           if row.get('kind') == 'activation' and row['id'] == body['proposal_id']), None)
        if closed is None:
            return None
        evidence = {'operation': operation, 'transaction_sha256': hashlib.sha256(raw).hexdigest(),
                    'chain_id': self.chain_id, 'height': state['height'],
                    'state_root': hashlib.sha256(canonical(state)).hexdigest(), 'closed_context': closed,
                    'transaction_outcome': 'unknown; context permanently closed'}
        with self.db:
            self.db.execute('INSERT INTO retired VALUES (?,?)', (operation, canonical(evidence)))
        return evidence

    def logical_id(self, operation):
        row = self.db.execute('SELECT envelope FROM operations WHERE id=?', (operation,)).fetchone()
        if row is None:
            raise ValueError('Unknown outbox operation')
        return protocol.transaction_id(protocol.parse_json(row[0]))

    def send(self, operation, kind, *, timeout=60, **fields):
        intent = canonical({'kind': kind, **fields})
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            saved = self.db.execute('SELECT intent, envelope, receipt FROM operations WHERE id=?', (operation,)).fetchone()
            if saved:
                if saved[0] != intent:
                    raise ValueError('Outbox operation was reused with a different intent')
            else:
                if self.pending() is not None:
                    raise ValueError('Resolve the existing pending transaction before signing another')
                account = self.query(self.url, '/account', {'public_key': self.owner.public_key})
                envelope = self.owner.sign({'chain_id': self.chain_id, 'nonce': account['nonce'], 'kind': kind, **fields})
                raw = canonical(envelope)
                self.db.execute('INSERT INTO operations VALUES (?, ?, ?, NULL)', (operation, intent, raw))
        return self.confirm(operation, timeout=timeout)

    def confirm(self, operation, *, timeout=60):
        if self.db.execute('SELECT id FROM retired WHERE id=?', (operation,)).fetchone():
            raise ClosedOperation('Native context permanently closed; preserve the original signed operation '+operation)
        row = self.db.execute('SELECT envelope, receipt FROM operations WHERE id=?', (operation,)).fetchone()
        if not row:
            raise ValueError('Unknown outbox operation')
        # CometBFT indexes SHA256(signed envelope bytes). The application's
        # logical transaction_id deliberately excludes the ECDSA signature.
        # They are different identifiers and cannot be used interchangeably.
        txid = hashlib.sha256(row[0]).hexdigest().upper()
        if row[1] is not None:
            return self.result(protocol.parse_json(row[1]), txid)
        deadline = time.monotonic() + timeout
        submitted = False
        while time.monotonic() < deadline:
            try:
                receipt = self.rpc(self.url, 'tx', {'hash': base64.b64encode(bytes.fromhex(txid)).decode(), 'prove': False}, timeout=10)
            except wire.Rejected as exc:
                if 'not found' not in str(exc).lower():
                    raise
            except (OSError, HTTPException):
                pass
            else:
                if str(receipt.get('hash', '')).upper() != txid or int(receipt.get('height', 0)) < 1:
                    raise ValueError('RPC returned an unrelated or uncommitted transaction')
                with self.db:
                    self.db.execute('UPDATE operations SET receipt=? WHERE id=?', (canonical(receipt), operation))
                return self.result(receipt, txid)
            if not submitted:
                try:
                    result = self.rpc(self.url, 'broadcast_tx_sync', {'tx': base64.b64encode(row[0]).decode()}, timeout=30)
                except wire.Rejected as exc:
                    if 'already exists in cache' not in str(exc).lower():
                        raise
                except (OSError, HTTPException):
                    pass
                else:
                    # Even CheckTx rejection can follow a previously accepted
                    # submission whose lookup response was lost. Keep the nonce.
                    if result.get('code', 0):
                        raise wire.Rejected('Pending transaction '+txid+': '+result.get('log', 'CheckTx rejected'))
                submitted = True
            time.sleep(.2)
        raise TimeoutError('Outcome remains unknown; retain and query transaction '+txid)

    @staticmethod
    def result(receipt, txid):
        if receipt['tx_result'].get('code', 0):
            raise wire.Rejected('Finalized transaction '+txid+': '+receipt['tx_result'].get('log', 'rejected'))
        return receipt

    def close(self):
        self.db.close()
