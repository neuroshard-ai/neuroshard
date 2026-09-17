"""Atomic read of an operator-owned full node, including committed WAL data.

This is local replay trust, not an inclusion proof from a remote RPC service.
The pin is the SHA-256 of the genesis file bytes, not its canonical JSON hash.
"""
import json
from pathlib import Path
import sqlite3

from .objects import digest
from .schema import integer, root

MAX_STATE_BYTES = 128 * 1024**2


def read(home, genesis_sha256):
    home = Path(home).resolve()
    root(genesis_sha256)
    path = home/'config/genesis.json'
    if path.stat().st_size > 2*1024**2:
        raise ValueError('Native genesis exceeds the local review bound')
    raw = path.read_bytes()
    if digest(raw) != genesis_sha256:
        raise ValueError('Local genesis differs from the pinned network')
    genesis = json.loads(raw)
    db = sqlite3.connect((home/'evolution.sqlite').as_uri()+'?mode=ro', uri=True, timeout=5)
    try:
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        size = db.execute('SELECT length(value) FROM state WHERE id=1').fetchone()
        if size is None or size[0] is None or not 1 <= size[0] <= MAX_STATE_BYTES:
            raise ValueError('Missing or oversized committed native state')
        raw = db.execute('SELECT value FROM state WHERE id=1').fetchone()[0]
    finally:
        db.close()
    state = json.loads(raw)
    if state['chain_id'] != genesis['chain_id'] or state['manifest'] != genesis['app_state']['manifest']:
        raise ValueError('Committed state differs from the pinned genesis')
    integer(state['height'], 1, 2**63-1)
    root(state['data_root'])
    return state
