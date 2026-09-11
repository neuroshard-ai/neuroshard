"""Read admission history from an operator's own committed native full-node state.

The genesis pin identifies the intended network. This is a local, trusted-node
read, not a state proof supplied by an arbitrary remote server. Never open the
database immutable: a running application's committed state may be in its WAL.
"""
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from neuroshard.dataflow.store import canonical
from neuroshard.evolution.objects import digest
from neuroshard.evolution.schema import integer, root


MAX_STATE_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class Snapshot:
    home: Path
    genesis_sha256: str
    state: dict

    @property
    def anchor(self):
        life = self.state['lifecycle']
        return digest(canonical({
            'genesis_sha256': self.genesis_sha256,
            'data_root': self.state['data_root'],
            'cursors': life['cursors'],
            'seen_documents': life['seen_documents'],
            'seen_batches': life['seen_batches'],
        }))

    def report(self):
        life = self.state['lifecycle']
        return {
            'chain_id': self.state['chain_id'],
            'genesis_sha256': self.genesis_sha256,
            'height': self.state['height'],
            'app_hash': digest(canonical(self.state)),
            'data_root': self.state['data_root'],
            'admission_anchor': self.anchor,
            'consumed_documents': len(life['seen_documents']),
            'consumed_batches': len(life['seen_batches']),
            'history_source': 'operator-owned committed full-node state',
            'scope': 'local replay trust; no remote inclusion proof or curation approval',
        }

    def ensure_current(self):
        latest = read(self.home, self.genesis_sha256)
        if latest.state['height'] < self.state['height'] or latest.anchor != self.anchor:
            raise ValueError('Native admission history changed during this run; retry against the current state')
        return latest


def read(home, genesis_sha256):
    home = Path(home).resolve()
    root(genesis_sha256)
    genesis_path = home / 'config/genesis.json'
    if genesis_path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError('Native genesis exceeds the local review bound')
    raw_genesis = genesis_path.read_bytes()
    if digest(raw_genesis) != genesis_sha256:
        raise ValueError('Local genesis differs from the independently pinned network')
    genesis = json.loads(raw_genesis)
    manifest = genesis['app_state']['manifest']
    if not manifest.get('lifecycle'):
        raise ValueError('This node does not use the native lifecycle profile')
    database = home / 'evolution.sqlite'
    # mode=ro refuses missing databases and never initializes node state. One
    # read transaction pins both the size check and value across concurrent commits.
    db = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True, timeout=5)
    try:
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        size = db.execute('SELECT length(value) FROM state WHERE id=1').fetchone()
        if size is None or size[0] is None or not 1 <= size[0] <= MAX_STATE_BYTES:
            raise ValueError('Missing or oversized committed native state')
        value = db.execute('SELECT value FROM state WHERE id=1').fetchone()[0]
    finally:
        db.close()
    state = json.loads(value)
    if state['chain_id'] != genesis['chain_id'] or state['manifest'] != manifest:
        raise ValueError('Committed state differs from the pinned genesis')
    integer(state['height'], 1, 2**63-1)
    root(state['data_root'])
    life = state['lifecycle']
    for field in ('seen_documents', 'seen_batches', 'cursors'):
        if not isinstance(life[field], dict):
            raise ValueError('Missing complete native admission history')
        for key in life[field]:
            root(key)
    for cursor in life['cursors'].values():
        integer(cursor, 0, 2**53-1)
    return Snapshot(home, genesis_sha256, state)


def add_arguments(parser):
    parser.add_argument('--native-home', type=Path,
                        help='Own fully replayed lifecycle node home; read committed history without changing the node')
    parser.add_argument('--genesis-sha256',
                        help='Independently pinned SHA-256 of that network genesis; required with --native-home')


def from_arguments(args):
    if bool(args.native_home) != bool(args.genesis_sha256):
        raise ValueError('--native-home and --genesis-sha256 must be supplied together')
    if args.native_home:
        if args.exclude_cohort:
            raise ValueError('Native state supplies complete history; omit manual --exclude-cohort files')
        return read(args.native_home, args.genesis_sha256)
    return None
