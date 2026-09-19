"""Rebuildable local contamination index over the complete admitted history.

The index is a reviewer's derived cache, never a peer-supplied proof. Missing
original objects prevent indexing and therefore prevent a successful review.
Nine disjoint SimHash bands retrieve every candidate within eight flipped bits;
the final comparison still checks the exact Hamming distance.
"""
import sqlite3
from pathlib import Path

from neuroshard.demo.protocol import parse_json
from .data import fingerprint
from .reference_data import identity, sha256
from . import serving_graph


def bands(signature):
    return [str(index) + ':' + str((signature >> (index*64//9)) &
            ((1 << ((index+1)*64//9 - index*64//9)) - 1)) for index in range(9)]


class HistoryIndex:
    def __init__(self, path=':memory:'):
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY, algorithm TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, metadata TEXT NOT NULL, signature TEXT NOT NULL,
                role TEXT NOT NULL, active INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS bands (
                band TEXT NOT NULL, document TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                PRIMARY KEY(band, document));
        ''')
        # Changed normalization or fingerprints require rebuilding derived rows.
        algorithm = identity({name: sha256(Path(__file__).with_name(name))
                              for name in ('expert_history.py', 'expert_data.py', 'data.py')})
        prior = self.db.execute('SELECT algorithm FROM settings WHERE id=1').fetchone()
        with self.db:
            if prior is not None and prior[0] != algorithm:
                self.db.execute('DELETE FROM documents')
            self.db.execute('INSERT OR REPLACE INTO settings VALUES (1,?)', (algorithm,))

    def synchronize(self, history, store):
        from .expert_data import document_identity
        added, read_bytes = 0, 0
        with self.db:
            self.db.execute('UPDATE documents SET active=0')
            for key, document in sorted(history.items()):
                commitment = identity(document)
                old = self.db.execute('SELECT metadata FROM documents WHERE id=?', (key,)).fetchone()
                if old is not None and old[0] == commitment:
                    self.db.execute('UPDATE documents SET active=1 WHERE id=?', (key,))
                    continue
                raw = store.get(document['object'])
                if len(raw) > 256 * 1024:
                    raise ValueError('Historical document exceeds the source review bound')
                original = parse_json(raw)
                serving_graph.fields(original, {'source', 'row', 'messages', 'license'},
                                     'Invalid historical source record')
                if (document['id'] != key or document_identity(original['messages']) != key
                        or original['source'] != document['source'] or original['row'] != document['row']
                        or document['role'] not in ('train', 'evaluation')):
                    raise ValueError('Historical source bytes differ from admitted provenance')
                signature = fingerprint('\n'.join(row['content'] for row in original['messages']))
                self.db.execute('DELETE FROM documents WHERE id=?', (key,))
                self.db.execute('INSERT INTO documents VALUES (?,?,?,?,1)',
                    (key, commitment, format(signature, '016x'), document['role']))
                self.db.executemany('INSERT INTO bands VALUES (?,?)', [(band, key) for band in bands(signature)])
                added += 1
                read_bytes += len(raw)
            # A different canonical snapshot cannot inherit a discarded fork's
            # data-membership decisions merely because the cache contains it.
            self.db.execute('DELETE FROM documents WHERE active=0')
        return {'documents': len(history), 'indexed_now': added, 'original_bytes_read': read_bytes,
                'history_root': identity(history)}

    def match(self, signature, distance, *, replay=False):
        if type(distance) is not int or not 0 <= distance <= 8:
            raise ValueError('History index supports Hamming distance zero through eight')
        values = bands(signature)
        query = '''SELECT DISTINCT d.id,d.signature,d.role FROM documents d
                   JOIN bands b ON b.document=d.id WHERE b.band IN (?,?,?,?,?,?,?,?,?)'''
        for key, candidate, role in self.db.execute(query, values):
            # Explicit replay may reuse training, but can never import a former
            # evaluation document into gradient-bearing supervision.
            if replay and role == 'train':
                continue
            if (signature ^ int(candidate, 16)).bit_count() <= distance:
                return key
        return None

    def close(self):
        self.db.close()
