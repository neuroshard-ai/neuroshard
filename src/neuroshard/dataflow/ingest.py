"""Bounded JSONL ingestion with a durable upload intent and immutable snapshots."""
import fcntl
import json
import os
import sqlite3
import tempfile
from pathlib import Path

from .store import canonical, digest

MAX_DOCUMENT_BYTES = 1024 * 1024
MAX_SHARD_BYTES = 8 * 1024 * 1024


def record(text):
    if not isinstance(text, str) or not text.strip() or len(text.encode()) > MAX_DOCUMENT_BYTES:
        raise ValueError("Document must contain 1–1048576 UTF-8 bytes")
    text = text.replace("\r\n", "\n")
    sha = digest(text.encode())
    return {"id": sha, "text": text, "split": "validation" if int(sha[:8], 16) % 20 == 0 else "train"}


class Ingestor:
    def __init__(self, journal, store):
        self.journal, self.store = Path(journal), store
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.journal)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY, metadata BLOB NOT NULL, cursor INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS pending (source TEXT PRIMARY KEY, start INTEGER, end INTEGER, payload BLOB);
            CREATE TABLE IF NOT EXISTS shards (source TEXT, start INTEGER, end INTEGER, sha TEXT, bytes INTEGER,
                                               records INTEGER, PRIMARY KEY(source,start));
        ''')

    def close(self):
        self.db.close()

    def _finish(self, source):
        pending = self.db.execute("SELECT start,end,payload FROM pending WHERE source=?", (source,)).fetchone()
        if not pending:
            return
        start, end, data = pending
        sha = self.store.put(data)  # Crash here is safe: retry verifies or creates the same object.
        with self.db:
            current = self.db.execute("SELECT cursor FROM sources WHERE id=?", (source,)).fetchone()[0]
            if current != start:
                raise ValueError("Journal cursor differs from pending upload")
            self.db.execute("INSERT INTO shards VALUES (?,?,?,?,?,?)", (source,start,end,sha,len(data),end-start))
            self.db.execute("UPDATE sources SET cursor=? WHERE id=?", (end,source))
            self.db.execute("DELETE FROM pending WHERE source=?", (source,))

    def ingest(self, path, provenance, batch_records=64, max_shards=16):
        if not 1 <= batch_records <= 1024 or not 1 <= max_shards <= 10000:
            raise ValueError("Bounded batch_records and max_shards are required")
        if (not isinstance(provenance, dict) or set(provenance) != {"origin","revision","license"}
                or any(not isinstance(v,str) or not v.strip() or len(v)>2048 for v in provenance.values())):
            raise ValueError("Provide origin, immutable revision, and license")
        path = Path(path)
        import hashlib
        content = hashlib.sha256()
        inputs = self.journal.parent / 'inputs'
        inputs.mkdir(exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.input-', dir=inputs)
        try:
            with os.fdopen(fd, 'wb') as target, path.open('rb') as f:
                for chunk in iter(lambda: f.read(1024**2), b''):
                    content.update(chunk); target.write(chunk)
                target.flush(); os.fsync(target.fileno())
            frozen = inputs / content.hexdigest()
            try:
                os.link(temporary, frozen)
            except FileExistsError:
                check = hashlib.sha256()
                with frozen.open('rb') as f:
                    for chunk in iter(lambda: f.read(1024**2), b''): check.update(chunk)
                if check.hexdigest() != content.hexdigest():
                    raise ValueError('Frozen input cache failed checksum')
        finally:
            os.unlink(temporary)
        meta = {**provenance, "input_sha256": content.hexdigest(), "format": "utf8-jsonl-text-v1"}
        source = digest(canonical(meta))
        # One process owns a journal while recovering, assigning, and committing offsets.
        with self.journal.with_suffix(self.journal.suffix + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with self.db:
                self.db.execute("INSERT OR IGNORE INTO sources VALUES (?,?,0)", (source,canonical(meta)))
            self._finish(source)
            start = self.db.execute("SELECT cursor FROM sources WHERE id=?", (source,)).fetchone()[0]
            rows, batches, buffered = [], 0, 0
            with frozen.open("rb") as f:
                index = 0
                while True:
                    raw = f.readline(MAX_DOCUMENT_BYTES * 6 + 1024)
                    if not raw:
                        break
                    if not raw.endswith(b"\n") and len(raw) >= MAX_DOCUMENT_BYTES * 6 + 1024:
                        raise ValueError("JSONL line exceeds the document limit")
                    if index < start:
                        index += 1; continue
                    item = json.loads(raw)
                    if not isinstance(item, dict) or set(item) != {"text"}:
                        raise ValueError("Each input line must be an object containing only text")
                    entry = record(item['text'])
                    size = len(canonical(entry)) + 1
                    if rows and buffered + size > MAX_SHARD_BYTES:
                        self._commit_batch(source,start,index,rows)
                        start,rows,batches,buffered = index,[],batches+1,0
                        if batches >= max_shards:
                            break
                    rows.append(entry); index += 1; buffered += size
                    if len(rows) == batch_records:
                        self._commit_batch(source,start,index,rows)
                        start,rows,batches,buffered = index,[],batches+1,0
                        if batches >= max_shards:
                            break
                if rows:
                    self._commit_batch(source,start,index,rows)
            return self.snapshot()

    def _commit_batch(self, source, start, end, rows):
        payload = b"".join(canonical(row)+b"\n" for row in rows)
        with self.db:
            self.db.execute("INSERT INTO pending VALUES (?,?,?,?)", (source,start,end,payload))
        self._finish(source)

    def snapshot(self):
        if self.db.execute("SELECT count(*) FROM pending").fetchone()[0]:
            raise ValueError("Recover pending uploads before publishing a snapshot")
        sources = {source:json.loads(meta) for source,meta in self.db.execute("SELECT id,metadata FROM sources ORDER BY id")}
        shards = [{"source":source,"start":start,"end":end,"sha256":sha,"bytes":size,"records":count}
                  for source,start,end,sha,size,count in self.db.execute("SELECT * FROM shards ORDER BY source,start")]
        value = {"schema":"neuroshard/dataset/v1","sources":sources,"shards":shards,
                 "split_rule":"sha256-utf8-mod20-validation-v1","tokenization":"raw-text; execution profile pins tokenizer"}
        raw = canonical(value)
        return {"sha256":self.store.put(raw),"manifest":value}


def verify_snapshot(store, root):
    value = json.loads(store.get(root))
    if value.get("schema") != "neuroshard/dataset/v1":
        raise ValueError("Unsupported dataset schema")
    identities, counts = {}, {"train":0,"validation":0,"duplicates":0}
    for shard in value["shards"]:
        raw = store.get(shard["sha256"])
        lines = raw.splitlines()
        if len(raw) != shard["bytes"] or len(lines) != shard["records"]:
            raise ValueError("Shard size or record count mismatch")
        for line in lines:
            item = json.loads(line)
            if item != record(item["text"]):
                raise ValueError("Document hash or split mismatch")
            if item["id"] in identities:
                counts["duplicates"] += 1
            else:
                identities[item["id"]] = item["split"]
                counts[item["split"]] += 1
    return {"dataset_root":root,"shards":len(value["shards"]),"unique_documents":len(identities),**counts}
