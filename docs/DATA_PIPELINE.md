# Immutable dataset collection and S3 recovery

The supported ingestion path is `neuroshard.dataflow`, introduced in 0.4.0. The old sequential-ID uploader remains disabled. The running LLM uses an immutable, genesis-pinned execution dataset; collection does not silently change it.

## Why the old writer was replaced

The read-only audit found 2,497,174 manifest rows but only 2,379,420 distinct shard IDs: 117,754 duplicate rows and 117,535 duplicated IDs. Some repeated IDs refer to different byte hashes. The manifest's reported token total is not verified usable training capacity. A mutable ID-derived filename can be overwritten, making earlier referenced bytes unavailable. Bucket versioning was not enabled, so missing historical versions cannot be reconstructed just from the manifest.

A bounded recovery inspected eight selected current shard objects, checked actual bytes against all manifest rows for their IDs, and preserved matching objects under `recovered/v1/sha256/`. Four conflicting historical hashes among this sample were unavailable. Recovery root: `6c184905d0c956bf76d7fa4f6eb775b5c8898623adffe3a61b41c5711a802a71`. This is an integrity-preservation sample, not a full audit or repair of the old corpus. The raw recovered tensors are quarantined: legacy licensing, source provenance and suitability still require review. They are not active LLM training data.

## Current collection

The collector is pinned to Smol-SmolTalk revision `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`, its training split, Apache-2.0 provenance and the exact model chat template. The deployed schedule collects at most 128 source records per day, up to 4,096 records in this collection home. The cap is deliberate; increasing it requires an explicit configuration change. `progress.json` records the cursor and most recent immutable snapshot. Reaching the cap produces `collection_budget_complete`; the timer cannot reset it.

The original audited snapshot used to prepare genesis is separately published and read-back verified: 512 distinct documents, 487 training and 25 validation, eight shards, root `e5562adf7f4cf8418004e70bb27d68b22e383acf2c0ab4fc68ded430bf6a2bf7`. The new collector can rediscover those source rows; hash-based document identity permits independent verification and deduplication. Do not blindly concatenate manifests from independent journals.

## Write and crash semantics

Objects use SHA-256 of their actual bytes as their name. S3 uploads use `If-None-Match: *` and a SHA-256 checksum. An already-existing object is accepted only after its actual bytes are read and checked. Authentication, network, checksum and manifest errors fail closed. The writer never treats an unreadable existing manifest as an empty dataset. It does not list an upload as committed before upload success.

A SQLite WAL/FULL journal records source identity, cursor, immutable input copy and pending shard bytes. It writes pending intent before upload, and advances the cursor only after upload and the corresponding transaction commit. A process lock prevents concurrent use of one journal. A failed write resumes its pending bytes; reusing the same content address does not overwrite another shard. Local storage uses exclusive immutable creation and fsync.

Upstream files are selected in sorted shard order from the pinned revision and must include SHA-256 metadata. Each file is limited to 512 MiB, downloaded to a local verified cache, and read with synchronous Parquet batches of 64 rows. No background dataset stream survives a bounded invocation. The supported source layout is `data/SPLIT-N-of-N.parquet`.

The outer collector also saves its rendered pending batch and its hash before publication, then advances its source cursor only after the snapshot exists. It recovers that batch without refetching changed upstream contents. A changed source revision requires a new collection home. Shards, documents and each invocation are bounded. The tests include interruption after durable upload but before journal progress, failed publication, concurrent creation, corrupted existing objects and denied S3 access.

S3 is one replaceable storage provider. Snapshot hashes establish content identity, not permanent availability. Mirrors and independent retention are still needed. Operator credentials that can delete objects remain an availability trust boundary; this release does not implement a decentralized storage market.

## Use and verify

For a local JSONL file containing `{"text":"..."}` records:

```bash
python -m pip install 'neuroshard-ai[data]'
neuroshard-data --objects ./objects ingest documents.jsonl \
  --journal ./collection.sqlite --origin https://SOURCE \
  --revision IMMUTABLE_REVISION --license SPDX_IDENTIFIER
neuroshard-data --objects ./objects verify SNAPSHOT_SHA256
```

Use `--bucket BUCKET` instead of `--objects` for S3. The normal AWS credential chain applies; credentials never belong in a dataset manifest, command argument or repository. `neuroshard-ai[collector]` adds the optional pinned upstream loader/tokenizer dependencies. A collector configuration and systemd examples live in [config](../config); local paths and budgets must be adapted before installation.

Before a new dataset becomes consensus input, verify all shard bytes and document IDs, confirm provenance, select/deduplicate deterministic train and evaluation data, tokenize with the pinned tokenizer, run multi-host conformance and evaluation, and explicitly publish the resulting execution profile. Existing chains continue using their original dataset. More S3 bytes by themselves do not make the live LLM better.
