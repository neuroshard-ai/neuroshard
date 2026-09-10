# Public native API

`https://neuroshard.com` reads a full node through bounded endpoints. No registration token is needed. Monetary amounts are decimal strings in atoms: **1,000,000 atoms = 1 NEURO**. API observations are not account inclusion proofs.

| Endpoint | Response |
|---|---|
| `GET /healthz` | Summary; HTTP 503 while syncing/stalled, 200 when ready |
| `GET /api/network` | Chain, height, model round/root, supply, membership, lease, health |
| `GET /api/manifest` | Genesis-bound execution and consensus profile |
| `GET /api/blocks?limit=10&before=H` | At most 20 blocks before exclusive cursor, newest first |
| `GET /api/block/H` | Block identifiers and transactions with execution codes; 0 means accepted |
| `GET /api/account?public_key=KEY` | Balance and next nonce; unused valid keys return zero |
| `GET /api/validators?limit=100&after=KEY` | Active validators, owners, bonds, voting powers; at most 100 |
| `GET /api/model` | Current model fields, round/root, last training loss |
| `GET /api/model/checkpoint.json` | Latest accepted checkpoint; [format](MODEL_CARD.md) |
| `GET /api/training?limit=50&before=H` | At most 100 accepted training receipts and indexing status |
| `GET /network/genesis.json` | Exact genesis bytes; compare the published digest independently |
| `GET /network/corpus.txt` | Genesis-bound corpus |
| `GET /work/status` | Optional sponsor's remaining attempts, recent workers, active operations |
| `POST /rpc` | Restricted JSON-RPC relay and bounded signed transactions |

Allowed RPC methods: `status`, `block`, `commit`, `validators`, `abci_query`, `broadcast_tx_sync`, and `broadcast_tx_commit`. State queries allow supported current-state paths without proofs. Administrative calls, arbitrary paths, batches, and historical state queries are rejected. Broadcast envelopes are limited to 16 KiB and signature-checked before relay.

A block header's `app_hash` commits the preceding application state, identified by `app_state_height`. Inclusion in a block alone does not imply successful execution; inspect the transaction code. `latest_block_height` corresponds to `latest_block_hash`; application `height` may be slightly earlier during concurrent reads.

The training-history index is rebuildable local SQLite data, separate from consensus. It walks contiguous blocks from genesis and records accepted training submissions. Missing/inconsistent data pauses indexing. `indexed_height`, `chain_height`, and `error` disclose lag or failure. It never fabricates points. Requests are paginated.

The gateway has 16 request slots, timeouts, bounded caches, and a 240-request/minute client budget. A loopback reverse proxy may supply `X-Real-IP`; the nginx template overwrites that header with the connection address. Non-loopback callers cannot override their address. Do not use a local proxy that passes an untrusted header unchanged.

The optional sponsor accepts signed `/work/poll` and `/work/result` messages. CLI workers check assignments against their own full node. The sponsor selects workers but cannot approve rewards. See the [operator guide](PUBLIC_TESTNET.md) for expiry, retries, and payment conditions.
