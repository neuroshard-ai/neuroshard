# Public native API — 0.4.0

The HTTPS gateway is a bounded view of a full node and a relay for signed native transactions. There is no account-registration, private-key custody, credit database or API-key billing service. Monetary API fields are decimal strings in integer atoms, except raw native RPC query results; use arbitrary-precision integers. One NEURO is 1,000,000 atoms.

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | Native readiness; 503 while unavailable, catching up or stalled |
| `GET /api/network` | Chain/genesis identity, height, exact observed block/hash pair, training round, supply and readiness |
| `GET /api/model` | Frozen/trainable parameter counts, model revision, training and serving roots, losses, execution profile |
| `GET /api/model/checkpoint.json` | Latest training adapter, chain identity and root; the serving adapter can be older |
| `GET /api/training?limit=50&before=ROUND` | Accepted update history, exact loss, roots, indexing progress and error |
| `GET /api/blocks?limit=10&before=HEIGHT` | Recent native blocks and transaction execution results |
| `GET /api/block/HEIGHT` | A committed block and decoded signed transactions |
| `GET /api/account?public_key=COMPRESSED_KEY` | Available `balance`, `locked` budget, `total` liquid balance and next `nonce` |
| `GET /api/validators` | Paginated native validator owners, power and bonds |
| `GET /api/inference` | Chain/genesis identity, serving root, default provider, heartbeat status, fixed price, fee and queue size |
| `GET /api/inference/request?id=REQUEST_ID` | Pending job, recent completed response or expiry/refund record |
| `GET /network/genesis.json` | Published native genesis |
| `GET /network/dataset.json` | Genesis-bound tokenized execution dataset |
| `GET /artifacts/sha256/SHA256` | Curated content-addressed model/data mirror; verify bytes against the manifest |
| `GET /work/status` | Optional project sponsor, stage availability, remaining attempts |
| `POST /work/poll`, `POST /work/result` | Signed outbound worker protocol; use the packaged worker |
| `POST /rpc` | Allowlisted native JSON-RPC queries and signed transaction relay |

The model/history/API describes the connected chain, not an independent proof. There are no account inclusion proofs. The explorer index is derived from retained native blocks and can lag or be unavailable independently of consensus. Request results are bounded to the latest 128 completed/expired jobs; archival reconstruction requires native history. Provider heartbeat status is a local service observation and cannot guarantee a future response.

The public relay supports bounded `status`, `block`, `validators`, `abci_query`, `broadcast_tx_sync` and existing allowed native methods. `/job` and `/jobs` are native query paths. It does not expose arbitrary Comet RPC, application gRPC, private keys or administrative methods. Inspect `publicnet/gateway.py` and `inference/gateway.py` for the exact method/parameter allowlist. Native transaction byte limits are stricter than the outer HTTP envelope.

Use the installed CLI or website to create a signed request. The body includes chain ID, nonce, chosen provider, serving checkpoint, public prompt, maximum tokens, integer price and block expiry. The client enforces a total price ceiling before signing. A successful broadcast response only establishes preliminary acceptance: poll the request to observe completion or expiry. After an unknown network outcome, query the same ID before paying again. Native Comet transaction hash and application request ID are different, as described in [the protocol](LLM_PROTOCOL.md).

Legacy registration APIs return 410 and old login/signup paths redirect to `/join`. `/reference/v03/api/...` exposes retained read-only views of the earlier native reference chain. Balances never move between these chain identities automatically.
