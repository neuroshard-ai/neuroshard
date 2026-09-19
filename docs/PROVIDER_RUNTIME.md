# Running a provider in the hosting research profile

This runtime serves assigned partitions of the complete answering graph through
authenticated HTTPS peers. It is implemented on `development/provider-market`.
The public 0.4.0 genesis does not enable these transactions. Use a freshly pinned
hosting research genesis and its exact source/runtime release. The operated LLM
recovery and independent-operator soak are still required before public release.

Each provider controls its own native wallet, TLS key and local full node.
The node must have finished synchronization. No SSH access to another provider,
shared wallet, operator allowlist or remote code installation participates in
serving. RPC authorization is restricted to the provider's own loopback node;
ABCI responses are not portable light-client proofs.

Create a local JSON configuration with the release's chain ID, manifest hash,
executor profile and object-length inventory. Mirror bases must be configured
locally. Each requested digest is already committed by the graph or policy;
the inventory cannot substitute different model bytes.

```json
{
  "home": "/srv/neuroshard-provider",
  "node_rpc": "http://127.0.0.1:26657",
  "chain_id": "RELEASE_CHAIN_ID",
  "manifest_root": "RELEASE_MANIFEST_SHA256",
  "advertise": "https://YOUR_PUBLIC_HOST:8443",
  "bind": "0.0.0.0",
  "port": 8443,
  "rank": 0,
  "profile": "/srv/release/executor.json",
  "inventory": "/srv/release/object-lengths.json",
  "source_home": "/srv/neuroshard",
  "mirrors": ["https://YOUR_CONFIGURED_MIRROR/objects"],
  "max_bytes": 17179869184,
  "prepare_seconds": 600,
  "frame_seconds": 60,
  "run_seconds": 7200,
  "collateral": 50000000,
  "fee": 1000,
  "offer_blocks": 10000
}
```

The numbers are an illustrative local configuration, not a tested public price
or latency promise. `rank` must exist in the accepted graph. `collateral` and
`fee` use native atoms; the coordinator also reserves the genesis claim bond.
Keep enough liquid balance for transaction fees. There is one active execution
slot per runtime, advertised with capacity one. Clients obtain concurrency by
reserving other available replicas. Do not reuse one wallet across concurrent
runtime processes or independent transaction outboxes.

```bash
python -m neuroshard.evolution.provider_runtime --config provider.json --identity
python -m neuroshard.evolution.provider_runtime --config provider.json --publish-offer
python -m neuroshard.evolution.provider_runtime --config provider.json
```

The first command creates local keys and prints only public registration fields.
Fund that native address on the intended research chain before publishing the
offer. Publishing reserves the configured collateral and creates the bounded
service offer. The final command discovers assignments for this key and rank,
restores/checks its own partition, acknowledges the native assignment and runs
the pinned model through authenticated peers. Its coordinator submits the
complete, jointly signed response. Payment still requires funded full replay.

Reissuing an identical publish command recovers the same signed operation; it
does not renew an expired offer. A fresh advertisement needs a changed offer
intent. Cancel an offer through `cancel_expert_offer` before a planned departure;
existing reservations remain obligations until completion, expiry or replacement.
Withdraw only unreserved collateral after the native cooldown.

On interrupted execution, providers wait for `replace_hosted_job` to commit a
new assignment. The client or any participant can replace a timed-out group
within its original fee ceiling and attempt/expiry limits. Late frames and
receipts cannot settle the new epoch. A pending transaction is recovered using
its original signed bytes; a committed epoch closure can retire that unknown
operation without inventing a successful receipt. Refunds follow native state,
not the process's local log.

Model files remain in the configured home for reuse; local cache management and
the host filesystem quota remain the operator's responsibility. Transcripts and
failure records are local and persist across restarts. Current assignments,
prompts and final responses are public ledger data. Providers, auxiliary model
owners and auditors see the text needed for execution. TLS protects connections;
this profile does not offer private conversations or ledger erasure.

Validation: adversarial native accounting, authenticated transport/retries,
partition-only HTTP restoration and five-process real neural equivalence cover
the implementation. No AWS hosts were required for those checks. The remaining
deployment trial must freeze the accepted LLM artifacts, failures, full costs,
latency/load bounds and resource expiry before GPU allocation.
