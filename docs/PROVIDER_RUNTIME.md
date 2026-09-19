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

Use the exact source commit and numerical profile published with the genesis.
For the accepted-model trial the GPU profile requires Python 3.10.12, an NVIDIA
A10G and the pinned CUDA/PyTorch environment. Other GPUs are not yet admitted by
that numerical commitment. Permissionless registration does not promise arbitrary
hardware compatibility. The metadata-only native node uses its separate CPU
environment; it does not download the full model.

From the pinned checkout, create both environments without installing over an
existing node:

```bash
python3 -m venv .neuroshard/native
.neuroshard/native/bin/python -m pip install -r docs/evolution-requirements.txt
python3.10 -m venv .neuroshard/provider
.neuroshard/provider/bin/python -m pip install -r docs/expert-execution-requirements.txt
```

The release must supply its genesis file/hash, reachable native peer, executor
profile and object-length inventory. Build the pinned CometBFT binary using the
[candidate instructions](CANDIDATE_OPERATIONS.md). For a newly agreed genesis,
start a non-voting full node with its own keys:

```bash
PYTHONPATH=src ATEN_CPU_CAPABILITY=default MKL_ENABLE_INSTRUCTIONS=SSE4_2 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .neuroshard/native/bin/python scripts/join_funded_candidate.py \
  --home /srv/neuroshard-full-node --genesis /srv/release/genesis.json \
  --genesis-sha256 RELEASE_GENESIS_SHA256 --engine /srv/release/cometbft \
  --peer NODE_ID@PUBLIC_PEER:26656 --base-port 26656
```

This helper now accepts the funded expert-graph profile as well as the earlier
lifecycle profile. It pins the source and genesis and preserves an existing
recognized home. It is for the fresh research deployment: following an old stake
history additionally needs a recent independently trusted checkpoint. Becoming a
validator requires native stake admission; running this follower grants no vote.
Keep RPC on loopback and expose the native peer and advertised HTTPS provider
port. Never share account, TLS or consensus private keys to join.

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
export PYTHONPATH="$PWD/src"
export ATEN_CPU_CAPABILITY=default MKL_ENABLE_INSTRUCTIONS=SSE4_2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.neuroshard/provider/bin/python -m neuroshard.evolution.provider_runtime --config provider.json --identity
.neuroshard/provider/bin/python -m neuroshard.evolution.provider_runtime --config provider.json --publish-offer
.neuroshard/provider/bin/python -m neuroshard.evolution.provider_runtime --config provider.json
```

The first command creates local keys and prints only public registration fields.
Fund that native address on the intended research chain before publishing the
offer. Publishing reserves the configured collateral and creates the bounded
service offer. The final command discovers assignments for this key and rank,
restores/checks its own partition, acknowledges the native assignment and runs
the pinned model through authenticated peers. Its coordinator submits the
complete, jointly signed response. Payment still requires funded full replay.

Keep the provider identity, TLS key and transaction journal on durable storage
with a private backup. The local native node has separate keys and stores to
preserve. Only `home/models` is a disposable cache suitable for instance-local
SSD storage: committed model bytes can be fetched again, but a new provider key
cannot spend the old key's balance or withdraw its collateral. Restore the same
identity and journal together, and require a new native assignment after an
interrupted execution.

Reissuing an identical publish command recovers the same signed operation; it
does not renew an expired offer. A fresh advertisement needs a changed offer
intent. Cancel an offer through `cancel_expert_offer` before a planned departure;
existing reservations remain obligations until completion, expiry or replacement.
Withdraw only unreserved collateral after the native cooldown.

The continuous runtime keeps one immutable model partition resident between
requests for the same graph. A new assignment installs fresh authenticated wires
and frame sequences while preserving those weights. Request KV caches and
conversation context do not carry between jobs. Changing the graph evicts the
older resident partition before loading its replacement.

`/hosting/quote` supplies a complete first-attempt debit bound: neural execution,
the selected providers' whole-request fees, full verification funding and a
bounded client transaction-fee allowance. Its matching checks shared collateral,
offer capacity and expiry. Quotes expire and native reservation rechecks them;
a quote alone does not reserve capacity. A rejected audited attempt needs fresh
verification funding rather than an automatic additional debit.

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
prompts, neural-call token IDs and final responses are public ledger data. Providers, auxiliary model
owners and auditors see the text needed for execution. TLS protects connections;
this profile does not offer private conversations or ledger erasure.

The [hosted chat client](HOSTED_CHAT.md) obtains a complete quote, funds native
verification and reserves provider capacity. The coordinator exposes a bounded
certificate-pinned `/v1/events` endpoint authenticated by the customer's wallet.
Visible drafts are provisional; only native replay settlement is authoritative.

Validation: adversarial native accounting, authenticated transport/retries,
partition-only HTTP restoration and five-process real neural equivalence cover
the implementation. No AWS hosts were required for those checks. The remaining
deployment trial must freeze the accepted LLM artifacts, failures, full costs,
latency/load bounds and resource expiry before GPU allocation.
