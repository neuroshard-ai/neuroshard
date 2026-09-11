# Operate the funded full-model candidate

This is a new-genesis integration candidate for the path toward mainnet. It does
not migrate public balances or change `neuroshard join`, which still selects the
published 0.4.0 adapter network. Read the [audit rules](FUNDED_AUDITING.md) before
accepting a funded obligation.

## Reproduce the complete lifecycle

Use Linux x86_64, Python 3.10–3.12 and Go 1.27.1. Check out the reviewed source
commit from the deployment or experiment record before creating this separate
environment. The candidate genesis commits every Python file in the package;
keep that checkout unchanged while its chain runs.

```bash
python3 -m venv venv_build
venv_build/bin/python -m pip install -r docs/evolution-requirements.txt
venv_build/bin/python -m pip install --no-deps .
mkdir -p .neuroshard/tools
GOTOOLCHAIN=local GOWORK=off GOFLAGS=-mod=readonly \
  go -C src/neuroshard/client/consensus build \
  -o "$PWD/.neuroshard/tools/cometbft" github.com/cometbft/cometbft/cmd/cometbft
.neuroshard/tools/cometbft version
```

The consensus build uses the repository's dependency lock and must report
`0.38.26`. The lightweight PyPI wallet installation alone does not provide this
candidate runtime. From the same checkout, choose a fresh trial directory:

```bash
ATEN_CPU_CAPABILITY=default MKL_ENABLE_INSTRUCTIONS=SSE4_2 \
  venv_build/bin/python scripts/experiment_lifecycle_native.py \
  --home /var/tmp/neuroshard-funded-trial \
  --engine "$PWD/.neuroshard/tools/cometbft" \
  --funded-audits --continue-cohort
```

The default is a small synthetic fixture, useful for failure and accounting
tests. It creates four fresh validators, one publisher group and a separate
auditor process under one operator. It admits two cohorts, trains, evaluates,
retains or promotes the serving root, refutes a forged response and settles
honest paid inference. The continuous operator takes the second cohort, is
killed after a durable reservation, and must recover to complete training and
evaluation. The trial then exercises validator loss, a quorum halt and catch-up.
It stops its processes and retains all state and logs.

The real-model options are `--model-root`, `--objects`, `--cohort`,
`--next-cohort`, `--workers-config` and optional `--growth-layers 4`. Cohort
proposals must already have undergone provenance and tokenization review.
Collection into S3 does not constitute native admission. Use complete,
nonoverlapping immutable proposals; the native cursor and duplication rules
still apply.

`--auditor-key` and `--auditor-command` support a separate host. The latter reads
a JSON argument list with `{genesis_sha256}`, `{sponsor}` and `{rpc}` placeholders.
It can launch a separately configured auditor through SSH. An independently
operated deployment should retain that auditor's private key on its own host;
the experimental driver expects a locally controlled identity and does not
provide a public independent-operator bootstrap ceremony.

## Run an auditor

An additional participant can first follow the agreed ledger with their own
non-voting full node. Obtain the genesis file, its checksum, the source commit
and a reachable native peer through the published deployment record:

```bash
venv_build/bin/python scripts/join_funded_candidate.py \
  --home /var/lib/neuroshard/candidate/follower \
  --genesis /absolute/path/to/agreed-genesis.json \
  --genesis-sha256 AGREED_GENESIS_SHA256 \
  --engine /absolute/path/to/cometbft \
  --peer NODE_ID@REACHABLE_HOST:PEER_PORT
```

Repeating this command resumes the retained home. A different genesis or an
unrecognized existing directory is rejected. New account funding, validator
bonding and worker contracts are separate native actions; following the chain
does not claim voting power or mining rewards. Inbound TCP access to an announced
peer must be tested from outside its host. The two-host trial uses SSH transport
for workers and audit-artifact transfer; native peer reachability is a separate
check. The September 11 trial also reconnects its second-host full node through
the primary host's public TCP peer endpoint, removes the SSH peer tunnel, and
compares common-height headers. A successful trial endpoint is not a commitment
to permanent service; use the current deployment record when joining.

The auditor requires a trusted local full node, the agreed genesis hash,
retrievable content-addressed artifacts, its own account and sufficient liquid
collateral and transaction fees. It accepts offers only from the configured
sponsors and up to its configured stage bound. Funding the account is an
ordinary native transfer; do not export a validator's consensus key for this.

Retain **liquid dispute funding in addition to the audit bond**. The reference
profile locks 5,000,000 atoms for auditing and needs another 2,000,000 for a
fraud challenge, plus acceptance and evidence transaction fees. The measured
head dispute uses 116 transactions: that particular obligation needs at least
7,117,000 atoms before acceptance, excluding any other spending. Larger or
repeated disputes need a larger reserve. The trial funds its auditor with
100,000,000 atoms initially.

The current daemon does not reserve this future dispute budget when accepting.
An auditor funded only for its audit bond can detect fraud but fail to submit a
challenge, then lose its bond for missing a report. A regression test records
this failure mode. Do not accept obligations without the separate reserve;
native prefunding of refutation and completion costs remains a release gate.

```bash
venv_build/bin/python -m neuroshard.evolution.audit_worker \
  --home /var/lib/neuroshard/candidate/auditor \
  --rpc http://127.0.0.1:54851 \
  --genesis-sha256 AGREED_GENESIS_SHA256 \
  --key /var/lib/neuroshard/candidate/auditor.key \
  --sponsor SPONSOR_ACCOUNT_PUBLIC_KEY \
  --objects /var/lib/neuroshard/candidate/audit-objects \
  --source http://127.0.0.1:54990
```

The source mirror serves only the object directory using `/<first-two-hex>/<hash>`.
Keep experimental HTTP and consensus RPC behind loopback or SSH forwarding.
Hashes identify the bytes; source availability and chain-query trust remain
separate requirements. Never serve the directory containing account keys,
validator state, databases or environment files.

The daemon stores its salt and complete replay result durably before committing.
Its SQLite outbox stores signed bytes before submission, looks up CometBFT's
SHA-256 of those exact bytes, and recovers the same operation after restart.
Application claim IDs are different content identities; they cannot substitute
for the native transaction hash. An unresolved transaction prevents signing a
new operation with another nonce.

An unknown result is recoverable; a permanently rejected pending operation needs
operator investigation. Inspect the exact hash, local full-node state and the
expired obligation. Do not delete the outbox, invent a replacement nonce, or
reuse the account concurrently from another process. The tooling deliberately
stops rather than guessing whether an ambiguous payment committed.

The daemon refuses to start with a different source or genesis. New genesis
files should explicitly set `initial_height` to `"1"`; CometBFT normalizes a
zero value when serving genesis over RPC, which changes its serialized hash.
An auditor fetches every required object and replays every stage before the
matching commitment. Missing inputs lead to native availability requests; an
incorrect stage leads to the existing uploaded-input fraud dispute.

## Run the continuous operator

`scripts/run_lifecycle_operator.py --config /absolute/path/operator.json`
operates an existing funded chain. The integration trial writes a concrete
`operator.json` for its retained homes; copy that deployment record with its
pinned source when rehearsing a restart.

The configuration contains `home`, `rpc`, `genesis_sha256`, `key`,
`worker_keys`, `auditors`, `objects`, `workers`, and `budget`. Worker entries
declare `capacity`, and either an authenticated `url` with `token_file` or a
local worker. The signing keys must belong to that operator's own worker group;
the coordinator is not an independent-key custody service. The absolute
`training_round_limit` and `minimum_balance` survive restarts through the
configuration and chain state. Increasing those budgets is an explicit operator
action, not an effect of restarting.

The operator funds complete audits before reserving training, recovers its
workers' durable operations, publishes all outputs before claiming, and completes
both sides of the quality evaluation. It then handles inference addressed to its
own provider key while waiting for another admitted cohort. Training and scoring
currently occupy the serial execution slot ahead of inference; a request can
expire and refund while those tasks run. This is an integration scheduler, not a
low-latency inference service.

The operator defaults to `budget.inference_token_limit = 1`, matching the
bounded generation path exercised in the real-model integration trial. Zero
disables serving; the native lifecycle itself caps requests at eight output
tokens. Increasing the operator limit requires measuring the full response
graph's execution and replay time, fitting the genesis reporting/deadline
windows, and agreeing a sufficient `--max-stages` with every auditor. A
longer native request does not itself establish that this operator can serve it.
Oversized jobs are skipped before funding an audit offer; they can expire and
refund through the native rules. This profile has not established practical
chat latency. The separate public 0.4.0 inference profile has different
generation limits.

The operator also refuses inference priced below its token-denominated audit and
submission costs unless `budget.allow_inference_subsidy` is explicitly `true`.
For four partitions, one auditor and the reference fees, a response that ends at
one token needs at least 502,000 atoms to cover those costs. This excludes compute,
storage and a margin. The trial's old 1,000-atom token price is deliberately
subsidized by its sponsor; it is not evidence of sustainable inference pricing.
A genesis price change requires a new reviewed profile. Underpriced jobs can
expire and refund instead of silently spending the operator's subsidy reserve.

It does not auto-vote new data, grow a model without a separate decision, or
fabricate new data when the approved queue ends. `waiting_for_admitted_data_or_inference`
is a healthy idle state. `training_budget_complete` is an intentional stop in
new training. A missing audit, failed quality gate or lack of funds cannot promote
an unchecked model. Preserve the serving checkpoint as well as learning history.

## Recovery and independent operation

### Starting without outside operators

One operator can bootstrap a public experimental network and test training,
settlement, recovery and joining. Recruiting other people is not a prerequisite
for that engineering work. Publish the actual ownership and hosting topology,
the agreed genesis and source, reachable peers, allocations and measured results.
Keep experimental balances distinct from any later permanent issuance decision.

The first outside participant can run a non-voting full node, compare headers
and application hashes, and report installation or disagreement logs. Funded
auditing and worker contracts can follow once their keys, costs and obligations
are understood. None of these roles needs website registration, and nobody
should give the bootstrap operator their private keys.

Independent consensus requires distributing **voting power**, not merely adding
accounts or processes. Four equal validators with four independent owners are
one useful initial topology: one validator's loss leaves more than two-thirds
online. Four validators controlled by one owner are still one control domain.
One outside validator with a small stake is valuable testing evidence, but does
not establish decentralized control. The [CometBFT consensus
rules](https://github.com/cometbft/cometbft/blob/v0.38.26/spec/consensus/consensus.md)
require more than two-thirds of voting power to commit, with Byzantine safety
assuming less than one-third faulty power. Shared hosting, key custody and
correlated outages must also be considered.

### Retained state and recovery

Keep source, genesis, account keys, validator signing state, native databases,
worker databases, outboxes and object stores in separate retained deployment
directories. Stop a validator before copying its consistent backup. Never run two
copies of its signing key. Restart nodes sequentially and compare a common-height
header and application hash; process uptime alone is not agreement.

Four equal validators tolerate one unavailable validator. Two unavailable
validators halt finality. Two hosts owned by one person do not create independent
consensus, and placing half the voting power on each host makes loss of either
host a quorum outage. Additional independent machines and ownership are part of
the trial, not facts that software can manufacture.

The [auditor service](../config/neuroshard-evolution-auditor.service) and
[operator service](../config/neuroshard-evolution-operator.service) are deployment
templates. Configure paths, user, source, budgets and mirrors before installing.
Both need running native nodes; they do not initialize or reset consensus state.
Monitor block age, audit deadlines, pending outboxes, service payments, available
collateral, evaluation completion, object-store growth and disk space. Runtime
checks stop new work below 2 GiB free; they are a final guard, not a retention
policy. Retaining each full-model update consumes substantial disk.

A source mismatch must fail before joining the candidate. There is no arbitrary
hot upgrade or automatic migration in this profile. Rehearse same-source crash
recovery first. A future consensus upgrade needs a reviewed activation or
migration specification and preserved balances and signing history. Independent
agreement logs, security review and a useful held-out serving checkpoint remain
required before changing the project's production-readiness claim.
