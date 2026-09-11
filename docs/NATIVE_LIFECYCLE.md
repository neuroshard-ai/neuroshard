# Native data and serving lifecycle — experimental profile

This opt-in protocol connects fresh-data admission, full-model training, quality decisions and paid generation to the same NeuroShard-native ledger. It requires a **fresh experimental genesis** containing `manifest.lifecycle`. It does not upgrade the public 0.4.0 chain, migrate its balances, or establish production readiness.

The [evolution protocol](EVOLUTION_PROTOCOL.md) describes partitioned training and growth. The [text protocol](TEXT_PROTOCOL.md) binds tokenization to the model. This profile adds the missing ledger transitions between those operations.

[Integration evidence](NATIVE_LIFECYCLE_RESULTS.md) records source commitments, admitted-data checks, numerical work, failures and release validation.

## What “fresh data” means

A new download is an input to curation. It is not an instruction to change consensus data. Here, fresh means previously unused documents and token batches from consecutive rows of explicitly pinned source revisions. It does not establish that their subject matter is recent, true, or useful. New source revisions are new source identities and require another admission decision.

1. A contributor collects bounded windows into the durable corpus registry. Exact and heuristic near-duplicate checks occur before tokenization. Original source, revision, row, license and raw-document hashes remain attached.
2. A contributor proposes a content-addressed cohort using `propose_data`, locking the same collateral as an execution claim. The transaction includes the source specifications, cursor ranges, document commitments and **all assigned token batches**. Cohort metadata is at most 1.5 MiB, execution metadata remains at most 512 KiB, and every transaction is at most 2 MiB.
3. Validators independently review provenance, license suitability, raw-text/token correspondence, train/evaluation separation and curation. `vote_data` records one explicit vote per snapshotted validator owner, weighted by its native bonded voting power. Multiple keys owned by the same account are aggregated.
4. Admission needs strictly more than two thirds of both the proposal's snapshot power and the current voting power, plus the activation delay. A vote is a curation attestation, not a cryptographic proof of source authenticity, permission, truth or model quality.
5. Activation occurs between cohorts, with no active reservation, execution claim or evaluation. Cursors, consumed document IDs and consumed token-batch roots advance atomically. Expired proposals refund 90% of their bond and burn 10%; proposal/vote fees are also burned. Proposals mint no tokens.

Validators do not fetch S3 or upstream URLs while deciding a block. The exact token bytes used for training/evaluation are carried in native transactions and state. Raw documents, tokenizer objects and model tensors remain content-addressed off-chain artifacts. S3 can mirror those artifacts; it does not decide their identity or admission. Availability of the raw evidence must be checked before a validator votes.

This is permissionless contribution and native-stake curation. It deliberately does not claim that arbitrary internet text can safely activate without review. An honest consensus participant that blindly approves source labels defeats that curation assumption.

## Bounded cohorts and replay

Each cohort contains up to 128 fresh training documents, exactly 32 retention documents and exactly 32 fresh-evaluation documents. All identities, source positions and token batches must be distinct within the cohort. Previously consumed document IDs or batches cannot be resubmitted as fresh. Evaluation source roles cannot be relabeled as training. Evaluation documents declare zero omitted targets.

Each document contains 1–4 labeled response windows. Evaluation windows share a row length of at most 128 tokens; each score transaction handles up to four windows, including a smaller last batch when necessary. A document may span multiple transactions. Full response coverage and correspondence to the committed raw document are part of admission review. Training omissions remain explicit, while incomplete evaluation documents are rejected. This retains bounded multi-turn responses without treating their chunks as independent evaluation examples.

The genesis fixes 1–64 training steps per cohort. Without history every scheduled batch is fresh and unique. Subsequent cohorts take `floor(steps / 4)` replay batches from the preceding admitted training cohort, spread through the schedule; the rest are fresh. Ranking is deterministic from the cohort root. This is a bounded replay baseline, not a representative lifelong-memory reservoir.

`reserve` binds the active data root, exact schedule position and batch, as well as the model, round and worker reward identities. Activation cannot change an in-flight job. The original daily/block-period issuance budget also applies. Once the cohort's steps settle, more reservations fail until evaluation completes or expires and another fresh cohort activates. Repeating the old corpus forever cannot renew the budget.

A stalled cohort has an absolute training deadline. Once outstanding work has ended, expiry closes the cohort and restores the serving model as the learning parent. Reservations and claims retain their own bounded deadlines. Consumed data and paid-work identities are not erased on timeout or rollback.

## Verifiable evaluation and the serving decision

The learning root and serving root are separate. Correct training can be paid even if its resulting model fails evaluation. Identity depth growth is allowed before a fresh cohort's first training step, remains challengeable, and mints no tokens.

After all prescribed training steps, `open_evaluation` binds the learning candidate, current serving baseline and active cohort. The cohort's public evaluation data was committed before training; the candidate is fixed before score claims begin. These are public tests, not secret or unpredictable held-out examples. Contributors can see them, so deliberate contamination and selection remain concerns even when the local collector excludes them from training.

`score` commits a forward graph for up to four assigned windows, one side (baseline/candidate), one group (retention/fresh) and one offset. Forward partition records bind model, input, output and ownership; a separate output-head record binds per-window response loss and next-token results. Each partition and the head can be challenged through the native upload/seal/referee path. Ordinary score admission checks graph structure; it does not execute the model. Every accepted graph therefore requires an honest online observer with complete coverage. Scores mint no tokens.

Depending on window counts, evaluation takes 32–128 score claims across both models and groups. Duplicate, skipped and substituted window scores are rejected. `finish_evaluation` fails until every assigned window has settled. The native decision then weights each window by its actual scored target count to form **one observation per document**, with exactly 32 paired documents per group. Every accepted score remains subject to the same optimistic assumption as accepted training.

The gate uses paired document loss changes and the existing approximate 99% normal-interval criteria: fresh improvement greater than 0.001 nats, with retention regression below 0.02 nats. Window losses are combined using exact rational arithmetic, then converted to a canonical document mean and quantized to integer micronats, with a conservative two-micronat allowance. The decision uses an integer squared-inequality comparison rather than platform-dependent statistical rounding. With `n = 32`, sum of deltas `S`, `Q = n*sum(delta²) - S²`, effective margin `m` and `G = n*m - S`, a group passes only when:

```text
G > 0 and G² * 1000² * (n - 1) > 2576² * Q
```

Both groups must pass. A passing decision changes the native serving root and its model metadata. Failure or timeout preserves the serving root and restores it as the next learning parent. Candidate, measurements and decision remain in a bounded history; paid computation history remains intact.

This gate measures one public cohort's teacher-forced response loss. It is not proof of better reasoning, factual accuracy, instruction following, safety, or continual improvement. Repeated cohort decisions also require a sequential-testing policy before interpreting a long run as a statistical guarantee.

## Paid full-model inference

`infer` pins the current serving root, named provider, tokenized prompt, generation limit, price cap and deadline. The requester pays the transaction fee and moves the maximum payment into ledger escrow. The experiment currently limits prompts to 2–192 tokens and generation to **1–8 tokens**, making the entire autoregressive graph fit within bounded metadata. It is not a production chat service.

`respond` can only come from the assigned provider. It locks execution collateral and commits every forward graph, including each output head, for the complete greedy generation. Graph checks bind every generated token to the next step's input and enforce the profile's stop tokens. A pending request continues to refer to its original model after a later promotion.

Following the dispute window, accepted generation pays the provider the fixed per-token price for actual output tokens and returns unused escrow. No inference tokens are minted. Unanswered requests refund escrow after expiry; the original transaction fee is not refunded. Refuted responses do not pay the provider and can be retried before the request deadline. Completed requests cannot settle twice.

This reference serializes execution claims, and settlement waits for disputes. It does not establish acceptable production latency or throughput. Model availability, provider discovery, customer-facing encoding/decoding, and public worker admission still need release integration.

## Disputes and expiry

The output head is part of verification, not a trusted coordinator calculation. Fraud disputes publish only the selected partition's components and boundary input through native transactions. Head-only disputes require the embedding, final norm and final hidden state, omitting transformer blocks owned by that worker. A referee reuses the worker's exact forward implementation. Out-of-memory/process failure must not be classified as numerical fraud.

Missing availability evidence after a complete response deadline can slash the publisher. An unfinished fraud accusation at the absolute claim deadline cancels the claim without paying or promoting it, refunds publisher collateral and burns the accuser's bond. This avoids allowing an uncompleted accusation to confiscate an honest publisher's bond. It does not solve adversarial censorship or establish that the test fee/bond parameters deter it economically.

Training/evaluation/inference audits and artifact replication need sustainable funding when participants are honest. The present implementation still relies on explicitly operated observers. One observer checking all stages is an experiment; it is not an independent permissionless audit market.

## Queries and reproduction

### Prepare a data proposal

From an installed checkout with the pinned numerical and collector dependencies, the existing model commands download the eight pinned seed files, verify their hashes, and import the model with its text contract:

```bash
python -m neuroshard.evolution download-seed --model-dir ./seed
python -m neuroshard.evolution import-model --model-dir ./seed --objects ./objects
```

Keep the printed model and tokenizer roots. Copy [the data configuration](../config/native-data.example.json) to the same private working directory and check its object path and tokenizer root against that output. Use a separate corpus from historical objectives. Save the native `/lifecycle` query's `cursors` object as `cursors.json`; use `{}` only for a genuinely new profile with zero initial cursors. Supply the chain's current data root.

```bash
python scripts/prepare_native_cohort.py \
  --config ./native-data.json --cursors ./cursors.json \
  --previous-data-root CURRENT_DATA_ROOT \
  --collect-records 256 --output ./proposal-1.json
```

The invocation scans at most 256 records per configured source and never goes beyond 4,096 unadmitted rows per source. A `needs_more_data` report includes rejected/selected counts; it does not emit an incomplete transaction. With a new output filename, a later invocation can add another bounded tranche using the same native cursor anchor. The durable corpus recovers previously collected documents even if an earlier export failed. `--collect-records 0` only prepares already collected data.

The exporter retains up to four response windows per document. It reports training omissions and requires complete evaluation responses. The bounded response/context policy still introduces selection bias. Do not treat its example Smol-SmolTalk source as a diverse lifelong learning corpus. Source authenticity, licensing, harmful/low-quality content, synthetic-data proportions and meaningful evaluation still require independent curation. The explicit Python `publish` helper mirrors the selected raw documents, batches, provenance and tokenizer to a content-addressed destination with read-back checks; copying an object does not approve it.

Before voting, each reviewer uses their own source/tokenizer policy and an independently obtained copy of the raw artifacts:

```bash
python scripts/review_native_cohort.py \
  --config ./reviewer-data-policy.json --proposal ./proposal-1.json \
  --cache ./reviewer-upstream-cache
```

The reviewer reconstructs every response window from original messages, checks document identities and source roles, enforces the held-out hash partition, and rejects heuristic near-duplicates within the cohort. By default it also checks every selected document against its pinned upstream Parquet row, verifies the downloaded file's SHA-256 against repository metadata, and confirms the repository revision. A cache can be reused, but its files are rehashed. Use a separate cache directory per review process. This source check trusts the upstream repository service and TLS; a license label is not proof of rights. `--offline` explicitly reports that upstream checking was skipped.

This report never votes or authorizes activation. The reviewer must separately check the live parent/cursors, historical contamination, license rights, harmful content, usefulness and evaluation suitability. Native consensus repeats its own parent, cursor, exact-duplicate and quorum checks. Treat `mechanical_evidence_verified` as evidence about bytes and tokenization, not an automatic curation decision.

### Run protocol and numerical checks

The native application exposes `/lifecycle`, `/data`, `/data/proposal`, `/evaluation` and `/inference`, alongside `/status`, `/candidate`, `/account` and `/work`. `/data` returns the exact activated batches; `/data/proposal` returns the content-addressed proposal for admission review. `/inference` accepts an optional `id` to retrieve a pending request or its retained result. These are ABCI query paths, not new public website endpoints.

Install the pinned [evolution dependencies](evolution-requirements.txt) and use a fresh experiment directory:

```bash
python -m pytest -q tests/evolution/test_lifecycle.py
python scripts/experiment_lifecycle_native.py \
  --home .neuroshard/lifecycle-check \
  --engine /path/to/cometbft
```

The script starts four isolated native validators with fresh keys, uses a 10,384-parameter synthetic model, checks every honest computation stage, and stops its processes on exit. Its tiny synthetic text identity is not a real tokenizer contract. It tests protocol mechanics; any passing synthetic quality decision is not evidence that the real 135M model improved. The result records physical hosts/operators explicitly.

For actual model arithmetic, `scripts/experiment_forward_profile.py` creates versioned evaluation and generation records and independently replays every stage. Supply `--workers-config` using the existing epoch endpoint format for workers on another machine, and `--verify-result` with copied objects for an independent replay on that host. This is a conformance check, separate from the synthetic native lifecycle and from a real-data model-quality experiment.

The native lifecycle script also accepts `--model-root`, `--objects`, `--cohort`, `--next-cohort` and optional `--workers-config` for two prepared real-data cohorts and an imported model. Add `--growth-layers 4` with four workers to exercise a capacity expansion before training. It derives the experimental genesis's initial source cursors from the first proposal. This path performs real training and complete multi-window evaluation through native settlement. Budget substantially more time and disk: every stage is independently replayed, and the forged-inference dispute uploads the real output-head dependencies through native blocks. It remains an isolated experiment with test keys, not a public migration command.

After that script has completed and stopped its validators, the bounded continuation check reopens the **same** node homes and signing state and trains the second admitted cohort:

```bash
python scripts/continue_lifecycle_native.py --home .neuroshard/lifecycle-check
```

Use the exact original package source, and supply the same `--workers-config` for an HTTP-worker trial. The continuation refuses running node ports, missing original keys/state, an unexpected ledger position, or an existing continuation result. It checks fresh/replay assignments, another four paid steps, unchanged serving identity pending evaluation, and common application hashes. It stops its own node processes on exit. It is an integration driver, not a fault-tolerant public coordinator or an unlimited training service.

## Requirements before public cutover

The code closes several protocol gaps, but the public release must still establish:

- The integrated lifecycle running with the real model, independently curated data and the exact tokenizer across supported hardware, including both accepted and rejected model decisions.
- Public worker/provider discovery, signed outbound work admission, reassignment on failure, bounded durable retries and an onboarding client that supports this profile.
- Independent validator operators and a voting-power layout that survives the intended host/operator failures. Four keys across two servers controlled by one operator do not provide that independence.
- Funded complete audit coverage and independent model replicas, with measured artifact recovery, dispute bandwidth, memory, latency and storage retention costs.
- Explicit balance-preserving migration from the public 0.4.0 ledger, replay/snapshot verification, rollback procedures and client/network-descriptor compatibility. Never edit a live genesis or reset signing state to simulate an upgrade.
- An independent security review, fault/load testing, monitored service budgets and issuance/economic parameters justified by actual costs. More peers do not automatically prove more independent hardware or better training throughput.

Until those conditions have evidence, this profile belongs on isolated integration networks. The current public testnet remains a separate experimental service.

The read-only topology check uses actual native voting power and an explicit inventory of host/operator labels:

```bash
python scripts/check_validator_topology.py --rpc http://127.0.0.1:26657 \
  --inventory ./validator-inventory.json
```

The inventory is `{"validators":[{"consensus_key":"64_LOWERCASE_HEX_CHARACTERS","host":"host-a","operator":"operator-a"}]}` with one entry per current voting key. The command pins one query height for pagination, rejects incomplete/duplicate inventories, and exits with status 2 if losing any one declared host or operator leaves at most two thirds of voting power. Labels are declarations that require independent verification. The check tests quorum topology, not the other release conditions.

The design keeps block validity deterministic as required by [CometBFT's ABCI application contract](https://github.com/cometbft/cometbft/blob/v0.38.26/spec/abci/abci%2B%2B_app_requirements.md). Keeping original-source data and bounded replay is motivated in part by documented risks of [recursive synthetic-data training](https://arxiv.org/abs/2305.17493) and [catastrophic forgetting](https://arxiv.org/abs/1811.11682); neither citation establishes that this particular learning policy works.
