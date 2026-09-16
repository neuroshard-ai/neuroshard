# Native jobs and serving for portable model shards

Status: implemented candidate; a new experimental genesis is required. This
does not migrate the public 0.4.0 network or import the research checkpoint as
paid work. The [balanced learning result](BALANCED_CONTINUATION_RESULTS.md)
supplies the numerical recipe for the integration trial.

## Admission and computation

The optional `portable_lifecycle` manifest enables `propose_shard_job` and
`vote_shard_job`. A proposal commits the currently settled parent, prepared
data, output job identity, numerical executor source commitment, reference
checkpoint, terminal cursor, window bound and quality policy. The quality
policy commits its serving baseline and evaluator/data/generation rules before
any paid update. Activation requires strictly more than two thirds of both
snapshotted and current native voting weight, aggregated by owner. It also
waits for the native activation delay. A timed-out proposal refunds its bond;
ordinary transaction fees remain burned.

Activation changes the prescribed computation, not model tensors or the paid
cursor. The first child may use the newly authorized job identity while its
parent remains the previously settled checkpoint. Architecture, parameter
births, optimizer semantics and shard layout cannot change. Each prepared job
can activate once; paid-step identities survive subsequent jobs and failed
quality. A live job permits one reservation or execution claim at a time.

Existing `reserve_shards`, worker receipts, funded audit escrow and native
commit/reveal verdicts settle one to four updates per window. The dedicated
executor checks the genesis and live reservation before allocating work, loads
only its owned parameters, preserves Adam state, and records boundary witnesses.
Every honest auditor replays every partition sequentially. Only accepted new
updates mint the prescribed NEURO reward. This still replicates computation;
it does not reduce verification to a signature check.

## Separate quality approval

After every activated update settles, `quality_shards` submits a report binding
the already-settled candidate, frozen serving baseline, prepared data, policy,
complete results and pass/fail decision. Its funded audit obligation is separate
from training. Auditors reproduce both checkpoints' scores and greedy answers,
check the prescribed decision, and attest the complete report. A valid report
may say **fail**: correctness of evaluation and model improvement are distinct.

Only a valid, passing report with a positive native audit quorum changes
`serving_root`. Quality approval mints zero tokens. A failed quality decision
keeps the serving checkpoint and already-earned training rewards. It closes
that job; the learned checkpoint and paid cursor remain available for a later
explicitly admitted recipe. A failed or unavailable claim releases the report
slot. Another publisher can report on the same fixed checkpoint and policy;
retry cannot select a different model, evaluator or dataset. This prevents an
invalid first report from permanently blocking a correct quality decision.

Quality witnesses are segmented at each generation and role-level loss scan.
Each segment uses the existing closed-graph verifier and event bound. The outer
commitment fixes segment order, complete coverage and the quality statement;
auditors must finish all segments. This bounds individual witness groups,
not total evaluation cost or total artifact storage.

Complete service manifests are exchanged as size-declared, hash-checked chunks.
The underlying control messages retain their 2 MiB limit; each decoded manifest
is capped at 128 MiB and their combined allocation at 256 MiB. Segmenting the
numerical witness alone does not make its complete metadata a small message.

The initial integration repeats the exact 64-update balanced recipe and its
already exposed final evaluation. It must reproduce the accepted research
learned-state root. It is an integration/reproducibility test, **not another
independent learning experiment**. No new data, optimization search or model
growth is authorized by that trial's policy.

## Paid sharded inference

`infer_shards` locks the caller's maximum price and fixes the serving checkpoint,
tokenizer, prompt tokens, greedy stopping rule and one provider per partition.
The checkpoint stays fixed if serving is promoted while that request is open.
`respond_shards` requires signed receipts from every assigned provider and
complete numerical replay before releasing payment. Providers split the fee
by owned parameter count, using the training adapter's deterministic rounding.
Payment transfers escrow; it does not mint tokens. Unused token budget returns
to the caller. Unanswered requests expire with a full escrow refund.

An inference provider must also arrange a funded audit offer. Its audit cost
can exceed the requested inference price. This candidate demonstrates correct
accounting and execution, not sustainable market pricing. Prompts and responses
are public, the current decoder is bounded greedy generation, and model size
does not imply general assistant competence.

## Operator entry points and artifact contract

`scripts/run_native_shards.py` executes training, quality and serving, and replays
one partition for an auditor. Its immutable execution descriptor commits the
original balanced recipe plus the native wrapper's sources. Training and
serving require a pinned genesis, a live assignment, and the existing assigned
worker key. The command writes checkpoint/witness commitments and signed
receipts; it does not submit payments itself.

The existing `scripts/replay_sharded_claim.py` backend accepts portable training,
quality and inference claims. Its operator-controlled catalog adds an
`execution` JSON path for this profile, retains all rank-specific checkpoint
and witness paths, and includes the original baseline for quality replay.
The backend checks hashes, executes each partition in a fresh subprocess and
retains completed partitions across restart. A missing file or crashed process
produces no validity vote; explicit numerical mismatches produce rejection.

The chain holds commitments, not multi-gigabyte model tensors. This trial stages
verified artifacts on operated hosts and preserves them separately. A public
artifact discovery/replication network and independent providers remain work
outside this candidate. Native quorum safety still assumes less than one third
Byzantine bonded weight and correct verifier software. One infrastructure owner
running several keys does not demonstrate independent ownership.

## Acceptance checks

- Changed parent, unsupported recipe, missing admission or reused job cannot train.
- Native window boundaries preserve full-model/Adam results against continuous execution.
- Every shard and communication segment must replay; fabricated tokens cannot
  reuse an honest witness.
- Training acceptance alone cannot promote serving; quality and inference mint zero.
- Failed quality preserves prior serving and earned rewards; absent audits never pay.
- In-flight inference retains its model version, pays assigned providers once,
  and refunds unused or expired escrow.
- Every transition preserves supply conservation and global work deduplication.

The existing genesis without `portable_lifecycle` retains the prior adapter's
behavior. Operators must explicitly choose this new manifest and its complete
consensus source hash; merging source does not upgrade a running chain.
