# Native replay quorum for portable shards

This is an opt-in experimental genesis profile using the existing NeuroShard
CometBFT application and NEURO ledger. It does not modify the running public
chain. The portable adapter admits bounded windows of one frozen computation;
future dataset activation, model growth and serving promotion are not enabled
by this adapter.

## Security statement

The active native validator bonds determine audit voting weight. Owners with
several consensus keys receive their combined weight, not extra votes per
signature. A sponsor cannot select a convenient set of attesters. Funding with
an empty auditor list derives the current owners directly from native state.

A positive verdict requires strictly more than two thirds of the snapshotted
weight. Every honest signer must replay **every shard in the window**. If less
than one third of that weight is Byzantine, a false computation cannot obtain
a positive quorum: every such quorum contains an honest verifier. This statement
depends on correct replay software, available committed inputs, the pinned
numerical runtime and honest signers. It is BFT validity, not a SNARK or a proof
that the signers used independent hardware. Threshold collusion can accept a
forgery; the tests explicitly preserve that counterexample.

The snapshot must still match active voting weights when work is reserved.
Snapshot bonds cannot be withdrawn while the bounded reservation or claim
remains live. They may leave consensus through its ordinary exit process, but
withdrawal waits for the audit obligation to finish. One owner may both train
and audit; the security assumption concerns bonded weight, not different key
strings pretending to represent different people.

Positive and negative verdicts use distinct committed statements. The signature
domain binds the chain, claim, auditor, covered graph, salt and verdict. Honest
auditors can reject a wrong execution without posting a second fraud-dispute
bond. A negative quorum rejects it. A minority rejection cannot veto an honest
positive quorum. Missing reports never authorize issuance. Silence does not
slash auditors in this profile: withholding or unavailable inputs are not proof
that an auditor misbehaved.

## What replay checks

The training window records boundary activations, boundary adjoints and collective
messages. Before numerical replay, every send must match its corresponding
receive, and each collective result must follow from all declared inputs. These
checks close the witness graph; they do not establish correct computation.

The numerical auditor loads one student partition, its Adam moments and the
corresponding frozen reference partition. It recomputes all local operations,
compares outgoing activations and gradients, checks the per-parameter clipping
norm contributions, and verifies every final parameter and Adam tensor. It then
repeats for the other partitions. The complete checkpoint and prepared data bind
the input model, cursor, architecture, optimizer groups, tokenization and batch
schedule. A self-consistent forged witness still fails numerical replay.

The operator verifier also compares the complete regenerated shard-manifest
hash with the output checkpoint's commitment, including ownership and RNG
metadata. Matching learned tensors alone cannot authorize arbitrary manifest
hashes: the next worker must be able to load the accepted checkpoint. A manifest
mismatch produces a negative report; an unavailable manifest produces no report.

The genesis profile also fixes `reference_root`, the complete frozen reference
checkpoint commitment. Omitting it means the prepared seed reference, encoded
as the canonical hash of JSON `null`. The claim, backend and every partition
report must agree on that reference. A different teacher cannot pass solely
because student input/output roots match. Moving to a subsequent cohort's new
reference requires explicit job activation; this adapter does not yet implement
that transition. The [continued-learning plan](CONTINUED_LEARNING.md) treats
that activation, then reserved-window receipts, as development order after a
quality pass. The [answer-balanced experiment](BALANCED_CONTINUATION_RESULTS.md)
now supplies a narrow learning pass; those integration steps remain unimplemented
in this adapter. Importing an existing checkpoint identity is not payment.

The current window is at most four updates. A dispute or audit starts from the
previous committed checkpoint; it does not replay the entire training history.
An auditor can process shards sequentially on one suitable GPU. Its memory need
is bounded by the largest shard, its optimizer, its reference shard and the
activation reserve. **Total replay computation is still required.** Mandatory
quorum auditing replicates that cost; it is not a claim of cheap verification.
Exact agreement is currently scoped to the recorded numerical profile, not
arbitrary CPU/GPU combinations.

## Ledger behavior

`reserve_shards` binds the current portable checkpoint, worker accounts and
funded audit obligation before worker receipts are accepted. `claim_shards`
provides the output checkpoint, transcript commitment and receipts for every
reserved shard. Receipts bind the reservation, both checkpoints, transcript and
rank. Training cannot change the architecture, layout, parameter birth cursors
or optimizer group membership.

Only a positive native quorum advances the learned checkpoint and issues the
fixed reward for each newly settled update. The cursor advances monotonically;
overlapping reservations, replayed steps and stale receipts fail. Rewards split
by owned parameter count with deterministic integer rounding. This is a defined
issuance rule, not a claim that parameter count perfectly prices hardware cost.
The serving checkpoint remains separate: verified computation does not imply
improved quality.

Audit fees come from sponsor escrow, not additional issuance. In the native
profile, the total stage budget is four times `price_per_stage`, independent of
the number of keys. Reports share it in proportion to snapshotted voting weight.
Unpaid fractions and unused stage capacity return to the sponsor. Splitting a
bond among identities cannot enlarge the total fee budget. A successful negative
quorum pays honest rejection reports; a conflicting positive report loses its
audit collateral. This requires actual demand to fund audits and does not
establish a sustainable token price or a competitive provider market.

## Auditor operation

The existing audit worker supports an operator-configured GPU backend:

```json
{
  "argv": ["/path/to/python", "scripts/replay_sharded_claim.py", "--catalog", "/path/to/catalog.json"],
  "timeout_seconds": 3600
}
```

Pass that local file with `--portable-backend` to the audit worker. Claim contents
go to the backend as JSON on stdin; they never become executable commands.
Without a backend, a portable audit worker does not accept an obligation.

The catalog identifies a frozen numerical checkout and GPU Python executable,
the prepared file, rank-specific seed directories, input/output checkpoint
paths, transcript manifests and per-rank witness directories. The script invokes
the pinned numerical auditor once per shard, sequentially, and binds its reports
to the native obligation. It distinguishes explicit replay mismatches from
missing files, OOM and process failures. Infrastructure failures produce no
validity report. Artifact discovery and fetching must supply these local,
hash-checked files before replay; this prototype does not provide a public DHT.

Replay progress is bound to the immutable claim, catalog, numerical source
commitments and operator-verifier source hash. Successfully completed partitions survive restart. A failed partial
partition runs in a fresh attempt directory; its earlier logs remain available.
Changing ledger deadlines or a local timeout does not invalidate completed work.
Changing the computation, reference, numerical sources or verifier does. Per-invocation reports
distinguish reused partitions and elapsed time from a new full replay.

A claim lock is inherited by the numerical subprocess. If the controller dies,
that child retains the lock until it exits, preventing a restarted controller
from launching an overlapping replay. A child result becomes reusable only after
the parent observes successful exit and durably records completion. The local
progress directory is trusted operator state, not a cache of strangers' reports.

The standard full-replay worker also supports the native quorum profile for
existing evolution graphs. Legacy sponsor-selected audit genesis files retain
their old behavior; they do not silently acquire this security model.

A live reservation or candidate currently occupies one global training cursor.
Unavailable inputs cannot earn a positive verdict, but can delay other work
until expiry. Refunds avoid treating infrastructure failure as proven fraud;
they also leave reservation capture insufficiently priced. Funded replay alone
does not solve this admission/liveness problem. Public rollout still needs
artifact-readiness requirements and a reviewed reservation/offer policy, rather
than assuming the native voting threshold prevents queue capture.

The remaining rollout work includes independently owned validator participation,
priced audit capacity, public artifact distribution, fresh-job activation and
integration of quality-approved growing checkpoints with serving. A single
operator running several keys cannot demonstrate decentralization of ownership.
