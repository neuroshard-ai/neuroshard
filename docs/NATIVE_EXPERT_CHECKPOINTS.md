# Preserve expert checkpoints in native metadata

The existing native portable profile describes a single model with a shared
update cursor. The learned experts retain parent parameters at Adam age 448 while
their own tails reach independent ages such as 1024 or 560. Rewriting those ages
would misrepresent the computation.

`expert_checkpoint` encodes only the active tail and a reference to the immutable
parent. It derives every frozen parameter record and every shard commitment, then
requires the reconstructed checkpoint and learned-state roots to match the exact
numerical checkpoint. A claim cannot supply a changed frozen tensor. The codec
validates parameter coverage, shapes, optimizer recipe, actual ages, ownership and
bounded update windows without importing PyTorch, Transformers or Safetensors.

The [archived-checkpoint check](../config/experiments/native-expert-checkpoint-results.json)
round-trips the first expert at step 1024 and the second at steps 0, 280 and 560.
Each complete checkpoint has 218 tensor references; the compact representation has
18. The terminal second checkpoint shrinks from 39,463 to 3,683 JSON bytes while
reconstructing its original identity. Tensor payloads are not compressed by this
operation.

Update identity binds the consumed numerical batch, parameter object identities,
optimizer recipe, actual cursor and numerical execution profile. It excludes job,
worker and serving-graph labels, so renaming a job cannot make the same prescribed
update payable again. The execution auditor must derive the batch root from the
canonical numerical inputs, excluding descriptive document metadata. This is not
a test for arbitrary mathematical equivalence across serialization or numerical
profiles.

The existing portable checkpoint validator was moved unchanged into a module
without neural imports and remains available through its previous import path.
Native portable settlement and lifecycle regression tests pass.

The opt-in `expert_work` genesis profile connects those commitments to the existing
native ledger and its funded, stake-weighted replay quorum. Its fixed job specifies
an immutable parent, fresh expert checkpoint, prepared input commitment, feature
bank, complete batch schedule and numerical profile. The protocol first requires
an accepted prefix-production claim. Only then can a worker reserve and claim up
to four prescribed expert updates. Claims include every intermediate commitment;
the ledger derives work identities and rejects paid work, omitted states, changed
batches and mismatched worker receipts.

Both production and training require complete native quorum verdicts. Missing or
negative verdicts cannot advance the expert or issue currency. Input admission
issues no currency; accepted training pays its actual tail owner. Serving remains
the genesis parent throughout this profile. General job activation, serving-graph
admission, quality promotion and paid graph inference still require integration.
This profile has passed ledger tests using actual small-model replay records. It
has not been activated on a public network or settled the real 560-update expert.

The [second-cohort learning gate](COMPOSED_COHORT.md) has now passed, and its
[model artifacts are public](../config/experiments/composed-cohort-public-artifacts.json).
This closes the model-side prerequisite for an operated integration of this
specific expert. The next required run must execute the prefix audit and each
training window through a real configured executor, settle the accepted work,
then separately admit the passing serving graph and pay for a raw-question
response. Its available checkpoint bytes, actual optimizer ages, composed-call
billing and serving identity must remain bound throughout. The published replay
reports alone cannot supply honest execution verdicts or activate serving.

The audit daemon requires an operator-configured execution backend for this
profile. A backend must execute the pinned numerical audit, return complete ordered
coverage, and bind the claim, parent, input/output checkpoints, prepared data,
feature bank and numerical profile. Passing a miner's report file through the
binding checker does not establish execution. The security assumption remains an
honest native voting quorum; these receipts are not cryptographic neural proofs.

The [bounded training executor](../src/neuroshard/evolution/sharded/expert_execution.py)
now implements the training side of that interface. It loads the configured
parent, immutable training records and feature bank, then restores the actual
input tail and its Adam state. It executes one to four prescribed updates at
their original global learning-rate cursors and checks every intermediate root
and bound measurement. A valid window atomically publishes its final weight/Adam
payloads; another process can continue from that boundary without replaying the
earlier training history. Existing output files do not replace computation on a
repeated audit. Missing or corrupted input, failed storage and an expired deadline
cannot produce an acceptance report. Intermediate steps remain commitments;
recoverable tensor payloads are stored at the audited window boundaries.

The module can run as the audit worker's local subprocess backend:
`python -m neuroshard.evolution.sharded.expert_execution --config /absolute/executor.json`.
Its input is the candidate JSON on standard input; its output is the bound native
replay report. The local configuration has format `neuroshard-expert-executor-v1`,
the chain's `profile`, the original `plan` and `prepared` objects, `max_seconds`,
and absolute `paths` for `inputs`, `objects`, `bank_home` and `checkpoint_store`.
Each checkpoint directory is keyed by its complete checkpoint root and contains
`checkpoint.json` plus the existing incremental owner payload format. The audit
worker's backend configuration supplies this command as an `argv` array and a
hard `timeout_seconds`. Keep the reviewed source and runtime pinned; configuration
and artifact paths must come from the operator, never from an untrusted claim.

The [CPU execution checks](../tests/evolution/test_expert_execution.py) reuse the
actual five-process small-model training fixture. They cover a second process
resuming the next window, exact output weights and Adam, repeated execution,
forged measurements, missing/corrupted inputs and failures during persistence or
at the deadline. This is a tested training backend, not a completed native
deployment: its GPU compatibility still needs a bounded check. Graph quality
promotion and paid graph inference remain unimplemented in this profile.

The same subprocess backend now also executes prefix-production claims. It
recomputes all three parent partitions sequentially, retains all resulting
feature payloads, and checks a canonical production record that excludes timing.
The CPU check feeds those retained outputs directly into the training executor;
changed production records and false feature roots are refuted. Prefix verdicts
accept or reject the complete production claim; they do not attribute fraud to
an individual partition owner. The
[bounded GPU probe](../config/experiments/native-expert-execution.json) freezes
336 prefix stages and two four-update windows against the existing numerical
trajectory, with an additional forged-measurement and missing-boundary check.
It permits one GPU for at most two hours within a $15 planning cap.

The second expert has durable weight/Adam checkpoints at 0/280/560. Reconstructing
the intervening commitments does not make their tensor payloads durably available.
An operated settlement must provide those bytes or maintain an actual replay
executor across consecutive windows. A missing execution backend cannot supply an
honest acceptance verdict.

The [window replay plan](../config/experiments/expert-window-replay.json) freezes a non-issuing reconstruction of all 560 updates from the archived feature bank. Each intermediate weight/Adam commitment uses the same Safetensors bytes as a normal checkpoint; the replay records 140 windows of at most four updates and checks the actual saved 0/280/560 roots. Its CPU test reproduced the full five-owner training result. The prefix has now received the separate execution audit described below.

The [first GPU attempt](../config/experiments/expert-window-replay-attempt-0.json)
matched the first 25 updates but was stopped early because checkpoint recording
could not finish the full trajectory within its frozen deadline. Its observations
are preserved and the instance, disk and security group retired. The replacement
uses a streaming encoder for the same restricted float32 Safetensors bytes,
avoiding large temporary byte strings. Compatibility tests compare its hashes and
lengths with the pinned writer, including ordinary complete weight/Adam
checkpoints. The numerical training kernel, data, update identity and runtime are
unchanged. The new plan additionally requires all 25 recorded intermediate GPU
checkpoints to match. The [complete GPU replay](../config/experiments/expert-window-replay-results.json)
passed all 560 updates and the
original final checkpoint in 1,188.22 seconds. All 140 windows and 560 distinct
work identities also passed the consensus-side metadata validator without
importing a neural runtime. Numerical reproduction uses the frozen source
`2cac1006c7685be000699c2c8f792fc6e28208d8`. Its complete evidence was archived and
read back successfully, and its GPU, disk and security group were retired.

The [prefix audit plan](../config/experiments/expert-prefix-audit.json) freezes the
other half of that verification: recomputing every cached prefix and parent
reference from the exact training records. Each of three stages loads one parent
partition and consumes the previous stage's committed output. The final stage
must reproduce the entire feature-bank identity consumed by training. The CPU
test matches the actual five-process producer and rejects a changed prefix even
when an attacker recomputes its file and manifest hashes. The
[GPU audit passed](../config/experiments/expert-prefix-audit-results.json): all
112 microbatches were recomputed through each of the three original parent
partitions, and the complete feature bank matched exactly. One GPU staged one
partition at a time, with at most 604,016,640 owned model parameters. Its numerical
source is commit `a9b19cec807ff0eaf606ce8a849cbd3f5e31be44`; reproducing this frozen
plan requires checking out that source. The evidence is archived with a complete
storage readback, and the GPU, volume and security group have been retired.
These are execution tools for a configured auditor; report files alone are not
proof, and neither tool enables native transactions or changes supply.
