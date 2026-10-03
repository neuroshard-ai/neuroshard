# Decentralized assistant — active TODO

**Updated October 3, 2026. Status: 3/6 milestones complete (A1, A2, A4).**

Build one useful conversational assistant that can learn new capabilities from
contributed data, execute across independently owned machines, and pay for useful
work through NeuroShard's native ledger. More peers first provide memory,
replicas, training and audits. Add parameters only when measured benefit pays
for the extra cost; keep each answer's active computation bounded.

The [architecture amendment](docs/ASSISTANT_ARCHITECTURE.md) defines the current
direction. The older [demonstration checklist](TODO.md) and all experiment reports
remain evidence, not readiness claims for this assistant.

## Current deliverable

- [x] Replace isolated checker qualification with a complete-assistant contract.
- [x] Implement a reusable document/tool/draft workspace with multi-turn execution,
  transcript replay, outcome scoring and separate evaluation goals.
- [x] Freeze parent, no-growth update and added-capacity comparisons; reserve
  separate training, integration, development and confirmation partitions.
- [x] Derive the Granite partition/storage/communication plan; identify native
  model semantics that the current Llama runtime does not implement.
- [x] Run the committed parent baseline. [Result](docs/ASSISTANT_WORKFLOW_BASELINE_RESULTS.md):
  2/24 workflows, 18/18 prior answers retained; baseline qualification failed.
  Conditional replays were skipped; the CPU allocation is retired.
- [x] Publish baseline outcomes and pin both successful workflows for retention.
- [x] Restore public testnet block production and settled training after the
  full-disk incident; make inference availability depend on a progressing ledger.
- [x] Implement [contributor onboarding](docs/CONTRIBUTOR_ALPHA.md): local signed
  multi-turn demonstrations, deterministic replay and explicitly reviewed training
  export. This is data preparation, not evidence of learned improvement.
- [x] Diagnose the failed baseline. Every Granite execution encoded prompts with
  transformers 5.5.4's GPT-2 pre-tokenizer instead of the checkpoint's
  `tokenizer.json`; 39 of 51 tool errors are misspelled tool names. The
  [canonical re-baseline](docs/ASSISTANT_WORKFLOW_CANONICAL.md) checks every encode.
- [x] Run the canonical re-baseline. [Result](docs/ASSISTANT_WORKFLOW_CANONICAL_RESULTS.md):
  9/24 workflows (was 2/24), 7/16 compound, zero tool errors, p95 177 s, 19/24
  anchors with all 18 prior successes retained; qualification failed at 2/8
  primitive. Remaining failures: version choice by listing position and the
  six-generation budget. Host retired; $0.86 compute.
- [x] Implement the [verified-experience method](docs/ASSISTANT_EXPERIENCE_LEARNING.md):
  verified self-generated conversations, coached practice distilled without the
  coaching, parent-answer replay, both trained arms, a success-rate selector, the
  development/confirmation gates and CPU evaluation with checkpoint upload.
- [x] Run collection, training and integration on one GPU host (user-authorized
  despite the failed primitive gate; A1 stays open). [Result](docs/ASSISTANT_EXPERIENCE_GPU_RESULTS.md):
  722 verified trajectories; on 64 integration episodes the parent completes 34
  greedily, the update and addition arms 42 each; $6.31 across all attempts.
- [x] Run CPU development evaluation of both routed systems. [Result](docs/ASSISTANT_EXPERIENCE_DEVELOPMENT_RESULTS.md):
  failed. Addition 14/24, update 15/24, parent 9/24; net +5 and parity with the
  update pass, but the total, one lost protected success and p95 latency (189 s
  with routing) fail. Confirmation stays sealed.
- [x] Run the declared decision-preference round. [Result](docs/ASSISTANT_EXPERIENCE_ROUND2_RESULTS.md):
  37 verified version-choice pairs raise integration to 55/64 for both arms (parent
  34, round 1 42); older-revision-first cases 22% to 76–79%.
- [x] Run development for the round-2 systems. [Result](docs/ASSISTANT_EXPERIENCE_DEVELOPMENT_ROUND2_RESULTS.md):
  failed on latency only. Addition 18/24 (parent 9/24, update 17/24), net +9 with
  no lost success, +1 over the update; p95 192 s against 180 s (189.8 s without
  routing). Confirmation stays sealed.
- [x] Declare a latency path and rerun development. [Result](docs/ASSISTANT_EXPERIENCE_DEVELOPMENT_CACHED_RESULTS.md):
  the prefix cache cuts p95 to 102.7 s, but the addition flips one fragile
  follow-up (17/24, one lost success); gate failed.
- [x] Round 3 (verified divergence preferences; [result](docs/ASSISTANT_EXPERIENCE_ROUND3_RESULTS.md)):
  only 2 pairs as declared, yet the addition reaches 58/64 integration episodes.
  [Development passed](docs/ASSISTANT_EXPERIENCE_DEVELOPMENT_ROUND3_RESULTS.md) under
  prefix-cache serving: 18/24, net +9, no lost success, p95 96.5 s.
- [x] Confirmation on the 96 sealed episodes, opened once. [Result](docs/ASSISTANT_EXPERIENCE_CONFIRMATION_RESULTS.md):
  failed on one check. Addition 88/96 (parent 49/96, update 86/96), net +39 with
  no lost parent success, p95 98.9 s; the lower 95% gain versus the update is
  −5.2 points against the −5 margin. A2 remains open for this capability.
- [ ] Second attempt ([declaration](docs/ASSISTANT_EXPERIENCE_LEARNING.md#second-attempt-goal-guided-repairs-and-fresh-confirmation)):
  round 4 repairs the addition's wrong document reads on training cases and keeps
  scorer-verified continuations; development under the prefix cache adds the A1
  served-system check. If both pass, a fresh 192-episode confirmation opens once.
  Budget ceiling: $1,000. [Round 4](docs/ASSISTANT_EXPERIENCE_ROUND4_RESULTS.md):
  49 verified repairs gave 15 pairs. On integration the update rose to 90.6%
  while the addition fell to 89.1% greedy and 84.8% sampled.
  [Development and the A1 served-system check passed](docs/ASSISTANT_EXPERIENCE_DEVELOPMENT_ROUND4_RESULTS.md):
  the addition solved 18/24 (update 19, parent 9) with p95 92.7 s, and the
  served system solved 7/8 primitive workflows with exact fresh-process replays.
  [The fresh 192-episode confirmation failed on 2 of 9 checks](docs/ASSISTANT_EXPERIENCE_CONFIRMATION2_RESULTS.md).
  The addition solved 174/192 (update 182, parent 106): +68 over the parent with
  a lower bound of +24.5 points. It lost 2 parent successes and trails the update
  at −7.3 points against the −5 margin. A2 remains open for this capability, and
  A1's served-system condition needs an accepted module, so A1 stays open too.
- [x] [Methodology study](docs/ASSISTANT_EXPERIENCE_LEARNING.md#methodology-study-before-a-third-attempt)
  on already-opened data. It compares a larger module (about 50M parameters)
  with a consensus of three small modules grown from disjoint verified
  experience, voting with the parent on every action, against the small module
  and the update. A declared rule picks the third attempt's method. Its fresh
  192-episode confirmation split is frozen before the study reports.
  [Neither method was carried forward](docs/ASSISTANT_EXPERIENCE_STUDY_RESULTS.md)
  ($5.18). Scores: update 74, small 71, large 69, committee 65. The committee
  alone lost no parent success, but its members, each trained on a third of the
  experience, solved only 47–56 of 64 integration cases (small 61).
- [x] [Grow verified experience](docs/ASSISTANT_EXPERIENCE_GROWTH.md) about
  threefold on two fresh 256-case training splits, collected in parallel with
  the round-1 recipe. Then compare a committee of members grown from disjoint
  thirds with one small module on the whole pool. The candidate enters the third
  attempt only if it loses no parent success on opened data.
  The first collection attempt stopped at verification because growth splits
  were not accepted as training goals ($7.01); a second attempt fixes only that.
  [The second attempt collected 702 and 672 verified trajectories](docs/ASSISTANT_EXPERIENCE_GROWTH.md#collections)
  ($9.88), growing the pool from 740 to 2114 sequences. [The growth study found no
  qualifying candidate](docs/ASSISTANT_EXPERIENCE_GROWTH_RESULTS.md) ($4.15). The update control rose to
  80 with no parent success lost; the small module scored 74 but lost 2 parent
  successes on development cases; the committee scored 59. No third attempt.
- [x] [Compositional practice](docs/ASSISTANT_EXPERIENCE_COMPOSE.md): the small
  module lost parent successes only on development cases whose correction adds
  an instruction it never saw combined that way. Two 240-case practice splits add
  one extra operation per case, in family and operation pairs that neither the
  development nor any confirmation split holds out. The small module enters the
  third attempt only if it then loses no parent success on opened data.
  [Both collections completed](docs/ASSISTANT_EXPERIENCE_COMPOSE.md#collections) with 596 and 592 verified
  trajectories ($9.04); the pool holds 3302 sequences. [The study found no third
  attempt](docs/ASSISTANT_EXPERIENCE_COMPOSE_RESULTS.md) ($2.25): the small module scored 73 (update 77)
  and lost 3 parent successes, all development cases whose correction adds
  "move the resulting due date one calendar day later".
- [x] [Development diagnostic](docs/ASSISTANT_EXPERIENCE_DIAGNOSTIC.md): serve the
  compositional study's update and small module unrouted on the 24 opened
  development cases on the canonical CPU hosts, keeping full transcripts, to see
  how the small module fails the added instruction and to measure the latency of
  the serving path a third attempt would use. No gate; the split stays sealed.
  [Result](docs/ASSISTANT_EXPERIENCE_DIAGNOSTIC_RESULTS.md) ($1.07): latency passes (p95 98.5 s,
  1.02x the update). In every failed latest case the module applies the review
  interval twice, starting from start plus interval; it does follow the added
  instruction. A check for dates never seen in the conversation flags 3 of 8
  failures and none of 35 successes.
- [x] [Verified date-base repairs](docs/ASSISTANT_EXPERIENCE_REPAIRS.md): sample the
  compositional small module on 528 training cases, repair date shifts that start
  from a base the round cannot justify, keep only verified continuations, and
  continue both arms on them. The third attempt runs only if the small module
  then loses no parent success on opened data.
  [Failed as declared](docs/ASSISTANT_EXPERIENCE_REPAIRS_RESULTS.md) ($4.90): only 9 of 190 training
  failures matched the error, each after an earlier wrong read, and no repair
  verified. The double-interval error appears only with the held-out correction.
- [x] [Third attempt](docs/ASSISTANT_EXPERIENCE_THIRD.md): A2 is amended so that a
  bounded, separable update is the learning unit. The round-4 update system, the
  control that solved 182/192 on the spent second confirmation with no parent
  success lost, is the candidate, unchanged. Its served version must pass the
  development gate and the A1 served-system check on CPU. Only then does the
  sealed 192-episode `confirmation3` split open, once. Budget ceiling: $40.
  [Development and the A1 served-system check passed](docs/ASSISTANT_EXPERIENCE_THIRD_RESULTS.md)
  ($0.58): 19/24 against the parent's 9, no parent success lost, 8/8 primitive
  workflows, p95 94.8 s, and every episode identical to the round-4 run.
  [The sealed confirmation passed](docs/ASSISTANT_EXPERIENCE_THIRD_RESULTS.md) ($7.10):
  183/192 against the parent's 119, no parent success lost, a lower 95% gain of
  +22.4 points and p95 103.0 s. A2 is established for this capability.
- [ ] [Repeated growth (A3)](docs/ASSISTANT_REPEATED_GROWTH.md): drafting is cohort 1;
  cohort 2 adds meeting scheduling in a calendar workspace that keeps every drafting
  tool, and cohort 3 will be an upgrade. Cohort 2 trains a separate update, a
  separate low-rank module and the shared update under one budget; the separate
  units must retain drafting better. Stage 0 first serves the opened development
  cases under the new interface on CPU, to confirm room to learn and to measure
  what the larger tool list does to drafting. Stage 1 is declared after it.
  [Stage 0](docs/ASSISTANT_REPEATED_GROWTH_STAGE0_RESULTS.md) ($2.15): scheduling is
  new, with the parent at 1/24 and the accepted version at 0/24, so there is room to
  learn. The larger tool list alone lowered drafting from 19 to 15 for the accepted
  version (parent 9 to 5); those four must be regained before cohort 2 is accepted.
  [Stage 1 is declared](docs/ASSISTANT_REPEATED_GROWTH_STAGE1.md): each user turn is
  routed to a unit with its own tool set, so drafting keeps the tools it was accepted
  with. The accepted version collects coached scheduling experience. One continued
  update serves the separate-update and shared versions; a low-rank module trains on
  the same mixture. Per-turn selectors are fitted from integration success rates.
  The development gate is 18/24 scheduling and 6/8 cross with no drafting success
  lost; the sealed confirmation decides acceptance and A3's comparison. Ceiling $100.
- [x] Port Granite execution to the shard runtime and validate numerical/cache
  agreement and checkpoint recovery before distributing a passing candidate.
  [Closed with A4](docs/A4_SHARDING_REVIEW.md).
  Port done: Granite owners hold only their layer ranges. On small checkpoints
  they are bit-identical to the complete model and token-identical to
  `generate`, and a killed owner resumes to the same tokens. The
  [first shard execution passed](docs/GRANITE_SHARD_EXECUTION_RESULTS.md):
  three owner hosts that each fetched only their own tensors (2.2–2.4 of 6.8 GB)
  reproduced all 230 canonical generations token for token. Each stayed at
  6.1–6.2 GB peak RSS against 11.3 GB on one host, and a lost owner resumed to
  the canonical tokens; $4.93. Training across owners reproduces the
  single-host trainer bit for bit on small checkpoints.
  [Shard training on the real model](docs/GRANITE_SHARD_TRAINING_RESULTS.md)
  matched a single-host reference bit for bit: losses, margins and all 16 LoRA
  tensors. The run still failed recovery, because the outage run's first forward
  pass differed in the last place from the uninterrupted run on the same hosts
  ($1.89). First-pass reproducibility needs a diagnostic before training
  recovery is repeated. [Shard serving passed](docs/GRANITE_SHARD_SERVING_RESULTS.md).
  The complete learned assistant (parent, round-4 addition on owner 2, gate,
  prefix cache) reproduced all 24 single-host development episodes token for
  token, at p95 101.0 s against 92.7 s on one host and ≤5.2 GB per owner ($1.75).
  Fresh processes computed identical first passes, so the training-recovery
  difference remains open. The [throughput execution failed](docs/GRANITE_SHARD_THROUGHPUT_RESULTS.md).
  Three episodes in flight kept every token identical but ran 0.97× as fast as
  one at a time ($3.09); owner 0's per-episode threads contended for CPU.
  [The second attempt passed](docs/GRANITE_SHARD_THROUGHPUT_RESULTS.md) with one
  compute thread on owner 0. Three episodes in flight gave 2.25× the throughput
  of one at a time and 2.20× the single host, with every token identical and
  owner busy time balanced within 3% ($2.20). The [ring determinism diagnostic](docs/GRANITE_SHARD_DETERMINISM.md)
  localized the recovery failure. In one launch of six, owner 0's first forward
  pass in a fresh process rounded differently, while every later pass matched
  ($0.92). [With a declared warm-up, twelve fresh launches were identical](docs/GRANITE_SHARD_DETERMINISM.md)
  ($1.76). [Shard training recovery then passed on its second attempt](docs/GRANITE_SHARD_TRAINING_RESULTS.md).
  After the arm's owner was lost at step 3 and relaunched from its checkpoint,
  training still finished bit-identical to the single-host reference, with all
  seven checks passing ($1.94). [Audited serving is declared](docs/GRANITE_SHARD_AUDIT.md).
  Owners 1 and 2 sign logs of their serving work, and two light auditors, each
  holding one owner's shard, replay those logs. In a second pass, owner 1 flips
  one bit of one declared message. Its auditor must name that message, a fresh
  verifier must accept the fraud proof, and owner 2 must audit clean. The first
  attempt failed at fetch because the remote runtime lacked the signing package
  ($0.32). [The second attempt passed all six checks](docs/GRANITE_SHARD_AUDIT_RESULTS.md).
  Each auditor held only its owner's 2.2 GB and replayed all 8830 logged forward
  passes exactly, at 0.85–0.89× the owner's busy time. The one-bit fault was
  named at the declared message after 11.5 s, a fresh verifier accepted the
  fraud proof, and owner 2 audited clean ($7.19). The raw logs and proof
  exceeded the evidence cap and were not archived.

**Current status:** the assistant has learned its first useful capability.
On fresh sealed workspace episodes, the learned update lifts it from 62% to 95%
with no parent success lost. A1, A2 and A4 are complete. A3, repeated growth, has
[begun](docs/ASSISTANT_REPEATED_GROWTH.md) with scheduling as its second cohort.

- **A2 complete.** The 1M-parameter added module twice came within 4–8 episodes
  of the 63M-parameter update without meeting the declared parity margin, and
  four studies found no low-rank candidate that kept every parent success. A2 was
  [amended](docs/ASSISTANT_EXPERIENCE_THIRD.md) so that any bounded, separable
  update is the learning unit. The unchanged round-4 update then
  [passed the sealed third confirmation](docs/ASSISTANT_EXPERIENCE_THIRD_RESULTS.md):
  183/192 against the parent's 119, every family at least 21/24, p95 103.0 s.
- **A1 complete.** The served version passed A1's development checks, and its
  learned unit is now accepted. A [review](docs/A1_FOUNDATION_REVIEW.md) maps every
  clause to published evidence.
- **A4 complete.** The pinned 3B assistant runs across owner machines that each
  fetch only their own tensors. Generation and training match the complete model
  bit for bit, and the learned assistant is served across owners exactly as on
  one host. Extra machines add pooled memory, recovery and 2.25× throughput. A
  [review](docs/A4_SHARDING_REVIEW.md) maps every clause to published evidence;
  all owners ran in one availability zone under one operator.
- **Toward A5.** Light auditors verified real-model serving and proved a one-bit
  fault. A [bonded optimistic serving ledger](docs/OPTIMISTIC_SERVING.md) now settles
  honest jobs after a challenge window and slashes an owner on a verified fraud
  proof. Two validator processes agree on small checkpoints; it is not yet on a
  chain or run by independent operators. [Settlement on the real assistant is
  declared](docs/GRANITE_SHARD_SETTLEMENT.md): six hosts, each party signing its own
  transactions, two validators replaying the same blocks with real proof checks. The
  first attempt failed at bootstrap: the module lacked the functions bootstrap calls
  ($0.33). The second failed at bond signing: torch loaded before the runtime was
  configured ($0.44). [The third attempt passed all eight checks](docs/GRANITE_SHARD_SETTLEMENT_RESULTS.md)
  ($7.57): the honest job settled after its window, the framing was rejected, owner
  1 was slashed by the real proof, and both validators reached the same state root
  with exactly the declared balances. The ledger now runs as an ABCI application
  behind CometBFT: four local validators holding shard 1 of the small checkpoint
  settled the same scenario through consensus and agreed exactly. [Settlement of
  real-model serving through CometBFT is declared](docs/GRANITE_SHARD_CHAIN.md): four
  validators on separate hosts, each holding only shard 1, agreeing on every block. The
  first attempt passed by consensus through the honest settlement, then lost the
  reply to the auditor's challenge to CometBFT's 11 s RPC write timeout ($10.72).
  [The second attempt passed all eight checks](docs/GRANITE_SHARD_CHAIN_RESULTS.md)
  ($10.81): the honest job settled by consensus, the framing was refused at
  admission, owner 1 was slashed by the real proof, and all four validators
  reached the same state root with exactly the declared balances. One operator
  still ran every host; this is not independent operation and gives no checklist
  credit. Before untrusted participants join, [the protocol was revised](docs/OPTIMISTIC_SERVING.md).
  Each owner log is now bound to its paid request through signed serving links,
  so a reused or relabelled log is refused or provably fraud. A proof bundle a
  validator lacks gives no verdict rather than a cached rejection, and challenges
  pass every cheap admission check before any replay. A review then found that
  owners committing one message of a session were paid the whole price, so
  settlement is now [metered](docs/OPTIMISTIC_SERVING.md#metered-settlement): a job
  buys a budget of token positions, each owner is paid for the positions its
  upstream signatures attest, and the user gets the rest back. A challenge now
  [locks a deposit](docs/OPTIMISTIC_SERVING.md#challenges-and-deposits) that is
  burned unless its proof verifies, and validators replay proofs in the background
  rather than at admission. Small-checkpoint and four-validator tests pass. The
  real-model runs above used the earlier protocol.
Contributors can join the existing CPU testnet or prepare reviewed demonstrations
now. That preview does not require all six assistant milestones to be complete.

Execution details and stopping rules: [verified-experience contract](docs/ASSISTANT_EXPERIENCE_LEARNING.md),
which supersedes the training and gate sections of the [workspace learning contract](docs/ASSISTANT_WORKFLOW_LEARNING.md).
No automatic training or admission follows a baseline. This is not a public assistant launch.

## Six completion criteria

- [x] **A1 — Usable foundation and reproducible execution.** [Complete](docs/A1_FOUNDATION_REVIEW.md). Pin weights,
  tokenizer, runtime and licenses; retain successful conversation, instruction
  and tool-use anchors; complete the workspace primitive baseline and process
  replay; publish feasible shard estimates. A1 no longer requires a third-party
  adapter to pass an unrelated quality benchmark. This is a prospective change:
  every failed reference remains failed, and no existing box is marked complete.
- [x] **A2 — One useful learned capability in the complete assistant.** [Established](docs/ASSISTANT_EXPERIENCE_THIRD_RESULTS.md)
  for the workspace capability. A bounded,
  separable learned update and automatic selection beat the unchanged parent on
  frozen development and fresh confirmation gates, while training a small fraction
  of the backbone's parameters. The update's tensors are stored apart from the
  frozen backbone and applied only when selected; serving the parent rolls it
  back. A low-rank module qualifies, and so does an update of declared
  projections. Per-answer preservation, actual training/serving costs and bounded
  latency are required. Tools supply execution, not hidden answers. Forced routing
  and retrieval-only gains do not satisfy this. Amended before any training; the
  previous "beat the update" margin moves to A3. [Amended again on October 2, 2026](docs/ASSISTANT_EXPERIENCE_THIRD.md),
  after both confirmations and before the third sealed split opened: the unit was
  "a new module" that had to match the equal-data update. Whether low-rank modules
  can match updates now joins A3's retention comparison.
- [ ] **A3 — Repeated useful growth and an upgrade/consolidation.** Three successive
  accepted cohorts, cumulative retention and cross-capability tasks; at least one
  new capability and one upgrade. Separate modules must retain earlier cohorts
  better than repeated shared-weight updates under the same budget. Demonstrate a
  beneficial update or consolidation against keeping the previous system under a
  declared resource budget. This establishes bounded growth, not unlimited
  intelligence or a no-forgetting theorem.
- [x] **A4 — Actual sharding and value from additional peers.** [Complete](docs/A4_SHARDING_REVIEW.md). No execution worker
  holds the complete backbone. Measure forward/backward/generation agreement,
  per-owner memory and traffic, outage recovery and a benefit from extra machines
  (pooled memory, throughput, training capacity or availability). Include placement,
  communication and failed work; throughput is not single-request latency.
- [ ] **A5 — Independent operation and funded settlement.** Four independently
  administered operators under the [hosting contract](docs/INDEPENDENT_HOSTING.md),
  separate keys, whole-graph admission/promotion, rejection/refund/recovery and
  funded honest auditing. Publish collusion assumptions and the verification bill.
  More AWS instances owned by us do not provide independent ownership.
- [ ] **A6 — Sustainable public assistant.** Versioned chat/tool/memory interface,
  consent and private-data boundaries, join/recovery guides, monitored quality,
  resource limits, funding, rollback and a public operating soak. Requires A1–A5;
  neither a research pass nor a token transaction substitutes for usability.

A1, A2 and A4 are complete. A3 follows the passing learned capability and can
begin now: it needs new capability cohorts, cumulative retention, and the
comparison of separate units with repeated shared-weight updates, now including
whether low-rank modules can match updates. Recruitment for A5 can proceed now.
Native 0.4.0 stays separate until a complete candidate passes its activation
contract.

## Evidence carried forward

- [Granite reference](docs/GRANITE_REFERENCE_RESULTS.md): parent 18/24; published
  adapter retained 18 anchors but failed its reference-regression gate.
- [Adapter audit](docs/GRANITE_ADAPTER_AUDIT_RECOVERY_RESULTS.md): implementation
  agreement confirmed; the quality failure is real, so integration debugging is closed.
- [Context reference](docs/GRANITE_CONTEXT_REFERENCE_RESULTS.md) and
  [answerability reference](docs/GRANITE_ANSWERABILITY_REFERENCE_RESULTS.md): closed
  failures. Correct evidence plus an added checker did not preserve complete behavior.
- [Evidence selection](docs/EVIDENCE_SELECTION_RESULTS.md): useful opened-set
  primitive, not learned capability or a replacement for complete conversations.
- All Granite results above were measured with transformers 5.5.4's GPT-2
  pre-tokenizer rather than the checkpoint's `tokenizer.json`
  ([details](docs/ASSISTANT_WORKFLOW_CANONICAL.md)). The pinned Switch checkpoint's
  own `tokenizer.json` also serializes GPT-2 splitting, so Switch runs now use the
  parent's splitting with the Switch control-token IDs preserved
  ([details](docs/ASSISTANT_WORKFLOW_CANONICAL_RESULTS.md#switch-checkpoint-tokenizer)).
  Their outcomes stand as published; they do not measure the models under their
  trained tokenization.
- Earlier learning, growth, sharding and settlement reports remain indexed in
  [the documentation](docs/README.md). Their successes and failures remain unchanged.
