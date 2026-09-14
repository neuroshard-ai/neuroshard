# From the reference chain to public NeuroShard

The intended product is a sovereign network whose participants train a shared model, serve it, and earn native rewards. The first reference chooses a concrete division of responsibility: NeuroShard-native consensus orders tasks and settlement; correctly executed neural work earns newly issued NEURO. This choice preserves useful-computation mining while giving the ledger a separately testable security assumption.

The [scaling design](SCALING_DESIGN.md) now specifies the architectural direction and release gates. Additional peers should first cover auditing, artifact replication, serving and training throughput; parameter growth follows sustained capacity and quality evidence. The [compact optimizer dispute](COMPACT_UPDATE_DISPUTES.md) is the first implemented bounded refutation primitive, with a complete-replay fallback.

The [v2 experiment report](https://github.com/neuroshard-ai/neuroshard/blob/108b4ba3d6c6fb5760ff211b447ee95a67fa9112/docs/PROTOCOL_EXPERIMENTS.md) records tests of earned-stake validator admission, both unbond windows, real consensus evidence, task collateral, paid inference, and numerical conformance across two different Xeon hosts. Its [protocol specification](PROTOCOL_CANDIDATE_V2.md) defines that reference lifecycle. The [revised working paper](https://neuroshard.com/papers/FINE2026_neuroshard_short.pdf) now covers the subsequent full-model experiments and distinguishes their results from the public release.

## Current implemented baseline — 0.4.0

The [LLM protocol](LLM_PROTOCOL.md) and [experiment records](LLM_EXPERIMENTS.md) extend the reference with a frozen 135M pretrained model, a 4,608-parameter trainable adapter, separate funded inference jobs, validation-gated serving, a minimal public client, browser payments and immutable S3 collection. A two-host native cycle earned rewards and spent them on the promoted model. The new live chain has its own genesis; old balances do not migrate. The requirements below for economical verification, robust evaluation and independent operators remain open.

The separate [model-evolution implementation](EVOLUTION_PROTOCOL.md) trains all 134.5M parameters, grows depth to 148.7M parameters, collects fresh/replay windows and evaluates research candidates. Its native application settles training and bonded growth claims, handles objective fraud/availability challenges and prevents repeated payment for the same prescribed computation. The [native lifecycle](NATIVE_LIFECYCLE.md) now connects rolling data, challengeable evaluation, serving decisions and bounded generation on isolated networks; its [results](NATIVE_LIFECYCLE_RESULTS.md) include unsuccessful quality experiments. The [funded candidate](FUNDED_AUDITING.md) adds prepaid complete replay and a [recovering operator](CANDIDATE_OPERATIONS.md). Public worker admission, independent audit selection, artifact retention and release integration remain gates before replacing the released network.

The [learning milestone](LEARNING_MILESTONE.md) supplies a public prepare/train/score contract. Its [completed 128-step experiment](LEARNING_MILESTONE_RESULTS.md) lowers mean response loss but fails the sealed-test confidence bound. Continual learning and the two-host reliability phase remain blocked. A follow-up requires a new committed plan and unused evaluation documents; the longer-term gates below also remain open.

The [GPU learning reference](LEARNING_REFERENCE.md) has [completed a 1.7B run](LEARNING_REFERENCE_RESULTS.md): conventional full-model AdamW training and exact same-host recovery work, but final-test gain remains inconclusive and answer regressions persist. The next comparison needs reviewed instruction targets, executable tasks and a new frozen holdout before translating a successful recipe into native distributed work. This reference does not change the rejected experiment, activate GPU consensus arithmetic or issue rewards.

The subsequent [cooperative experiment](COOPERATIVE_LEARNING_RESULTS.md) passes a separate narrow task-learning contract: clean targets improve exact answers from 23/256 to 169/256, and a shared two-GPU run reaches 168/256. Invoice arithmetic remains failed. Both ranks agree on parameters, but synchronous updates are 4.39× slower than one GPU; inference replicas deliver 1.67× aggregate throughput and survive a stopped provider with bounded retries. This result establishes neither economical training scale nor native permissionless GPU work.

The [four-worker follow-up](LOCAL_TRAINING_WINDOWS_RESULTS.md) reduces communication by 93.8% and recovers every rank exactly after a process crash, but the local-window candidate fails retention and the declared quality margin against DDP. Single-GPU and synchronized four-GPU controls learn the new narrow tasks with acceptable exposed retention. The later [shared-gradient study](LEARNING_METHOD_STUDY_RESULTS.md) passes a full-length quality screen with 91.4% less traffic than dense synchronization, but its initial implementation is slower than one GPU.

The [batched follow-up](BATCHED_LEARNING_STUDY_RESULTS.md) then trains the fixed 1.7B model 1.45× faster on two GPUs than an efficiently batched single GPU, while passing its fresh-task and exposed-retention screen. It uses 38% more allocated GPU seconds. Shared AdamW with compressed gradients is now the leading measured learning candidate; it still needs complete compression-state recovery, continual learning from the resulting checkpoint, broader quality evaluation and pooled model memory. Parameter growth and native integration stay gated on those results. More protocol surface is not the next experiment.

## 1. Publish a falsifiable technical claim

The [adaptive shard trial](ADAPTIVE_SHARDS_RESULTS.md) preserves a 1.7B model's complete learned state across a two-to-three-owner redistribution and reproduces the following updates exactly. Initial learning passes, but both later continuation candidates lose generated answers, and the 1.85B candidate fails its gain margin over the fixed-depth control. Growth mechanics work; useful growth remains unproven. A one-GPU auditor replays every partition sequentially; an isolated native chain rejects a forged checkpoint and settles the genuine window once. The [native quorum profile](NATIVE_SHARD_REPLAY.md) derives audit weight from validator bonds rather than sponsor-selected identities. This is still complete replay under one infrastructure owner. Reliable answer learning, public artifact distribution and independent ownership remain requirements.

The next learning experiment is the [continued-learning contract](CONTINUED_LEARNING.md): continue from the passing phase-A checkpoint at fixed size, freeze the actual parent/optimizer/tokenizer/runtime artifacts, and require generated-answer improvement with per-family retention on unused data. Native payment, if that method passes, is a later activated reserved-window replay rather than importing the existing checkpoint.

The defensible initial claim is: **a prescribed model update, produced by mutually untrusted pipeline stages, can be independently reproduced and settled exactly once by a native blockchain.** The small-model reference and the isolated 1.7B GPU trial exercise this claim with complete replay and local native validators. They do not demonstrate economical decentralized LLM pretraining, public admission, or a new proof-of-work consensus algorithm.

For each research claim, publish a matched manuscript, source revision, environment manifest, measurements and reproduction instructions. Keep executable code and compact input plans in this repository; host manuscripts and generated outputs separately with immutable references. Invite external attempts to falsify the exact claims. Preserve unsuccessful attack tests and failed reproducibility cases as well as successful runs. Establish prior work against permissionless training, collaborative model sharding, and reproducible ML disputes; avoid claiming that any one of those ideas is new by itself.

The source and working draft are public research artifacts. Formal venue submission should follow independent review and evidence for the stronger claims below; it is separate from publishing reproducible development work in the repository.

## 2. Make the reference survive independent machines

Run four validators under independent operators and at least two workers on separate machines. Freeze an execution profile, publish golden gradient/activation vectors, and test mismatched CPUs and libraries. Record accepted checkpoint roots, common-block certificates, and a common held-out evaluation. Test latency, interrupted pipeline stages, conflicting proposals, duplicated messages, restarted applications, and unavailable task data.

Exit criteria: every honest validator either derives the same prescribed state or rejects an incompatible task; no wrong or repeated claim is paid; progress resumes within documented bounds when the required resources return. A worker failure must lead to resumable or reassigned work from the last accepted checkpoint, without manufacturing a proxy gradient.

## 3. Move expensive verification off the consensus critical path

The full-replay reference is an oracle for correctness, with intentionally poor scaling. A production candidate should commit a task's execution graph and keep training completion separate from ordinary block production. Workers publish retrievable state before acceptance. Independent verifiers execute selected obligations; a dispute narrows to a bounded, reproducible operation. The chain settles only after the specified verification and availability rules finish.

The key experiment is **dependency-aware verification**: can the network check a disputed activation or adjoint at a lower total cost than redoing its entire history? Measure independent execution, retained states, retrieval bandwidth, dispute latency, false positives across hardware, and the cost of an adversary who submits maximally expensive disputes. A cheap final referee alone is insufficient.

Useful design options to test:

- Fix compatible training windows and numerical profiles; offer several task sizes so smaller devices can complete a well-defined obligation before its deadline.
- Pay separately for training, replay, and available state. Devices that cannot hold a model stage can contribute a protocol-verified service with a finite budget.
- Verify boundary dependencies before accepting downstream stage updates. Keep a complete job as the failure domain until stage attribution is justified.
- Introduce concurrency across independent tasks or replicas before introducing stale-parent updates. Specify the exact aggregation and optimizer restart policy for each synchronization point.

Continue using the reference as a differential oracle. A faster verifier earns adoption by accepting the same valid transitions and rejecting the tested invalid ones under explicit assumptions.

## 4. Specify open membership and economic security

A public network needs a published admission rule for each role, genesis resource distribution, consensus weight, activation and withdrawal delays, objective penalties, and bounded outstanding liabilities. Linear bonded voting power prevents identity splitting from creating extra consensus weight; it does not prevent ownership concentration. Track the latter explicitly.

Task assignment must resist cheap lease monopolization and reward copying while allowing newcomers to acquire resources. Evaluate a fixed task budget against reservation bonds, refundable fees, and value-proportional admission costs. Any bootstrap allocation must be documented and must not turn a permanent operator into an implicit admission authority.

For each accepted task, account for worker, verifier, availability, and consensus compensation inside an explicit emission/fee budget. Simulate low demand, expensive verification, withholding, majority resource concentration, and externally motivated sabotage. Correct execution and token issuance do not establish that the trained model is useful or that participants have positive real returns.

Exit criteria: no outstanding claim spends the same reserved collateral twice; all penalties have an objective adjudication path; the emission bound holds under adversarial replay and reconfiguration; participation and ownership assumptions are visible to prospective operators.

## 5. Run an explicitly experimental public testnet

Only after the prior gates, publish a genesis proposal and protocol constants ahead of the start. Distribute validator operation, document all allocations, make historical state independently retrievable, and rehearse software upgrades and recovery. Testnet rewards remain separate from any later public issuance decision.

Measure learning quality over enough steps and seeds to compare with centralized and cooperative baselines at equal total cost. Include pipeline traffic, all verifier work, data retrieval, storage, idle time, retries, and rejected work. A useful efficiency result must survive that accounting.

The milestone is independently operated, auditable training with measured costs and stated failure bounds. Token launch timing follows evidence that the network can sustain that behavior.
