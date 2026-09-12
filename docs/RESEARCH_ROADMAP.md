# From the reference chain to public NeuroShard

The intended product is a sovereign network whose participants train a shared model, serve it, and earn native rewards. The first reference chooses a concrete division of responsibility: NeuroShard-native consensus orders tasks and settlement; correctly executed neural work earns newly issued NEURO. This choice preserves useful-computation mining while giving the ledger a separately testable security assumption.

The [scaling design](SCALING_DESIGN.md) now specifies the architectural direction and release gates. Additional peers should first cover auditing, artifact replication, serving and training throughput; parameter growth follows sustained capacity and quality evidence. The [compact optimizer dispute](COMPACT_UPDATE_DISPUTES.md) is the first implemented bounded refutation primitive, with a complete-replay fallback.

The [v2 experiment report](https://github.com/neuroshard-ai/neuroshard/blob/108b4ba3d6c6fb5760ff211b447ee95a67fa9112/docs/PROTOCOL_EXPERIMENTS.md) records tests of earned-stake validator admission, both unbond windows, real consensus evidence, task collateral, paid inference, and numerical conformance across two different Xeon hosts. Its [protocol specification](PROTOCOL_CANDIDATE_V2.md) defines that reference lifecycle. The [revised working paper](https://neuroshard.com/papers/FINE2026_neuroshard_short.pdf) now covers the subsequent full-model experiments and distinguishes their results from the public release.

## Current implemented baseline — 0.4.0

The [LLM protocol](LLM_PROTOCOL.md) and [experiment records](LLM_EXPERIMENTS.md) extend the reference with a frozen 135M pretrained model, a 4,608-parameter trainable adapter, separate funded inference jobs, validation-gated serving, a minimal public client, browser payments and immutable S3 collection. A two-host native cycle earned rewards and spent them on the promoted model. The new live chain has its own genesis; old balances do not migrate. The requirements below for economical verification, robust evaluation and independent operators remain open.

The separate [model-evolution implementation](EVOLUTION_PROTOCOL.md) trains all 134.5M parameters, grows depth to 148.7M parameters, collects fresh/replay windows and evaluates research candidates. Its native application settles training and bonded growth claims, handles objective fraud/availability challenges and prevents repeated payment for the same prescribed computation. The [native lifecycle](NATIVE_LIFECYCLE.md) now connects rolling data, challengeable evaluation, serving decisions and bounded generation on isolated networks; its [results](NATIVE_LIFECYCLE_RESULTS.md) include unsuccessful quality experiments. The [funded candidate](FUNDED_AUDITING.md) adds prepaid complete replay and a [recovering operator](CANDIDATE_OPERATIONS.md). Public worker admission, independent audit selection, artifact retention and release integration remain gates before replacing the released network.

The next executable research contract is the [learning milestone](LEARNING_MILESTONE.md): a public, fail-closed plan for useful learning, then continual learning, then a two-host reliability measurement. It does not replace the longer-term gates below.

## 1. Publish a falsifiable technical claim

The defensible initial claim is: **a prescribed model update, produced by mutually untrusted pipeline stages, can be independently reproduced and settled exactly once by a native blockchain.** The reference tests this claim on a small model with full replay and fixed local validators. It does not demonstrate economical decentralized LLM pretraining, public admission, or a new proof-of-work consensus algorithm.

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
