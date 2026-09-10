# From the reference chain to public NeuroShard

The intended product is a sovereign network whose participants train a shared model, serve it, and earn native rewards. The first reference chooses a concrete division of responsibility: NeuroShard-native consensus orders tasks and settlement; correctly executed neural work earns newly issued NEURO. This choice preserves useful-computation mining while giving the ledger a separately testable security assumption.

The [v2 experiment report](PROTOCOL_EXPERIMENTS.md) now records tests of earned-stake validator admission, both unbond windows, real consensus evidence, task collateral, paid inference, and numerical conformance across two different Xeon hosts. Its [protocol specification](PROTOCOL_CANDIDATE_V2.md) defines the supported lifecycle. Independent operators, broader hardware support, and economical verification of a complete training graph remain open gates; the manuscript still describes the earlier reference.

## Current implemented baseline — 0.4.0

The [LLM protocol](LLM_PROTOCOL.md) and [experiment records](LLM_EXPERIMENTS.md) extend the reference with a frozen 135M pretrained model, a 4,608-parameter trainable adapter, separate funded inference jobs, validation-gated serving, a minimal public client, browser payments and immutable S3 collection. A two-host native cycle earned rewards and spent them on the promoted model. The new live chain has its own genesis; old balances do not migrate. The requirements below for economical verification, robust evaluation and independent operators remain open.

## 1. Publish a falsifiable technical claim

The defensible initial claim is: **a prescribed model update, produced by mutually untrusted pipeline stages, can be independently reproduced and settled exactly once by a native blockchain.** The reference tests this claim on a small model with full replay and fixed local validators. It does not demonstrate economical decentralized LLM pretraining, public admission, or a new proof-of-work consensus algorithm.

Prepare the short manuscript, source revision, environment manifest, raw measurements, and one-command reproduction together. Invite external attempts to falsify the exact claims. Preserve unsuccessful attack tests and failed reproducibility cases as well as successful runs. Establish prior work against permissionless training, collaborative model sharding, and reproducible ML disputes; avoid claiming that any one of those ideas is new by itself.

Before selecting a publication venue or posting the concept publicly, complete the stronger experiment below and have the final manuscript reviewed. Venue selection and public posting are later decisions; nothing is published by the local demo.

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
