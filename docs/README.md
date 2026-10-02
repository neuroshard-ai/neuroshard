# Protocol and development documentation

Start with the [contributor preview](CONTRIBUTOR_ALPHA.md) to run a testnet worker
or submit a reproducible assistant correction. The public CPU chain's September 27
[recovery](NETWORK_RECOVERY_20260927.md) restores existing small-model operation;
the newer assistant is not yet active on that chain.

The active [assistant checklist](../TODO_ASSISTANT.md) follows the
[complete-assistant architecture](ASSISTANT_ARCHITECTURE.md). The next deliverable
is the [workspace learning contract](ASSISTANT_WORKFLOW_LEARNING.md): a complete
parent conversation baseline, followed by trained update and added-capacity
comparisons with automatic selection and individual retention. A Granite-specific
shard port is required before activation. All six milestones remain open. The [first workspace baseline](ASSISTANT_WORKFLOW_BASELINE_RESULTS.md)
completed at 2/24, retaining all 18 prior assistant answers but failing primitive
quality and latency. Every Granite execution, including that one, encoded prompts
with the wrong pre-tokenizer; the [canonical re-baseline](ASSISTANT_WORKFLOW_CANONICAL_RESULTS.md)
reaches 9/24 with every encode checked but still fails primitive qualification. Training follows the
[verified-experience contract](ASSISTANT_EXPERIENCE_LEARNING.md): the model learns
from its own conversations that the deterministic workspace scorer accepts.

The earlier [Granite reference](GRANITE_REFERENCE_RESULTS.md),
[context reference](GRANITE_CONTEXT_REFERENCE_RESULTS.md) and
[answerability comparison](GRANITE_ANSWERABILITY_REFERENCE_RESULTS.md) remain
closed with their published failures. The [adapter audit](GRANITE_ADAPTER_AUDIT_RECOVERY_RESULTS.md)
confirmed implementation agreement. Repeatedly qualifying that adapter is no
longer a prerequisite for a new learning method; no failed score is reclassified.

[Preserved interpretation](PRESERVED_INTERPRETER.md) accesses a learned neural
expert while retaining original instruction-following weights. The
[four-owner final passes](PRESERVED_INTERPRETER_RESULTS.md): 949/1,024 newly
worded knowledge answers, with all earlier outputs and losses reproduced exactly.

[Frozen feature reuse](FROZEN_FEATURES.md) develops repeated training of an
added shard without recomputing the immutable distributed prefix each time.

[Knowledge rehearsal](KNOWLEDGE_REHEARSAL.md) isolates repeated exposure while
retaining the existing generated-answer and retention requirements.

| Start here | Purpose |
| --- | --- |
| [Assistant development plan](../TODO_ASSISTANT.md) | Active six-milestone plan: capable foundation, useful modular learning, repeated growth, sharding, independent operation and public release. |
| [Architecture amendment](ASSISTANT_ARCHITECTURE.md) / [workflow contract](ASSISTANT_WORKFLOW_LEARNING.md) | Current complete-assistant direction, controls, execution and shard requirements. |
| [Canonical re-baseline](ASSISTANT_WORKFLOW_CANONICAL.md) / [result](ASSISTANT_WORKFLOW_CANONICAL_RESULTS.md) | Tokenizer defect in all Granite executions. Corrected parent: 9/24 workflows, zero tool errors, primitive qualification failed on version choice and the call budget. |
| [Verified-experience learning](ASSISTANT_EXPERIENCE_LEARNING.md) / [GPU result](ASSISTANT_EXPERIENCE_GPU_RESULTS.md) | A2 method: scorer-verified self-generated conversations, coached practice, parent-answer replay and success-rate selection. Integration: parent 34/64, both arms 42/64. [Development](ASSISTANT_EXPERIENCE_DEVELOPMENT_RESULTS.md): 14/24 addition, failed. [Round 2](ASSISTANT_EXPERIENCE_ROUND2_RESULTS.md): verified decision preferences, integration 55/64; [development](ASSISTANT_EXPERIENCE_DEVELOPMENT_ROUND2_RESULTS.md) 18/24 with no lost success, failed on latency only; [cached serving](ASSISTANT_EXPERIENCE_DEVELOPMENT_CACHED_RESULTS.md) p95 102.7 s but 17/24. [Round 3](ASSISTANT_EXPERIENCE_ROUND3_RESULTS.md): [development passed](ASSISTANT_EXPERIENCE_DEVELOPMENT_ROUND3_RESULTS.md), 18/24, no lost success; [confirmation](ASSISTANT_EXPERIENCE_CONFIRMATION_RESULTS.md) 88/96 (parent 49/96), failed on the update non-inferiority bound. Second attempt declared: goal-guided repairs, A1 served-system check, fresh 192-episode confirmation. [Round 4](ASSISTANT_EXPERIENCE_ROUND4_RESULTS.md): 15 verified repair pairs; integration update 90.6%, addition 89.1% greedy / 84.8% sampled; [development and A1 served check passed](ASSISTANT_EXPERIENCE_DEVELOPMENT_ROUND4_RESULTS.md) (18/24, 7/8 primitive); [second confirmation](ASSISTANT_EXPERIENCE_CONFIRMATION2_RESULTS.md) failed: addition 174/192, update 182, parent 106; 2 lost parent successes and −7.3 points vs update. [Methodology study](ASSISTANT_EXPERIENCE_STUDY_RESULTS.md): neither the larger module (69) nor the committee (65) beat the small module (71; update 74), so neither is carried forward; the committee alone lost no parent success. [Experience growth](ASSISTANT_EXPERIENCE_GROWTH.md) declared: two fresh training splits collected in parallel, then committee members on disjoint thirds of about three times the experience against one small module on all of it. [Result](ASSISTANT_EXPERIENCE_GROWTH_RESULTS.md): no third attempt; update 80 with no parent success lost, small 74 but 2 lost on development, committee 59. [Compositional practice](ASSISTANT_EXPERIENCE_COMPOSE.md) declared: training instructions with one extra operation, in pairs no evaluation split holds out. [Result](ASSISTANT_EXPERIENCE_COMPOSE_RESULTS.md): no third attempt; small 73 with 3 lost parent successes, update 77. [Development diagnostic](ASSISTANT_EXPERIENCE_DIAGNOSTIC.md) / [result](ASSISTANT_EXPERIENCE_DIAGNOSTIC_RESULTS.md): CPU latency passes; the small module applies the review interval twice when chaining date shifts, and a check for never-seen dates flags 3 of 8 failures and no successes. [Date-base repairs](ASSISTANT_EXPERIENCE_REPAIRS.md) / [result](ASSISTANT_EXPERIENCE_REPAIRS_RESULTS.md): failed as declared; the error does not occur on training cases, so no repair verified. |
| [Bonded optimistic serving](OPTIMISTIC_SERVING.md) | Owner bonds, signed log commitments, a challenge window and slashing on verified fraud proofs; validators replay only challenged proofs. Tested end to end on small checkpoints with two agreeing validators; not yet on chain. [Real-model settlement](GRANITE_SHARD_SETTLEMENT.md) / [result](GRANITE_SHARD_SETTLEMENT_RESULTS.md): passed on the third attempt; honest serving of the 3B assistant paid after its window, a framing rejected, a cheating owner slashed by a real proof, two validators agreeing exactly. |
| [Verifiable network design](VERIFIABLE_NETWORK_DESIGN.md) | How exact sharded execution becomes a decentralized assistant: BFT ledger for small facts, Raft-style owner logs, optimistic replay audits by light auditors with fraud proofs, user-held first stage for privacy, and behavioral consensus of modules for growth. |
| [Granite shard execution](GRANITE_SHARD_EXECUTION.md) / [result](GRANITE_SHARD_EXECUTION_RESULTS.md) | A4 first execution passed: three owner hosts, each fetching only its own tensors, reproduced all 230 canonical generations token for token at 6.1–6.2 GB peak RSS each (11.3 GB on one host), and a lost owner resumed to the canonical tokens. [Shard training](GRANITE_SHARD_TRAINING.md) / [result](GRANITE_SHARD_TRAINING_RESULTS.md): the addition arm trained across owners equals a single-host reference bit for bit; after a warm-up fix, recovery from a lost owner is exact too. [Shard serving](GRANITE_SHARD_SERVING.md) / [result](GRANITE_SHARD_SERVING_RESULTS.md): the complete learned assistant served across three owners reproduced all 24 single-host development episodes token for token. [Throughput](GRANITE_SHARD_THROUGHPUT.md) / [result](GRANITE_SHARD_THROUGHPUT_RESULTS.md): first attempt 0.97× (failed); second attempt 2.25× over one-at-a-time serving (2.20× over the single host) with every token identical. [Ring determinism](GRANITE_SHARD_DETERMINISM.md): owner 0's first forward pass in a fresh process occasionally rounds differently; with a declared warm-up, twelve fresh launches were identical. [Audited serving](GRANITE_SHARD_AUDIT.md) / [result](GRANITE_SHARD_AUDIT_RESULTS.md): passed on the second attempt. Light auditors holding one shard each replayed all 8830 signed forward passes exactly at 0.85–0.89× the owner's compute; a one-bit fault was named at the declared message, its fraud proof accepted by a fresh verifier, and the honest downstream owner audited clean. |
| [Modular reference](MODULAR_REFERENCE.md) | Historical BAR contract; baseline failed and decoder agreement was confirmed. |
| [Tool-interface diagnostic](MODULAR_TOOL_INTERFACE.md) / [result](MODULAR_TOOL_INTERFACE_RESULTS.md) | Corrected instructions yield 3/3 calls and three exact replays on opened cases; no A1 credit. |
| [Fresh reference comparison](MODULAR_REFERENCE_FRESH.md) | 24 new cases; paired gains/losses, exact replay and latency on one temporary CPU host. |
| [Earlier demonstration checklist](../TODO.md) | Original six criteria and bounded evidence; independent hosting remains open. |
| [Programming expert trial](PROGRAMMING_EXPERT_TRIAL.md) | One new neural skill, executable answers, learned selection, preserved general responses and bounded serving latency. |
| [Programming fallback comparison](PROGRAMMING_FALLBACK_COMPARISON.md) / [result](PROGRAMMING_FALLBACK_RESULTS.md) | Research baseline: leftover extra attempt +4/32 versus parent and parent repair; not promoted. |
| [Second-capability growth](PROGRAMMING_GROWTH.md) | Independent second tail plus heuristic integration; campaign closed. |
| [TIES composition](PROGRAMMING_GROWTH_TIES.md) | Preserved 29 incumbent successes, scored 31/64 including leftover 54; failed declared gate. Stopped. |
| [Elect-sign composition](PROGRAMMING_GROWTH_ELECT.md) | Same tails, sign-election without trim; matched TIES 31/64. Stopped. |
| [Feedback-status selector](PROGRAMMING_SELECTOR_V4.md) | Added only on parent extraction-error; 29/64, unique added 0/3. Stopped. |
| [Learned integration](LEARNED_INTEGRATION.md) | Last-layer expert and gate versus matched no-expansion control. Method frozen. |
| [Stage-1 execution freeze](LEARNED_INTEGRATION_EXECUTION.md) | CPU 135M last-layer train/score loop frozen. No GPU. |
| [Stage-1 learned-integration result](LEARNED_INTEGRATION_RESULTS.md) | Expansion 0/32 vs control 5/32. Failed development. Confirmation closed. |
| [Staged integration](STAGED_INTEGRATION.md) / [baseline result](STAGED_INTEGRATION_RESULTS.md) | Stopped before training: 0/32 under the frozen exact-string, eight-token output rule. |
| [Staged answering](STAGED_ANSWERING.md) / [execution result](STAGED_ANSWERING_RESULTS.md) | Baseline protected 15 answers; timed out after 127/128 updates before candidate scoring. No GPU. |
| [Staged execution recovery](STAGED_ANSWERING_RECOVERY.md) / [result](STAGED_ANSWERING_RECOVERY_RESULTS.md) | Completed and rejected: expansion 0/32 versus control 2/32; 11 of 15 protected answers lost. Interrupted and repeated work charged; candidate stopped. |
| [Block expert competence](BLOCK_EXPERT.md) / [baseline result](BLOCK_EXPERT_RESULTS.md) | Stopped before training: parent 6/64 retention answers, below eight required. Added blocks have no learning-quality result. |
| [Block-expert measurement](BLOCK_EXPERT_MEASURE.md) / [result](BLOCK_EXPERT_MEASURE_RESULTS.md) | Added blocks 10/64, control 11/64, both lost 8/8 protected answers. Stopped. |
| [Append-only execution result](APPEND_ONLY_EXECUTION_RESULTS.md) | Oracle gates passed; selector read the hidden answer. Constant label also scores 9/64. Not deployable. |
| [Observable selection](OBSERVABLE_SELECTION.md) | Next rule: choose a shard without the gold answer and beat the training-label baseline. Not trained. No GPU. |
| [Observable reasoning experiment](OBSERVABLE_REASONING.md) / [result](OBSERVABLE_REASONING_RESULTS.md) | Completed and rejected: added blocks 7/96, trained control 10/96, constant 16/96. Task routing worked; no qualifying protected baseline. |
| [Independent hosting](INDEPENDENT_HOSTING.md) | Item 4 soak freeze: equal-power CPU genesis and stranger-provider join. No GPU. |
| [Neural-work research](NEURAL_WORK_RESEARCH.md) / [results](NEURAL_WORK_RESULTS.md) | Exact linear training with challenge-bound intermediate mining tickets, adversarial checks and full-cost measurements; experimental, no consensus activation. |
| [Retired alpha guide](JOIN_ALPHA.md) | Pinned source, ledger observation and historical setup; GPU service closed September 20, 2026. |
| [Alpha deployment result](OPERATED_ALPHA_RESULT.md) | Measured latency, automatic recovery, public evidence, funding and remaining limits. |
| [Operated alpha](OPERATED_ALPHA.md) | Atomic funded admission, provider maintenance, automatic recovery and the distinction between AWS hosts and independent operators. |
| [Provider LLM service](PROVIDER_LLM_SERVICE.md) | Frozen accepted-model concurrency, streaming, recovery and complete-cost trial, including failed setups. |
| [Provider runtime](PROVIDER_RUNTIME.md) | Native discovery, assigned-partition restoration, authenticated execution and recovery. |
| [Hosted chat](HOSTED_CHAT.md) | Lightweight client, complete price caps, provisional streaming, settled conversation history and public-data limits. |
| [Finite sponsorship](FINITE_SPONSORSHIP.md) | Complete learning/serving cost scope, replay-quorum assumptions and bounded research funding. |
| [Continual admission results](CONTINUAL_ADMISSION_RESULTS.md) | Three prospective admitted cohorts, automatic continuation, retained answers, an equal-resource growth comparison and complete native replay; checklist items 1 and 2 complete. |
| [Public testnet](PUBLIC_TESTNET.md) | Install the client, join, earn test NEURO and request inference. |
| [LLM protocol](LLM_PROTOCOL.md) | Supported training, payments, consensus and model-serving rules. |
| [Model card](MODEL_CARD.md) | Capabilities, limits, evaluation and provenance. |
| [API](API.md) | Native ledger and inference interfaces. |
| [Data pipeline](DATA_PIPELINE.md) | Immutable ingestion, collection and verification. |
| [Node deployment](DEPLOYMENT.md) | Pinned runtimes, service templates, recovery and monitoring. |
| [Model evolution](EVOLUTION_PROTOCOL.md) | Experimental full-model training, growth, verification and reproduction. |
| [Text protocol](TEXT_PROTOCOL.md) | Model/tokenizer identity, response windows, document evaluation and text conformance. |
| [Native lifecycle](NATIVE_LIFECYCLE.md) | Curated fresh data, training, quality decisions and paid generation on isolated networks. |
| [Lifecycle evidence](NATIVE_LIFECYCLE_RESULTS.md) | Real-model integration, failed promotion, recovery and continued training. |
| [Compact optimizer disputes](COMPACT_UPDATE_DISPUTES.md) | Bounded SGD refutations, complete-audit requirements and measured costs. |
| [Funded auditing](FUNDED_AUDITING.md) | Prepaid complete replay, reporting windows, collateral, payments and refunds. |
| [Funded audit evidence](FUNDED_AUDIT_RESULTS.md) | Two-host execution, recovery, public peer connectivity, costs and failure analysis. |
| [Audit admission RFC](AUDIT_ADMISSION_RFC.md) | Proposed atomic reservations and refutation funding; not activated. |
| [Candidate operations](CANDIDATE_OPERATIONS.md) | Recovering operators and auditors, full-node joining and bootstrap ownership. |
| [Scaling design](SCALING_DESIGN.md) | Compute groups, audit funding, capacity-backed growth and release gates. |
| [Inherited ledger rules](PROTOCOL_CANDIDATE_V2.md) | Reference consensus, accounting, bonds and dispute assumptions. |
| [LLM experiments](LLM_EXPERIMENTS.md) | Measured outcomes, including failures, with historical evidence links. |
| [Research requirements](RESEARCH_ROADMAP.md) | Remaining conditions for a stronger public deployment. |
| [Learning milestone](LEARNING_MILESTONE.md) | Frozen useful-learning, continual-learning and two-host scaling contract. |
| [Learning result](LEARNING_MILESTONE_RESULTS.md) | Completed 128-step run, rejected quality gate, all generations and downloadable checkpoint. |
| [GPU learning reference](LEARNING_REFERENCE.md) | Complete-conversation full-model training, held-out evaluation, checkpoint recovery and a bounded GPU trial. |
| [GPU reference results](LEARNING_REFERENCE_RESULTS.md) | Completed 1.7B training, exact same-host recovery, inconclusive test gain, all answer pairs and target audit. |
| [Cooperative experiment](COOPERATIVE_LEARNING.md) | Frozen target-quality, two-GPU training and replicated-serving comparisons. |
| [Cooperative results](COOPERATIVE_LEARNING_RESULTS.md) | Measured task learning, shared parameter agreement, communication overhead and inference failover. |
| [Four-worker methods](LOCAL_TRAINING_WINDOWS.md) | Frozen local-window training, complete group checkpoints and replica-serving comparisons. |
| [Four-worker results](LOCAL_TRAINING_WINDOWS_RESULTS.md) | Reduced communication, exact process-crash recovery, failed quality preservation and a corrected compression feasibility probe. |
| [Shared-gradient learning](LEARNING_METHOD_STUDY_RESULTS.md) | Passed task/retention screen with lower communication; buffer equivalence and stronger batching controls. |
| [Batched comparison](BATCHED_LEARNING_STUDY_RESULTS.md) | Passed full-length quality/retention screen and 1.45× faster training; 38% more GPU seconds, fixed model size. |
| [Persistent model shards](SHARDED_TRAINING.md) | Train and generate through disjoint model partitions with complete optimizer/RNG checkpoints. |
| [Shard recovery results](SHARDED_TRAINING_RESULTS.md) | Physical-host replacement and exact recovery of a 1.7B model across two GPU shards. |
| [Adaptive shard experiment](ADAPTIVE_SHARDS.md) | Portable Adam state, reference regularization, redistribution and gated model growth. |
| [Adaptive shard results](ADAPTIVE_SHARDS_RESULTS.md) | Measured redistribution, complete GPU replay, native settlement and quality decisions. |
| [Native shard replay](NATIVE_SHARD_REPLAY.md) | Bond-weighted complete replay, sponsor-funded audits and bounded GPU update issuance. |
| [Portable jobs and serving](PORTABLE_LIFECYCLE_RFC.md) | Native recipe activation, separately audited quality approval and escrow-paid inference across model shards. |
| [Continued learning](CONTINUED_LEARNING.md) / [result](CONTINUED_LEARNING_RESULTS.md) | Three-shard continuation completed; generated-answer gate failed despite lower loss. |
| [Calculation-step learning](REASONED_LEARNING.md) / [result](REASONED_LEARNING_RESULTS.md) | Large generated-answer gain, failed sorting/filtering retention; all final answer pairs published. |
| [Weight consolidation](CONSOLIDATED_LEARNING.md) / [result](CONSOLIDATED_LEARNING_RESULTS.md) | Prior answers retained and large new-task gain; two sorting regressions still fail the final gate. |
| [Answer-balanced continuation](BALANCED_CONTINUATION.md) / [result](BALANCED_CONTINUATION_RESULTS.md) | Complete frozen gate passes: new answers 378→462, all 188 correct prior answers retained, three model shards. |
| [Preserved interpretation](PRESERVED_INTERPRETER.md) | Original-model interpretation combined with a learned expert; distributed quality evaluation. |
| [Second expert with interpretation](INTERPRETED_COHORT.md) | Separate learning owner, continued earlier serving and exact retention; execution requires the preceding final to pass. |
| [Native expert lifecycle result](NATIVE_EXPERT_LIVE_RESULT.md) | 560 distinct audited updates, 560 NEURO, separate graph promotion, earned-token inference and complete public ledger replay; checklist item 3 complete. |
| [Ordinary serving diagnostic](ORDINARY_SERVING_DIAGNOSTIC.md) | Inference-only planned-path screen on ordinary development questions; gold controls separate knowledge, selection, decomposition and assembly. Not a training, promotion or new-final result. |

This directory contains technical documentation and pinned dependency profiles. Manuscripts, publication figures and raw experiment dumps are outside the tracked tree. The [published paper](https://neuroshard.com/papers/FINE2026_neuroshard_short.pdf) remains available. Historical evidence is linked to revision `108b4ba3d6c6fb5760ff211b447ee95a67fa9112`, preserving access without mixing generated outputs into the current checkout.

`eval/data/input.txt` is a deliberate exception: the small licensed Tiny Shakespeare corpus is a test and source-compatibility fixture used by the reference implementation. Its path and bytes are retained; its [license and digest](../THIRD_PARTY.md) are recorded. Network genesis/data manifests under [networks](../networks) are also required protocol inputs, not disposable training output.

Example settings and compact reproduction plans live under [config](../config). Run `python scripts/check_repository.py` after staging moves to check tracked-file boundaries and local Markdown links. CI also checks package contents so local archives, website files and manuscripts cannot enter a distribution.

- [Compose the learned second expert](COMPOSED_COHORT.md): exact prompt preservation and actual two-call answers; frozen read-only experiment.

- [Native expert checkpoint representation](NATIVE_EXPERT_CHECKPOINTS.md): exact frozen ages, compact references and numerical work identity.
