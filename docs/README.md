# Protocol and development documentation

| Start here | Purpose |
| --- | --- |
| [Live LLM checklist](../TODO.md) | Six fixed completion goals, the active milestone and evidence of completion. |
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
| [Native continuation integration](NATIVE_CONTINUATION_RESULTS.md) | The same learning recipe settles 64 updates, promotes only after quality replay and serves an answer paid from earned tokens; full ledger replay passes. |

This directory contains technical documentation and pinned dependency profiles. Manuscripts, publication figures and raw experiment dumps are outside the tracked tree. The [published paper](https://neuroshard.com/papers/FINE2026_neuroshard_short.pdf) remains available. Historical evidence is linked to revision `108b4ba3d6c6fb5760ff211b447ee95a67fa9112`, preserving access without mixing generated outputs into the current checkout.

`eval/data/input.txt` is a deliberate exception: the small licensed Tiny Shakespeare corpus is a test and source-compatibility fixture used by the reference implementation. Its path and bytes are retained; its [license and digest](../THIRD_PARTY.md) are recorded. Network genesis/data manifests under [networks](../networks) are also required protocol inputs, not disposable training output.

Example settings and compact reproduction plans live under [config](../config). Run `python scripts/check_repository.py` after staging moves to check tracked-file boundaries and local Markdown links. CI also checks package contents so local archives, website files and manuscripts cannot enter a distribution.
