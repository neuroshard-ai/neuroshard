# Decentralized assistant — active TODO

**Updated September 27, 2026. Status: 0/6 milestones complete.**

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
- [ ] Declare a latency path (fewer or faster generations) and rerun development;
  open confirmation only after a complete development pass.
- [ ] Port Granite execution to the shard runtime and validate numerical/cache
  agreement and checkpoint recovery before distributing a passing candidate.

**Current status:** the canonical parent completes 9/24 workflows but fails
primitive qualification, so A1 remains open. After verified experience and 37
verified decision preferences, the added module completes 18/24 development
workflows (parent 9/24) with every parent success preserved and beats the 60×
larger update control; only the 180 s latency limit fails.
Contributors can join the existing CPU testnet or prepare reviewed demonstrations
now. That preview does not require all six assistant milestones to be complete.

Execution details and stopping rules: [verified-experience contract](docs/ASSISTANT_EXPERIENCE_LEARNING.md),
which supersedes the training and gate sections of the [workspace learning contract](docs/ASSISTANT_WORKFLOW_LEARNING.md).
No automatic training or admission follows a baseline. This is not a public assistant launch.

## Six completion criteria

- [ ] **A1 — Usable foundation and reproducible execution.** Pin weights,
  tokenizer, runtime and licenses; retain successful conversation, instruction
  and tool-use anchors; complete the workspace primitive baseline and process
  replay; publish feasible shard estimates. A1 no longer requires a third-party
  adapter to pass an unrelated quality benchmark. This is a prospective change:
  every failed reference remains failed, and no existing box is marked complete.
- [ ] **A2 — One useful learned capability in the complete assistant.** New module
  and automatic selection beat the unchanged parent and match the equal-data
  no-growth update on frozen development and fresh confirmation gates, while
  training a small fraction of its parameters. Per-answer preservation, actual
  training/serving costs and bounded latency are required. Tools supply execution,
  not hidden answers. Forced routing and retrieval-only gains do not satisfy this.
  Amended before any training; the previous "beat the update" margin moves to A3.
- [ ] **A3 — Repeated useful growth and an upgrade/consolidation.** Three successive
  accepted cohorts, cumulative retention and cross-capability tasks; at least one
  new capability and one upgrade. Separate modules must retain earlier cohorts
  better than repeated shared-weight updates under the same budget. Demonstrate a
  beneficial update or consolidation against keeping the previous system under a
  declared resource budget. This establishes bounded growth, not unlimited
  intelligence or a no-forgetting theorem.
- [ ] **A4 — Actual sharding and value from additional peers.** No execution worker
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

A1 baseline and A4 port preparation proceed together. A2 implementation proceeds
alongside the re-baseline; its execution follows a usable baseline. A3 follows a
passing learned capability. Recruitment for A5 can
proceed now. Native 0.4.0 stays separate until a complete candidate passes its
activation contract.

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
