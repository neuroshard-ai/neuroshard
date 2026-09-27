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
- [ ] Run the committed parent baseline: 24 workflows plus prior assistant anchors,
  two conditional process replays, one CPU allocation, two hours / $6 allowance.
- [ ] Publish baseline outcomes and pin every successful workflow for retention.
- [ ] Implement the declared training and gate stages, validate trajectories and
  memory, commit their execution inventory, then compare the three complete arms.
- [ ] Port Granite execution to the shard runtime and validate numerical/cache
  agreement and checkpoint recovery before distributing a passing candidate.

Execution details and stopping rules: [workspace learning contract](docs/ASSISTANT_WORKFLOW_LEARNING.md).
No automatic training or admission follows the baseline. This is the first
implementation of that contract, not a public assistant launch.

## Six completion criteria

- [ ] **A1 — Usable foundation and reproducible execution.** Pin weights,
  tokenizer, runtime and licenses; retain successful conversation, instruction
  and tool-use anchors; complete the workspace primitive baseline and process
  replay; publish feasible shard estimates. A1 no longer requires a third-party
  adapter to pass an unrelated quality benchmark. This is a prospective change:
  every failed reference remains failed, and no existing box is marked complete.
- [ ] **A2 — One useful learned capability in the complete assistant.** New module
  and automatic selection beat unchanged parent and no-growth update on frozen
  development and fresh confirmation gates, with per-answer preservation, actual
  training/serving costs and bounded latency. Tools supply execution, not hidden
  answers. Forced routing and retrieval-only gains do not satisfy this criterion.
- [ ] **A3 — Repeated useful growth and an upgrade/consolidation.** Three successive
  accepted cohorts, cumulative retention and cross-capability tasks; at least one
  new capability and one upgrade. Demonstrate a beneficial update or consolidation
  against keeping the previous system under a declared resource budget. This
  establishes bounded growth, not unlimited intelligence or a no-forgetting theorem.
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

A1 baseline and A4 port preparation proceed together. A2 execution follows a
usable baseline; A3 follows a passing learned capability. Recruitment for A5 can
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
- Earlier learning, growth, sharding and settlement reports remain indexed in
  [the documentation](docs/README.md). Their successes and failures remain unchanged.
