# Live LLM checklist

NeuroShard's goal is one useful, openly available assistant whose learning and
serving capacity can grow through contributed model shards, coordinated by its
own permissionless blockchain.

This is the fixed six-item completion checklist. **0/6 complete. Task 3 is active.**
A passing experiment counts as supporting evidence; a top-level box is checked
only when all of its completion criteria have a committed implementation and
reproducible evidence. Changes to these criteria must be recorded explicitly,
before the affected work, rather than changing the target after a result.

- [ ] **1. Intelligence across growing shards**

  Done when ordinary questions automatically select and combine relevant shards,
  including questions requiring knowledge from different learned experts; at
  least three successive admitted learning cohorts pass prospectively frozen
  response-quality and retention gates. Include a comparison against spending
  the same total resources without growth, and a declared rule for adding,
  updating or consolidating capacity. Evaluation must cover broader assistant
  tasks as well as specialized knowledge, without supplying answer labels to
  routing. More stored parameters alone do not satisfy this item.

  Evidence so far: two specialized experts, improved held-out answers, exact
  measured retention, and no complete backbone on any one owner. Routing and
  the two-question composition grammar remain explicit. See the
  [published learning result](https://github.com/neuroshard-ai/neuroshard/releases/tag/research-cohorts-20260915).

- [ ] **2. Continuous data admission and learning**

  Done when the network repeatedly admits immutable data with provenance,
  deduplication and contamination/poisoning checks; funds and assigns a bounded
  training job; verifies the work; evaluates the candidate on precommitted
  acceptance rules; and promotes or rejects it without manually editing genesis
  or orchestrating each cohort. Demonstrate restart, rejected data and a rejected
  candidate while the accepted serving model remains available.

  Evidence so far: immutable ingestion and experimental lifecycle components
  exist. The growing expert architecture still uses fixed prepared jobs.

- [ ] **3. Live native training-to-inference lifecycle — ACTIVE**

  Done when the already validated second expert completes this entire path on
  an operated NeuroShard-native candidate chain: available numerical inputs and
  checkpoints; real configured execution audits; accepted training claims;
  exactly-once NEURO rewards; a separate quality decision admitting the serving
  graph; and a raw-question inference request paid from earned NEURO. Bind the
  actual model, optimizer ages, tokenizer, routing and composed-call billing.
  Publish the ledger and execution evidence. Restart/replay, a forged claim,
  missing data and a repeated claim must leave supply and serving state correct.

  Evidence so far: exact prefix production and all 560 updates were replayed;
  140 bounded windows and native settlement guards pass. This expert has not
  earned NEURO or been activated for native paid serving. Historical fixed-model
  integration is supporting evidence, not completion of this expert lifecycle.

  Implementation progress: the native-format training executor now loads actual
  weights and Adam, executes a bounded window, and atomically saves a resumable
  boundary. A separate-process CPU check reproduces the next checkpoint. See
  [execution and checkpoint details](https://github.com/neuroshard-ai/neuroshard/blob/research/native-expert-graphs/docs/NATIVE_EXPERT_CHECKPOINTS.md).
  The backend also recomputes and retains the complete prefix production.
  Next: check this backend on the pinned GPU profile, then connect native claims,
  quality promotion and paid graph inference. Reuse the published
  model and original training job; this milestone does not need a new training
  campaign.

- [ ] **4. Reliable permissionless shard hosting**

  Done when independent providers can discover work, acquire assigned shards,
  join, leave and replace unavailable providers through the protocol. Demonstrate
  required-backbone-shard loss, coordinator loss, state recovery and continued
  service for the growing graph across separate failure domains and supported
  hardware/network profiles. Complete a declared load/availability soak with
  independently administered operators; no one operator may control one third
  or more of consensus voting power in the demonstrated release deployment.

  Evidence so far: operated sharding, earlier host-replacement experiments and
  continued older serving after the newest expert exited. The latest five-owner
  experiment used one operator and controlled cloud networking.

- [ ] **5. Affordable verification and sustainable incentives**

  Done when every accepted obligation has available evidence, complete funded
  verification and a bounded liability; adversarial tests cover forged work,
  duplicate work, withholding and failed audit obligations. Publish the quorum
  and collusion assumptions and measure complete costs, including verification,
  retention, storage and serving. Configured fees or finite sponsorship must
  cover those measured obligations within the declared issuance limits. Signatures
  and token issuance alone are not evidence of independent computation or demand.

  Evidence so far: supply invariants, work deduplication, funded native replay
  quorums and real replay measurements. Intermediate replay commitments do not
  yet make every intermediate tensor payload available to a new auditor.

- [ ] **6. Usable chat service**

  Done when the accepted growing graph serves ordinary multi-turn requests with
  streaming, versioned context/tokenizer handling, bounded prices, correct
  expiry/refunds and concurrent clients. Meet a published, prospectively frozen
  latency/load target on supported deployments. State and test prompt visibility,
  storage and privacy behavior; users must understand which providers or auditors
  receive their data. Publish representative responses and failure behavior.

  Evidence so far: the public 0.4.0 client supports a smaller experimental model;
  the larger expert graph has operated generation and public reproducible weights.
  Those pieces are not yet the live growing chat service.

## Work discipline

Only one milestone is active. Its necessary dependencies stay attached to that
milestone; they do not create additional top-level tasks.

Before an expensive run, record its exact decision, reused artifacts, cheapest
adequate preflight, success/failure rule and time/spending cap. A rerun needs an
identified cause and a concrete change. Do not repeat long quality evaluation
when the relevant model, inputs, numerical path and acceptance decision are
unchanged. Real execution required by a new audit path remains a separate claim
that must be checked.

Every update records delivered behavior, evidence, remaining blockers and
resources consumed. Failed attempts stay visible. A release meeting all six
items is the live-LLM milestone; future quality and capacity improvements continue
under the same admission rules.

## Progress log

- 2026-09-15: Established the six fixed items from the agreed live-LLM gaps.
  All remain open; task 3 is active. Prior expert learning, retention, replay and
  public artifact evidence are credited above. No new GPU run was needed to
  establish this checklist.
- 2026-09-15: Task 3 implementation now has actual bounded training execution and
  recoverable window-boundary payloads. Thirteen focused execution and native
  settlement regression checks passed in 37.55 seconds, including a fresh process
  restoring Adam and completing the next window, and rejection of corrupted
  state, storage failure and an expired deadline. The 11 original numerical
  kernel files remain unchanged. No GPU was launched. Native settlement of this
  expert, prefix-backend integration, quality promotion and paid graph inference
  are still required; task 3 remains open.
- 2026-09-16: Full CI passed for the resumable training backend. Added complete
  native prefix execution with timing-independent production commitments and
  retained feature bytes. Sixteen focused checks passed in 56.26 seconds.
  The next GPU probe is frozen around the existing parent and updates 0–8,
  with one GPU, a two-hour limit and a $15 planning cap. All six milestones
  remain in scope; dependencies determine execution order and none is complete.
