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
  The pinned GPU check now passed complete prefix execution and updates 0–8,
  with actual retained boundaries, forged-measurement rejection and missing-data
  refusal. Graph quality promotion and paid inference now have a candidate
  implementation with local five-process execution checks. Next: operate the
  complete native job and its GPU graph serving integration. Reuse the published
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
  yet make every intermediate tensor payload available to a new auditor. The new
  executor has now durably retained and read back its first two window boundaries.

- [ ] **6. Usable chat service**

  Done when the accepted growing graph serves ordinary multi-turn requests with
  streaming, versioned context/tokenizer handling, bounded prices, correct
  expiry/refunds and concurrent clients. Meet a published, prospectively frozen
  latency/load target on supported deployments. State and test prompt visibility,
  storage and privacy behavior; users must understand which providers or auditors
  receive their data. Publish representative responses and failure behavior.

  Evidence so far: the public 0.4.0 client supports a smaller experimental model;
  the larger expert graph has operated generation and public reproducible weights.
  The new candidate binds and audits all neural calls used by a raw question,
  including interpretation and composition. Those pieces are not yet the live
  growing chat service.

## Work discipline

All six milestones are in scope. Task 3 currently determines execution order;
work on billing, availability and serving supports the other fixed milestones
without creating additional top-level tasks.

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
- 2026-09-16: The real-job preflight caught an identifier-domain mismatch missed
  by the small earlier-cohort fixture. Stopped the driver before neural execution,
  fixed preservation of the original interpreted-cohort job, and verified all
  three real claim contexts. Twelve executor checks passed in 37.58 seconds.
  The replacement probe retains the same model, data, success criteria, GPU and
  spending deadline; the first attempt remains recorded.
- 2026-09-16: The bounded GPU backend probe passed in 854.73 seconds after setup.
  Complete prefix production, updates 0–8, forged measurements and unavailable
  input handling met the frozen criteria. Retained and hash-read-back artifacts
  total 5.98 GB. The GPU and attached resources were retired; estimated compute
  was $0.65 plus storage. No NEURO was issued and no graph was activated.
- 2026-09-16: Added the candidate graph lifecycle, complete neural-call billing,
  real five-owner serving/audit execution, and a quality auditor that measures
  new answers while checking unchanged retained computations. The latter checked
  all 2,048 original retained records in 1.87 seconds without another GPU run.
  Native full-job settlement and GPU serving integration remain open. See
  [implementation and evidence boundaries](https://github.com/neuroshard-ai/neuroshard/blob/research/native-expert-graphs/docs/NATIVE_EXPERT_SERVING.md).
- 2026-09-16: Froze the complete native integration in
  [native-expert-live.json](https://github.com/neuroshard-ai/neuroshard/blob/research/native-expert-graphs/config/experiments/native-expert-live.json).
  All 140 original window contexts and four published serving traces pass the
  metadata preflight. The fresh four-validator genesis boots with zero issuance.
  The bounded run assigns actual training to the expert owner and replay to
  three auditors, retains every accepted boundary before pruning local copies,
  and checks separate quality promotion and inference paid from earned NEURO.
  Allowance: five GPUs, 28 vCPUs, six hours, $75 planning cap, automatic shutdown.
  Numerical success and completion of task 3 are not yet claimed.

- 2026-09-16: The complete integration stopped during serving startup, before
  generation or training. The runtime guard caught a different CuDNN library
  selected by the background launcher. After correcting that path, the smaller
  owner exhausted the remaining startup interval under file-cache pressure.
  All five GPUs and volumes were retired; estimated compute was $1.85. The
  [recorded failure](https://github.com/neuroshard-ai/neuroshard/blob/research/native-expert-graphs/config/experiments/native-expert-live-results.json) remains
  visible. A [same-target retry](https://github.com/neuroshard-ai/neuroshard/blob/research/native-expert-graphs/config/experiments/native-expert-live-retry.json)
  carries the original library path and releases clean unowned cache before
  serving. Its deadline and combined $75 cap include the first attempt.

- 2026-09-16 02:13 UTC: The corrected five-GPU integration is running under
  the original 07:50 UTC stop deadline and combined $75 cap. The first attempt
  retired every GPU and attached volume. Implementation remains isolated in
  [draft PR #50](https://github.com/neuroshard-ai/neuroshard/pull/50), preserving
  the source commitments of the existing public chain.
