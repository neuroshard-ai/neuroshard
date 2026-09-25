# Live LLM checklist

NeuroShard's goal is one useful, openly available assistant whose learning and
serving capacity can grow through contributed model shards, coordinated by its
own permissionless blockchain.

This is the fixed six-item completion checklist. **5/6 complete. Tasks 1, 2, 3 and 5 are complete against their bounded demonstration criteria; task 6 is complete for the funded operated alpha. Task 4 remains open.**

**Availability, September 20, 2026:** the operated GPU alpha is retired at the
owner's request. Its evidence and completed demonstration criteria remain;
the separate ledger stays scheduled through September 26 for expiry/refunds.
See the [retirement record](config/experiments/operated-alpha-retirement.json).

**Current learning work:** the [programming expert trial](docs/PROGRAMMING_EXPERT_TRIAL.md)
failed automatic serving. The leftover [fallback comparison](docs/PROGRAMMING_FALLBACK_COMPARISON.md)
passed +4/32 and remains the research baseline because later challengers failed
acceptance, not because it had the highest observed score. The
[second-capability growth](docs/PROGRAMMING_GROWTH.md) campaign is **closed**:
isolation passed (+7/32), unit-merge failed, four heuristic selectors failed,
and two sign-consensus mixes failed their declared gates.
[TIES](docs/PROGRAMMING_GROWTH_TIES.md) preserved all 29 incumbent successes
and reached 31/64, including leftover 54 which neither original tail produced,
then failed unique-added 276 and 265. That is partial combination, not
acceptance. Complementary coverage still exists in the unchanged tails. Four
heuristic selectors failing does not establish that learned routing cannot
work. Stop work on these two tails and the opened 64 cases. Do not open the
original 128-task final. Serving stays the leftover incumbent extra. The subsequent
experiment was [learned integration of new capacity](docs/LEARNED_INTEGRATION.md):
train a new module and its gate together versus a matched no-expansion control,
and score generated answers on a fresh split. The stage-1 method is frozen. The 135M CPU run
[failed development](docs/LEARNED_INTEGRATION_RESULTS.md): expansion 0/32,
control 5/32, general 8/8. Confirmation was never opened. No GPU is authorized.
Serving stays the leftover incumbent extra. A fresh dataset by itself would
not address the failure. The new [staged-integration candidate](docs/STAGED_INTEGRATION.md)
trains an expert with guaranteed access, freezes it, then trains its gate. Its
separate 135M CPU arithmetic mechanism contract includes checkpoints, route
traces, per-answer preservation, isolated peak memory and actual CPU spend.
It [stopped before training](docs/STAGED_INTEGRATION_RESULTS.md) on an unsuitable
output-format baseline. The separate [staged-answering study](docs/STAGED_ANSWERING.md)
uses complete-answer scoring and fresh operands, with unchanged training and gates.
Its [execution timed out](docs/STAGED_ANSWERING_RESULTS.md) after 64 expert and
63 gate updates. The baseline protected 15 answers; candidate quality was not scored
in that interrupted run. Its separate
[recovery completed and failed](docs/STAGED_ANSWERING_RECOVERY_RESULTS.md):
expansion 0/32 versus control 2/32, with 11 of 15 protected answers lost.
This candidate is stopped and receives no checklist credit. The separate
[block-expert competence study](docs/BLOCK_EXPERT.md) tests added Transformer
blocks on fresh arithmetic questions against a matched-cost control, before any
selector training. It [stopped at the baseline](docs/BLOCK_EXPERT_RESULTS.md)
with 6/64 protected answers against a minimum of eight; no expert was trained.
The separate [measurement contract](docs/BLOCK_EXPERT_MEASURE.md) records parent
retention and unfinished replies, then trains anyway. It
[failed](docs/BLOCK_EXPERT_MEASURE_RESULTS.md): added blocks 10/64, control
11/64, and both lost all 8 protected answers. The next rule is
[append-only growth](docs/APPEND_ONLY_GROWTH.md): the parent keeps protected
answers, and a new shard is used only where the parent missed. Its
[CPU execution](docs/APPEND_ONLY_EXECUTION_RESULTS.md) met the oracle gates and
tied a constant training label at 9/64. The selector read the hidden answer, so
this is not a deployable assistant. The next rule is
[observable selection](docs/OBSERVABLE_SELECTION.md). It has not trained. Explicit expert competence alone does not prove automatic
serving or preservation. The separate systems track is
[independent hosting](docs/INDEPENDENT_HOSTING.md): a CPU protocol preflight of
equal-power genesis and stranger-provider join, then an independent-operator
soak that requires four independently administered operators. This operator's
AWS account cannot satisfy that criterion. No GPU is authorized.
This does not change the six checklist criteria, the 0.4.0 genesis, or item 4.
The opened-development 15/32 diagnostic remains not admission evidence.
A passing experiment counts as supporting evidence; a top-level box is checked
only when all of its completion criteria have a committed implementation and
reproducible evidence. Changes to these criteria must be recorded explicitly,
before the affected work, rather than changing the target after a result.

- [x] **1. Intelligence across growing shards — COMPLETE**

  Completed: **three prospectively admitted cohorts in the accepted model lineage**.
  Conversation reached 15/16 single and 14/16 combined answers; audit reached
  15/16 and 14/16; storage reached 16/16 and 14/16. Each preserved every measured
  previously correct retained answer, with three exact complete quality audits.
  Ordinary requests select and combine learned shards without evaluator labels;
  fixed broader skills and conversation retention also pass. Under equal seven-host,
  9,000-second budgets, addition preserved all retained answers while replacement
  lost four; replacement served more requests. The declared growth rule remains
  quality and preservation within the resource budget.

  The three accepted extensions span two research geneses. The latest two ran
  consecutively and automatically under one unchanged frozen continuation.
  Rejected trials and development repairs receive no cohort credit. This completes
  the bounded demonstration criterion; broad assistant capability and independent
  public operation are not inferred. See [complete results, replies and reproducible evidence](docs/CONTINUAL_ADMISSION_RESULTS.md).

  Done when ordinary questions automatically select and combine relevant shards,
  including questions requiring knowledge from different learned experts; at
  least three successive admitted learning cohorts pass prospectively frozen
  response-quality and retention gates. Include a comparison against spending
  the same total resources without growth, and a declared rule for adding,
  updating or consolidating capacity. Evaluation must cover broader assistant
  tasks as well as specialized knowledge, without supplying answer labels to
  routing. More stored parameters alone do not satisfy this item.

  <details>
  <summary>Earlier evidence and failures (historical snapshots)</summary>

  Evidence so far: two specialized experts, improved held-out answers, exact
  measured retention, and no complete backbone on any one owner. The admitted
  graph retains explicit routing and composition. A new jointly trained neural
  interface improved raw single-fact answers to 13/16, but combined answers
  reached only 2/16 and failed its gate; general retention passed. See the
  [interface result](docs/EXPERT_INTERFACE_TRIAL.md) and the earlier
  [published learning result](https://github.com/neuroshard-ai/neuroshard/releases/tag/research-cohorts-20260915).
  Continuing that interface reduced training loss without improving combined
  answers and failed retention. The next [owned planner](docs/OWNED_PLANNER.md)
  raised correct development answers from 13/56 to 48/56 while preserving the
  answering weights. All 16 mixed requests retrieved both correct facts, but
  the final composer mishandled six responses. The subsequent learned output
  plan improved previously unopened final answers from 129/144 to 136/144;
  mixed answers improved from 23/32 to 29/32 without category regression.
  Its exact-plan gate failed: four correct mixed replies omitted a full stop,
  and three mixed plans were invalid. The frozen result remains failed.
  Repeated admitted cohorts and matched-resource growth remain unproved.
  Continued expert jobs can now start from accepted weights with fresh Adam.
  Distributed feature production and native replay use that accepted expert as
  the frozen reference. A new quality profile measures retained answers across
  changed weights; it no longer requires an updated expert to remain identical.
  The first continued GPU trial raised new single-fact answers from 2/16 to
  7/16, but lost eight previously correct retained answers and failed its gate.
  Its final set stayed unopened; no serving model was promoted. See the
  [continued-learning result](docs/CONTINUAL_EXPERT_LEARNING.md#measured-result-rejected).
  The next method isolates new expert weights and appends a learned binary
  routing gate while preserving the accepted router. Six-process numerical
  validation now executes both earlier and expanded learned services on the
  same owned shards, without duplicating the backbone. The four-GPU trial
  completed 512 updates: new single answers improved from
  2/16 to 11/16 and composed answers from 0/8 to 4/8, preserving all 80 previously
  correct retained answers. It failed its accuracy gate; the final stayed unopened.
  Fresh replay reproduced the final four updates exactly. See the
  [isolated-learning result](docs/CONTINUAL_EXPERT_LEARNING.md#isolated-learning-result).
  A separately frozen [replacement control](config/experiments/expert-replacement-control.json)
  reuses the exact new terminal weights while replacing the earlier expert.
  Learned route aliases let both routes use that fixed capacity; six-process
  execution checked actual routing, replay and owner payments. The first GPU
  comparison failed during setup before generating answers. A separate later
  [semantic training trial](docs/CONTINUAL_EXPERT_LEARNING.md#semantic-learning-and-replacement-result)
  completed that comparison. Its development gate passed (13/16 single, 6/8
  composed), then the previously unopened final failed (13/16 single, 5/8
  composed; 6/8 required). The added expert preserved all 80 previously correct
  answers. Replacing B with identical trained weights scored 44/96 retained
  answers: 38 previously correct answers lost and two newly correct. This
  comparison matches training, not total lifetime costs. No native promotion
  occurred; three successive admitted cohorts and broader ordinary-question
  quality remain unproved. The next measurement is an inference-only
  [ordinary serving diagnostic](docs/ORDINARY_SERVING_DIAGNOSTIC.md) of the
  complete planned path, with gold standalone controls that separate knowledge,
  selection, decomposition and assembly. It does not train, open a new final
  or reuse the exposed semantic final.
  That screen completed with 1/15 ordinary answers passing: 12 selection
  failures and two decomposition failures. The
  [ordinary access candidate](docs/ORDINARY_ACCESS_TRIAL.md) restores the earlier
  ordinary-question base and fits the new selector on separate training inputs.
  Its CPU preflight selects all 21 standalone gold routes (13 distinct
  questions). The frozen GPU run completed all 15 conversations: strict passes
  improved from 1 to 7, with zero selection failures, five knowledge failures
  and three decomposition failures. Forced experts answered 10/13 distinct
  questions correctly. C's three wrong facts exactly reproduce its earlier
  development output tokens. All six owner traces agree. CPU recovery repaired
  a missing failure category without changing pass flags; the new run's replay
  verdict was not persisted before aggregation failed. All resources were
  retired. Subsequent request preservation and subject-only reference resolution
  improve 7→9→10/15 without changing expert weights. The completed engineering
  screen has zero selection/decomposition/assembly failures, all 21 standalone
  routes correct, all seven retained cases correct, and exact automatic/forced
  replay with six identical owner transcripts. The
  [C-only learning repair](docs/CONTRACT_LEARNING_REPAIR.md) then improves
  10→14→15/15 through crossed training contracts and source-grounded claim/job
  scope contrasts. All 13 distinct forced controls and all 16 C development
  questions pass; no previously correct tested answer is lost. Six complete
  serving transcripts agree, and fresh training/inference replay passes.
  This repairs the exposed ordinary diagnostic. It does not count as a new
  admitted cohort or independent holdout; repeated prospective learning and
  operated native admission remain the next proof.

  The first full [ordinary native cohort](docs/ORDINARY_NATIVE_CAMPAIGN.md#completed-full-cohort-rejected)
  completed 128 verified updates: single answers improved 3→13/16 and combined
  answers 2→9/16, preserving every previously correct retained answer. It failed
  the 12/16 combined-answer gate and was rejected. A subsequent inference-only
  literal-question diagnostic reached 11/16 combined but regressed two diagnostic
  answers; it also failed. Selection and wording-sensitive knowledge remain
  unresolved. Neither result counts toward the required three admitted cohorts.

  [Semantic question access](docs/SEMANTIC_QUESTION_ACCESS.md) now repairs the
  opened complete-service diagnostic: 15/16 single, 13/16 combined, zero retained
  answers lost and exact replay. A small frozen semantic encoder selects a
  canonical training question; the existing expert generates its answer from
  weights. The question index contains no answers. This is a development repair,
  with no new final, neural expert training or native promotion. Three fresh
  admitted cohorts and the matched-resource comparison remain required.

  The [prospective semantic campaign](docs/PROSPECTIVE_SEMANTIC_CAMPAIGN.md)
  completed 128 verified updates, improving singles 1→13/16 and combined
  answers 0→11/16 with zero retained-correct losses. Three fresh quality audits
  agreed; native quality rejected the candidate. Subsequent selector repairs
  use that opened set as development only. No fresh cohort was admitted.
  The complete repaired service reaches 15/16 single and 13/16 combined answers
  on those opened cases, with no regressions, no retained-correct losses and
  exact replay. It learns question distinctions inside the retrieved expert
  while preserving the earlier fallback. This is a development pass; three
  newly admitted cohorts and the resource comparison are still required.

  </details>

- [x] **2. Continuous data admission and learning — COMPLETE**

  Completed: the real sharded LLM automatically rejected its bootstrap and a
  substituted immutable source, then prepared, funded, trained, audited and
  promoted both fresh cohorts without a genesis or recipe change between jobs.
  Native cursors consumed 768 distinct training documents, with replay from
  actually trained windows. The durable publisher resumed across 3,040 process
  starts; earlier controller/native-process restart and host-restoration evidence
  remains linked in the [completed result](docs/CONTINUAL_ADMISSION_RESULTS.md).
  There were 226 during-work serving probes. Full replay reproduced 28,912
  headers, 869 signed transactions and exactly 257 issued trial NEURO; paid
  inference from the final graph settled without additional issuance.

  Scope: automatic admission from a prospectively approved, source-backed feed,
  with explicit provenance, duplicate/contamination checks and source-substitution
  rejection. Arbitrary-web truth, independent curation and public service
  reliability are not established by this bounded operated demonstration.

  Done when the network repeatedly admits immutable data with provenance,
  deduplication and contamination/poisoning checks; funds and assigns a bounded
  training job; verifies the work; evaluates the candidate on precommitted
  acceptance rules; and promotes or rejects it without manually editing genesis
  or orchestrating each cohort. Demonstrate restart, rejected data and a rejected
  candidate while the accepted serving model remains available.

  <details>
  <summary>Earlier evidence and failures (historical snapshots)</summary>

  Evidence so far: immutable ingestion and experimental lifecycle components
  exist. The growing expert architecture still uses fixed prepared jobs.
  Bounded planner windows now connect actual replay to the existing native
  reservation, funded audit and issuance machinery. Five-process CPU validation
  covers three fresh audits, exact rewards and unchanged serving state. This
  optional research profile still lacks automatic repeated planner admission
  and separate planner promotion; it does not complete this item.
  The continual expert quality rule also accumulates every prior admitted
  evaluation conversation while keeping its acceptance thresholds and initial
  assistant anchors fixed. Admission rejects omitted or rewritten history.
  Native preparation now turns consecutive pinned source rows into reviewable
  jobs, preserves exact accepted replay provenance, carries prior evaluations
  forward and refuses stale state. Its inputs drove actual CPU updates and
  fresh-process replay. Production source integration and passing cumulative GPU
  learning results are still required.
  A durable publisher controller now reconciles repeated native jobs, funds
  execution, submits measured quality failures for audit, and proceeds after
  rejection. Recovery checks reproduce committed-before-acknowledgement
  crashes without duplicate transactions. Configured curator reviews share the
  auditor's sole signer and journal their own approvals or rejections. These
  recovery checks use explicit numerical fixtures. A further two-cohort
  integration now executes real prefix production, training and three neural
  replays per claim, including six-process quality evaluation. It issued only
  the four verified updates, rejected both quality failures, advanced source
  cursors automatically and preserved the serving graph without changing
  genesis. The operated repetition now adds publisher restart, actual repeated-
  source rejection and nine unchanged responses from persistent accepted shards.
  Four CometBFT validators agreed, and full replay reproduced all 74 transactions
  across 449 headers. These small synthetic models and local source fixtures do
  not establish operated continuous LLM learning. See the
  [implementation and published evidence](docs/AUTOMATIC_EXPERT_COHORTS.md).
  Structured immutable feeds now preserve original messages for that native
  preparation path. Restart-safe publication, cross-window reads and independent
  source review passed; all nine objects for a three-window, 192-conversation
  feed passed full public S3/CDN hash checks. This supplies the transport and
  preparation bridge; operated GPU integration and useful promotion remain open.

  The full ordinary cohort now exercises real immutable data, GPU shards, 96
  training audits, three complete quality audits and native rejection while
  accepted serving remains available. Full ledger replay matched 446 signed
  transactions and 8,995 headers; 129 verified updates issued 129 NEURO including
  bootstrap. No useful cohort was promoted. Repeated useful automatic admission
  remains open; the rejection machinery alone does not complete this task.

  The [prospective semantic campaign](docs/PROSPECTIVE_SEMANTIC_CAMPAIGN.md)
  now reproduces bootstrap rejection with three exact audits, refuses a
  hash-consistent substituted source answer, and automatically admits the next
  real GPU job. A controlled restart after twelve verified updates preserves
  checkpoint and issuance; accepted shards return identical responses while
  the controller and research validator processes are down. An earlier startup
  outage remains recorded. Useful repeated promotion is still pending.

  </details>

- [x] **3. Live native training-to-inference lifecycle — COMPLETE**

  Done when the already validated second expert completes this entire path on
  an operated NeuroShard-native candidate chain: available numerical inputs and
  checkpoints; real configured execution audits; accepted training claims;
  exactly-once NEURO rewards; a separate quality decision admitting the serving
  graph; and a raw-question inference request paid from earned NEURO. Bind the
  actual model, optimizer ages, tokenizer, routing and composed-call billing.
  Publish the ledger and execution evidence. Restart/replay, a forged claim,
  missing data and a repeated claim must leave supply and serving state correct.

  Completed 2026-09-16: actual prefix production and all 140 four-update
  windows settled, issuing exactly 560 NEURO. A separate full-audit quality
  decision promoted the original validated graph. A raw composed request used
  earned NEURO, paid both neural calls and refunded unused escrow. Forged work,
  withheld bytes, duplicates, restart and recovery onto replacement hosts were
  checked. Full application replay verified 1,728 signed transactions, 16,359
  headers and all four saved validator states. Numerical and ledger evidence,
  replay instructions and the public 2,856-tensor catalog are
  [published](https://github.com/neuroshard-ai/neuroshard/releases/tag/research-native-expert-20260916).
  See the [result and its limits](docs/NATIVE_EXPERT_LIVE_RESULT.md).

  This was a finite candidate operated by one administrator. Its GPU resources
  are retired after preservation; independent operation and unattended public
  chat remain covered by the other items. The integration reproduces the
  previously published expert's quality gate and does not count as another
  successful learning cohort.

- [ ] **4. Reliable permissionless shard hosting**

  Current development: [independent hosting](docs/INDEPENDENT_HOSTING.md) freezes
  the remaining item-4 soak. The CPU protocol preflight uses four equal genesis
  validators (each share strictly below one third) and a non-genesis provider
  join without SSH. The independent-operator soak stays unauthorized until four
  independently administered operators exist. Voting share is aggregated by
  administrator, not by key. AWS machines under our account do not satisfy that
  criterion. The [operated LLM alpha](docs/OPERATED_ALPHA_RESULT.md)
  remains the last one-administrator hosting evidence; its GPU service retired
  on September 20.

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

  In development: native provider registration, graph/rank offers, collateral
  and capacity reservation, complete prepaid audit assignment and timed
  coordinator replacement now have adversarial state-transition coverage.
  Receipts bind assignment epochs; failed or expired service refunds its unused
  budgets. Five separately keyed CPU processes now execute actual model
  partitions through certificate-pinned HTTPS, matching the fixed-group
  executor for ordinary and multi-turn fixtures. Providers discover their own
  native assignments, restore committed partitions without SSH and jointly
  sign the complete response.   Operated LLM recovery has since passed below. The item-4
  [independent hosting](docs/INDEPENDENT_HOSTING.md) freeze now specifies the CPU
  protocol preflight (four equal validators, stranger-provider join). The
  independent-operator soak remains unauthorized. See
  [the provider-market design](docs/PROVIDER_MARKET_RFC.md).

  The subsequent [native provider preflight](docs/PROVIDER_NATIVE_PREFLIGHT.md)
  passes actual registration, partition restoration, signed execution and
  settlement across five provider processes and four CometBFT validators.
  A killed coordinator and a killed backbone owner are each replaced by a fresh
  key and endpoint. Each recovered request pays once after three complete
  numerical replays. Application replay matches 660 headers, 84 transactions
  and all four stored states. This uses a small synthetic graph on one host;
  accepted-LLM load/cost evidence and independent operation remain required.

  The [accepted LLM trial](docs/PROVIDER_LLM_SERVICE.md#seventh-allocation-complete-serving-and-recovery-pass)
  now passes fresh-coordinator and fresh-backbone replacement across seven hosts
  in five availability zones. Requests recover and settle once in 622.34 and
  641.97 seconds after three complete numerical replays each. Six ordinary
  requests also meet the frozen latency gates. All disposable resources are
  retired, and complete application replay matches all seven saved states.
  These are process-loss trials with replacement on surviving hosts, not a
  declared independent-operator availability soak. Provider selection and
  audit-offer saturation resistance remain open engineering work. Separate
  EC2 hosts do not establish separate administrators.

- [x] **5. Affordable verification and sustainable incentives — COMPLETE within finite sponsorship**

  Done when every accepted obligation has available evidence, complete funded
  verification and a bounded liability; adversarial tests cover forged work,
  duplicate work, withholding and failed audit obligations. Publish the quorum
  and collusion assumptions and measure complete costs, including verification,
  retention, storage and serving. Configured fees or finite sponsorship must
  cover those measured obligations within the declared issuance limits. Signatures
  and token issuance alone are not evidence of independent computation or demand.

  Evidence so far: supply invariants, work deduplication, funded native replay
  quorums and real replay measurements. Task 3 published the complete retained
  window payload catalog and settled all 140 windows with full audits. Complete
  retention/storage/serving costs and finite funding are now reconciled below.
  The planned-service operator now quotes and meters all prompt and output
  work, including planning and composition, with exact owner shares and a
  bounded reservation. Real five-process replay reproduced the same receipt
  and rejected a forged response. These execution quotes explicitly require
  separate verification and retention funding; they are not ledger payments.

  The [finite sponsorship](docs/FINITE_SPONSORSHIP.md) now declares complete
  allocation, controller, network and retention accounting under a $1,000
  bounded research budget. Actual hosted requests prepay complete audits and
  settle all unused execution/provider/audit funds without inference issuance.
  The [public complete-cost report](docs/FINITE_SPONSORSHIP.md) reproduces
  **$792.54 against $1,000**, including three learning allocations, all seven
  provider attempts, both controllers and the entire public object pool's
  93-day retention. Existing checks cover forged/duplicate work, withholding,
  missing auditors, rejection payments and exact refunds; quorum/collusion
  assumptions are published. This meets this item's finite-sponsorship option.
  It does not establish independent auditors, cheap verification at arbitrary
  model size, a token price, outside demand or perpetual funding. The demonstrated
  admission saturation attack remains open under permissionless hosting.

- [x] **6. Usable chat service — COMPLETE for the funded operated alpha**

  The [public alpha](docs/JOIN_ALPHA.md) served the accepted growing graph
  with two concurrent customers, streaming, bounded complete prices and pinned
  graph/tokenizer conversation history. All ten frozen deployment requests pass,
  including automatic coordinator/backbone process recovery. Six ordinary cases
  show first text in **33.15–42.17 seconds** and complete generation in
  **34.75–49.34 seconds**. A fresh public observer and wallet completed two
  paid turns, preserved context and resumed a deliberate timeout with the same
  nonce. All four saved ledger states replay exactly, with zero inference issuance.
  The first memory-instruction reply was poor and remains published; the following
  turn retained the phrase. This is a service/context check, not broad answer quality.

  [Results, replies, failures and reproducible evidence](docs/OPERATED_ALPHA_RESULT.md)
  are public. Admission and GPU service closed early on September 20, 2026;
  the separate ledger remains scheduled through September 26, 2026 at 23:49 UTC
  for expiry/refunds. The demonstrated service was a source CLI,
  with public conversations and 64-token outputs, under one administrator.
  These are a finite alpha's measured guarantees, not perpetual availability,
  independent ownership or ChatGPT-level capability. The existing done-when
  criteria below are unchanged.

  Done when the accepted growing graph serves ordinary multi-turn requests with
  streaming, versioned context/tokenizer handling, bounded prices, correct
  expiry/refunds and concurrent clients. Meet a published, prospectively frozen
  latency/load target on supported deployments. State and test prompt visibility,
  storage and privacy behavior; users must understand which providers or auditors
  receive their data. Publish representative responses and failure behavior.

  <details>
  <summary>Earlier chat evidence before the operated alpha</summary>

  Evidence so far: the public 0.4.0 client supports a smaller experimental model;
  the larger expert graph has operated generation and public reproducible weights.
  The new candidate binds and audits all neural calls used by a raw question,
  including interpretation and composition. Those pieces are not yet the live
  growing chat service. An experimental local service now pins gate and owned
  adapter checkpoints, preserves multi-turn context, and delivers only checked
  token chunks. Five CPU processes checked discarded delivery, context overflow
  and changed owner weights. Eight GPU responses then passed full checking,
  including corrected numerical disagreements. One-pass prompt prefill and
  sixteen-token target blocks subsequently passed the frozen latency and traffic
  bounds on all eight exposed GPU workloads: 128-token responses took 8.37–9.56
  seconds and full replay took 1.63–1.68 seconds. The weights still fail their
  learning-quality gate. Public access,
  native billing and load targets remain open. See [block streaming](docs/BLOCKED_STREAMING.md).

  </details>

## Work discipline

All six milestones remain in scope. Items 1–3 and 5 are complete against their
fixed demonstration criteria, with item 5 using its finite-sponsorship option.
Item 6 is complete for the bounded operated alpha. Reliable permissionless
hosting remains item 4; all existing completion criteria are unchanged.

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

- 2026-09-21: **Revised independent-hosting soak to four operators. No GPU.**
  Three operators cannot each hold strictly less than one third of voting power.
  Soak requires four independently administered operators and aggregates share
  by administrator. Replaces `5dc22aab…`. Contract `8213fdfd…`. Soak unauthorized.
  Not admission.

- 2026-09-21: **Froze independent hosting for item 4. No GPU.**
  CPU protocol: four equal genesis validators, stranger-provider join, no key
  may hold the complete backbone. Independent soak unauthorized. AWS under one
  account does not satisfy. Contract `5dc22aab…`. Not admission.

- 2026-09-21: **Stage-1 learned integration failed development. Stop. No GPU.**
  Expansion 0/32, control 5/32 (517, 733, 807, 896, 924), parent 0/32,
  general 8/8, retention 0=0. Serving budget passed. Confirmation closed.
  Score `07d4ac47…`. Execution freeze `e87456a6…`. Not admission.

- 2026-09-21: **Froze the stage-1 CPU train/score loop. Not scored. No GPU.**
  Last-layer expert plus gate versus matched last-layer MLP, 128 teacher-forced
  steps on unused MBPP, then generated development/retention/general only.
  Confirmation closed. Execution freeze `e87456a6…`.

- 2026-09-21: **Recorded the stage-1 CPU execution freeze. Not run. No GPU.**
  Seed `12fd25f7…` file hashes including `model.safetensors` `5af571cb…`.
  Eight general-retention identities from Smol-SmolTalk rows 13712, 10926,
  17751, 6645, 4043, 4233, 20506, 20999. Confirmation closed. Execution freeze
  `1f838d4d…`. Method freeze now `dc76e6ad…`.

- 2026-09-21: **Froze the learned-integration stage-1 method. No GPU.**
  Last-layer top-1 expert plus trained gate versus matched last-layer MLP
  training. Confirmation closed. General document identities wait for a later
  execution freeze. `scripts/run_learned_integration.py` refuses launch.
  Method freeze `6a21a3df…`.

- 2026-09-21: **Closed independent-tail programming growth. Specified learned
  integration.** Stop the two tails and the opened 64 cases. TIES remains
  rejected after partial combination (29 incumbent preserved, 31/64, leftover
  54). Incumbent extra stays the research baseline because challengers failed
  acceptance. Next experiment trains a new expert and its gate together versus
  a matched no-expansion control on unused MBPP IDs. Contract
  `config/experiments/learned-integration.json` (`6801d1a2…`). No GPU. Not
  admission. Original 128-task final stays closed.

- 2026-09-21: **Selector v4 (public-feedback extraction-error) failed.**
  Decisions `7bd95df8…`. Score 29/64, unique added 0/3, incumbent 29/29.
  37/38 extras were execution-error; the one extraction-error was not
  unique-added. `stop-this-picker`. Not admission. No GPU. These 64 cases
  remain opened. Serving stays leftover incumbent extra.

- 2026-09-21: **Declared public-feedback-status selector v4.** Added only when
  the parent public-example status is extraction-error; otherwise incumbent.
  A priori fail-class rule on the unused allowed feedback field. Not fitted.
  CPU screen not yet scored. No GPU.

- 2026-09-21: **Elect-sign composition failed its screen and matched TIES.**
  Source `8ea046a`. Extractable 38/38. Full-test 31/64. Unique added 1/3
  (recovered 503; missed 276, 265). Incumbent 29/29 including 249. Leftover 54
  again. Same opened-task outcomes as TIES; trim was not the unique-added loss.
  Score `7491a09b…`. `stop-this-composition`. Close task-vector sign-consensus
  on these tails. First GPU attempt died on SSH; retry decoded (~10 min) and
  retired. Not admission. Serving stays leftover incumbent extra.

- 2026-09-21: **Declared elect-sign disjoint mean of the same frozen tails.**
  TIES trim is the step that deletes small unique directions. This keeps
  sign-election and the disjoint mean and drops trim. CPU merge pinned
  (`65058087…`); extras not yet decoded. Not admission. No training.

- 2026-09-21: **TIES composition failed its screen.** Source `9e85585`.
  Extractable 38/38 (unit merge was 18/38). Full-test 31/64 versus incumbent 29
  / always-added 31 / oracle 32. Unique added 1/3 (recovered 503; missed 276,
  265). Incumbent 29/29 including 249. Leftover task 54 passed under TIES and
  under neither frozen tail. Score `86a6a489…`. `stop-this-composition`. Do not
  iterate keep or scale on these 64. GPUs retired (~7 min). Not admission.
  Serving stays leftover incumbent extra.

- 2026-09-20: **Declared TIES composition of the two frozen programming tails.**
  Unit merge left 18/38 extras unparseable. Raw last-four-layer deltas disagree
  in sign on 33.6% of jointly nonzero parameters. TIES keep 0.2 / disjoint mean
  is frozen (`6a994d62…`); extras not yet decoded. Not admission. No training.

- 2026-09-20: **Selector v3 (failed-parent AST shape vs train gold programs)
  failed.** Decisions `b08ab22d…`. Score 29/64, unique added 0/3, incumbent
  29/29. Chose added on 8 extras, none unique-added. `stop-this-picker`. Not
  admission. No GPU. These 64 cases remain opened. Serving stays leftover
  incumbent extra.

- 2026-09-20: **Selector v2 (question+parent Jaccard agreement) failed.**
  Decisions `e89117ac…`. Score 29/64, unique added 1/3 (only 276), incumbent
  28/29 (lost 249). Agreement fired on 7 extras. `stop-this-picker`. Not
  admission. No GPU. These 64 cases remain opened.

- 2026-09-20: **Declared selector v2: Jaccard agreement of question and failed
  parent.** Separate contract from stopped picker `963de13`. Added only when
  both views strictly prefer added; disagreement selects incumbent. Same 64
  opened cases remain diagnostic. Not trained, no GPU yet.

- 2026-09-20: **Nearest-train Jaccard selector failed its CPU screen.** Picker
  commit `963de13`. Decide hashed 38 calls (`344c68ac…`) before joining tails.
  Score 30/64 versus incumbent 29 / always-added 31 / oracle 32. Recovered
  unique added 276 and 265; missed 503; lost leftover success 249. Picker p95
  3.6 ms, zero errors. Scorer lookup now uses added extras only on the 38
  visible-fail rows; decisions were not regenerated. `stop-this-picker`. Not
  admission. No GPU. These 64 cases remain opened.

- 2026-09-20: **Selector evaluation contract frozen (`8e39b74`).** Independent
  CPU rescore matched complementarity (incumbent 29, always-added 31, oracle 32).
  Picker inputs, incumbent-default, CPU qualification and fresh MBPP 511–600
  confirmation slice are pinned. First picker specified as nearest-train Jaccard
  (train prompts only; not fitted on opened labels). Not screened. No GPU.

- 2026-09-20: **Programming-tail complementarity: unique added coverage exists.**
  Inference-only on freeze `dcc6693`, added extras 38/38 extractable. Unique
  added 3 (tasks 276, 503, 265), unique incumbent 1 (249), both 6, neither 28.
  Oracle union 32 versus incumbent policy 29 (+3). Not a serving policy, not
  admission. GPUs retired (~8 min, ~$0.55). Next is selection between the two
  unchanged tails, not another trained tail. Merge remains failed.

- 2026-09-20: **Programming unit-merge growth failed after isolation passed.**
  Freeze `dcc6693` trained 256 parent-init updates. Isolation +7/32 on leftover
  development. Growth lost leftover specialist wins 115/169/249/258, scored
  −3/32 new leftover answers versus the incumbent extra, and merged extras
  added zero successes beyond the parent (18/38 extractable). GPUs retired;
  all five launch attempts included in accounting (~$3). Not promoted. The
  next measurement is inference-only complementarity of the two unchanged
  tails on the 38 opened extras.

- 2026-09-20: **Completed item 6 for the funded operated alpha.** The accepted
  graph passes ten deployment requests, ordinary concurrent latency gates and
  both automatic process-loss recoveries. A new public observer/customer passes
  two paid conversation turns, actual streaming and timeout/resume without a
  duplicate payment. Application replay matches 3,369 headers,
  697 accepted transactions and four saved states;
  serving issues no tokens. [Evidence and access](docs/OPERATED_ALPHA_RESULT.md)
  are published with the finite service window and $800 ceiling. Checklist
  **5/6 complete**; independent hosting and its soak remain item 4.

- 2026-09-19: **Completed item 5 under finite sponsorship; advanced items 4 and 6.**
  Native providers acquire only assigned model portions and serve the accepted
  graph through paid, streamed, replay-verified requests. The final seven-GPU
  trial passes all ordinary latency and coordinator/backbone recovery gates;
  all ten responses preserve the preceding source's neural results exactly.
  All experimental resources are retired. Complete costs are $792.54 under
  the frozen $1,000 cap, including failed attempts and full retention. Independent
  administration, market admission resistance, the declared availability soak
  and persistent public deployment remain open. Checklist **4/6 complete**.

- 2026-09-19: **Completed items 1 and 2.** The accepted conversation → audit →
  storage lineage now has three prospective admissions across two research
  geneses, with zero measured retained-correct losses. The latest two cohorts
  ran automatically, passed all three complete numerical audits, and promoted
  natively. Equal-budget growth preserved answers that replacement lost.
  Full replay passes 28,912 headers and 869 signed transactions; the final graph
  served paid inference. The public evidence and complete measured scope are in
  [the completion report](docs/CONTINUAL_ADMISSION_RESULTS.md). Seven experiment
  GPUs and their volumes are retired; compute upper estimate $59.59, with
  storage/transfer separate. Checklist **3/6 complete**.

- 2026-09-17: The complete ordinary interface now passes **25/25 retained
  knowledge, 10/12 general skills and 8/8 conversations**, with no earlier
  correct answer lost and exact replay across actual owned shards. A separately
  frozen prospective assistant screen then passes **10/12 skills and 6/8
  conversations**, including exact replay; its four failed answers remain
  recorded. The twenty fresh cases join cumulative retention unchanged. The
  renewed three-cohort native prescription preserves all original neural
  training/final bytes, domain gates and thresholds. This is a measured
  answering improvement; repeated useful native cohorts remain unproved and
  items 1 and 2 stay open. [Results and limits](docs/ORDINARY_NATIVE_CAMPAIGN.md).

- 2026-09-17: Completed the C scope continuation: **15/15 ordinary answers**,
  **13/13 unique forced controls**, and **16/16 C preservation questions**.
  All earlier correct tested answers survived, with identical neural outputs
  for all seven retained A/B/general conversations. The 64 updates took 575.6
  seconds. Four training owners and six serving transcripts agree; fresh
  replay of updates 60–64 and both inference replays pass. A preceding setup
  attempt failed before training because publication receipts lacked the
  fetch destination; the corrected handoff passes local regression checks.
  [Evidence and limits](docs/CONTRACT_LEARNING_REPAIR.md) retain both attempts.
  Both allocations and their volumes are deleted; compute estimates are $1.68
  for the passing run and $0.67 for setup failure, with storage/transfer separate.
  No new final, cohort, issuance or native promotion; tasks 1 and 2 remain open.

- 2026-09-17: Training C on crossed input contracts improved ordinary answers
  **10/15 → 14/15**, forced answers **10/13 → 12/13**, and standalone C retention
  **13/16 → 15/16**, losing no previously correct answer in either inventory.
  Four training owners and six serving transcripts agree; fresh replay of
  updates 104–108 and both inference replays pass. The full gate still fails on
  one claim-window versus whole-job scope confusion. The 108-update run took
  985 seconds; all four instances and volumes retired, compute at most $2.28
  excluding storage/transfer. [Result and next bounded scope repair](docs/CONTRACT_LEARNING_REPAIR.md)
  preserve the exposed-development status: no new final, cohort, promotion or
  issuance. Tasks 1 and 2 remain open.

- 2026-09-17: Repaired ordinary access using verbatim standalone requests and
  bounded neural subject resolution with deterministic substitution. The first
  revision scored 9/15 and failed one reference; the second scored 10/15 with
  only the five existing knowledge failures remaining. Both retain all earlier
  correct cases, agree across six owners, replay exactly and meter every neural
  call. Across the same 28 ordinary/control requests, neural calls fall from
  69 to 48 and prompt tokens from 19,207 to 7,866. All ten allocated instances,
  their volumes and security groups were deleted; combined compute estimate
  $2.03, storage and transfer separate. [Result manifests](config/experiments/request-planning-results.json)
  bind the local evidence archives. No expert weights changed and no new final,
  issuance or promotion occurred. Tasks 1 and 2 remain open.

- 2026-09-17: Completed one frozen five-host ordinary-access allocation, with
  retained neural weights, a restored ordinary router, a C gate fitted on
  separate training inputs, and forced-expert controls. Recovered all 15 saved
  transcripts after an aggregate-scorer failure; strict passes improved 1→7,
  while remaining knowledge/planning failures keep access closed. Forty-four
  focused checks pass. All GPU instances, volumes and the temporary security
  group were deleted; compute estimate $1.16, other charges separate. No next
  learning cohort, native promotion or issuance was performed.

- 2026-09-17: Froze an inference-only
  [ordinary serving diagnostic](docs/ORDINARY_SERVING_DIAGNOSTIC.md). Fifteen
  development questions cover retained A/B knowledge and planner-cohort
  development facts, with gold standalone controls and no explicit two-question
  grammar. CPU tests classify knowledge, selection, decomposition and assembly
  without neural execution. The GPU driver remains unlaunched: five owners, one
  hour, $25 cap, no training, no new final. Tasks 1 and 2 remain open.

- 2026-09-17: Published the completed semantic learning and replacement
  comparison. All 512 GPU updates completed; a fresh process reproduced the
  final four updates exactly. Six primary owners and five replacement owners
  agreed; rescoring with the frozen implementation reproduced every recorded
  score and gate. Development passed, but final composed accuracy failed.
  Preserved all 80 previously correct answers with the addition; replacement
  lost 38 and gained two. Published all result metadata and the two retained
  numerical boundaries. All four GPUs, disks and the temporary security group
  were deleted; compute upper bound $7.05, storage/transfer separate. Tasks 1
  and 2 remain open; the exposed final cannot serve as a fresh final again.

- 2026-09-17: Completed and [published the continued-expert trial](docs/CONTINUAL_EXPERT_LEARNING.md#measured-result-rejected).
  All 192 sharded updates executed; the last four replayed exactly in a fresh
  process. New answers improved, but accuracy and retention failed. The final
  test remained unopened, with no native activation, issuance or serving change.
  Preserved 3.22 GB of final/input-boundary tensors with full public hash checks.
  All four GPUs and temporary resources were retired; compute estimate $3.16,
  storage/transfer separate. Added native cohort preparation and sealing with
  actual numerical execution/replay checks. Tasks 1 and 2 remain open.

- 2026-09-17: Prepared a fixed [continued-expert learning trial](docs/CONTINUAL_EXPERT_LEARNING.md)
  using 16 fresh protocol facts, 96 actually trained replay conversations,
  accepted-expert distillation and 192 updates. Four-process execution preflight
  and eight preparation/data-review checks passed. The final questions stay
  closed after a failed development gate. This records the experiment contract,
  not a new GPU quality result or another completed milestone.

- 2026-09-16: Removed three blockers to repeated expert learning: restarting
  every job from the original parent, distilling old answers from that same
  parent, and using unchanged weights as the only retention criterion. Accepted
  expert initialization, actual generated-answer retention and complete prior
  evaluation coverage now have implementation checks. Seventeen numerical
  execution checks passed, including four-process production matching native
  prefix replay byte for byte; eleven data-review and graph-execution checks
  passed, including comparison of two expert revisions without replacing the
  serving weights. Thirteen admission/checkpoint regression checks also passed.
  This is implementation evidence, not a new model-quality gain. No GPU was
  allocated for these changes and neither active milestone is marked complete.

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
  [implementation and evidence boundaries](docs/NATIVE_EXPERT_SERVING.md).
- 2026-09-16: Froze the complete native integration in
  [native-expert-live.json](config/experiments/native-expert-live.json).
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
  owner exhausted the remaining startup interval during cold file reads.
  All five GPUs and volumes were retired; estimated compute was $1.85. The
  [recorded failure](config/experiments/native-expert-live-results.json) remains
  visible. A [same-target retry](config/experiments/native-expert-live-retry.json)
  carries the original library path and releases clean unowned cache before
  serving. Its deadline and combined $75 cap include the first attempt.

- 2026-09-16 02:54 UTC: The corrected five-GPU serving probe reproduced four
  published responses and all six neural calls across three availability zones.
  Actual prefix production received the configured three-auditor replay quorum.
  The first eight updates earned exactly eight candidate-chain NEURO. Forged
  measurements earned zero; a withheld auditor checkpoint blocked the next
  payment; restoration resumed execution. Duplicate rejection and restart kept
  the same checkpoint and eight-token issuance. The full 560-update run continues
  toward separate quality approval and paid raw inference under the original
  deadline. See the dated [progress evidence](config/experiments/native-expert-live-progress.json).
  This is one administrator, a specialized graph and partial lifecycle evidence;
  all six top-level items remain open.

- 2026-09-16: Completed item 3 and published its numerical and signed-ledger
  evidence. All 560 distinct updates earned exactly 560 NEURO; the separately
  admitted graph answered using earned-token escrow. Full replay checked 1,728
  transactions and 16,359 headers against four validator states. All five
  recovery GPUs, their volumes and their temporary security group are retired.
  Three integration attempts used a conservative $31.96 compute estimate;
  storage and previous learning runs are separate. The other five items remain
  open. Development also adds prospective repeated-cohort admission, original-
  data/tokenizer review and execution by three learned expert owners. These
  mechanics do not establish three useful learning cohorts or independent
  operators. A bounded ordinary-question decomposition screen passed 10/12
  development cases; two lost subject context and require correction before
  the method is used to route expert calls.

- 2026-09-16: Added a separately committed conversation executor that separates
  neural question decomposition from learned shard selection, executes the
  selected owned models with request-local KV caches, and records every neural
  call for replay. Seven cache checks passed, followed by a five-process failed-
  planner/replay check after integration. These cover tiny CPU models, not GPU
  numerical equivalence or assistant quality. The typed planner screen failed
  11/16 cases; its complete responses remain committed. The frozen
  [cached conversation trial](config/experiments/cached-composition-trial.json)
  now tests the actual model partitions, mixed answers and complete replay.
  No additional top-level milestone is complete.

- 2026-09-16: Added a rebuildable historical-data index that checks new cohorts
  against all admitted documents, reuses verified fingerprints across restart,
  rejects near copies of old evaluation examples and preserves explicit replay.
  Five source-data checks passed, including actual prefix production and
  independent numerical replay. Added verified object restoration with replica
  fallback; four HTTP fault checks passed. The first cached conversation GPU
  attempt stopped during one artifact download, before neural execution; all
  resources are retired. The [retry](config/experiments/cached-composition-retry.json)
  retains every neural input and pass rule, and changes only artifact delivery.

- 2026-09-16: The cached-conversation retry retrieved all assigned tensors but
  exposed a startup race: small experts opened their subgroup before a parent
  finished cold weight reads. A new readiness barrier loads both local models
  before forming inference groups. Five graph checks passed, including a
  deliberately delayed parent and exact cached replay. Both failed GPU
  allocations are fully retired. The next run keeps the original deadline,
  questions, weights and scoring rules under a new executor commitment. A
  [public serving catalog](https://github.com/neuroshard-ai/neuroshard/releases/download/research-native-expert-20260916/serving-tensors.json)
  now provides CDN/GitHub replicas for all 472 required tensors.

- 2026-09-16: The five-owner cached conversation trial completed and failed its
  quality gate (26/32 plans, 1/8 complete answers). All measured cached tokens
  matched and tensor traffic fell 93.23%; complete response replay passed. Eight
  diagnostic pairs reproduced the failures without caching and recovered learned
  facts when the training prompt contract was restored. The earlier router
  screen replaced domain phrases instead of removing them. Development now binds
  expert input contracts, supports a bounded planner reminder and freezes
  answer-blind raw-question router fitting on the existing entity-disjoint split.
  These changes require the next regression result; no milestone is checked.
  [Complete failed responses and diagnosis](config/experiments/cached-composition-results.json).

- 2026-09-16: Corrected input-only routing passed the frozen development gate:
  161/164 raw protocol questions, 1,920/1,920 raw directory questions, every
  original route and all 146 retained general questions. Calibration used only
  fitting inputs; class-balanced integer learning countered paraphrase imbalance.
  The CPU fit and evaluation took 114.21 seconds. The next five-owner regression
  combines this router with committed expert prompt contracts and a bounded
  planner reminder, under the original GPU deadline and combined $25 cap.
  These are development routing results; broad response learning remains open.

- 2026-09-16: Added explicit learned-route mappings for all four installed
  numerical paths. General conversation and the earlier trained structured-data
  model are distinct choices; an assistant fallback cannot silently replace
  that learned skill path. Eight focused graph checks passed in 61.71 seconds,
  including actual five-process execution of both backbones. A four-class router
  fit is frozen on the same data split. The separate three-class conversation
  regression's first allocation stopped at an Ubuntu package lock before neural
  execution; its $0.76 estimated compute and complete retirement are recorded.

- 2026-09-16: Four-model routing passed on the unchanged split: 163/164 raw
  protocol questions, 1,920/1,920 raw directory questions, all 91 structured
  questions and all 55 general conversations. The fit/evaluation took 163.89
  CPU seconds. Complete request data and earlier turns now reach the selected
  general or structured model; a single-request path preserves its raw input
  rather than substituting the planner's shorter rewrite. The actual five-process
  context-preservation check passed in 21.90 seconds. GPU conversation evidence
  for this extended path is still pending.

- 2026-09-16: Published all five owners' conversation and review outputs. The
  corrected three-route service reached 6/8 answers and 28/32 plans. Input-scope
  checks recovered general science answers; the four-model service reached 7/9
  complete answers with exact replay. Two remaining planner errors prevent
  promotion. A review pass worsened 28/32 drafts to 1/32 and is excluded from the
  serving path. All five GPUs, volumes and the temporary security group were
  retired; this allocation cost at most $2.34 compute, storage separate. See
  [the complete results](config/experiments/conversation-routing-results.json).

- 2026-09-16: Implemented an experimental trainable connection between owned
  model activations. It generates one token stream through both backbones and
  every expert tail, sharing the trained prefix once. It starts as the unchanged
  general model and accepts projected source activations instead of rewritten
  subquestions. Seven numerical checks passed in 18.16 seconds: actual five-owner
  generation, exact initial seed tokens, reproducible continuation, two-owner
  gradients matching joint execution, causal masking and request-local caches.
  These are CPU numerical checks; this method has not yet passed an LLM learning
  or quality gate and has no native admission. Task 1 remains open.

- 2026-09-17: The automatic numerical loop completed two real training cohorts
  against four operated CometBFT validators: four updates earned four trial
  NEURO, both measured quality failures were rejected, and serving stayed pinned.
  A driver waiting for the original audit deadline initially missed the shortened
  reveal window; that failed attempt is retained. Auditor recovery now follows
  current deadlines and may retire an unknown signed audit only when committed
  history proves its exact context closed. Five new recovery checks passed.
  The extended numerical integration also passed with publisher restart, actual
  duplicate-source rejection and nine unchanged responses from persistent
  accepted shard processes while cohorts executed (128.51 seconds). The operated
  repetition passed in 406.98 seconds. Four validator app hashes agreed and a
  fresh replay matched all 74 transactions, 449 headers and the exact final state.
  [Source and evidence are public](https://github.com/neuroshard-ai/neuroshard/releases/tag/research-automatic-cohorts-20260917).
  These are synthetic models and one administrator;
  useful cumulative LLM learning and automatic production data operation remain
  open, so neither task 1 nor task 2 is checked.
