# Live LLM checklist

NeuroShard's goal is one useful, openly available assistant whose learning and
serving capacity can grow through contributed model shards, coordinated by its
own permissionless blockchain.

This is the fixed six-item completion checklist. **1/6 complete. Tasks 1 and 2 are active.**
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
  same owned shards, without duplicating the backbone. The four-GPU trial completed 512 updates: new single answers improved from
  2/16 to 11/16 and composed answers from 0/8 to 4/8, preserving all 80 previously
  correct retained answers. It failed its accuracy gate; the final stayed unopened.
  Fresh replay reproduced the final four updates exactly. See the
  [isolated-learning result](docs/CONTINUAL_EXPERT_LEARNING.md#isolated-learning-result).
  A separately frozen [replacement control](config/experiments/expert-replacement-control.json)
  reuses the exact new terminal weights while replacing the earlier expert.
  Learned route aliases let both routes use that fixed capacity; six-process
  execution checked actual routing, replay and owner payments. The first GPU comparison
  failed during setup before generating answers. The next frozen trial includes
  it in the main controller. It avoids duplicate training and does not equate
  lifetime costs.

- [ ] **2. Continuous data admission and learning**

  Done when the network repeatedly admits immutable data with provenance,
  deduplication and contamination/poisoning checks; funds and assigns a bounded
  training job; verifies the work; evaluates the candidate on precommitted
  acceptance rules; and promotes or rejects it without manually editing genesis
  or orchestrating each cohort. Demonstrate restart, rejected data and a rejected
  candidate while the accepted serving model remains available.

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
  quorums and real replay measurements. Task 3 published the complete retained
  window payload catalog and settled all 140 windows with full audits. Complete
  retention/storage/serving costs and a sustainable funding policy remain open.
  The planned-service operator now quotes and meters all prompt and output
  work, including planning and composition, with exact owner shares and a
  bounded reservation. Real five-process replay reproduced the same receipt
  and rejected a forged response. These execution quotes explicitly require
  separate verification and retention funding; they are not ledger payments.

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

## Work discipline

All six milestones remain in scope. Item 3 is complete. Automatic useful
composition and repeated admitted learning now determine execution order;
billing, availability and usable serving address the other fixed milestones
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
