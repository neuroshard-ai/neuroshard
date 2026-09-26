# Modular assistant development checklist

**Direction agreed September 26, 2026. Status: 0/6 milestones complete.**

Build one useful, openly available personal assistant whose capabilities can
improve through contributed data and neural modules, with weights partitioned
across peers and coordination through NeuroShard's native blockchain. More
devices should provide useful training capacity, pooled memory and reliable
serving. Model growth must earn its cost through measured improvement.

This is the active checklist for the new learning direction. The earlier
[six-item demonstration checklist](TODO.md) and its results remain historical
evidence. Its **5/6** is not a readiness estimate for this assistant. Existing
implementations may satisfy parts of this plan after validation against the
selected model and execution profile; their previous completion does not
automatically close a new milestone.

## Track completion here

- [ ] **A1. Establish a capable foundation and reproduce a modular reference.**
- [ ] **A2. Add one useful capability without degrading the assistant.**
- [ ] **A3. Repeat useful growth and demonstrate an upgrade or consolidation.**
- [ ] **A4. Run the accepted model across shards and measure the value of more peers.**
- [ ] **A5. Operate the learning and payment loop with independent participants and funded verification.**
- [ ] **A6. Release and sustain the resulting public assistant.**

**Next action:** run the [committed decoder agreement audit](docs/MODULAR_DECODER_PARITY.md)
after exact-commit CI, then revisit the foundation decision. The
[recovered A1 baseline](docs/MODULAR_REFERENCE_FRESH_RESULTS.md)
finished **11/24** and failed its unchanged quality gate. The modular checkpoint
and scheduled replays were not run; temporary resources were retired. The
[recovery amendment](docs/MODULAR_REFERENCE_FRESH_RECOVERY.md) preserved and charged
the earlier interrupted work. The
[tool-interface diagnostic](docs/MODULAR_TOOL_INTERFACE_RESULTS.md) finished:
3/3 correct calls and three exact independent replays, on opened cases. This
resolves their formatting failure but cannot close A1. The
[original BAR baseline](docs/MODULAR_REFERENCE_BASELINE_RESULTS.md)
remains failed at 4/9, including 0/3 valid tool calls. The original
[execution amendment](docs/MODULAR_REFERENCE_EXECUTION.md) and larger modular
comparison have not been run. The
[observable-reasoning candidate](docs/OBSERVABLE_REASONING_RESULTS.md) is closed.

**Order:** A1 → A2 → A3 is the learning priority. Plan A4's memory and network
constraints during A1; perform the distributed comparison after A2 produces a
passing candidate. Independent-operator recruitment can proceed separately;
A5's operational proof needs the selected model and execution profile. A6
requires all preceding milestones.

## Architecture decisions

| Decision | Working rule |
| --- | --- |
| General ability | Start from a capable, openly licensed pretrained assistant. Keep 135M as an implementation fixture. |
| Neural growth | Train compatible expert layers and integrate them inside one model using a validated modular recipe. A choice between finished answers alone does not establish neural composition. |
| Shared weights | Specify which parameters may adapt at each stage. Protect accepted behavior with evaluation and rollback; do not freeze all shared weights forever by assumption. |
| Hardware ownership | A learned expert and a hardware shard are different things. An expert can span machines; a machine can host parts of several experts. |
| Network scale | Use compute groups with measured communication limits. Additional groups can replicate serving and train candidates independently. |
| Cost per answer | Bound active experts, tokens, routing work and network communication. Sparse activation alone does not guarantee constant latency as the catalogue grows. |
| Data | Separate source-backed retrieval, user-controlled personal memory, and data admitted for weight training. Improvement from retrieval is reported separately from learned capability. |
| Native chain | Reuse consensus, commitments, funding and settlement. Correct work and improved assistant quality remain separate decisions. |

## A1. Capable foundation and reference reproduction

**Deliver:** a model/recipe decision record, reproducible baseline, and reference
comparison using published artifacts before training a new expert.

Choose a reference from BAR, FlexOlmo or another documented modular method
based on reproducibility and the assistant use case. BAR is a candidate, not
an adopted dependency or a guarantee of retention. Its reported combined model
has a lower chat score than its initial model; that tradeoff must be measured.

**Done when:**

- Model, tokenizer, code and weight revisions, licenses, artifact availability,
  parameter counts, training stages and hardware requirements are documented.
- The recipe identifies expert initialization, permitted changes to attention,
  embeddings and output head, how shared weights combine, router training,
  and compatibility between model versions. Differences from upstream are explicit.
- Baseline conversation, instruction following and tool-use results exist.
  Every protected category has a predeclared, nonempty set of successful tasks;
  format scoring cannot turn an unusable baseline into a retention claim.
- A published baseline and modular checkpoint run through the same evaluation
  harness. Reproduction tolerances and resource limits were declared before
  inspecting outputs; deviations are explained before proceeding.
- Memory and communication estimates show a feasible route to A4. No full
  reproduction of a large pretraining pipeline is required to close this item.

**Evidence:** pending. **Stop:** unavailable artifacts, unusable baseline,
unexplained reproduction failure or an unaffordable execution profile blocks A2.

## A2. One useful new capability

**Deliver:** a newly trained tool-use extension integrated into the complete
assistant. It should select tools, supply correct arguments and complete
previously unseen combinations of operations in an isolated test environment.

**Done when:**

- A committed plan fixes the data splits, recipe, model interfaces, sample
  sizes, numerical success margins, retention criteria, serving limits and
  total budget before training. The development set and unopened evaluation
  set have distinct roles; familiar examples do not count as new evidence.
- Compare the unchanged parent, a matched-budget no-growth update, and the
  added-capacity candidate. Each receives the same tools, descriptions and
  external information. Account for actual training and serving expenditure.
- Automatic end-to-end task success beats both controls by the frozen margin
  and uncertainty criterion. Forced-expert diagnostics do not count as serving.
  Selection receives no evaluation labels or protected task identities.
- All predeclared must-keep tasks remain successful, and each broader chat,
  instruction-following and retained-skill gate passes. An aggregate score
  cannot hide a failed category. Publish individual regressions.
- The complete model, router, tokenizer and generation policy pass the memory,
  latency and cost gates together. The previous accepted version remains usable.

**Evidence:** pending. **Stop:** any failed gate rejects this candidate. Diagnose
training fit versus generalization before proposing a different learning method;
do not respond to weak experts with more routing heuristics alone.

## A3. Repeated growth, upgrade and consolidation

**Deliver:** a recorded lineage of three successive accepted cohorts under this
new architecture. A2 may count as the first; older fact-cohort demonstrations do
not substitute for this lineage.

**Done when:**

- Each cohort passes its prospectively frozen new-task and cumulative
  retention gates. Evaluation includes tasks combining earlier and later
  capabilities, as well as general assistant behavior.
- At least one subsequent cohort upgrades an existing capability, and at
  least one adds a distinct capability. Every comparison starts from the
  complete previously accepted system, not the original seed.
- Adding, updating and consolidating have declared admission rules. At least
  one upgrade or consolidation passes against retaining the previous version
  under a matched resource budget. Failed consolidation remains a failure.
- Active compute, routing overhead, storage, audit cost and response latency
  stay within the declared growth envelope. Shared-weight changes are evaluated
  as changes to the whole assistant, even when old expert weights are untouched.

**Evidence:** pending. **Stop:** repeated new-skill gains that degrade earlier
behavior, or growth that escapes the resource envelope, do not complete A3.
Three passing cohorts establish bounded repeated growth, not unlimited scaling.

## A4. Actual sharding and useful additional peers

**Deliver:** the accepted assistant and its training checkpoint state run across
separate machines, with no execution worker holding the complete backbone.

**Done when:**

- Reuse and adapt the existing partitioning/recovery implementation. Record
  per-worker weights, optimizer state, activations, caches and peak memory.
- Sharded generation and training meet a declared agreement standard against
  the reference execution; exact versus tolerance-based checks are explicit.
- A declared worker outage and replacement recover within measured limits
  without silent checkpoint changes, lost accepted updates or duplicate rewards.
- A controlled comparison shows what extra peers buy: pooled memory, faster
  bounded training, greater serving throughput or better availability. Include
  network transfers, setup, failures and total device time in that comparison.
- Replicas and placement keep request latency within the assistant's budget.
  More throughput is not reported as a faster individual response.

**Evidence:** pending. Existing [sharding and recovery results](docs/SHARDED_TRAINING_RESULTS.md)
must be revalidated for the new architecture.

## A5. Independent operation, admission and economics

**Deliver:** independent participants operate this model's funded data → training
→ verification → quality → promotion/rejection loop on NeuroShard's own ledger.

**Done when:**

- Adapt the existing lifecycle rather than create another parallel chain
  implementation. Bind the entire answering system in each promoted version;
  data provenance, deduplication and poisoning/contamination checks precede jobs.
- Show accepted and rejected updates, unavailable workers/artifacts, disputes,
  restart, refunds and recovery while the last accepted model remains available.
- Meet the existing [independent-hosting contract](docs/INDEPENDENT_HOSTING.md):
  four independently administered operators, with separate keys and machines.
  Additional instances controlled by this administrator do not count.
- Publish the verification threat model, collusion assumptions and measured
  attack results for the new numerical profile. No optimistic claim or
  signature is presented as an unconditional computation proof.
- Measured honest auditing, failed work, storage, transfers and serving fit a
  declared funded operating envelope, including payments when there is no fraud.
  Full replay may be used if its cost fits; cheaper verification needs its own
  evidence. Finite sponsorship and recurring demand are reported separately.
- Operator admission and job assignment function without private manual
  installation by this project's administrator. Correct work payment does not
  depend on a candidate being promoted; budgeted rejection is accounted for.

**Evidence:** pending. Prior native settlement remains reusable evidence, not
proof of independent operation or sustainable large-model verification.

## A6. Public assistant and release

**Deliver:** a versioned public assistant with documented capabilities, operating
limits, funding and a reproducible route for new participants to join.

**Done when:**

- Real multi-turn conversations, approved tool use and streamed responses use
  the accepted distributed model. Personal memory and prompt privacy have an
  explicit implementation and disclosure; raw private conversations are not
  silently treated as public training data or ledger payloads.
- End-to-end quality, task completion, time to first token, response speed,
  concurrency, outages and cost per successful request pass a predeclared soak.
- The public client, website and operator instructions match the running model,
  network and limits. Published checkpoints and source reproduce the release.
- Reviewed source is merged to `main` with CI passing. Network/model activation
  follows its separately reviewed version, migration and rollback plan. Merging
  code alone never upgrades genesis, token accounting or the serving checkpoint.
- Ongoing hosting and audit costs have actual funding, automatic spend limits,
  and a clear failure/recovery policy. New peers can participate under A5's rules.

**Evidence:** pending. Completion establishes a working public assistant under
published bounds; frontier-model parity and million-device scale remain claims
that require separate evidence.

## Rules for updating this checklist

Check a milestone only when all its criteria have linked implementation,
committed outputs, reproducible scoring and a cost record. Record failed runs
without rewriting their contracts. A specification, green CI, paper result or
downloaded checkpoint alone cannot close a capability milestone.

Before each run, commit its own execution plan with numeric quality/retention
gates, data policy, seed/revisions, full runtime dependencies, resource limits,
stop rule and progress/result locations. Resolve model and budget choices under
the existing authorization before launch. This checklist starts no jobs and
changes none of the earlier CPU-only study freezes or closed evaluation sets.

If a criterion or method must change, record the reason and its effect on past
evidence before the replacement run. Do not silently change this checklist's
completion target. Update the table below when a milestone's evidence changes.

| Date | Milestone | Decision / evidence |
| --- | --- | --- |
| 2026-09-26 | A1–A6 | New direction recorded; all milestones open. Start with reference artifacts and recipe review. |
| 2026-09-26 | A1 | BAR selected. [Decision and reproduction contract](docs/MODULAR_REFERENCE.md) committed before any scored output. Milestone remains open. |
| 2026-09-26 | A1 | Execution amendment adds complete artifact/source binding, enforced worker limits, historical cost accounting and independent replay. Original questions and quality gates remain fixed; no milestone credit. |
| 2026-09-26 | A1 | Original dense baseline completed: conversation 3/3, instruction 1/3, tool use 0/3. Usability gate failed. [Diagnostic and raw evidence](docs/MODULAR_REFERENCE_BASELINE_RESULTS.md) recorded; amended execution and modular comparison remain unstarted. |
| 2026-09-26 | A1 | [Interface audit and separate diagnostic](docs/MODULAR_TOOL_INTERFACE.md) correct omitted call-format instructions, retain the failed record, and freeze three opened CPU cases with successful-call replay. No capability result or milestone credit. |
| 2026-09-26 | A1 | [Corrected interface result](docs/MODULAR_TOOL_INTERFACE_RESULTS.md): 3/3 correct tool calls and three exact replays; 103.4 minutes of new CPU worker time. Opened-case interface check only; A1 remains open. |
| 2026-09-26 | A1 | [Fresh comparison](docs/MODULAR_REFERENCE_FRESH.md) declares 24 cases, baseline usability, full replay, per-answer retention and latency on one disposable 128 GiB CPU host. No result or milestone credit at freeze. |
| 2026-09-26 | A1 | [Pre-execution CI correction](docs/MODULAR_REFERENCE_FRESH_CI.md): first queue stopped before allocation due to a test subprocess import path. No generation or EC2 cost; method and budget unchanged. |
| 2026-09-26 | A1 | [Execution recovery](docs/MODULAR_REFERENCE_FRESH_RECOVERY.md): first CPU allocation stopped after nine generations on process-specific parser error text. Record all receipts, stabilize rejection without changing scores, and freeze one restart within the remaining combined time/cost budget. No completed quality result or milestone credit. |
| 2026-09-26 | A1 | [Recovered baseline result](docs/MODULAR_REFERENCE_FRESH_RESULTS.md): 11/24 (conversation 4/8, instruction 4/8, tools 3/8). Quality gate failed; no modular evaluation or scheduled replay. All saved scores rechecked, resources retired; combined fresh-allocation compute $0.445142. Study closed, A1 open. |
| 2026-09-26 | A1 | [Decoder audit](docs/MODULAR_DECODER_PARITY.md): strengthen local serialized-weight parity checks and freeze standard-generation/logit comparisons on opened BAR-7B outputs. One CPU host, two-hour/$6 incremental allowance, no training or credit. Actual checkpoint agreement pending. |

## Research informing this direction

- [BAR: modular post-training](https://arxiv.org/abs/2604.18473) and
  [released model suite and results](https://huggingface.co/allenai/BAR-7B):
  independently trained experts, stage-specific shared-parameter adaptation and
  router integration. Reported chat falls from 48.9 to 38.7 in the combined model;
  specialist gains do not prove our retention criteria.
- [FlexOlmo](https://arxiv.org/abs/2507.07024): compatible expert training against
  a shared model and composition without pooled training data. Its pretraining
  setting is not interchangeable with assistant post-training.
- [Petals](https://arxiv.org/abs/2312.08361): distributed transformer inference
  and fine-tuning across heterogeneous hosts. This is systems precedent, not
  a solution to permissionless verification, incentives or model improvement.
