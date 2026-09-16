# Native expert graph serving

The first full GPU integration attempt stopped during startup; its
[failure record](../config/experiments/native-expert-live-results.json) is public.
The [corrected run](../config/experiments/native-expert-live-retry.json) retains
the original model, job, numerical profile, native genesis and acceptance rules.
Its result is pending.

The runtime fingerprint includes the CUDA libraries actually loaded by the
process. The tested AMI's login shell selects a different CuDNN library from a
background service with no `LD_LIBRARY_PATH`. Deployment must preserve the
original library selection and pass the fingerprint check before allocating
model weights. Installing the Python requirements alone does not establish that
the loaded runtime is identical. The original profile remains the acceptance
criterion; a launcher mismatch must not be accepted by changing that profile.

The candidate implementation connects the measured parent, preserved interpreter
and learned experts to a separate native quality decision and escrowed inference.
It supports the published
[composed graph](COMPOSED_COHORT.md), including its explicit routing and two-question
grammar. General automatic expert selection remains item 1 of [the checklist](../TODO.md).
This implementation has local numerical and ledger checks; it is not yet an
operated native deployment of the complete 560-update expert.

## Commitments and promotion

`serving_graph` wraps the original research descriptor with the complete parent
and compact expert metadata, interpreter tensor inventory, tokenizer file hashes,
EOS/context bounds, exact interpretation prompt and executor commitment. It
reconstructs the published graph without rewriting parent or expert optimizer
ages. Ownership and parameter counts come from the actual tensor shapes.

The opt-in `expert_lifecycle` profile freezes the earlier graph, the already
validated terminal candidate, and the quality policy before the native replay
job. This is integration of an existing validated trajectory, not admission of
arbitrary future cohorts. Continuous admission still requires item 2.

`claim_expert` continues to pay accepted, nonduplicate prescribed updates without
changing serving. `quality_expert` is possible only after the complete expert and
its feature production have settled. It requires a separate funded native audit
quorum covering the bound quality report. An accepted passing report changes
`serving_root`; an accepted failing report keeps the previous graph and closes
that candidate's quality decision. Neither outcome mints additional tokens.
Unavailable or forged reports preserve serving and permit a retry of the same
frozen candidate and policy.

## Raw questions and complete billing

`infer_expert` commits the accepted graph, raw question, token limit and every
participating provider. It locks a price covering the maximum of every neural
call selected by the committed execution rules:

| Path | Neural calls charged |
| --- | --- |
| Ordinary question | Parent generation |
| Directory question | Preserved interpretation, then directory expert generation |
| Protocol question | Protocol expert generation |
| Explicit composed protocol question | Both protocol expert generations |

`respond_expert` binds actual greedy token sequences and prompt roots for all
those calls, the rendered text and every participating owner's signed receipt.
The numerical auditors reconstruct prompts, execute the graph, check the actual
outputs and reproduce the transcript commitment. A matching signature or a
well-formed token list alone does not settle payment.

The charged amount is actual generated tokens, including EOS and interpretation,
times the request's unit price. Each call's fee is split by participating parameter
count, with deterministic integer remainders. Unused experts receive no inference
payment. The payer receives unused escrow; an expired unfulfilled request receives
the full escrow. Network transaction fees and separately funded audit fees remain
distinct. In-flight requests retain their original graph after promotion.

Questions, outputs and provider receipts are public ledger data in this profile.
The current five-owner control group receives the question, including owners
unused by that particular neural path; configured auditors receive their replay
inputs. This is not a private-prompt service or the final streaming chat API.

## Numerical executor and retention

`sharded.graph_execution.GraphNetwork` loads only each owner's parent/expert and
interpreter portions. It reuses the evaluated generation, interpretation and
composition kernels, verifies installed source and runtime, and records every
actual neural call. Owners agree on the request before selecting process groups
and compare complete outputs afterward. An oversized context is rejected without
leaving unused owners waiting at request completion. No silent truncation occurs.

`sharded.graph_service` re-executes inference claims. `sharded.graph_quality`
re-executes the new questions against both baseline and candidate, using the
original response-quality gates. It checks retained inputs by comparing the
committed deterministic computations: selected models, input construction,
tokenizer, interpretation, executor and numerical profile must remain identical.
Conversation checks additionally bind the original causal input and target mask.
This preserves old errors as well as correct answers. It does not establish
hardware equivalence, artifact availability or improved answers on new inputs.

The published retained cohort contains 1,024 knowledge questions, 768 skill
questions and 256 conversation records. A metadata preflight found all their
computations unchanged in 1.87 seconds. This avoids repeating thousands of
unchanged neural calls during every quality audit. Newly routed inputs still
require actual generation. This is a new audit method, separately checked from
the earlier numerical retention run.

`scripts/run_native_expert_service.py` is a bounded operator queue for the five
owners, not a customer-facing daemon. It supports generation and separate fresh
inference/quality audits, sends idle heartbeats, records results, and stops on an
unavailable operation. Deployment must impose an external deadline and stage the
exact committed inputs. Its initial transport still uses a fixed process group;
dynamic provider replacement and coordinator recovery remain item 4.

## Evidence boundaries

The development branch also supports prospectively committed jobs. Set expert
work to `neuroshard-prospective-expert-work-v1`, replacing `feature_root` and
`batch_roots` with `batch_count`. The prepared records, ordered schedule,
initial checkpoint, optimizer recipe and numerical profile remain fixed before
execution. The existing executor's `--produce-features` mode computes and
durably reads back the new feature bank. `claim_expert_prefix` binds its root
and every batch in the worker receipts and native replay obligation. Only an
accepted prefix populates `expert_work.execution_profile(state)` for training;
a missing or rejected audit leaves training disabled. The producer and auditor
share numerical kernels but execute in separate processes from available bytes.

`neuroshard-prospective-expert-lifecycle-v1` replaces the known `candidate_graph`
with a `candidate_template` containing the exact initial expert. A matching
`neuroshard-prospective-expert-graph-quality-v1` policy commits that template and
the evaluation inputs before training. The terminal graph is materialized from
the settled checkpoint of the complete prescribed job. Incomplete work cannot
be served, and a failed quality decision preserves the accepted graph. Neither
prefix acceptance nor quality promotion issues training rewards. These formats
remove precomputed-output dependencies; repeated cohort activation and general
growth admission remain separate unfinished work. Existing operated profiles
continue to use their frozen source and formats.

Validation of this development path includes new prefix production, independent
execution replay, forged batch rejection, then training from the retained actual
features. The producer discovers the same numerical result without receiving
its expected root. Separate lifecycle checks bind the newly derived terminal
graph and preserve issuance, refunds and serving through rejected audits. These
small-model checks establish execution behavior, not a fresh large-model quality
gain or completion of the live-LLM checklist.

The real five-process CPU checks execute all four serving paths, replay their
outputs, reject fabricated response text and a falsely reported quality pass,
and check that old paths remain identical. Ledger checks cover separate quality
approval, failed/unavailable audits, pinned in-flight graphs, composed budgets,
refunds and supply conservation. The ledger tests seed a post-training boundary;
they do not claim that the real 560-update expert has earned native rewards.
The focused graph, expert-work and portable-lifecycle suite passed 43 checks in
155.36 seconds, including the bounded operator queue's idle and shutdown paths.

The GPU training backend separately
[passed complete prefix production and updates 0–8](../config/experiments/native-expert-execution-results.json).
The next operated integration must verify this serving wrapper on the pinned GPU
profile, settle the complete original expert job, admit its audited graph, and pay
for an answer using earned NEURO. Quorum honesty and actual operator independence
remain explicit assumptions; adding machines under one administrator does not
demonstrate independent operators.
