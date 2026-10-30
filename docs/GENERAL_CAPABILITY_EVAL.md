# General-capability evaluation: a protected baseline and promotion gate

October 30, 2026. **Design, declared before any model run.** No number in this document describes
`a4-general` or any other model. The harness is in
[`general_eval`](../src/neuroshard/evolution/general_eval/), the open development split in
[`data/development`](../src/neuroshard/evolution/general_eval/data/development/), and the gate template in
[`general-capability-gate-template.json`](../config/experiments/general-capability-gate-template.json).

The project lead's rule: *promote a candidate only when it improves across a protected general-assistant
evaluation covering reasoning, coding, writing, factual questions, planning and tool use, without
unacceptable regressions.* Until now every set has been narrow and templated: documents lookups and chat
constraints (`assistant_documents_data`, `assistant_chat_data`), and A3's drafting and scheduling. Those
sets showed repeatedly that training and testing on a few templates per family measures the templates.
This evaluation is broad, freshly written, mostly program-scored, and sealed.

## What is measured

Six areas. Each item is one JSON object: the user's turns, a hidden scoring block, and for tool use a
fixture ([schema](../src/neuroshard/evolution/general_eval/items.py)). Only the user's words, tool
schemas and tool results ever reach a system.

| area | measures | item format | scored by |
| --- | --- | --- | --- |
| reasoning | multi-step arithmetic, algebra, rates, dates, logic puzzles, probability, counting | a word problem; the reply ends with `Answer: <value>` | exact number (stated tolerance), choice letter or normalised text from the last `Answer:` line; numeric items fall back to the last number |
| coding | writing a Python function or class from a spec, fixing a bug, meeting a complexity bound | a spec; the reply has a fenced code block | hidden tests run in the [sandbox](../src/neuroshard/evolution/general_eval/sandbox.py): type-exact comparisons, timeouts, no network |
| writing | doing an ordinary writing task while following its explicit constraints | an email, summary, rewrite, description, letter... with 1-4 constraints | program-checked constraints (paragraphs, words, sentences, bullets, start, end, include/exclude, JSON keys, lowercase, title, no commas); quality by the blind pairwise judge (below) |
| factual | short answers about stable facts, plus false premises and unknowable questions | a question; a short reply ending `Answer:` | accepted-answer set with normalisation and whole-word matching; any listed distractor fails the item |
| planning | producing a plan that satisfies explicit constraints: schedules, orderings, staff rotas, budgets, routes | a planning request; the plan as JSON | a validator per type checks every constraint; `optimal` items must also match the exhaustive-search optimum |
| tool use | choosing and sequencing tools, using results correctly, not acting on ambiguity or impossibility | a request plus a deterministic fixture | tools executed on the fixture; the effects (refunds, messages, events) must equal the expected ones exactly, and the reply must state the expected facts |

Tool use has two tool sets ([tools](../src/neuroshard/evolution/general_eval/tools.py)):

- **`fixture/1`** is generic function calling over customers and orders, currency, weather, contacts and
  messages, a calendar, a catalogue and arithmetic. It has distractors: two customers called Jordan Lee*,
  two contacts called Priya, a shipped order that cannot be refunded, an out-of-stock product, and a
  calendar clash.
- **`documents/2`** is the served documents route's own workspace and scorer, unchanged, over fixture
  documents. They include a superseded policy and a one-way flight.

Every scorer is a declared rule, and every result can be rescored from its transcript
(`runner.rescore`, which re-runs the code tests).

### The development split (open)

There are 120 items, 20 per area, all written for this evaluation on October 30, 2026
([sources](../scripts/general_eval_dev/), [builder](../scripts/build_general_eval_development.py)). The
digest is `split_sha256 = 1433201b…` (recomputed by `build_general_eval_development.py --check`). Each
reference is checked when the split is built:

- all 20 coding references pass their hidden tests in the sandbox;
- all 20 planning references satisfy their validators, and every budget and route marked `optimal` is
  confirmed optimal;
- 12 reasoning answers are recomputed from their formulas.

Tool use is 14 `fixture/1` items and 6 `documents/2` items. Every writing item carries the `writing/1`
judge rubric.

Development is for building and debugging the harness, prompts and scorers, and for measuring per-item
cost and discordance. With 20 items an area's 95% interval is about ±22 points, so development decides
nothing. It is in the repository, so **no candidate may train on it**. Training pipelines must run an
8-gram overlap check against it, as earlier rounds did against their confirmation wordings.

One development defect was found and fixed before any model ran. `writing-dev-001` asked for "exactly
three paragraphs" in an email, which is ambiguous once there is a greeting and a sign-off. The item now
asks for the body only.

## Where judgement is unavoidable: the blind pairwise protocol

Constraint checks say whether a letter is two paragraphs long, not whether it is a good letter. Writing
quality, and any future open-ended items, are judged pairwise
([judge](../src/neuroshard/evolution/general_eval/judge.py)):

1. The judge sees the request and the two systems' final replies as "Response 1" and "Response 2", in
   seeded random order, and then again swapped. It never sees system names, routes, tool calls, or which
   reply is the candidate.
2. It answers `1`, `2` or `tie` under the pinned rubric `writing/1`. The rubric excludes the
   program-checked constraints and length for its own sake.
3. A prompt scores 1 if the candidate wins in both orders, 0 if it loses both, and 0.5 otherwise. The
   order-inconsistency rate is reported as the judge's position bias.
4. **Calibration before use.** About 150 prompt pairs, stratified by family, are each rated by three
   humans under the same rubric, blind and in both orders. The judge is usable for a gate only if all
   three hold:
   - agreement with the human majority is at least 75%;
   - Cohen's κ is at least 0.5;
   - position consistency is at least 80%.

   The calibration sample comes from the development split and from the incumbent's own replies, never
   from sealed items.
5. The judge model, its prompt digest and every verdict are recorded. The judge must not be a model that
   wrote any training data for the candidate. L5-L9's user-side phrasings were written by Claude Sonnet
   4.6, so a Claude judge is a mild risk for wording style. A judge from another family, or a local
   open-weight judge, avoids it.

Judged quality is a non-regression check on writing, not the area's score. The area's score is the
program check.

## Splits and the sealed set

| split | size | where | use |
| --- | --- | --- | --- |
| development | 120 (20 per area) | this repository | harness work, cost and discordance measurement; never trained on |
| sealed-1 | **3,000 (500 per area)** | off-repo, encrypted, with the custodian | baseline of `a4-general`, then one opening per candidate |
| reference | public benchmarks, labelled `provenance.source = reference` | downloaded at run time, never committed | context only; never part of a gate |

**Authoring the sealed set.** A custodian authors it. The custodian takes no part in training candidates
and has not seen their data recipes beyond the published declarations.

- Items are written fresh: by hand, or from private generators whose code, templates and seeds stay
  private. A public generator would let anyone train on its families, which is the template-measuring
  failure again.
- Items cover the same areas and kinds as development, but use new families, topics and wording styles.
- Any drafting help from a model uses a model that wrote no training data for candidates. Every drafted
  item is verified by the builder's reference checks.
- 10% of items are checked by a second person.
- Items share no normalised 8-gram with development, with any training or phrasing set in this
  repository (`config/experiments/*phrasings*`), or with the reference benchmarks.

**Commitment.** Before anything runs, the custodian publishes `items.commitment(items, salt, 'sealed-1')`
as `config/experiments/general-capability-sealed-1-commitment.json`. It holds per-area counts and a
salted SHA-256 over the canonical file. The salt is 32 or more random bytes, kept private until opening,
so the digest cannot be used to test guesses. Per-item hashes are not published.

**Opening.** The custodian runs the declared candidate and the incumbent on the canonical host (see the
cost model). The auditor receives the file and the salt and checks `verify_commitment` before accepting
any score. Only the per-area counts, intervals and gate checks are published. Per-item outcomes and
transcripts stay with the custodian and the auditor, because they reveal items.

**Reuse.** Each candidate is opened once. The project's custom so far, a fresh sealed split for every
confirmation, would cost about 3,000 newly authored items per candidate. The proposal instead reuses
sealed-1 for at most three candidate openings, releasing only aggregate counts each time (about 6 × 9
bits per opening). It is retired after the third opening, or at once if any item-level material leaks.
It then becomes an open development set, and sealed-2 replaces it. This is an open question for the
project lead (below).

**Contamination.**

- *Fresh items by default.* Facts are public by nature; what is fresh is the wording, the combination,
  and every reasoning, planning, coding and tool fixture.
- *Public benchmarks only as labelled reference sets*, run separately and never in a gate. Granite's
  training data very likely contains them or close variants. Licences recorded here must be verified at
  the source before any download:

  | set | area | licence (to verify) |
  | --- | --- | --- |
  | GSM8K | reasoning | MIT |
  | HumanEval | coding | MIT |
  | MBPP | coding | CC BY 4.0 |
  | IFEval | writing constraints | Apache-2.0 |
  | TriviaQA | factual | Apache-2.0 |
  | BFCL | tool use | Apache-2.0 |
  | MMLU | factual | MIT |

  None is committed to this repository, and none may be trained on.
- *Training data checks.* Every candidate's declaration lists its training corpora's digests. The gate
  refuses a candidate whose corpora share an 8-gram with development or (checked privately by the
  custodian) with sealed-1.

## Sizes from statistical power

Greedy decoding is deterministic, so all the variance is in which items were drawn. Candidate and
incumbent run on the same items, so the right test is paired. What matters is the *discordant fraction*
`d`: the share of items that exactly one of the two systems passes. Past rounds measured it for unit
changes on the same parent:

- L5 and L7 on development: 6 of 66 (9%);
- L7 against the parent on confirmation 4: 11 of 132 (8%).

The planning range below is 8–15%, and the first development and sealed runs will measure it. Every
number is recomputed by [`stats.power_table`](../src/neuroshard/evolution/general_eval/stats.py) and
asserted in the tests.

**Items per area to detect a 5-point paired change** (McNemar, Connor's formula, 80% power):

| d | two-sided α = 0.05 | one-sided α = 0.05 |
| --- | --- | --- |
| 0.08 | 249 | 196 |
| 0.10 | 312 | 246 |
| 0.12 | 375 | 295 |
| 0.15 | 469 | 369 |
| 0.20 | 626 | 493 |

For comparison, two independent samples near a 50% pass rate would need 1,570 per system. Pairing is
what makes a 5-point target affordable.

**Non-regression across six areas is the binding constraint.** A truly equal area passes a 5-point
margin, checked with a one-sided 95% bound, with these probabilities. "All six pass" is for a candidate
equal in every area:

| items per area | d = 0.08 | d = 0.12 | d = 0.15 | all six pass, d = 0.12 |
| --- | --- | --- | --- | --- |
| 400 | 0.971 | 0.893 | 0.826 | 0.507 |
| 500 | 0.989 | 0.943 | 0.893 | 0.704 |
| 600 | 0.996 | 0.971 | 0.935 | 0.838 |

At 400 items per area, a harmless candidate would be blocked by chance about half the time at d = 0.12.
**Sealed-1 therefore has 500 items per area (3,000).** That size:

- detects a 5-point change per area for any `d` up to about 0.16 (two-sided, 80% power);
- gives a harmless candidate a 70–94% chance of clearing all six margins (d = 0.12 down to 0.08);
- puts each area's absolute rate within at most ±4.4 points (Wilson, 95%, at a 50% rate).

The macro average over six areas has a standard error of about 0.6–0.7 points, so the macro improvement
check detects a gain of about 1.5–1.8 points: for example, 5 points in one area with no change in the
other five, plus a little noise.

**Writing quality (judged).** Each prompt's score is 0, 0.5 or 1, with variance (1 − t)/4 for tie rate
t. At t = 0.3, detecting a 5-point preference shift needs 550 prompts, and 7 points needs 281. All 500
sealed writing prompts are judged, which is 1,000 judge calls per comparison. A 7-point non-regression
margin then has about 98% power for an equal candidate.

## The promotion rule

[`gate.decide`](../src/neuroshard/evolution/general_eval/gate.py) applies a declaration copied from the
[template](../config/experiments/general-capability-gate-template.json), filled and committed before the
split opens. A candidate is promoted only if every check passes:

1. **Same items.** Both systems completed exactly the items of the committed sealed split. A system error
   or an unsupported item counts as a failed item, and their counts are capped separately.
2. **No unacceptable regression, in each area.** The lower end of the stratified paired-bootstrap 90%
   interval of (candidate − incumbent) must be at least −5 points. That interval is a one-sided 95%
   bound, with items resampled within families and 10,000 draws at a declared seed. Each area must pass
   on its own. This is an intersection-union test, so no multiplicity correction applies.
3. **Improvement.** By default (`macro`), the lower bound of the macro-average difference must be above
   0. Alternatively (`targets`), a declaration names the areas it claims to improve, and each must have a
   Bonferroni-adjusted lower bound above 0.
4. **Judged writing.** The candidate's blind pairwise preference must have a lower 95% bound of at least
   0.43. The judge must have passed calibration, with position consistency of at least 80%.
5. **Preservation of the narrow sets.** Each existing set must meet its own declared minimum on a *fresh*
   sealed split, because confirmation7 is spent:
   - documents ≥ 85%;
   - chat ≥ 90%;
   - not-found ≥ 90%;
   - A3's drafting, scheduling and cross-capability minimums as A3 declared them.

   A3 must also reproduce exactly where its units are unchanged, as the general release check required.
   The template carries the general-route numbers (113/132, 87/96 and 22/24). It leaves A3's as nulls,
   which `decide` refuses until they are filled.
6. **Served-equivalence.** As before, the candidate must then pass its own release qualification through
   paid shards before the owners' vote.

The incumbent's sealed-1 rows are computed once, by the baseline run, and reused while `a4-general`
remains the incumbent. Greedy runs on the canonical host are deterministic, and the rows are bound to the
split commitment and the source commit.

## What "the served version" means here, and what the harness runs

`a4-general` routes every turn. A top router chooses A3 (whose own router then picks drafting or
scheduling), `documents` (L9 on the parent, policy `documents5`) or `chat` (C1 on the parent, policy
`chat-tools/1`). The chat route has the calculator and `check_text`, at most 400 new tokens per
generation, and 8 model turns.

[`ServedSystem`](../src/neuroshard/evolution/general_eval/runner.py) runs each item through
`assistant.public.Session` with the version's pinned route policies, exactly as `network.Conversation`
does: route per turn, then that route's responder, system prompt, tools and limits.
[`served.a4_general`](../src/neuroshard/evolution/general_eval/served.py) binds the policies, checked
against the descriptor's digests. `served.local_route` computes the accepted message-mean router feature
from the frozen parent. The general release check showed this local stack reproduces paid network
conversations token for token (324/324) on the canonical CPU host, so a local run there *is* the served
behaviour.

What this exposes about the served version, all of which is measured rather than worked around:

- **Caller-defined tools are not served.** No route accepts tools from a request, so for a served system
  `fixture/1` items are recorded as `unsupported` and scored as failures, counted separately. Served tool
  use is the documents route plus the chat route's calculator and `check_text`. A model-level
  approximation (`ModelSystem`: parent + unit under a plain system prompt, given the fixture tools) can
  be reported beside it, labelled as such, and never gated.
- **General requests routed to A3.** If the router sends, say, a planning request to drafting or
  scheduling with no user files, A3's calendar workspace cannot be built. The item is recorded as a
  failure (`routed to … without a workspace`), and each area's route histogram is reported.
- **Length.** A chat-route answer longer than 400 tokens in one generation does not terminate, and fails.

`ModelSystem` takes any `generate(messages, tools) -> reply` and owns the tool loop. Reply means text, or
`cached_responder`'s record. It is for local models, stubs and approximations. Both systems produce the
same report: per area, counts, Wilson 95% intervals, families, routes, unsupported and errors; overall
and macro rates; the sandbox isolation actually used; and cost fields (instance, $/h, wall and model
seconds, generations, generated tokens, compute $, judge calls and $). `baseline_claim` is false unless a
declared baseline run sets it.

**Code sandbox.** Each solution runs in a child `python -I` in an empty temporary directory with:

- an empty network namespace (`unshare --user --net`; required for sealed runs);
- an audit hook refusing sockets, subprocesses, exec/fork/spawn and ctypes;
- limits on CPU, address space, file size and open files;
- a wall-clock timeout that kills the process group;
- a scrubbed environment.

The hidden tests are read and deleted before the solution runs. Results carry a per-run nonce sent on
stdin, and comparisons are type-exact, so a custom `__eq__` cannot pass. It is built against careless
generated code, not code crafted to forge results from inside the interpreter. Implausible pass patterns
are reviewed by hand.

## Cost model and the recommended baseline run (a proposal: nothing here is approved or spent)

Measured anchors from earlier runs:

- **Canonical CPU host, local served stack** (r7i.4xlarge, Sapphire Rapids AMX bf16, 8 threads,
  $1.058/h). Confirmation 7 ran 360 tool-using episodes sequentially in 9,742 s: 24–33 s per episode,
  for 115–153 generated tokens over 2.8–4.2 generations
  ([resources](../config/experiments/assistant-general-l9-confirmation-resources-finished.json): 9,929 s,
  $2.92).
- **Network path:** about 85 s per paid conversation now, and 109 s in the general release check (324
  conversations, 9.8 h). That check's temporary client cost $4.32, on top of the owner hosts, the shadow
  chain and the operator's time.
- **GPU** (A10G, g5.2xlarge, $1.212/h; L40S, g6e.xlarge, $1.861/h). Earlier unit rounds trained and
  evaluated development in about 17 minutes for $0.52–$0.64.

General items generate more than confirmation 7's lookups did: code, letters and worked reasoning. The
estimate is about 350 generated tokens per item, over about 2.5 generations. Scaling confirmation 7's
measured rate gives **45–90 s per item on the canonical host (central 60 s)**.

| how | per item | full baseline, 3,120 items (sealed-1 + development) | faithful? |
| --- | --- | --- | --- |
| network, paid conversations | 85–109 s | 74–94 h serving; about $33–41 client time plus owner hosts and the shadow chain | yes, but it touches the network; not needed, given proven equivalence |
| **canonical CPU host, local served stack** | 45–90 s (60) | 39–78 h of host time: **$41–$83 (central $55)**; 6 hosts in parallel take about 9 h wall at the same cost plus about $3 setup | **yes**: the served numerics, token for token |
| one GPU (A10G), local served stack in transformers | 10–20 s | 9–17 h: $11–$21 | approximate: same weights, routing and policies; different numerics, so not token-identical to serving |
| this 4 vCPU development box (Cascade Lake, no AMX) | about 200–400 s | 170–350 h | no: other ISA and thread count, so not served numerics; and the Granite weights are not here |

Judging on a candidate comparison adds 1,000 judge calls. With a hosted model at about $3 per million
input tokens and about 1.5k tokens per call, that is roughly $5; a local open-weight judge on the GPU
host costs only its time. Human calibration is about 150 pairs × 3 raters × 1 minute, or about 8
person-hours. Authoring sealed-1 is custodian time, plus at most about $10–30 if model drafting help is
approved.

**Recommendation (needs approval):**

1. **Step 0, development baseline, about $3.** Run `a4-general` on the 120 development items on one
   canonical host through `served.a4_general` with `local_route` and the pinned units: about 2 h with
   setup. First, reproduce 10 confirmation7 conversations token for token, to prove the host is
   canonical.

   This measures the real per-item time, the route histogram, the discordance proxy and scorer defects.
   It publishes development numbers only, labelled as development. Any defect found is fixed before
   sealed-1 is committed.
2. **Step 1, sealed-1 baseline, about $55 (range $41–$83).** Run it on canonical hosts, sharded across up
   to six hosts for about a 9 h wall, after the commitment and this declaration are committed. Publish
   per-area counts and intervals, the route histograms, unsupported and error counts, and cost.
3. Do not use the network path for the baseline. Do not use the GPU for anything that is gated. A GPU run
   may be added as a cheaper engineering signal, reporting its agreement with the canonical CPU results
   on development.

The total requested for steps 0 and 1 is a ceiling of **$90 compute, no Bedrock**. Judging and human
calibration are only needed once a candidate exists.

## Verification so far (no model numbers)

- **Tests:** 35, all passing, in `tests/evolution/test_general_eval_{sandbox,scorers,harness}.py`. They
  cover:
  - the sandbox refusing loopback and external connections, with each layer alone and both together,
    against a live local listener that must receive nothing;
  - busy-loop and sleep timeouts, the memory limit, refused subprocesses and ctypes, hidden tests being
    unreadable, forged result lines, and custom `__eq__`;
  - every scorer, with passing and failing replies;
  - every development reference plan validating, and optimality being enforced;
  - the tool loop and its fixture effects;
  - the served Session path on the a4 policies: route policies and tools, unsupported caller tools, and
    misrouting;
  - rescoring, the salted commitment, judge blindness, order balance and calibration, the gate
    promoting and blocking, and the power table.
- **Smoke run** ([report](../config/experiments/general-capability-smoke-report.json), 3 development
  items per area, CPU, about 30 s):
  - a scripted oracle replying with references passes 18/18 and rescores to 18;
  - the test suite's tiny random Granite-architecture model, behind the real `cached_responder`, passes
    0/18, both directly and under a4's route policies with a stand-in router;
  - on that served path the 3 `fixture/1` items are recorded as unsupported, as designed.

  This proves the pipeline only. **It is not a baseline.**

## Risks

- **Discordance may be larger than planned.** A general-purpose candidate that changes more than one
  unit may disagree with the incumbent on 15–20% of items. Then 500 per area detects about 6 points, not
  5, and harmless candidates fail non-regression more often. Measure on the first opening, and resize
  sealed-2 if needed.
- **Scorer strictness is part of the measurement.** Exact facts, Answer-line extraction and whole-word
  matching will mark some correct but oddly formatted replies wrong, as confirmation 3's bold endings
  did. The same rule applies to both systems in a paired gate, so the bias mostly cancels in differences
  but not in absolute baselines. Rescore audits of a sample of failures must be published with the
  baseline.
- **Abstention items are marker-matched.** False-premise and unknowable questions are scored by phrase
  lists. Those are the least robust items; keep them a small share and audit their failures.
- **Served-version gaps dominate tool use.** `a4-general` cannot take caller tools, so the tool-use
  baseline will be low by construction. A candidate that adds a tool interface will "improve" there. That
  is the intended reading, but it must be stated whenever the number is quoted.
- **Routing into A3** can turn general requests into hard failures. The route histogram will show it.
- **Sandbox threat model.** The sandbox is built against careless code, not adversarial code inside the
  interpreter. It is acceptable for a 3B assistant whose code is also read when results look implausible.
  A stronger isolation layer (container or gVisor, with `--network none`) can wrap the same interface.
- **Judge bias.** Position bias is measured. Length and self-family bias are mitigated by the rubric and
  the judge choice, not eliminated.
- **Fact drift.** Factual items were chosen to be stable. The custodian re-checks the accepted sets
  before each opening.

## Open questions for the project lead

1. Is sealed-1 reused for up to three candidate openings with aggregate-only release (the proposal), or
   is a fresh sealed set authored for every candidate (the strict reading of current practice, with
   about 3,000 items per candidate)?
2. Who is the custodian, and who is the auditor? Neither may train candidates.
3. Should improvement be `macro` (the default), or must each candidate name target areas?
4. Margins: 5 points for program-checked areas and 7 points for judged writing. Are these acceptable?
5. Should generic function calling (`fixture/1`) count in the gated tool-use score for served versions
   now, while it is unsupported, or be reported separately until a version serves caller tools?
6. Should general items run with an empty user workspace (the design), or with a standard file set that
   a real user would have? The answer changes what happens when a request is routed to A3.
7. Approval of steps 0 and 1 above: at most $90 compute and no Bedrock. The judge (hosted or local) and
   paid human raters are to be decided when the first candidate exists.
