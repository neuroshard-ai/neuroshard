# General route: the parent on the user's documents and on general chat

October 11, 2026. This is the first step toward a useful product: a third route, beside drafting and
scheduling, for ordinary requests. Requests come in two kinds: questions about the user's own documents,
answered with search and read tools, and general chat. The route serves the unchanged Granite 4.1 3B
parent, with no update and no module. Runs used one canonical CPU host each, except L4's and L5's training on one A10G. Compute
across the first seven runs was $5.59; L5 added $0.52, plus $0.19 for a host retired after a setup defect,
and its Bedrock data generation cost about $6.

## What was built

- [`assistant_documents`](../src/neuroshard/evolution/assistant_documents.py): a workspace over free-text
  documents. Its tools are `search_documents` (deterministic BM25 ranking), `read_document`, `calculate`,
  `shift_date`, `submit_answer` and `report_not_found`. An answer must state the expected facts and cite
  exactly the documents it relied on. A missing answer must be reported, never guessed.
- [`assistant_documents_data`](../src/neuroshard/evolution/assistant_documents_data.py): eleven question
  families over 12 fictional personal and work documents per workspace. The questions cover lookups,
  sums across invoices, date offsets and two kinds of unanswerable question. Every workspace contains
  distractors: a superseded itinerary, last year's policy, and a paid invoice next to unpaid ones. There
  are 132 training, 66 development and 132 sealed confirmation cases. The confirmation uses phrasings
  that appear in neither of the other splits.
- [`assistant_chat_data`](../src/neuroshard/evolution/assistant_chat_data.py): eight families of ordinary
  requests, each with a constraint a program can check. They are bullet counts, word limits, JSON keys,
  required and forbidden words, an ending phrase, lowercase, and a final number. There are 96 training,
  48 development and 96 sealed cases.

## Results

| run | documents | chat |
| --- | --- | --- |
| [baseline](../config/experiments/assistant-general-baseline-report.json), development, interface v1 | 18/66 | 48/48 |
| [documents v2](../config/experiments/assistant-general-documents2-report.json), development | 61/66 | — |
| [sealed confirmation](../config/experiments/assistant-general-confirmation-report.json), opened once | **118/132 (89.4%)** | **84/96 (87.5%)** |

**The baseline's documents failures were mostly the interface, not the model.** The model usually found
and read the right document. Then 41 of its 48 failures answered in prose, with the document ID, instead
of calling `submit_answer`. Interface v1 also refused to let the model cite a document it had only seen in
search results. Version 2, fixed on development data before the confirmation was frozen, makes three
changes. It accepts a prose answer that states the facts and cites document IDs. It accepts citations of
documents found or read. And its policy says to search first. Not-found must still be reported explicitly.
The baseline's own transcripts still rescore to their declared 18/66.

**The confirmation did not pass its gate.** The gate was declared before the run opened:

| check | minimum | result | |
| --- | --- | --- | --- |
| documents | 85% | 89.4% | pass |
| chat | 90% | 87.5% | fail |
| not-found reported | 90% | 87.5% (21/24) | fail |

The failures are the model's, and each is specific:

- **Chat arithmetic, 0/12.** The chat route has no calculator, and the parent miscomputes multi-step
  prices; "6 packs at 22 with 6 off" came out as 90. Every other chat family passed 12/12 on wording no
  run had seen.
- **Unanswerable return flights, 3 of 12 wrong.** The model reported the outbound flight as the return
  flight instead of saying it had none.
- **Claim deadlines, 8 of 12 wrong.** The model skipped the policy and assumed 30 days, or shifted the date
  backwards. It made the same errors on development: 1/6.
- **Three further misses**, in hotel codes and annual leave.

## Versions 3 and 4

The first confirmation's failures were addressed in two steps. Each was judged on a fresh sealed split,
with new values and a phrasing set used nowhere before, opened once against the same declared minimums.

| version | change | documents | chat | not-found | gate |
| --- | --- | --- | --- | --- | --- |
| [v3](../config/experiments/assistant-general-v3-confirmation-report.json), confirmation 2 | calculator on chat; three general rules in the documents policy | 108/132 (81.8%) | 89/96 (92.7%) | 13/24 (54.2%) | fail |
| [v4](../config/experiments/assistant-general-l4-confirmation-report.json), confirmation 3 | L4, a documents module learned from 132 verified demonstrations | 118/132 (89.4%) | 78/96 (81.3%) | 22/24 (91.7%) | fail |
| parent alone, confirmation 3 (reported) | v3 policies, no module | 120/132 (90.9%) | — | 21/24 | — |

- **The calculator fixed what it was for.** Chat arithmetic went from 0/12 to 12/12 on confirmation 2.
- **Prompt rules did not generalise.** On confirmation 2's new wording, return flights fell to 1/12, with the
  outbound leg reported as the flight home. Claim deadlines fell to 2/12.
- **L4 learned its templates, not the skill.** On development (GPU) it beat the parent 65 to 64 with chat
  unchanged, so it advanced as declared ([report](../config/experiments/assistant-general-l4-report.json),
  $0.45). On confirmation 3's new wording it scored 118 against the parent's 120, gaining 4 cases and
  losing 6. On claim deadlines it was worse: 3/12 against 8/12. 132 templated demonstrations of eleven
  families are too narrow; this is the same lesson as the router study.
- **Two evaluation defects in confirmation 3**, recorded and not rescored:
  - Six of six "ending" failures ended with the requested words in bold (`**Go for it**.`), and the
    checker does not strip markdown.
  - The new arithmetic wording ("a box of 24 pens costs 37 each box") is ambiguous: 8 of 12 replies
    multiplied pens by box price.

  The declared result stands as a fail; the checker and the wording are fixed for any further
  confirmation.

Across the three confirmations, the unchanged parent is a consistent general assistant. Under v2/v3
interfaces it answers 82–91% of document questions with correct citations, and it meets formatting
requests reliably. Its misses repeat: rule application (claim windows), absent facts that resemble
present ones (return flights), and emphasis around required endings. The authored grammars are now the
limit. Training and testing on a few templates per family keeps measuring the templates.

## Routing

A three-way router (general, drafting, scheduling) was fitted on Granite features of the training turns,
using the packaged [`assistant_turn_router`](../src/neuroshard/evolution/assistant_turn_router.py) with
the general route as its fallback. On development it routed all 318 familiar turns correctly: 114
general, 100 drafting and 104 scheduling. On reworded real requests it routed all 38 drafting turns
correctly and 38 of 39 scheduling turns; the one miss went to general, which is the safe direction. Its
held-out accuracy was 1.000 ([report](../config/experiments/assistant-general-router-report.json)).
Adding a general route does not disturb the accepted units' routing.

## What this means

The parent is already a capable general assistant for formatting and instruction following, and with the
v2 interface it can answer most questions over a user's documents with citations. It is not yet good
enough to ship by the bar set in advance. The fixes are concrete:

1. **Give chat a calculator.** Arithmetic belongs to a tool, as it did for scheduling's free-slot tool, not
   to the 3B model's head.
2. **Learn the two document failures.** They are inferring a missing return flight from an outbound one,
   and applying a policy's claim window. This is a learned unit for the general route, trained on
   verified demonstrations from the training split. Like every earlier cohort, it is judged on a fresh
   sealed split, which this run did not touch because the confirmation is now spent.
3. **Then serve it.** The general route needs owners to keep the parent's cache separate from the
   units'. That is a new served version and needs its own release qualification, as A3 did.

## Version 5: varied wording from a larger model (L5)

Confirmation 3 showed that the authored grammars were the limit, so L5 changes the data and nothing else.

**Data.** Claude Sonnet 4.6 on Amazon Bedrock wrote only the *user's* side: natural ways to ask each of the
eleven document questions and each chat request
([`assistant_phrasings`](../src/neuroshard/evolution/assistant_phrasings.py),
[script](../scripts/generate_general_phrasings.py),
[sets](../config/experiments/assistant-general-phrasings.json), every response recorded in the
`.responses.json.gz` beside it). It never saw a workspace, a fact or an answer. Workspaces, facts,
constraints and expected values are built by our code as before. Every assistant message trained on is
our verified solver's demonstration or the parent's own reply that passed the checker. Each template:

- must keep its placeholders and contain no digits, names or document IDs;
- was filled with values and judged by a separate call that sees only the request and the intended task
  (arithmetic also had to be solved independently twice and match our formula, which removes
  confirmation 3's ambiguous pens-per-box wording);
- belongs to exactly one of three disjoint sets: training (casual style), development (15% of training
  templates held out), and the sealed `confirmation4` (terse/formal style). `confirmation4` shares no
  normalised six-word span with training or with any authored phrasing of an earlier split.

| route | train templates | development | confirmation4 | judged and kept |
| --- | --- | --- | --- | --- |
| documents | 831 | 146 | 279 | 1,397 of 1,415 |
| chat | 757 | 134 | 297 | 1,284 of 1,290 |

Generation and judging used 582k input and 250k output tokens (about $6). For new splits only, the ending
check also strips markdown emphasis and quotes, which fixes confirmation 3's other defect
without rescoring any recorded result.

**L5** is L4's recipe (A2 layers and rank, 256 steps, mixture 6:2) on the 660 `generated-train`
demonstrations, all of which pass the scorer. The replay is 283 of the parent's 320 chat replies that
passed ([declaration](../config/experiments/assistant-general-l5.json),
[report](../config/experiments/assistant-general-l5-report.json)). It ran on one A10G in 17 minutes for
$0.52.

| generated-development (held-out wording) | parent | L5 | minimum |
| --- | --- | --- | --- |
| documents | 60/66 | 62/66 (93.9%) | 85% |
| chat | 43/48 | 42/48 (87.5%) | 90% |
| not-found | 8/12 | 10/12 (83.3%) | 90% |

- **Varied wording helped documents on wording it never saw.** L5 gained 4 cases and lost 2, against
  L4's net loss on new wording. Return flights went from 2/6 to 4/6 and claim deadlines from 4/6 to 5/6.
- **It does not advance, as declared:** chat and not-found are below their minimums, and it lost two
  chat cases. `confirmation4` stays sealed.
- **A data defect.** Every casual-style ending, include and exclude template (267 of 267) leaves the value
  unquoted ("wrap up with Keep at it as literally the final words"). All four L5 ending failures copied
  the trailing instruction. Both exclude failures ("leave the word and out") are failed by the parent too.
  The judge passed these templates. It is fixed prospectively by quoting the value when a case is built.
  L5 is not rescored.

Next: L6 is the same recipe on the corrected chat wording, judged by the same rule on generated-development.
The remaining not-found misses are two answers that give the outbound flight as the flight home.

## Version 6: quoted chat values (L6)

L6 is L5 with one change: generated chat cases quote the word or phrase they constrain
([declaration](../config/experiments/assistant-general-l6.json),
[report](../config/experiments/assistant-general-l6-report.json); one A10G, $0.53).

| generated-development | parent | L6 | minimum |
| --- | --- | --- | --- |
| documents | 60/66 | 62/66 (93.9%) | 85% |
| chat | 45/48 | 46/48 (95.8%), none lost | 90% |
| not-found | 8/12 | 10/12 (83.3%) | 90% |

- **Quoting fixed endings:** 6/6 for both the parent and L6, against 2/6 for L5.
- **L6 does not advance, as declared.** Not-found is one case short, so `confirmation4` stays sealed. Both
  misses are requests like "do my files have anything about flying back from Hobart". L6 answers with the
  outbound itinerary.
- generated-development has now judged two units, and its two not-found misses were read to design the
  next one. The next unit is therefore judged on a fresh development set.

## Version 7: contrasting flight experience (L7), judged on fresh wording

L7 is L6 plus 90 further verified demonstrations on new workspaces: 60 return-flight not-found and
30 outbound flight-time, so the extra experience contrasts the two
([declaration](../config/experiments/assistant-general-l7.json),
[report](../config/experiments/assistant-general-l7-report.json); one A10G, $0.52). It is judged on
`generated-development2` ([sets](../config/experiments/assistant-general-phrasings-development2.json)).
That set was written after L6 in a third style (one line of background, then the request), judged the
same way, and contains no template from any earlier set.

| generated-development2 | parent | L7 | minimum |
| --- | --- | --- | --- |
| documents | 61/66 | 63/66 (95.5%) | 85% |
| chat | 46/48 | 46/48 (95.8%), identical case for case | 90% |
| not-found | 8/12 | 11/12 (91.7%) | 90% |

**L7 advances, as declared.** Return flights went from 2/6 to 5/6. It gained four document cases and lost
two (an invoice sum and a notice period). Its digest `5c4e428b…` is pinned in the
[sealed confirmation4 declaration](../config/experiments/assistant-general-l7-confirmation.json), which
also freezes the case IDs and contents before opening.

### Confirmation 4 (sealed, opened once)

One canonical CPU host, 2.0 hours, $2.11
([report](../config/experiments/assistant-general-l7-confirmation-report.json)). The frozen case hashes
matched, and every episode rescored identically.

| confirmation4 (terse/formal wording) | result | minimum |
| --- | --- | --- |
| documents, parent + L7 | 127/132 (96.2%) | 85% |
| chat, parent + calculator | 90/96 (93.8%) | 90% |
| not-found | 20/24 (83.3%) | 90% |
| *documents, parent alone (reported)* | *118/132; not-found 16/24* | — |

**Not shipped, as declared:** not-found is below its minimum. It is still the best general route so far.
On wording in a style no training or development set used, L7 gained 10 document cases over the parent
and lost 1. Documents and chat clear their bars with margin.

The four not-found misses are one form: terse requests that presuppose the flight ("I have a flight home
from Osaka; tell me when it departs"). L7 answers each with the outbound departure time. Training was
casual and development2 opened with background, so neither contained this presupposing imperative. The
chat misses are "exclude" (8/12, the forbidden word used anyway, as for the parent in every split) and
one each of include and JSON.

Next: an absent fact must not be answerable by a present one that resembles it. This is now the general
route's one failing family. Neither more templates of the same styles nor more passes over them will
settle it. The next round gives experience in every style to date, adds a sealed confirmation5, and
considers a check in the workspace itself.
