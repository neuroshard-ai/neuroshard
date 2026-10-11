# General route: the parent on the user's documents and on general chat

October 11, 2026. This is the first step toward a useful product: a third route, beside drafting and
scheduling, for ordinary requests. Requests come in two kinds: questions about the user's own documents,
answered with search and read tools, and general chat. The route serves the unchanged Granite 4.1 3B
parent, with no update and no module. Every run used one canonical CPU host. Total compute was $1.76:
$0.56, $0.35 and $0.85 for the three runs.

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
