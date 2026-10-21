# Design: answering from the latest approved revision of a document (revision 3)

Status: **proposal for review, revised after the October 21 design review. Not implemented.** It targets the
release after `pilot-1`.

Revision 3 (after the review of revision 2) makes these changes:
- families are scoped by `(project, family)`;
- a demo-fixture migration with explicit family IDs is added;
- historical selection includes superseded versions, with defined interval outcomes;
- "in effect now" is separated from "latest approved";
- the no-approved answer states the actual statuses.

Revision 2 addresses three review points:
- it separates unknown metadata from "no approved revision" (R1);
- it gives exact definitions of family identity and of "as of" (R2);
- it evaluates three conditions on one protected test set (R3).

## What the soak found, and what is established

In the pilot-1 soak, every wrong answer gave a value from revision 1 of the Cedar 6581 plan, when the latest
approved revision is 2. The cause is in the interface, before the model:
- the demo workspace's documents carry structured `project`, `revision` and `status` fields;
- the general documents interface (`interfaces.documents_world`) exposes only `id`, `title` and `content`;
- all three Cedar 6581 revisions share the title "Cedar 6581 delivery plan", and each one's text says "use the
  status and revision in the document metadata", which is exactly what was removed.

A counterfactual makes this precise. Swapping the two approved revision numbers changes the correct document
from `doc-88b36697f8` to `doc-292283f7a9`, yet the interface's output is identical in both worlds. The exposed
documents therefore cannot determine the authoritative latest revision.

This establishes **information loss**. It does not yet show that exposing the metadata fixes the 27 mismatches;
condition 2 below measures that. Every soak reply came through the documents route (route label
`documents`). This design does not rely on how other routes expose these fields.

## Principles

1. **Selection is deterministic code over structured metadata, not a model judgment.**
2. **Metadata comes only from the workspace's structured fields, never from document text.** A document whose
   text says "approved" is not approved unless its `status` field says so. The fields belong to the user's
   workspace on their own device.
3. **No draft is ever presented as current. No currency is ever claimed without complete metadata.**
4. **Named-version, comparison and "as of" questions stay possible**, within what the fields record.

## Definitions

**Metadata fields:** structured fields of a workspace document, never parsed from text:

| Field | Values |
| --- | --- |
| `family` | Family identity |
| `project` | The project the family belongs to |
| `revision` | Integer ≥ 1 |
| `status` | `approved`, `draft` or `superseded` |
| `effective_from`, `effective_until` | ISO dates; `effective_until` is exclusive |

- **Family identity is explicit and scoped by project.** A family is exactly the set of documents with the same
  **`(project, family)`** pair, both compared byte for byte. One project may hold several unrelated families,
  and the same `family` value in two projects names two different families. "Cedar 6581" and "Cedar 6581
  Annex" are separate. A document without `family` or `project` belongs to no family. There is no fallback
  from titles, because titles collide and stripping a revision suffix is not a well-defined rule.
- **Approved** means `status == 'approved'`, exactly. **Revision order** is the integer `revision`.
- **A family is complete** when every member has `family`, `status` and an integer `revision`.

### Per-family outcome

Selection over a family returns exactly one of these. It never discards an unknown member to make the rest
look authoritative.

| Outcome | When | Answer |
| --- | --- | --- |
| `known_current` | Complete, with exactly one approved member at the highest approved revision | That document, cited |
| `known_no_approved` | Complete, no approved member | States the actual statuses: for example, "no approved version; revision 3 is a draft" or "every version is superseded; the last approved was revision 2" |
| `ambiguous` | Complete, two approved members at the highest approved revision | Names both and asks which applies |
| `incomplete` | Some member lacks `status` or an integer `revision` | Names the unorderable members; no claim of currency |
| `no_metadata` | No document has a `family` | Answers from text, saying it cannot tell which version is the latest approved |

### Highest approved, effective today, and effective on a date

These are three different questions, and each has its own tool:

1. **"The latest approved revision"** → `current_document(project, family)`: the highest approved revision,
   under the outcomes above. It may be future-effective.
2. **"What is in effect now"** → `effective_document(project, family, today)`: the date case below, at the
   user's current date.
3. **"What applied on a date"** → `effective_document(project, family, date)`.

**What the fields mean historically:**
- `superseded` means the version was approved and later replaced. Its interval
  `[effective_from, effective_until)` records when it applied, and that interval is the only evidence of
  history.
- A member with `effective_from` but no `effective_until` is open-ended: it applies from its start onward.

**Selection on a date `d`** considers every `approved` or `superseded` member, never a `draft`:

| Situation | Outcome |
| --- | --- |
| Exactly one candidate's interval contains `d` | `known_effective`: that document, cited |
| Two or more intervals contain `d` (overlap) | `ambiguous`: names them; no choice is made |
| No interval contains `d`, all intervals well formed | `none_effective`: "no recorded version applied on d" |
| A candidate lacks `effective_from`, or has `effective_until ≤ effective_from` (malformed or reversed) | `incomplete`: the date question is declined |
| A superseded candidate with no interval | `incomplete`: its period is unknown, so the date question is declined |
| A future-effective approved member, `d` before its start | Not a candidate for `d`; the earlier member's interval decides |

Reading a named revision ("what did revision 1 say?") reads that document and cites it, and claims nothing
about when it applied.

### The demo needs a metadata migration

None of the four current demo documents has `family`, so passing the existing fields through would give
`no_metadata` for all of them, and the promised selection would not happen. The fix:

- **A versioned enriched fixture**, `assistant_workflow_data` sample 0 v2, adds explicit IDs:
  - `family: "delivery-plan"` on the three Cedar 6581 documents, with `project: "Cedar 6581"`;
  - `family: "delivery-plan"` on the Cedar 6581 Annex document, with `project: "Cedar 6581 Annex"`.

  They are separate families, because selection uses `(project, family)`. Effective dates are added only
  where the fixture's authors state them, never derived from document text.
- **The original fixture is kept unchanged,** so earlier transcripts replay exactly.
- **All three evaluation conditions use the same enriched fixture.** `/3` ignores the added fields, so
  condition 1 measures today's interface on identical inputs.
- **No title inference anywhere in `/4`.** A workspace without explicit IDs stays `no_metadata`.

## Interface change: `workspace-documents/4`

- `documents_world` passes the metadata fields through, structured, when present. No other fields pass.
- `search_documents` results include the metadata fields and the family's outcome. `read_document` includes
  the metadata fields.
- The new tools `current_document(project, family)` and `effective_document(project, family, date)` return a
  document with its outcome, or the outcome alone. They never return a draft as current or effective.
- A workspace without metadata behaves as `/3`, except that a "latest approved" request is answered with that
  limit stated.
- **Client requirement:** a `/4` route is served only to clients that declare `/4`. Owners enact it only as
  part of a version whose policy declares `/4`. `/3` remains a declared interface, so every earlier training
  and evaluation transcript replays exactly.

## Policy and training

- The route prompt gains three rules:
  - "latest approved" questions call `current_document`;
  - "in effect now" or "current" questions call `effective_document` with today's date, since the highest
    approved version may not be in effect yet;
  - "as of" or "on a date" questions call `effective_document` with that date. Named-revision and comparison
    questions read those documents.

  Every answer cites the selected id.
- A new unit is trained on `/4` transcripts through the normal growth loop: a verified recipe, a sealed
  evaluation, and an owner vote. Earlier units are untouched.

## Failure cases

| Case | Expected behaviour |
| --- | --- |
| Rev 3 draft, rev 2 approved, complete | `known_current`, rev 2, cited |
| Rev 1 superseded `[2027-01-01, 2027-02-01)`, rev 2 approved from 2027-02-01; "what applied on 2027-01-15?" | `known_effective`, rev 1, cited |
| Two intervals overlap on the date | `ambiguous` |
| No interval covers the date | `none_effective` |
| Reversed interval, or superseded member with no interval | `incomplete`; the date question is declined |
| Rev 3 approved from 2027-06-01; "what is in effect" on 2027-05-20 | `effective_document` returns rev 2, not rev 3 |
| Complete family, every member superseded | `known_no_approved`, stating that they are superseded |
| Same `family` value in two projects | Two families; never merged |
| Complete family, nothing approved | `known_no_approved` |
| Two approved at rev 2 | `ambiguous`, names both |
| One member lacks `status` | `incomplete`; no claim of currency |
| No `family` fields | `no_metadata`; says it can't tell which is latest approved |
| Two families share a title | Separate: identity is `family`, not title |
| "Cedar 6581" vs "Cedar 6581 Annex" | Separate families and projects |
| "What did revision 1 say?" | Reads rev 1, cites it, claims nothing about currency |
| "What applied on 2027-05-20?", effective dates present | The member effective on that date |
| Same question, no effective dates | Declined |
| Future-dated approval | Approved, not yet effective |
| Text says "approved", `status` is draft | Not approved |
| Old and current revisions hold the same value | Must cite the current one; the citation check catches a stale source |

## Acceptance tests

1. **Tool unit tests:** every row above, with shuffled ids and document order, and shared titles.
2. **One frozen, human-written test set,** not written by the training author model, covering:
   - unrelated families in one project;
   - duplicate and conflicting versions;
   - missing approval;
   - misleading text;
   - named-version and "as of" questions;
   - old and current revisions holding the same value.

   Every case is scored on **both the fact and the cited document id**.
3. **Three prespecified conditions on that same set:**
   1. the current unit with `/3`;
   2. the same unit with `/4` and its tools;
   3. a new unit trained on `/4`.

   1 versus 2 measures the combined interface and tool change without training. 2 versus 3 measures training.
   Separating metadata visibility from the new tools would need a fourth condition; that narrower claim is not
   made.
4. **Protection of the set:**
   - the training recipe, the evaluation contract and the scorer are frozen before the set is first run;
   - case-level outputs are never used for training or debugging;
   - if results from the set guide development, a fresh final set is written for the reported result.
5. **The pilot-1 smoke workload rerun on the enriched v2 fixture:** Cedar questions answered from revision 2
   (61, 2027-05-14), citation checked. On the original fixture, `/4` gives `no_metadata`, as specified.
6. **Every earlier `/3` evaluation and training transcript** replays unchanged on its own version.

These results are reported separately from any claim of improved general capability.
