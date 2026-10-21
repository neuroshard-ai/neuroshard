# Design: answering from the latest approved revision of a document (revision 2)

Status: **proposal for review, revised after the October 21 design review. Not implemented.** It targets the
release after `pilot-1`.

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

- **Family identity is explicit.** A family is exactly the set of documents with the same `family` value,
  compared byte for byte. One project may contain several unrelated families. A document without a `family`
  belongs to no family: there is no fallback from titles, because titles collide and stripping a revision
  suffix is not a well-defined rule.
- **Approved** means `status == 'approved'`, exactly. **Revision order** is the integer `revision`.
- **A family is complete** when every member has `family`, `status` and an integer `revision`.

### Per-family outcome

Selection over a family returns exactly one of these. It never discards an unknown member to make the rest
look authoritative.

| Outcome | When | Answer |
| --- | --- | --- |
| `known_current` | Complete, with exactly one approved member at the highest approved revision | That document, cited |
| `known_no_approved` | Complete, no approved member | "There is no approved version; the newest is a draft, revision N" |
| `ambiguous` | Complete, two approved members at the highest approved revision | Names both and asks which applies |
| `incomplete` | Some member lacks `status` or an integer `revision` | Names the unorderable members; no claim of currency |
| `no_metadata` | No document has a `family` | Answers from text, saying it cannot tell which version is the latest approved |

### Highest approved, and effective on a date

These are different questions:
- `current_document(family)` returns the highest approved revision, under the outcomes above.
- `effective_document(family, date)` returns the approved member whose `[effective_from, effective_until)`
  contains `date`. It is used only when every approved member has `effective_from`; otherwise it returns
  `incomplete`, and the answer declines the "as of" inference.
- A future-dated approval is approved but not yet effective. A superseded member stays readable, but is never
  current.
- Reading a named revision ("what did revision 1 say?") reads that document and cites it, and claims nothing
  about when it applied.
- Historical approval is only what the fields record. Without effective dates, "what was current in May?" is
  declined.

## Interface change: `workspace-documents/4`

- `documents_world` passes the metadata fields through, structured, when present. No other fields pass.
- `search_documents` results include the metadata fields and the family's outcome. `read_document` includes
  the metadata fields.
- The new tools `current_document(family)` and `effective_document(family, date)` return a document with its
  outcome, or the outcome alone. They never return a draft as current.
- A workspace without metadata behaves as `/3`, except that a "latest approved" request is answered with that
  limit stated.
- **Client requirement:** a `/4` route is served only to clients that declare `/4`. Owners enact it only as
  part of a version whose policy declares `/4`. `/3` remains a declared interface, so every earlier training
  and evaluation transcript replays exactly.

## Policy and training

- The route prompt gains two rules:
  - for "current" or "latest" questions, call `current_document` and answer from it, citing its id;
  - for named-revision, comparison or "as of" questions, read those documents, or call `effective_document`.
- A new unit is trained on `/4` transcripts through the normal growth loop: a verified recipe, a sealed
  evaluation, and an owner vote. Earlier units are untouched.

## Failure cases

| Case | Expected behaviour |
| --- | --- |
| Rev 3 draft, rev 2 approved, complete | `known_current`, rev 2, cited |
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
5. **The pilot-1 smoke workload rerun:** Cedar questions answered from revision 2 (61, 2027-05-14), citation
   checked.
6. **Every earlier `/3` evaluation and training transcript** replays unchanged on its own version.

These results are reported separately from any claim of improved general capability.
