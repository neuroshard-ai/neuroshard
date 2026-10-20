# Design: answering from the latest approved revision of a document

Status: **proposal for review (October 20, 2026). Not implemented.** It targets the release after `pilot-1`.

## What the soak found, and why

In the pilot-1 soak, every wrong answer gave a value from revision 1 of the Cedar 6581 plan when the latest
approved revision is 2. The cause is in the interface, before the model:

- The demo workspace's documents carry structured `project`, `revision` and `status` fields.
- The general documents interface (`interfaces.documents_world`) exposes only `id`, `title` and `content`, so
  the route never sees them.
- All three Cedar 6581 revisions share the title "Cedar 6581 delivery plan". Each one's content even says
  "use the status and revision in the document metadata", which is exactly what was removed.
- The model therefore could not tell the revisions apart. The 27 mismatches show the information was missing;
  they are not evidence about the model's ability to choose revisions.

Every soak reply came through the documents route; its logged route label is `documents`. Whether the drafting
route's interface exposes these fields is not established here, and this design does not rely on it.

## Principles

1. **Selection is deterministic code over trusted metadata, not a model judgment.** Where structured metadata
   exists, the tool chooses the revision, and the model reads what was chosen and cites it.
2. **Metadata comes from the workspace's structured fields, never from document text.** A document whose text
   says "approved" is not approved unless its `status` field says so. The fields belong to the user's
   workspace on their own device; their provenance is the workspace itself.
3. **The model is never silently given a draft.** If no approved revision exists, it is told so.
4. **Historical and comparative questions stay possible:** "revision 1", "as of May", "what changed".

## Definitions

- **Document family:** documents with the same `project` and the same `family` field. Where `family` is
  absent, the title with any revision suffix removed stands in for it. `project` is compared exactly:
  "Cedar 6581" and "Cedar 6581 Annex" are different families.
- **Approved:** `status == 'approved'`, exactly. Anything else (`draft`, `superseded`, missing) is not
  approved.
- **Revision order:** the integer `revision` field. Two approved documents in one family with the same
  revision, or an approved document with no integer revision, make the family **ambiguous**.
- **Latest approved revision of a family:** the approved document with the highest revision, if the family
  is not ambiguous.

## Interface change: `workspace-documents/4`

- **Structured metadata is passed through.** `documents_world` passes through the fields `project`, `family`,
  `revision`, `status` and `effective`, when present, as structured fields. No other fields pass.
- **`search_documents` results** include those fields, plus a `family_status` per result: `latest_approved`,
  `older_approved`, `not_approved`, `ambiguous` or `no_metadata`.
- **A new tool, `current_document(project, family?)`,** returns the latest approved revision of the family,
  with its metadata. It instead returns an explicit `{status: 'no_approved_revision'}` or
  `{status: 'ambiguous', candidates: [...]}`. It never returns a draft as current.
- **`read_document(id)`** is unchanged, apart from adding the metadata fields to its result.
- **Workspaces without metadata** (most real documents) behave exactly as in `/3`. Every result is
  `no_metadata`, and the model answers from text, as today. Nothing is inferred from text.

## Policy and training

- The route's tool prompt adds one rule: for a question about the current or latest version, call
  `current_document` and answer from it, citing its id.
- For an explicitly historical question (a named revision, "as of", "previous", "changed"), read the named
  revision or compare revisions; don't call `current_document`.
- The model needs a new unit trained on `/4` transcripts, grown through the normal loop: a verified recipe, a
  sealed evaluation, and an owner vote. Earlier units and transcripts are untouched. `/3` remains a declared
  interface, so every earlier training and evaluation replays exactly.
- The scorer for these cases checks the cited document id, not just the value, so a stale citation fails even
  when the stale and current revisions hold the same number.

## Failure cases

| Case | Expected behaviour |
| --- | --- |
| Rev 3 is a draft, rev 2 approved | `current_document` → rev 2; the answer cites rev 2 |
| No approved revision | "There is no approved Cedar 6581 plan; revision 3 is a draft" |
| Two approved docs at rev 2 | `ambiguous`: the answer names both and asks which applies |
| Cedar vs Cedar Annex | Separate families, never merged |
| "What did revision 1 say?" | Reads rev 1 and cites it; `current_document` not used |
| Document text says "approved", `status` is draft | Treated as not approved; the text is not authority |
| Real workspace with no metadata | As today (`no_metadata`); no claim about revisions is made |
| Old revision and current one share the number | The answer must still cite the current one; the citation check catches a stale source |

## Compatibility

- This is not a consensus change. It needs a new declared interface version and a new grown unit, and is
  served only once owners enact a version whose policy declares `/4`.
- Clients for `/3` keep working with versions that declare `/3`.

## Acceptance tests

1. **Tool unit tests:** every row of the failure table, with shuffled ids and document order.
2. **A held-out evaluation written by people** (not by the training author model), with these properties:
   - every case cites the expected document;
   - the set includes unrelated families, duplicate and conflicting versions, missing approval, misleading
     text, historical questions, and same-value old and current revisions.

   Scoring checks both the claimed fact and the cited id.
3. **Before training,** the served model with `/4` tools but no new unit, measured on that set. This
   separates the gain from exposing metadata from the gain from training.
4. **The pilot-1 smoke workload rerun:** the Cedar questions answered from revision 2 (61, 2027-05-14), cited.
5. **Every earlier `/3` evaluation** replays unchanged on the earlier versions.

These are evaluated and reported separately from any claim of improved general capability.
