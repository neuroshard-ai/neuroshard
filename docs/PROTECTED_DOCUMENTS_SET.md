# Writing the protected documents evaluation set

This set decides whether a documents change actually answers better. It's the only evidence for a quality claim, so
it's written by people, kept apart from training, and frozen before its first run
(`docs/DESIGN_LATEST_REVISION.md`, acceptance test 2).

## Rules that keep it protected

- **Writers never see the training corpus, its prompts or its outputs.** The corpus is written by a model
  (Claude Sonnet 4.6). The set must not be, nor edited from its output.
- **Each case has two people:** a writer (`written_by`) and a different checker (`checked_by`). The checker answers
  the question from the documents alone before reading the gold.
- **The set lives outside this repository** (for example a private bucket). Only `freeze`'s manifest is committed:
  the set's SHA-256, the scorer's and the document tools'.
- **Freeze before the first run.** Then run each condition once. Case-level outputs are never used to train or
  debug; if they guide development, write a fresh set for the reported result.

## What a case is

One small workspace, written like real paperwork (leases, invoices, rotas, policies, letters), and one question
phrased the way a person would type it. JSON, one object per case, in a file with `"format":
"neuroshard-protected-documents-set/1"` and a `"cases"` list:

```json
{
  "id": "p-001",
  "category": "latest_approved_with_newer_draft",
  "written_by": "Initials or name",
  "checked_by": "A different person",
  "question": "What's the monthly rent in the lease we actually agreed?",
  "documents": [
    {"id": "doc-3f2a91c0", "title": "Lime Street lease", "content": "…Rent is 1,150 a month…",
     "project": "Lime Street lease", "family": "lease", "revision": 1, "status": "superseded"},
    {"id": "doc-77b0e4d2", "title": "Lime Street lease", "content": "…Rent is 1,200 a month…",
     "project": "Lime Street lease", "family": "lease", "revision": 2, "status": "approved"},
    {"id": "doc-c91d5e07", "title": "Lime Street lease", "content": "…Rent is 1,260 a month…",
     "project": "Lime Street lease", "family": "lease", "revision": 3, "status": "draft"}
  ],
  "expected": {"kind": "answer", "facts": ["1,200"], "source_ids": ["doc-77b0e4d2"],
               "selection": "latest_approved", "project": "Lime Street lease", "family": "lease"}
}
```

- **Document IDs:** `doc-` plus at least 8 hex digits, unique within the case.
- **Metadata fields** (`project`, `family`, `revision`, `status` from `approved`/`draft`/`superseded`,
  `effective_from`, `effective_until` as `YYYY-MM-DD`): set them as a user's device would, and leave them out or
  wrong where the case is about missing or wrong metadata.
- **`expected.kind`:**
  - `answer`: `facts` are the exact strings the answer must contain, and `source_ids` exactly the documents it must
    cite. `selection` says how the right document is chosen: `latest_approved`, `effective` (with `date`),
    `named` (the question names a revision) or `other`.
  - `not_found`: the documents don't contain it.
  - `decline`: no document can honestly be chosen. `must_cite` lists documents the reply must name (for example
    the two conflicting versions), and `must_not_state` lists values it must not assert.

## Coverage

At least **8 cases in each category**, about 150 in total, spread over at least 30 different workspaces:

| Category | What it tests |
| --- | --- |
| `latest_approved_with_newer_draft` | The newest revision is a draft; the answer is the latest approved one |
| `latest_approved_newest` | The newest revision is the approved one |
| `single_revision` | A family with one document |
| `unrelated_families` | Several unrelated families in one project |
| `ambiguous_versions` | Two approved documents at the same revision (a `decline`) |
| `no_approved_revision` | Nothing approved yet (a `decline` saying so) |
| `incomplete_metadata` | A member lacks status or revision (a `decline`, no claim of currency) |
| `misleading_text` | Text says "approved" or "final" but the status field says draft |
| `named_revision` | "What did version 1 say?": read that one, no currency claimed |
| `as_of_date` | "What applied on 3 March?", with effective dates |
| `same_value_old_and_current` | Old and current hold the same value; only the current may be cited |
| `shared_titles` | Different families with the same title |
| `not_found` | The documents don't say |

Vary the wording: "latest approved", "the one we signed", "current", "final", and questions that never say which
version. Typos and casual phrasing are welcome.

## Checking and freezing

```bash
python -c "import json,sys; from neuroshard.evolution import protected_documents as p; \
  print(p.check_set(json.load(open(sys.argv[1]))['cases']))" SET.json
python -c "import sys; from neuroshard.evolution import protected_documents as p; \
  print(p.freeze(sys.argv[1], sys.argv[2]))" SET.json MANIFEST.json
```

`check_set` refuses a case whose facts are in none of its cited documents, whose gold differs from what the
selection code chooses, or that has the same writer and checker. Fix the case, never the code. Commit only
`MANIFEST.json`.

## How it's scored

The primary score (`protected_documents.score`) treats every condition alike:
- **Answer cases:** every fact appears, and the cited IDs are exactly the gold sources.
- **Not-found cases:** not found is reported.
- **Decline cases:** the required documents are named, and no forbidden value is stated.

The route's own stricter scorer (quoted evidence under `/3` and `/4`) is reported beside it.
