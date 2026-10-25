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
"neuroshard-protected-documents-set/2"` and a `"cases"` list:

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

## When the selection code disagrees with your gold

For `latest_approved` and `effective` golds, the kit compares your expected document with what the deterministic
selection code chooses (`selector_report`). **A disagreement never changes your gold, and you never edit a gold to
match the code.** The case's checker (not its writer) decides from the documents and the specification
(`docs/DESIGN_LATEST_REVISION.md`), and records it in the case:

```json
"adjudication": {"by": "Checker's name", "decision": "gold_correct", "note": "Rev 2 is the highest approved"}
```

- **`gold_correct`:** the code is wrong. The case stays, and it counts as a confirmed selection-code error.
- **`unresolved`:** the specification doesn't decide it. The case stays, is listed and is reported separately.
- **The gold itself was wrong:** the checker rewrites it from the documents and the specification, never from the
  code's output, and checks it again.

The set can't be frozen while any disagreement has no recorded decision.

## Checking and freezing

```bash
python -c "import json,sys; from neuroshard.evolution import protected_documents as p; \
  print(p.check_set(json.load(open(sys.argv[1]))['cases'])); \
  print(p.selector_report(json.load(open(sys.argv[1]))['cases']))" SET.json
python -c "import sys; from neuroshard.evolution import protected_documents as p; \
  print(p.freeze(*sys.argv[1:]))" SET.json PROTOCOL.json MANIFEST.json
```

**What the machine checks (`check_set`):**
- IDs are well formed and unique, and every case has two different people.
- Every gold fact is a whole value in a cited document.
- Each case has the structure its category claims. For example, `latest_approved_with_newer_draft` needs a draft
  newer than the latest approved revision; `ambiguous_versions` needs two approved documents at one revision, and
  its `kind` must be `decline`.
- At least 8 cases and 4 distinct workspaces per category, and 30 distinct workspaces overall. A workspace is
  identified by its documents' content, so copies don't count.
- No question is asked twice of the same workspace.

**What people attest:** realistic phrasing, semantic diversity, and the correctness of each gold. The report
lists the two kinds of coverage separately.

**The protocol file** names every condition to be compared, each with its interface, policy file and model root,
plus the seed of the human audit. For example:

```json
{"seed": 20261026, "conditions": [
  {"name": "served", "interface": "workspace-documents/2",
   "policy": "config/experiments/assistant-general-policy-documents5.json", "model_root": "080c9a10…"},
  {"name": "trained-v4", "interface": "workspace-documents/4", "policy": "…", "model_root": "…"}]}
```

**What `freeze` records** (no case content):
- the set's digest and the protocol's;
- the git commit, and the digest of every file the score and the selection depend on: this kit, the documents
  tools, `document_revisions.py`, `interfaces.py`, `public.py` and the episode loop;
- each condition's policy digest.

It refuses uncommitted changes to any of those files. `verify` checks all of them again before each run. Commit
only `MANIFEST.json` and the protocol.

## How it's scored

The score reads **the reply the user sees** under every condition. A structured answer, if the condition makes
one, must agree with it: the same cited documents, and the same gold facts.

| Kind | Passes when |
| --- | --- |
| `answer` | Every gold fact is stated as a whole value and not negated just before it, the cited IDs are exactly the gold sources, and nothing in `must_not_state` is stated |
| `not_found` | The reply says the documents don't contain it, submits no answer, and states nothing in `must_not_state` |
| `decline` | The reply names every `must_cite` document, says why none can be chosen, and states nothing in `must_not_state` |

**How values and citations are compared.** Values are compared by type: numbers as numbers (12,000 isn't 1,200),
dates as dates (14 May 2027 is 2027-05-14), and words at word boundaries. Citations must be complete IDs of
documents in the case.

**Human audit.** A mechanical rule can't tell whether a sentence really asserts a fact. So a pass that mentions a
value from another version, or negates the gold somewhere, is marked contested. Every contested reply, plus a fixed
20% random sample of all the others, is judged by a person who can't see which condition produced it
(`audit_sample`). The reported result is the machine score corrected by those judgments, with their agreement rate.
Protocol correctness (the route's own scorer, which under `/3` and `/4` also requires quoted evidence) is reported
separately.
