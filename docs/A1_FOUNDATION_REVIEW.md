# A1 closure review: usable foundation and reproducible execution

October 3, 2026. This review checks [A1](../TODO_ASSISTANT.md) against published
results only. It runs nothing new and changes no result.

## The criterion

> Pin weights, tokenizer, runtime and licenses; retain successful conversation,
> instruction and tool-use anchors; complete the workspace primitive baseline and
> process replay; publish feasible shard estimates.

Since September 28, 2026, the usable-foundation check judges the version that
would be served: the pinned parent plus its accepted, verified learned unit,
under the declared serving runtime and selection. On development it needs at
least 6 of 8 primitive workflows with one per primitive family, the original
anchor gate with every protected anchor routed to the parent, p95 at most 180 s,
and two fresh-process replays matching exactly
([rule](../config/experiments/assistant-experience-learning.json)).

## Clause by clause

| Clause | Evidence | Met |
| --- | --- | --- |
| Pinned weights | `ibm-granite/granite-4.1-3b` at revision `c0650403…`, every file SHA-256 checked against [the inventory](../config/experiments/granite-reference-artifacts.json). | Yes |
| Pinned tokenizer | The checkpoint's own `tokenizer.json` pipeline (`ec4c2b41…`) with its fixtures, after the [canonical re-baseline](ASSISTANT_WORKFLOW_CANONICAL_RESULTS.md) found the wrong pre-tokenizer. Every run checks every encode: 1,690 in the update's confirmation. | Yes |
| Pinned runtime | Python 3.12.12, torch 2.10.0+cpu, transformers 5.5.4, tokenizers 0.22.2, the required CPU instructions and numerical environment, all [pinned](../config/experiments/assistant-workflow-canonical-execution.json); every evaluation refuses a different runtime. | Yes |
| Licenses | Granite 4.1 3B is Apache-2.0, recorded in the inventory and in [third-party provenance](../THIRD_PARTY.md); project code is Apache-2.0. | Yes |
| Anchors retained | The canonical anchor gate holds: 19 of 24 anchors, with none of the 18 earlier successes lost. The served version routes anchors to the parent. Forced onto the anchors, the update loses one (`granite-instruction-counts`); that is never served. | Yes |
| Workspace primitive baseline | The canonical baseline completed (bare parent 9/24, 2/8 primitive). The served version solved 8 of 8 primitive development workflows, two per family ([third attempt](ASSISTANT_EXPERIENCE_THIRD_RESULTS.md)). | Yes |
| Process replay | Both declared episodes replayed exactly in fresh processes, and all 24 development episodes matched the round-4 run of the same system on another host. | Yes |
| Latency | p95 94.8 s on development and 103.0 s on the sealed confirmation, both including selection, against 180 s. | Yes |
| Feasible shard estimates | Published as measurements by [A4](A4_SHARDING_REVIEW.md): owners fetch 2.2–2.4 GB each and peak at 4.8–6.2 GB serving. | Yes |
| Accepted, verified learned unit | The round-4 update passed the [third confirmation](ASSISTANT_EXPERIENCE_THIRD_RESULTS.md) (A2): 183/192 against the parent's 119, with no parent success lost. | Yes |

## Verdict

**Every clause is met by published evidence, so A1 is complete.** The served
assistant is pinned end to end, keeps the parent's anchors, solves the workspace
primitives and replays exactly.

## What A1 does not establish

- **Prose quality.** Scoring checks tool use, values and final artifacts; it does
  not grade prose.
- **One workspace.** The tasks are authored fictional workflows in an in-memory
  workspace with no external actions.
- **Execution class.** Exact replay holds within the pinned runtime and CPU
  instruction class.
- **A public assistant.** Interfaces, consent, monitoring and operations are A6.
