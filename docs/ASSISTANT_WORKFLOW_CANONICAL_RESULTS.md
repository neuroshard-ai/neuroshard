# Canonical-tokenizer workspace re-baseline result

**Finished September 27, 2026 at 08:12 UTC; qualification failed.** Execution freeze
`449aecaa9d3665a1a57cb781f5b16d9f2aae0848` passed exact-commit CI. The
[contract](ASSISTANT_WORKFLOW_CANONICAL.md) and its thresholds are unchanged.
Confirmation remains unevaluated.

| Measurement | Defective tokenizer | Canonical tokenizer | Required |
| --- | ---: | ---: | ---: |
| Original anchors | 18/24 | 19/24 | Original gate |
| Prior anchor successes retained | — | 18/18 | Published, not a stop |
| Complete workflows | 2/24 | 9/24 | Not a learning comparison |
| Primitive workflows | 2/8 | 2/8 | At least 6/8, one per family |
| Compound conversations | 0/16 | 7/16 | Measured |
| Correct follow-up rounds | 0/11 | 7/16 | Measured |
| Tool-error messages | 51 | 0 | Measured |
| Episode p95 | 199.97 s | 177.11 s | At most 180 s |
| Worker peak RSS | 11.01 GiB | 10.56 GiB | At most 64 GiB |

All 436 runtime encodes matched `tokenizer.json`. In the same worker, the old
`AutoTokenizer` path failed the conformance check, confirming the defect in the
pinned runtime. The anchor gate passes and adds `granite-tool-convert`. Both prior
workflow successes pass again; seven compound conversations are new successes.
Qualification fails on primitive workflows (date and difference 0/2 each), so no
conditional replay ran. Development headroom exists (9/24, below 20).

## What the traces show

These are diagnostics on opened development episodes, not admission evidence.
Tool use is now mechanically clean; the 15 failures reduce to two behaviors:

- **Version choice by position.** 20 of 24 first reads chose the first approved
  document in the ID-sorted listing. When the older approved revision is listed
  first, the parent usually reads it: 10 episodes cited revision 1 where the
  latest approved revision 2 was required, then faithfully used its values.
- **The six-generation budget.** Six rounds ended with an exact draft but no reply,
  because the parent spent its sixth generation on `save_draft`. It makes one tool
  call per response and sometimes shifts a date twice instead of once, although
  two calls per response are allowed.

One further follow-up added the review interval twice. Each behavior is a procedure
stated in the [coaching card](ASSISTANT_EXPERIENCE_LEARNING.md): compare revision
numbers, combine date offsets and use both call slots. Whether training learns
them is not established by this run.

## Decision

Keep this result failed; the parent alone is not yet a qualified workspace
assistant, so A1 stays open. The earlier interpretation that the parent could not
use tools or follow corrections came from the tokenizer defect. The remaining
failures are specific and verifiable. No training, candidate or native change followed.

## Evidence and cost

The [raw result](../config/experiments/assistant-workflow-canonical-result.json)
has SHA-256 `de899e8bc1f21fddd6416460f68ab77843f19f686606df56e72973642c6c5d30`.
The [report](../config/experiments/assistant-workflow-canonical-report.json) records
per-case diagnosis, source/CI binding and resource receipts. The frozen scorer
independently reproduces the saved outcomes, anchors and encode count.

Workspace generation used 206 model calls, 264,616 input tokens and 8,750 output
tokens. Primary worker wall time was 2,817.99 seconds. Conservative instance time
was 2,927.64 seconds: **$0.86073 compute**, within the $6 allowance. AWS
verification at 08:13 UTC confirms the instance terminated, no tagged volume
remains and the temporary security group no longer exists.
