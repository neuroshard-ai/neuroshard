# Complete workspace-assistant baseline result

**Finished September 27, 2026 at 04:35 UTC; baseline failed.** Execution freeze
`6e44dcaa766978684af8740fd5d16ce94f35bcd8` passed exact-commit CI. The
[contract](ASSISTANT_WORKFLOW_LEARNING.md) and its quality gates remain unchanged.
All 24 development episodes are opened. Confirmation remains unevaluated.

| Measurement | Result | Required |
| --- | ---: | ---: |
| Original assistant anchors | 18/24 | Original gate and all prior 18 successes |
| Protected prior answers retained | 18/18 | 18/18 |
| Complete primitive workflows | 2/8 | At least 6/8 and one in each primitive family |
| Complete compound conversations | 0/16 | Measured baseline, no separate minimum |
| Complete workflows overall | 2/24 | Not a learned-capability comparison |
| Episode p95 | 199.97 seconds | At most 180 seconds |
| Isolated worker peak RSS | 11.01 GiB | At most 64 GiB |

The two successful episodes are `workflow-bd845e98b603` (copy) and
`workflow-cdda343cccd8` (sum/doubling). Their identities and outcomes are pinned
in the report for retention. This does not establish a usable compound-workflow
baseline or finish A1. Conditional model replays did not run because qualification
failed. Independent deterministic transcript rescoring did run and matches.

## What the saved traces show

These are post-run diagnostics on opened data, not new admission evidence:

- Seven initial user turns completed correctly. None of the eleven executed
  follow-up turns completed correctly; five compound cases stopped before their
  follow-up. Completing the first turn alone cannot pass a compound episode.
- Twenty-two episodes read a document. Eleven first read revision 1 and eleven
  revision 2. Ten initial reads selected a source outside the requested set;
  the remaining revision-1 read belongs to a task that needs both revisions.
  For example, the date and copy failures selected an older approved plan despite
  the request for the latest approved version, then faithfully used its wrong values.
- Eleven episodes emitted invalid tool-call syntax or schemas, producing 41
  rejection messages. Some follow-ups invented tool names such as `read_annotated`
  and repeatedly retried them. Ten additional calls failed argument/object checks.
  Some project lookups used a changed project string and therefore found no record.
- Fourteen rounds exhausted their six-generation limit; one generation failed
  to terminate within its output budget. These are serving failures, not an EC2
  crash or a timed-out study. Two rounds had a correct saved draft but no completed
  reply; they remain failures under the frozen whole-round rule.

All 131 executed tool calls and draft states replay deterministically from saved
model responses. This checks the environment and scoring, not the causal source
of every model error. In particular, the traces do not establish that more training
or constrained output alone would fix version selection and follow-up reasoning.

## Runtime defect found after publication

The pinned `transformers` 5.5.4 `AutoTokenizer` did not use Granite's serialized
pre-tokenizer ([upstream issue](https://github.com/huggingface/transformers/issues/45812)).
All 193 recorded prompts reproduce exactly under that tokenizer and none under the
checkpoint's `tokenizer.json`. Identifiers and digit runs were split differently
from training (`save_draft` as `save`, `_`, `draft`; `2027` as `20`, `27`). Of the
51 tool errors above, 39 call a misspelled tool name and most of the rest corrupt
dates, document IDs or project numbers. This result stays failed. The
[canonical re-baseline](ASSISTANT_WORKFLOW_CANONICAL_RESULTS.md) repeats the
measurement with every encode checked: 9/24 workflows and zero tool errors.

## Decision and next boundary

Keep this result failed. No new weights were trained, no candidate was compared,
and no shared model or native ledger was updated. The old assistant anchors
survive, but this parent plus this interface is not yet a reliable workspace assistant.

The next implementation target is grounded tool selection and stable conversation
state: real tool names and object references, explicit document-version decisions,
and reliable correction of an existing draft. Examine the actual native tool
conversation contract before attributing all failures to model capacity. Any
changed serving policy needs its own declared comparison; diagnostics on these
24 episodes cannot become fresh evidence. Do not silently increase budgets,
count draft-only completions as passes, or resume training under the failed baseline.

## Evidence and cost

The [raw result](../config/experiments/assistant-workflow-baseline-result.json)
has SHA-256 `a76de0e5f88ba03db1dfa7f5a9a3f934ec00d458ef7b62ed25e6e15a87de1bcc`.
The [report](../config/experiments/assistant-workflow-baseline-report.json) records
per-case diagnosis, preserved successes, source/CI verification and resource receipts.
The frozen scorer independently reproduces the saved outcome and accounting.

Workspace generation used 193 model calls, 249,237 input tokens and 8,641 output
tokens. These counts include repeated prompts; original-anchor generations are
additional and their cost is included in the worker/instance totals. Primary worker
wall time was 2,781.42 seconds. Conservative instance time was 2,861.00 seconds:
**$0.84113 compute**, with storage/transfers billed separately, within the $6
planning allowance. AWS verification at 04:52 UTC confirms the instance terminated,
no tagged volumes remain, and the temporary security group no longer exists.
No new allocation is queued.
