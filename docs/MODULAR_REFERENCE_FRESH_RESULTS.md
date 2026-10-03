# Fresh BAR reference result — September 26, 2026

The recovered baseline completed all 24 primary answers on freeze `0494826`.
It scored **11/24**, below the frozen **16/24** minimum. Tool use also missed
the **4/8 per-category** minimum. The controller correctly stopped before
baseline replay or any BAR-5x7B download/generation. The reference comparison
is incomplete and A1 remains open.

| Category | Correct | Required |
| --- | ---: | ---: |
| Conversation/context | 4/8 | 4/8 |
| Instruction following | 4/8 | 4/8 |
| Tool use | 3/8 | 4/8 |
| Total | 11/24 | 16/24 |

This is a quality-gate rejection, not another scorer crash. All 25 workers
(one preparation and 24 generation workers) completed. The malformed literal
that stopped the earlier study remained an incorrect answer and no longer
aborted scoring. The fixed rejection semantics did not change any pass/fail
status on the nine previously generated replies.

## What the result establishes

All 24 saved replies were independently rescored on the controller; their
source/task bindings, pass/fail reasons and native tool-validation receipts
agreed. This is verification of recorded outputs, **not independent neural
replay**. Scheduled replay did not begin because the baseline gate failed.
The larger model has no score, no measured gains/losses and no latency result.

Failures include substantive mistakes: the inventory task answered 15 rather
than 17; the room choice violated the minimum seating requirement; the cheapest
route cost was wrong; and filtering omitted an even number. Other failures
include format violations and malformed tool calls. The label-replacement call
used `approved, not archived` as the replacement value instead of `approved`.
Loosening call syntax alone would not address the ordinary reasoning failures.

Baseline p95 reply time was **77.931 seconds**; p95 first-token time was
**3.913 seconds**. Peak recorded RSS was **14,819,889,152 bytes (13.80 GiB)**.
These are single-request CPU reference measurements, not distributed serving
throughput or public-assistant latency.

## Evidence and costs

- [Unaltered run result](../config/experiments/modular-reference-fresh-result.json),
  SHA-256 `2d4cf929bd791f056c40190378802648f0d8d064ed0f60039c8a6193b0f5d6fa`.
- [Rescored report, worker outcomes and resource receipts](../config/experiments/modular-reference-fresh-report.json).
- [Recovery amendment](MODULAR_REFERENCE_FRESH_RECOVERY.md) and
  [unchanged case plan](../config/experiments/modular-reference-fresh.json).
- [Exact-source CI](https://github.com/neuroshard-ai/neuroshard/actions/runs/36250720669)
  passed before allocation.

The recovery used **718.334 seconds** of evaluation and **152.889 seconds** of
preparation. Including the interrupted fresh study, preparation totals
296.377 seconds. Baseline cumulative evaluation across the previously charged
studies is **12,306.292 seconds**. Older diagnostic preparation and the original
unknown download remain disclosed in their existing records.

The instance, volume and security group were retired at **15:42:14 UTC**.
Conservative compute is **$0.290307** for this allocation and **$0.445142** for
both fresh-study allocations combined, totaling 1,514.089 instance-seconds.
Storage and transfer charges are not included in those compute figures.

## Decision

Close this study with its failed baseline gate. No model was trained or admitted;
no assistant milestone is complete. Do not rerun these opened cases with adjusted
prompts, formatting rules or gates and call that fresh success.

Before attributing this result solely to the pretrained model, audit the custom
layer-streamed decoder against the standard upstream implementation. Exact
repetition of our own decoder cannot establish that agreement. If parity holds,
the foundation/interface needs reconsideration before further modular growth
work. Either a decoder correction or a replacement foundation requires a separate
declared experiment; this result starts no new allocation.
