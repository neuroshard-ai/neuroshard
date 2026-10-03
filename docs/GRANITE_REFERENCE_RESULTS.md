# Granite reference result — September 26, 2026

The inference-only comparison on freeze `551238a` completed, but **failed its
reference-preservation gate**. The parent passed assistant qualification. The
modular checkpoint preserved all 18 successful assistant answers and improved
requirement checking from 8/16 to 12/16. It also lost two correct reference
checks, exceeding the declared maximum of one. A1 remains open.

| Measure | Parent | Modular | Frozen requirement |
| --- | ---: | ---: | --- |
| Conversation | 7/8 | 7/8 | Parent ≥5/8 |
| Instruction following | 5/8 | 5/8 | Parent ≥5/8 |
| Native tool calls | 6/8 | 6/8 | Parent ≥5/8 |
| Assistant total | 18/24 | 18/24 | Parent ≥18/24; keep every success |
| Requirement checks | 8/16 | 12/16 | Modular ≥12/16 |
| Correct positive / negative checks | 7/8 / 1/8 | 6/8 / 6/8 | Modular ≥6/8 each |
| Previously correct reference checks lost | — | **2** | **≤1** |
| p95 generation time | 5.063 s | 5.016 s | Combined ≤120 s |

All 24 ordinary assistant replies matched exactly in text and token IDs between
the parent and the modular checkpoint with adapters inactive. That includes the
six failures; matching outputs do not make those answers correct. The module's
reference gains comprise six newly correct checks and two regressions, net +4.
This is useful evidence of an integrated specialization, not a passing study.

## Where it failed

The requirement-check adapter wrongly rejected these two valid responses:

- `granite-reference-words-yes`: “River stones stay cool.” meets the four-word
  requirement. The parent returned `yes`; the adapter returned `no`.
- `granite-reference-case-yes`: “BLUE, GOLD, TEAL” meets the uppercase requirement.
  The parent returned `yes`; the adapter returned `no`.

The parent was poor at rejecting violations: only 1/8 negative checks passed.
The adapter improved that to 6/8 while reducing positive-check accuracy from
7/8 to 6/8. This explains why a better overall score can still fail the frozen
regression rule. The rule is not changed after observing this tradeoff.

The ordinary assistant still made an availability mistake and a time-arithmetic
mistake, returned fenced JSON on two strict-format tasks, changed capitalization
in a reminder argument, and emitted a malformed length-conversion call. These
remain failures under the declared scoring. Passing this small qualification
set is not proof of a reliable general assistant.

## Verification and limits

All 80 primary replies completed and were rescored independently on the
controller. Source hashes, task identities, individual reply receipts and saved
route traces were checked. Ordinary modular calls used route 0; the explicit
requirement checks activated only their intended adapter and the base prefix.
This verifies recorded execution evidence, not an independent operator's proof.

The reference gate failed, so the six conditional reload/replay calls were
correctly skipped. No new module was trained; no automatic selection or
multi-step composition was tested. No model, token rule or network was promoted.
The published model and runtime were executed on one CPU host, not distributed
across peers. The earlier BAR result is a different case set and is not a
head-to-head benchmark against these scores.

## Evidence and cost

- [Unaltered result](../config/experiments/granite-reference-result.json), SHA-256
  `774f68e6fbf57286fa1a0eca0692acf811af181c64fc473f39fc0321c7beb361`.
- [Rescored report and cleanup receipts](../config/experiments/granite-reference-report.json).
- [Research decision and frozen comparison](ASSISTANT_FOUNDATION_DECISION.md).
- [Exact-source CI](https://github.com/neuroshard-ai/neuroshard/actions/runs/36260376207)
  passed before allocation.

Worker wall time, including preparation, was 325.340 seconds; process CPU time
was 1,418.732 seconds. Cumulative peak RSS was 8,962,760,704 bytes (8.35 GiB);
this is not isolated per-model peak memory. p95 first-token times were 0.424 s
for the parent and 0.410 s for the modular arm on these short requests.

The instance, volumes and security group were retired at **18:15:55 UTC**.
An additional AWS read at 18:20:22 UTC confirmed termination and no remaining
tagged volumes/security group. Conservative compute cost was **$0.121118** for
411.966 instance-seconds. Storage and transfer charges are separate. No GPU
was allocated and no new job is queued.

## Decision

Close this study with its failed reference gate. Keep the qualified parent and
measured preservation as partial A1 evidence. Do not replace a now-qualified
foundation merely because this reference adapter failed, or tune this opened
set into a pass.

Before another paid run, review the two false rejections against the published
adapter's invocation contract using the saved inputs and source. If the interface
is correct, this is a behavioral tradeoff. Any changed method or evaluation
policy then needs a new prospective contract; this study stays failed. A1 is
incomplete, so new expert training and A2 remain gated.
