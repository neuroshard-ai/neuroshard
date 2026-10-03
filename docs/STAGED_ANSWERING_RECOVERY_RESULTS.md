# Staged answering recovery: completed, candidate rejected

The [execution amendment](STAGED_ANSWERING_RECOVERY.md), committed as
`c30625c145512198f8b43035dcf86a504f9b6130`, completed on September 21, 2026.
It restored the saved expert, restarted the gate from its declared initialization,
reproduced all 63 previously recorded gate updates exactly, saved the 64-step
gate checkpoint, and ran the matched-cost control and automatic answer comparison.
The [original timeout](STAGED_ANSWERING_RESULTS.md) remains a separate result.

**The candidate failed both new-answer gain and preservation. Nothing is admitted.**

| Automatic serving arm | New modular-addition answers | Ordinary-addition retention answers |
| --- | ---: | ---: |
| Parent | 0/32 | 15/32 |
| Added expert and learned gate | 0/32 | 4/32 |
| No-expansion control | 2/32 | 3/32 |

The expansion preserved 4 of the 15 protected parent answers and lost 11,
without gaining any other retention answers. Admission required at least two
more correct new answers than both parent and control, and preservation of every
protected answer. Frozen-incumbent, gate-only training, memory, latency and
matched-cost checks passed; they cannot compensate for failed answer quality.

## What the saved replies and routes show

All 64 expansion replies ended with EOS and were parseable under the frozen
scorer. This failure is not an incomplete-answer or output-format result.
The automatic gate selected the added expert for all **174/174 generated-token
decisions**, including EOS: 76 on new questions and 98 on retention questions.
It bypassed the incumbent MLP on every one of those output decisions.

For example, the parent answered `15 + 15` with `30`; the expansion answered
`45`. On `(17 + 23) mod 7`, whose answer is `5`, the expansion answered `17`.
The reconstructed training probe fell from loss 5.26347 to 1.11075. That is a
training signal, not evidence of generalization to these held-out answers.
No separate forced-expert serving comparison was run, so this record does not
establish that routing alone caused the failure. Guaranteed expert access during
training followed by gate training did not produce useful automatic integration
under this recipe.

## Execution and accounting

Recovery took 22.97 minutes of wall time. The recorded lifecycle, including the
prior baseline and interrupted training plus this recovery, consumed 35.70 CPU
minutes. Earlier development, calibration and CI are outside that study total.
No GPU was used.

The control's optimization budget was **916.3823 CPU seconds**: the entire
598.805385-second failed training process plus the entire 317.576915-second
recovery gate process. This charges the 63 discarded gate updates, repeated
updates, checkpointing, probes, and the interruption's unseparated remainder.
The control completed 198 updates using 919.6242 optimization CPU seconds,
overshooting the target by 3.2419 seconds in its final update.

Expansion/control p95 reply times were 1.1040/0.9397 seconds. Their isolated
evaluation peak RSS values were 1,135,378,432/1,139,511,296 bytes. Both bounds
passed, but these measurements describe a rejected answering system.

## Reproduction and disposition

The [complete result record](../config/experiments/staged-answering-recovery-result.json)
contains raw replies and routes, training histories, checkpoint manifests,
process receipts, costs and the frozen scorer's verdict. The unchanged parent
receipt is in the earlier
[timeout record](../config/experiments/staged-answering-timeout-record.json).
Offline verification reproduced the scores, all 63 recovered-prefix updates,
cost accounting and checkpoint-file hashes. The execution sources still match
the committed recovery freeze.

The recorded disposition is **stop this candidate**. This closes the interrupted
comparison; it does not authorize further tuning on its opened cases, GPUs,
1.7B execution, promotion, issuance or checklist credit. The original programming
final remains closed. [Item 4](INDEPENDENT_HOSTING.md) still requires four
independently administered operators and is separate from this learning result.
