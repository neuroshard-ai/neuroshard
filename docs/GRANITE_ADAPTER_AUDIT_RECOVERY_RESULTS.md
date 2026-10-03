# Granite adapter audit completed: matching implementation, unchanged quality failure

September 26, 2026. **The implementation audit passed. A1 remains open.**
The amended run finished and retired its resources at **20:14:53 UTC**.
Execution commit `40f065165ece473bb55c3977d6ba2c5fe0610f7c` passed
[CI](https://github.com/neuroshard-ai/neuroshard/actions/runs/36267377327)
before allocation. No further job is queued.

## Result

| Check | Result |
| --- | --- |
| Backbone comparisons | **243/243**, with exact original-vocabulary rows |
| Adapter matrices, including scaling and padding | **320/320** |
| Non-target adapter tensors | **240/240 zero** |
| Added vocabulary | Exactly 12 rows; finite and recorded, with input/output head tied |
| Fresh standalone / Switch calls | **16 / 16**, both completed |
| Exact paired texts, output tokens, termination and scores | **16/16** |
| Original Switch output and route replays | **16/16** |
| Standalone matches to the stopped attempt | **16/16** |
| Correct requirement checks | **12/16 for both implementations** |

Both paths still reject the valid four-word response and the valid uppercase
response. Those are the same two regressions that exceeded the original
one-loss limit. Both also retain the earlier mistakes on the negative exclusion
and ending-word cases. The quality result did not change.

The added vocabulary rows are **not zero**: all 30,720 elements are nonzero,
with values from −0.06298828125 to 0.055419921875. Their identical input/head
hash is recorded in the report. The amendment preserved these rows in inference;
it did not mask logits or alter generation to obtain agreement.

## What this establishes

The published standalone and embedded adapter agree on the opened cases under
the declared BF16 profile and explicitly aligned activation. Input, task,
source, worker receipt and activation bindings were independently checked, and
all 32 saved scores were recomputed. The tensor-check receipts cover the complete
declared backbone mapping and adapter inventory.

This closes the integration investigation for this profile. The observed errors
are reproduced with the standalone adapter and are not unique to embedding it
in Switch. This does not identify their training cause, prove equivalent logits
for every input, or establish generalization to new questions.

The default standalone PEFT invocation trap remains documented: its automatic
sequence detector would not activate on these prompts. The explicitly aligned
invocation was fixed in advance and worked in both audit attempts. It does not
make automatic module selection a solved problem.

The [original reference](GRANITE_REFERENCE_RESULTS.md) remains failed. Its
assistant retention evidence remains the earlier 18/18 successful answers;
these audit runs did not repeat the 24 assistant cases or train a new capability.
There is no model promotion, native-network change, admission evidence or
checklist credit.

## Accounting

| Measurement | Amended run | Both audit attempts |
| --- | --- | --- |
| Worker wall time | 217.42 s | 409.57 s |
| Process CPU time | 488.33 s | 771.18 s |
| Conservative instance time | 322.71 s | 616.20 s |
| Conservative compute | **$0.094877** | **$0.181161** |

New cumulative process peak RSS was 9,059,344,384 bytes; this is not an isolated
per-model peak. Storage and transfer charges remain separate from compute.
The instance was terminated, volumes deleted and security group removed.
An independent AWS check at **20:29:41 UTC** confirmed cleanup.

## Next decision

Stop rerunning or tuning these 16 cases. A1 still needs a passing functional
reference under prospective criteria. The next decision is which useful
assistant capability to evaluate through the complete served path on fresh
cases, using the now-audited implementation. Executable task checks must remain
independent of the learned checker. A separate contract is required before a
new run; this result authorizes no expert training.

Evidence: [unchanged raw result](../config/experiments/granite-adapter-audit-recovery-result.json),
[independent rescore and cleanup report](../config/experiments/granite-adapter-audit-recovery-report.json).
The [stopped attempt](GRANITE_ADAPTER_AUDIT_RESULTS.md) and
[execution amendment](GRANITE_ADAPTER_AUDIT_RECOVERY.md) remain recorded.
