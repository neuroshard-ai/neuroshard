# Staged answering: execution timeout

The CPU study on commit `310aeb8` stopped at its frozen 600-second training-worker
limit. **127 of 128 planned updates were recorded. No candidate answer comparison
ran, so there is no quality or retention verdict.** All GPUs remained off.

| Stage | Result |
| --- | --- |
| Parent baseline | 0/32 development; 15/32 retention |
| Protected answers | All 15 correct retention identities saved before training |
| Expert training | 64/64 updates; weights and optimizer/RNG checkpoint saved |
| Gate training | 63/64 recorded updates; no gate checkpoint saved |
| Matched-cost control | Not run |
| Automatic candidate answers | Not scored |
| Admission or promotion | None |

The baseline took 168.73 wall seconds. Expansion training stopped after 600.06
wall seconds; total recorded study wall time was 768.82 seconds. The two child
processes consumed 766.88 CPU seconds, including setup and artifact writes.
The 30-minute overall limit was not reached: the per-worker limit stopped the
study. The expert's file hashes match its manifest.

Expert optimization used 294.25 CPU seconds and the recorded gate updates used
279.78. Setup, training probes, checkpoint writes, and the interrupted work also
cost time. The original 600-second worker allowance had inadequate margin.
Lower training losses are not a generated-answer result.

The [public record](../config/experiments/staged-answering-timeout-record.json)
contains the original baseline replies, all recorded training steps, checkpoint
manifest, process receipts, and failure. The [method and execution freeze](STAGED_ANSWERING.md)
remain unchanged. This run stays timed out; it must not be relabeled as a pass.

## Next execution work

Use a separately recorded execution amendment if this method continues. Keep
the expert/gate schedule, data assignments, automatic serving, protected answers,
and quality gates fixed. The saved expert can be restored, but the unsaved gate
phase must start again from its declared initialization. Recovery must preserve
the RNG/optimizer semantics and charge the interrupted and repeated work
explicitly; the matched-cost control cannot ignore retry expenditure.

The amendment needs a realistic worker budget and a global cap before it runs.
This document authorizes no additional run. There is no reason to change the
learning method based on this timeout, and no answer result supports promotion,
larger-model execution, or a completed checklist item.
