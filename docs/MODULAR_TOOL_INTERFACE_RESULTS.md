# BAR tool-interface result — September 26, 2026

The corrected interface produced **3/3 correct, valid tool calls**, compared
with 0/3 in the original diagnostic. All three independent generations replayed
with identical text, termination, scores and token IDs. The run finished at
**05:48:28 UTC** under committed source `2664b12`, after its CI passed.

| Opened case | Original | Corrected interface | Replay |
| --- | --- | --- | --- |
| Add two integers | Invalid format | Correct named arguments | Exact |
| Weather in Paris | Prose; token limit | Correct function and city | Exact |
| Account balance | Prose; token limit | Correct account and currency | Exact |

This resolves the observed tool-format failure on these three cases. The model
weights were unchanged. Only the system instruction changed; questions, function
definitions, greedy decoding, EOS handling, numerical profile and 32-token cap
remained fixed. Scoring was not relaxed. The separate native validator also
accepted every call. No function was executed against an external service.

These are opened diagnostic cases, not a fresh capability benchmark. The
[original failed baseline](MODULAR_REFERENCE_BASELINE_RESULTS.md) stays failed;
the new calls cannot be spliced into it to claim a new passing baseline.
**A1 remains open; no model is promoted and no training occurred.**

## Reproducible evidence

- [Committed diagnostic contract](MODULAR_TOOL_INTERFACE.md).
- [Unmodified result](../config/experiments/modular-tool-interface-result.json),
  SHA-256 `cf569c62e199f82cb50f2d382b8aec84370158f316689a04f61200e7fff1ce3b`.
- [Full replay replies and execution receipts](../config/experiments/modular-tool-interface-replays.json).
- [Successful CI for the execution commit](https://github.com/neuroshard-ai/neuroshard/actions/runs/36215733419).

After completion, all six saved replies were independently rescored, their
source/task bindings verified, and all three replay comparisons recomputed.
All seven preparation/generation receipts report successful execution.
The accounting was recomputed from those receipts and the original baseline.

New primary and replay execution cost **6,125.411 seconds**; cached artifact
verification cost **78.283 seconds**. Total new worker time was **103.4 minutes**.
The original baseline's **5,203.386 seconds** remain charged, bringing cumulative
evaluation to **11,328.797 seconds**, within the declared eight-hour envelope.
Historical download duration remains unknown. Peak generation RSS was
**6,744,805,376 bytes (6.28 GiB)**. No GPU or additional instance was launched.

Primary calls took about **14–20 minutes each** in this streamed CPU execution.
That is an interface correctness diagnostic, not usable assistant latency.
The controller exited successfully; no diagnostic worker remains running.

## Next decision

Use the corrected interface in a separately frozen A1 comparison, with fresh
conversation, instruction and tool cases, nonempty protected successes, and
an explicit compute/latency budget. Establish the baseline before running the
published modular checkpoint. This result does not authorize an automatic
larger-model launch, a training run or a checklist completion.
