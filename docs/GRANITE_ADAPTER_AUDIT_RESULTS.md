# Granite adapter audit: partial result and structural stop

September 26, 2026. **The audit stopped; A1 remains open.** The exact execution
commit `302c254e8b81ad41ea4f1f0116e788a1d5f49cb8` passed
[CI](https://github.com/neuroshard-ai/neuroshard/actions/runs/36263183266).
All temporary resources were retired at **19:03:09 UTC**. No retry is queued.

## Completed evidence

All 16 standalone PEFT generations completed with explicitly aligned activation.
Independent rescoring found **12/16 correct**. All 16 texts, output token sequences,
termination flags and scores exactly match the **earlier** Switch run, including
the word-count and uppercase false rejections that failed its retention gate.

Source, task, request/reply and actual activation receipts were checked. Input
tokens and prompt hashes match the earlier parent prompts. The default PEFT
activation mismatch remains recorded separately; it was corrected through the
explicit offset declared in this audit, without changing the inputs or weights.

This is an informative comparison with saved outputs. It is not a completed
same-run comparison: **zero fresh Switch generations** ran in this allocation.

## Why execution stopped

The first backbone check raised:

```text
composed backbone differs: model.embed_tokens.weight
```

The audit incorrectly compared whole embedding tensors by digest. The published
configurations declare **100,352 parent vocabulary rows** and **100,364 Switch
rows**, adding 12 control-token rows. The pinned upstream composer explicitly
resizes token embeddings after transferring base weights. The tied output head
also uses the expanded vocabulary. Whole-tensor equality is therefore the wrong
test here. This is a defect in our audit, not evidence of a corrupted model.

The stopped receipt does **not** show whether every original vocabulary row
matches, nor whether there are other tensor differences. Do not infer that
expansion is the only difference. The remaining backbone checks, adapter tensor
checks and fresh Switch replay are incomplete.

The [original audit contract](GRANITE_ADAPTER_AUDIT.md) stopped as declared. It
must not be restarted under the same freeze. A separate execution amendment
should compare all original vocabulary rows exactly, account explicitly for
the added control-token rows and tied head, and retain exact checks elsewhere.
It must preserve and charge this attempt before any further allocation.

## Accounting and interpretation

| Measurement | Result |
| --- | --- |
| Standalone calls | 16/16 completed; 12 correct |
| Exact matches to earlier Switch outputs | 16/16 |
| Fresh Switch calls | 0/16 |
| Worker wall / process CPU | 192.14 s / 282.84 s |
| Cumulative process peak RSS | 9,152,724,992 bytes; not an isolated model peak |
| Conservative instance time / compute | 293.48 s / **$0.086284** |
| Remaining instances / volumes / security group | None; independently checked at 19:17:21 UTC |

Storage and transfer charges are separate from compute. There was no training,
promotion, quality-gate change or checklist credit. The earlier reference study
remains failed. The standalone outputs reproduce its errors under the aligned
BF16 invocation; the full integration audit remains incomplete.

Evidence: [unchanged raw result](../config/experiments/granite-adapter-audit-result.json),
[rescore, provenance and cleanup report](../config/experiments/granite-adapter-audit-report.json).
