# First programming-selector picker

**Status: implementation specified; not yet committed as an execution freeze
in git, and not screened.** This candidate is nearest-training-prompt Jaccard.
It was not fitted on opened diagnosis labels. The evaluation contract remains
[`PROGRAMMING_SELECTOR_CONTRACT.md`](PROGRAMMING_SELECTOR_CONTRACT.md). A CPU
screen without a pinned picker execution freeze is invalid.

## Rule

After the parent public example fails, compare the original question to every
frozen incumbent train prompt and every frozen added-tail train prompt using
the programming-expert alphanumeric word-set Jaccard (the same word rule as
near-duplicate detection, without the 0.8 threshold). Select `added` only when
the nearest added prompt is **strictly** closer. Equal scores, empty overlap,
invalid input, picker errors and the one-second deadline select `incumbent`.
This picker does not abstain.

It reads only `question`. The failed parent program, public example and public
feedback are accepted as unused allowed fields so the serving object stays
exactly the frozen four-key input.

No evaluation-question, task-ID, row-order, split-label, hidden-test, reference
or tail-answer lookup is present. Train prompt assets store texts only.

## Why this candidate, not a fitted rule

The two tails were trained on disjoint public MBPP slices (official train
601–974 versus leftover remainder 12–510). Routing by nearest training prompt
is the serving-time analogue of that split. It does not use the disclosed
unique IDs 276, 503, 265 or 249, and it does not search thresholds on the
opened 38 extras.

The CPU screen may still fail. Failure stops this picker.

## Artifacts

- [`programming-selector-picker.json`](../config/experiments/programming-selector-picker.json)
- [`programming-selector-assets.json`](../config/experiments/programming-selector-assets.json)
- [`programming-selector-picker-freeze.json`](../config/experiments/programming-selector-picker-freeze.json)

```sh
PYTHONPATH=src python scripts/prepare_programming_selector.py \
  --mbpp MBPP.jsonl
# After this picker freeze is committed:
PYTHONPATH=src python scripts/screen_programming_selector.py \
  --phase decide --home STUDY --mbpp MBPP.jsonl
PYTHONPATH=src python scripts/screen_programming_selector.py \
  --phase score --home STUDY --mbpp MBPP.jsonl
```

Do not train a tail or a selector. Do not launch GPUs from a screen pass.
The original 128-task final stays closed.
