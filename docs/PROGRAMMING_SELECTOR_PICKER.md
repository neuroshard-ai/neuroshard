# First programming-selector picker

**Status: CPU screen failed. Stop this picker. No GPU.**
Nearest-training-prompt Jaccard did not recover complementary coverage.
The evaluation contract remains
[`PROGRAMMING_SELECTOR_CONTRACT.md`](PROGRAMMING_SELECTOR_CONTRACT.md).
This is not admission, not a second extra decode, and not a new trained tail.

## Measured screen

Decide hashed 38 picker calls before joining tail answers
(`344c68ac…`). Score then joined the frozen parent, incumbent and added
traces. Controls reproduced the diagnosis: parent 22, incumbent 29,
always-added 31, oracle 32.

| Gate | Required | Measured |
| --- | ---: | ---: |
| unique added recovered | 3 | **2** |
| incumbent successes preserved | 29 | **28** |
| old successes preserved | 12 | **11** |
| selected full-test | 32/64 | **30/64** |
| picker p95 | ≤ 50 ms | 3.6 ms |
| picker errors | 0 | 0 |

The picker chose added on 15 extras and incumbent on 23. It recovered unique
added tasks 276 and 265, missed 503 (chose incumbent), and lost unique
incumbent 249 (chose added), which is one of the 12 leftover successes.
`next` is `stop-this-picker`. `admission_evidence` false.
`gpu_authorized_by_screen` false.

Evidence: `config/experiments/programming-selector-decisions.json` (`344c68ac…`),
`config/experiments/programming-selector-screen-score.json`, and
`.neuroshard/programming-selector-screen-20260920/`.

## Scorer correction

The first score attempt raised `KeyError` on parent-pass rows. Added extras
exist only for the 38 public-example failures. The scorer now loads an added
output only on those extras. Picker rule, assets, and the decide-phase file
were not regenerated; the decision hash remains `344c68ac…`. The correction
does not change the failed gates and does not justify another attempt.

## Rule (unchanged, failed)

After the parent public example fails, compare the original question to every
frozen incumbent train prompt and every frozen added-tail train prompt using
the programming-expert alphanumeric word-set Jaccard (the same word rule as
near-duplicate detection, without the 0.8 threshold). Select `added` only when
the nearest added prompt is **strictly** closer. Equal scores, empty overlap,
invalid input, picker errors and the one-second deadline select `incumbent`.
This picker does not abstain.

It reads only `question`. Train prompt assets store texts only. It was not
fitted on opened diagnosis labels.

A later method needs its own declared experiment. These 64 cases remain opened.
Do not train a tail or a selector. Do not launch GPUs. The original 128-task
final stays closed.
