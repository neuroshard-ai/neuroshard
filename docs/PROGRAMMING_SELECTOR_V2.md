# Second programming-selector picker

**Status: CPU screen failed. Stop this picker. No GPU.**
This was a separately declared experiment, not a retry of `963de13`.
Complementary coverage still exists in the two unchanged tails. This agreement
rule did not connect it. The original 128-task final stays closed.

## Measured screen

Decide hashed 38 calls (`e89117ac…`) before joining tail answers. Controls
reproduced the diagnosis: parent 22, incumbent 29, always-added 31, oracle 32.

| Gate | Required | Measured |
| --- | ---: | ---: |
| unique added recovered | 3 | **1** (276; missed 503, 265) |
| incumbent successes preserved | 29 | **28** (lost 249) |
| old successes preserved | 12 | **11** |
| selected full-test | 32/64 | **29/64** |
| picker p95 | ≤ 50 ms | 5.7 ms |
| picker errors | 0 | 0 |

Question Jaccard preferred added on 15 extras; parent Jaccard on 15; both
agreed on 7. Those 7 included unique added 276 and unique incumbent 249.
`next` is `stop-this-picker`. `admission_evidence` false.

## Rule (unchanged, failed)

Select `added` only when nearest-train Jaccard on **both** the question and the
failed parent program strictly prefer added. Disagreement defaults to
incumbent. Public-example feedback was unused. Not fitted on opened labels.

A later method needs its own declared experiment. Do not iterate this family
on these 64 cases. They remain opened development history. The next experiment
is [learned integration](LEARNED_INTEGRATION.md).
