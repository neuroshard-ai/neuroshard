# Third programming-selector picker

**Status: CPU screen failed. Stop this picker. No GPU.**
This was a separately declared experiment, not a retry of `963de13` or of the
v2 prompt-Jaccard agreement picker. Complementary coverage still exists in the
two unchanged tails. This AST-shape rule did not connect it. The original
128-task final stays closed.

## Measured screen

Decide hashed 38 calls (`b08ab22d…`) before joining tail answers. Controls
reproduced the diagnosis: parent 22, incumbent 29, always-added 31, oracle 32.

| Gate | Required | Measured |
| --- | ---: | ---: |
| unique added recovered | 3 | **0** (missed 276, 503, 265) |
| incumbent successes preserved | 29 | **29** (kept 249) |
| old successes preserved | 12 | **12** |
| selected full-test | 32/64 | **29/64** |
| picker p95 | ≤ 50 ms | 2.0 ms |
| picker errors | 0 | 0 |

The picker chose added on 8 extras and incumbent on 30. Those 8 did not include
any unique-added task. Score matches the incumbent extra policy. `next` is
`stop-this-picker`. `admission_evidence` false.

## Rule (unchanged, failed)

Select `added` only when the failed parent program's AST node-type set is
strictly nearer a frozen added-tail training gold program than any incumbent
training gold program. Unparseable parents and ties default to incumbent.
Question, public example and public feedback were unused. Not fitted on opened
labels.

Do not iterate this family on these 64 cases. Do not train a tail or a
selector. Do not launch GPUs. These 64 cases remain opened development data.
Serving stays the leftover incumbent extra.
