# Fourth programming-selector picker

**Status: CPU screen failed. Stop this picker. No GPU.**
This was a separately declared experiment, not a retry of Jaccard, agreement,
AST-shape, TIES, or elect-sign. Complementary coverage still exists in the two
unchanged tails. This fail-class rule did not connect it. The original 128-task
final stays closed.

## Measured screen

Decide hashed 38 calls (`7bd95df8…`) before joining tail answers. Controls
reproduced the diagnosis: parent 22, incumbent 29, always-added 31, oracle 32.

| Gate | Required | Measured |
| --- | ---: | ---: |
| unique added recovered | 3 | **0** (missed 276, 503, 265) |
| incumbent successes preserved | 29 | **29** (kept 249) |
| old successes preserved | 12 | **12** |
| selected full-test | 32/64 | **29/64** |
| picker p95 | ≤ 50 ms | 0.02 ms |
| picker errors | 0 | 0 |

37 extras were `execution-error`. One was `extraction-error` (task 220, not
unique-added). Score matches the incumbent extra policy. `next` is
`stop-this-picker`. `admission_evidence` false.

## Rule (unchanged, failed)

Select `added` only when the parent public-example feedback status is
`extraction-error`. Every other frozen failure class selects incumbent.
Question, failed parent program, and public example text were unused. Not
fitted on opened labels.

Do not iterate this fail-class map on these 64 cases. Do not train a tail or a
selector. Do not launch GPUs. These 64 cases remain opened development data.
Serving stays the leftover incumbent extra.
