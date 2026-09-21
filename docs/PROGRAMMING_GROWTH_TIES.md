# TIES composition of two programming tails

**Status: decoded and failed. Stop this composition.**
Unit task-vector addition failed because the merged extra was often not even
valid Python (18/38 extractable versus 38/38 for each frozen tail). On the
frozen last-four-layer tensors, **33.6%** of jointly nonzero signed parameters
have opposite signs. TIES replaced that sum: keep the top 20% magnitude of each
delta, elect a consensus sign, and average only the values that agree. It did
not train. It did not open the original 128-task final. It was not a selector.

Serving stayed one extra decode. Parent public-example pass still returned the
parent. Failure spent that extra on the TIES mix instead of the unit mix.

## Frozen rule (failed)

`θ = θ_parent + mean({trim_0.2(θ_inc − θ_parent), trim_0.2(θ_add − θ_parent)}`
restricted to the elected sign). Keep `0.2` and scale `1.0` were taken from the
TIES paper defaults and were not tuned on the opened 64 cases.

CPU merge of the frozen checkpoints is pinned in
`config/experiments/programming-growth-ties-expert.json`
(`9c06f100…`). The GPU worker loaded those hashes; it did not re-merge.

## Screen (opened 64, not admission)

Generated the 38 extras whose parent public example already failed. Joined the
saved parent, incumbent and added traces. Required versus measured:

| Gate | Required | Measured |
| --- | ---: | ---: |
| TIES extras extractable | 38/38 | **38/38** |
| unique added recovered | 3 | **1** (503 yes; missed 276, 265) |
| incumbent successes preserved | 29 | **29** (kept 249) |
| old successes preserved | 12 | **12** |
| selected full-test | 32/64 | **31/64** |

Parent 22, incumbent 29, always-added 31, oracle union 32. TIES scored 31: it
recovered unique-added 503, kept unique-incumbent 249, and created leftover
task 54 which neither frozen tail had. It did not recover 276 or 265. That is
partial combination. It failed the declared gate and remains rejected. The
incumbent extra remains the research baseline because this challenger failed
acceptance, not because 31 was lower than 29. `next` is
`stop-this-composition`. `admission_evidence` false.

Score `86a6a489…`, extras `3de80b42…`, record `29887a86…`. Source `9e85585`.
Four g5.xlarge hosts, two-hour /$50 cap, ~7 minutes, then retired. No native
issuance.

Do not iterate keep or scale on these 64 cases. Trim discarded small unique
directions that sign-election itself had already made extractable. [Elect-sign
without trim](PROGRAMMING_GROWTH_ELECT.md) then matched this screen, so trim
was not the unique-added loss. Task-vector sign-consensus of these two tails
is stopped. These 64 cases remain opened development history. Serving stays
the leftover incumbent extra. The next experiment is
[learned integration](LEARNED_INTEGRATION.md).
