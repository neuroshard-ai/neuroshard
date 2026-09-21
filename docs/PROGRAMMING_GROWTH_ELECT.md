# Elect-sign composition of two programming tails

**Status: decoded and failed. Stop this composition.**
TIES (trim 0.2, elect sign, disjoint mean) restored extractable Python (38/38)
and scored 31/64, then missed unique-added 276 and 265. This experiment kept
sign-election and the disjoint mean and dropped magnitude trim. It did not
train. It did not open the original 128-task final. It was not a selector.

Serving stayed one extra decode. Parent public-example pass still returned the
parent. Failure spent that extra on the untrimmed elect mix.

## Frozen rule (failed)

`θ = θ_parent + mean({θ_inc − θ_parent, θ_add − θ_parent}` restricted to the
elected sign of the untrimmed sum). Unique (one-sided) parameters are kept.
Opposite-sign parameters follow the sign of the sum. Scale `1.0` is frozen.
There is no keep fraction.

CPU merge of the frozen checkpoints is pinned in
`config/experiments/programming-growth-elect-expert.json`
(`65058087…`). The GPU worker loaded those hashes; it did not re-merge.

## Screen (opened 64, not admission)

Generated the 38 extras whose parent public example already failed. Joined the
saved parent, incumbent and added traces. Required versus measured:

| Gate | Required | Measured |
| --- | ---: | ---: |
| elect extras extractable | 38/38 | **38/38** |
| unique added recovered | 3 | **1** (503 yes; missed 276, 265) |
| incumbent successes preserved | 29 | **29** (kept 249) |
| old successes preserved | 12 | **12** |
| selected full-test | 32/64 | **31/64** |

Parent 22, incumbent 29, always-added 31, oracle union 32. Elect scored 31 and
matched TIES on every opened task outcome: recovered 503, kept 249, created
leftover 54, missed 276 and 265. Trim was not what discarded those unique-added
tasks. `next` is `stop-this-composition`. `admission_evidence` false.

Score `7491a09b…`, extras `3d1e80f2…`, record `8be39e8c…`. Source `8ea046a`.
First allocation died on SSH to rank 2; the retry
`neuroshard-programming-growth-elect-20260921b` decoded and retired. No native
issuance.

Do not iterate sign-election, keep, or scale on these 64 cases. Task-vector
sign-consensus of these two tails is stopped. These 64 cases remain opened
development data. Serving stays the leftover incumbent extra.
