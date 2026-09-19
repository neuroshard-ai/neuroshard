# Calculation-step learning: substantial gain, failed answer retention

The three-shard, fixed 1.7B experiment completed on 14 September 2026. It learned
many new correct arithmetic answers, but **failed its frozen acceptance gate**
because sorting and filtering answers regressed. The candidate is not approved
for serving or native settlement. The [complete result](../config/experiments/reasoned-learning-results.json)
includes all 384 final answer pairs, including every regression.

## Final measurements

The [frozen method](REASONED_LEARNING.md) combines calculation-step supervision
with replay KL and protection of correct reference token margins. Both models
receive identical prompts and a 256-token greedy generation limit. Calculation
steps are model-generated; no calculator or answer repair supplies predictions.
Original prior-task prompts and their strict JSON scoring remain unchanged.

| Measurement | Parent | Candidate | Decision |
| --- | ---: | ---: | --- |
| New-task correct answers, 256 cases | 192 | 242 | Gain passes; sorting floor fails |
| New-task invoice totals, 64 cases | 2 | 53 | 51 gains, no losses |
| New-task sorting, 64 cases | 62 | 61 | One gain, two losses |
| Prior-task correct answers, 128 cases | 93 | 90 | Three losses |
| Prior-task filtering, 32 cases | 30 | 29 | One loss |
| Prior-task sorting, 32 cases | 31 | 29 | Two losses |
| Conversation response loss, 128 documents | 0.612478 | 0.614999 | Retention bound passes |

New-task lookup and filtering remain 64/64 each. Prior lookup remains 32/32 and
prior invoice totals remain 0/32. The primary set has 52 wins and two losses,
with exact one-sided McNemar *p* = 8.249e-14. That strong aggregate improvement
cannot override the declared per-family floors. Conversation loss rises by
+0.002522 nats; its 95% bootstrap upper bound is +0.003677, below +0.02.

Three regressions emit `top_two` instead of the required `ids` key while naming
the correct items. Another violates the requested ordering, and another selects
the wrong ranked item. These remain errors under the original contract; the
scorer does not rename keys, reorder outputs or relax the acceptance rule.

## What this establishes

The learning intervention produces a substantial generated-answer improvement
on new examples of the public task families. Soft preservation losses do not
establish reliable answer retention on unseen examples. The 64-case prior
development probe missed regressions that appeared in the final set.

| Development checkpoint | New correct / 64 | Prior correct / 64 | Conversation loss change |
| --- | ---: | ---: | ---: |
| Parent | 49 | 45 | 0 |
| 192 | 59 | 45 | +0.008336 |
| 256 | 58 | 45 | +0.007226 |
| 320 | 60 | 45 | +0.006628 |
| 384, the only selectable endpoint | 61 | 46 | +0.006686 |

Every development check passed. The actual endpoint was committed before final
evaluation; no intermediate checkpoint replaced it after these results arrived.
The joint intervention changes supervision, preservation penalties and training
budget, so it does not isolate their causal contributions. It is a narrow task
experiment, not evidence of general assistant quality or automatic model growth.

The improved arithmetic also uses more inference work. New-task generation
increases from 3,140 to 11,390 tokens and from 206.97 to 819.53 seconds on this
implementation. Total parent/candidate evaluations take 363.91/976.95 seconds.
The extra calculation tokens are part of the method's cost.

## Execution and artifacts

All 256 new updates completed across three A10G hosts in separate availability
zones under one operator. Each worker held only its own student, reference and
Adam partition. The model has 1,711,376,384 parameters and boundaries
`[0, 6, 15, 24]`; peak allocated GPU memory was 12.83/12.97/12.97 GiB.
The training phase, including development checks, took 6,971.52 seconds.
All three workers produced identical checkpoint and evaluation measurements.

- Numerical source and plan freeze: `760765a`.
- Prepared inputs committed at `5958498`, identity
  `6dd8e4f66ce615df39d75d3dba71b1833635507c9e2857c7bc404ad34cb2267d`.
- Actual endpoint committed at `190cd9b489d1eddb6f9579218226c41143e458d3`.
- Parent checkpoint:
  `094138fb6e3e2a8962f8455b0bf81de3f2fbe82029222df76aa71e2e22e14d49`.
- Candidate checkpoint:
  `1401f5831fefd3f7c01e4badae094f6af327923d278b4c50ab16678b9b55e826`.
- Candidate learned-state root:
  `3aac7e95f483a153703e941a5036223622f83a70f947ce9e579f16859f406cc9`.

Validation passed 425 tests, repository checks and distribution checks. Both CI
runs passed for the selection commit. Reproduction must use the frozen numerical
sources; later changes are not interchangeable with those bytes.

All four checkpoints, including complete Adam state, were backed up and verified
by full S3 readback. The final evidence archive has SHA-256
`2e1cf7e8d8ac9146c9623ffac1501dca021ce14dbd4194fd3ebcdca62df0b188`.
Large tensors and full execution evidence remain in operator-controlled storage;
the linked repository result publishes every final task answer pair.

All three temporary GPU instances, root disks and their security group were
confirmed deleted at 16:52:12 UTC. Both CPU network hosts remained running.
Compute through confirmed termination is estimated at $9.12. A conservative
planning total including transfer and a $10 allowance is $36.18, below the $100
cap. These are estimates, not an invoice; retained S3 storage accrues separately.
No NEURO was issued and no public serving root changed.
