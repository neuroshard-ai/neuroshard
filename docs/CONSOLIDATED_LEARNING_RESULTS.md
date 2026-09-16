# Consolidation: prior answers retained, sorting gate failed

The three-shard experiment completed on 14 September 2026. Its selected 1.7B
checkpoint improves fresh task answers from 375/512 to 451/512 and preserves
every previously correct answer in a separate 256-case prior-task set. It
**fails the complete frozen gate** because sorting falls from 118/128 to 116/128
on the primary set. The [complete result](../config/experiments/consolidated-learning-results.json)
publishes all 768 final answer pairs and every evaluated development pair.

## Method and selection

The [frozen method](CONSOLIDATED_LEARNING.md) combines the accepted phase-A
weights with the rejected [calculation-step checkpoint](REASONED_LEARNING_RESULTS.md).
Each owner computes the declared FP32 interpolation for its own parameters and
retains the fast checkpoint's complete Adam state. There are no additional
gradient updates, routing between models, calculator calls or answer repairs.
The result is one fixed-size model with 1,711,376,384 parameters.

| Development candidate | Prior correct / 512 | Lost prior answers | New correct / 128 | Decision |
| --- | ---: | ---: | ---: | --- |
| Parent | 378 | 0 | 97 | Baseline |
| 75% learned update | 378 | 1 | Not evaluated | Reject |
| 50% learned update | 378 | 0 | 118 | Select |

The 75% candidate gains one prior arithmetic answer and loses another. The
zero-loss rule rejects that exchange despite the unchanged total. The 50%
candidate preserves the entire correct prior set and gains 21 new arithmetic
answers; conversation loss changes by +0.001206 nats. It is the first complete
development pass. The 25% and 12.5% candidates are therefore not attempted.
The actual checkpoint is committed before final evaluation and is not replaced
after the failed final result.

## Final measurements

Parent and candidate receive identical prompts and a 256-token greedy limit.
The earlier strict JSON scoring rules still apply. Fresh cases and conversation
documents exclude every role from all three preceding experiments by identity
and normalized content. Instruction families remain public; this measures new
examples of those families, not general assistant quality.

| Measurement | Parent | Candidate | Decision |
| --- | ---: | ---: | --- |
| New-task correct answers, 512 cases | 375 | 451 | Gain passes; sorting floor fails |
| New-task invoice totals, 128 cases | 4 | 82 | 80 wins, two losses |
| New-task sorting, 128 cases | 118 | 116 | Two losses |
| Prior-task correct answers, 256 cases | 194 | 194 | No individual wins or losses |
| Conversation response loss, 256 documents | 0.539504 | 0.540975 | Retention bound passes |

New-task lookup remains 128/128 and filtering 125/128. All prior families retain
their complete correct-answer sets: lookup 64/64, filtering 63/64, sorting 64/64
and invoice totals 3/64. The primary set has 80 wins and four losses, with exact
one-sided McNemar *p* = 1.049e-19. Conversation loss changes by +0.001470 nats;
its 95% bootstrap upper bound is +0.002425, below the frozen +0.02 limit.

One lost sorting response uses `top_two` instead of `ids` while selecting the
right items. The other selects the wrong second item. They occur under two
different instruction wordings, including one already used in training. The
two arithmetic losses contain incorrect additions in generated calculation
steps. All remain errors. The large arithmetic gain cannot override the
separate sorting floor.

The result supports retaining the new arithmetic skill with less interference
on the measured prior sample. A larger development probe and smaller update
still do not establish preservation on every fresh task set. The final data
are now exposed and cannot approve a replacement candidate.

## Execution and preserved state

Three A10G hosts in separate availability zones ran under one operator. Each
held only its own parent, learned and optimizer partitions; boundaries were
`[0, 6, 15, 24]`. Owners held 503,343,104 / 604,016,640 / 604,016,640 model
parameters. The complete screen took 2,149.92 seconds, with peak allocated GPU
memory of 8.33 / 10.21 / 10.21 GiB. All owners agreed on the selected checkpoint
and the complete evaluation measurements.

The initial parent evaluation hit a 180-second transport timeout before any
answer-generation event or completed report. One worker was observed blocked
on I/O. The failed logs and exit codes are retained. Preloading and SHA-256
checking both owned input checkpoints preceded a successful retry of the
unchanged numerical evaluation. This is an operational recovery, not a second
candidate selection; the first attempt has no completed report to compare.

Completed parent/candidate evaluations took 770.56 / 1,996.71 seconds. New-task
generation used 6,408 / 22,358 tokens and 450.46 / 1,682.22 seconds. The additional
calculation tokens are part of the method's cost.

- Numerical source and plan freeze: `89ea9edc4fc43b0588b3609731b2e30c3e261554`.
- Prepared-input commit: `acb0c0a`; prepared identity
  `af7526b07a8046ecbb0337875030eed5e2d43d48ab1b6069697ea9bee42da205`.
- Actual selection commit: `59b1f0fcdb836a005f7d979afd12be999cc28858`.
- Selected checkpoint:
  `4f5fcde935dba361a83a65f63c1db48c479796debd574d8ea9f7e10dc959bddb`.
- Selected learned-state root:
  `917895b27de4ebaf9b418ca268314e70cb62e2f1c78bc2cefed7b5be020511b5`.

Validation passed 435 tests, repository and distribution checks. Both CI checks
passed at the selection commit. Reproduction must use the pinned numerical
source bytes. Both attempted checkpoints, including full Adam state, have
verified S3 readbacks. The full evidence archive has SHA-256
`322a6003c2e25f0a2116ebc0757d76155f0a04182dcd63eb444002218d9f234d`.
Large tensors and full execution evidence remain in operator-controlled storage.

All three temporary instances, root disks and the dedicated security group were
confirmed deleted at 19:06:13 UTC; both CPU network hosts remained running.
Estimated compute is $6.23. The conservative planning total including transfer
and a $10 allowance is $23.42, below the $100 cap. These are estimates rather
than an invoice; retained S3 storage accrues separately. No NEURO was issued and
no serving promotion occurred. Useful growth, broad assistant quality and
public network integration remain open requirements.
