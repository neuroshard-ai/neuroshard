# Sharded continuation passes its independent learning gate

The 14 September 2026 experiment passes its complete frozen final gate. One
1.7B model, trained and evaluated across three disjoint GPU partitions, improves
new-task answers from **378/512 to 462/512** while preserving every one of the
**188 previously correct answers** in a separate 256-case prior-task set.
Conversation retention also passes. The [complete result](../config/experiments/balanced-continuation-results.json)
contains all 768 final answer pairs and the development comparisons.

This establishes a successful continuation on four public, generated task
families. It does not establish general assistant quality, indefinite improvement,
useful parameter growth, independent ownership or a public training economy.
There is one arithmetic regression in the new-task set; the passing rule requires
non-decreasing family totals, not preservation of every new-task answer.

## Working recipe

The [frozen continuation](BALANCED_CONTINUATION.md) starts from the rejected
[consolidated checkpoint](CONSOLIDATED_LEARNING_RESULTS.md), including Adam
state, and performs 64 additional full-model updates. The accepted adaptive
phase-A checkpoint remains the final quality baseline. The resulting recipe
combines supervised calculation steps, consolidation, broader instruction
coverage, verified replay and per-answer supervised weights.

Each document receives `loss_weight = 1 / targets`; every recorded batch has a
supervised denominator of 64. This corrects the earlier objective's dominance
by long arithmetic responses. The fixed mixture emphasizes sorting and retains
previously trained arithmetic and conversation examples. Existing token-average
reference KL and correct-token margin penalties anchor every training example
to the consolidated input. These penalties are not document-normalized.

All 4,096 training documents are used once. Learning rate peaks at `1e-6`;
microbatch size is eight. All four public instruction wordings occur in fresh
training and development. Finals use new records under those wordings, not an
unseen-wording holdout. Fresh records exclude every preceding role by identity
and normalized content. Replay comes from hash-checked records used by the
completed calculation-step training schedule.

This is one combined recipe, not an ablation isolating the effect of loss
normalization. Inference uses ordinary greedy generation with a 256-token limit.
There is no calculator, output repair, constrained decoder or model routing.
Model depth and parameter count do not grow in this experiment.

## Selection before final evaluation

| Development measurement | Original phase A | Terminal candidate |
| --- | ---: | ---: |
| Prior answers, 512 cases | 379 | 380; zero lost answers |
| New answers, 128 cases | 94 | 116; zero lost answers |
| New sorting, 32 cases | 30 | 31 |
| New arithmetic, 32 cases | 0 | 21 |

Against the consolidated training input, new answers improve 114→116: sorting
29→31 and arithmetic remains 21/32, with no individual losses. Conversation
response loss changes by +0.002171 nats, below the +0.01 development limit.
Only the terminal checkpoint at step 448 is selectable. Its complete development
report and both training checkpoint commitments were committed before opening finals.

## Independent final results

| Measurement | Original phase A | Candidate | Frozen decision |
| --- | ---: | ---: | --- |
| New answers, 512 cases | 378 | 462 | Pass |
| New lookup, 128 cases | 128 | 128 | Pass |
| New filtering, 128 cases | 126 | 126 | Pass |
| New sorting, 128 cases | 120 | 121 | Pass |
| New arithmetic, 128 cases | 4 | 87 | Pass; exceeds 64-answer floor |
| Prior answers, 256 cases | 188 | 188 | Pass; zero individual losses |
| Conversation response loss, 256 documents | 0.544084 | 0.546741 | Retention bound passes |

The primary set has 85 wins and one loss: a net gain of 84 answers, or 16.41
percentage points. Exact one-sided McNemar *p* is `1.12445e-24`. Sorting adds
one correct answer and loses none. Arithmetic adds 84 correct answers and loses
one. In that lost case, the model incorrectly adds `226 + 72` as `300`, then
returns `372` instead of `370`. This remains an error in the published result.

All prior families preserve their complete correct-answer sets: lookup 64/64,
filtering 62/64, sorting 61/64 and arithmetic 1/64. This low strict-answer
arithmetic baseline is preserved; the substantial arithmetic gain is in the
new role that permits calculation steps. The model is not universally accurate.

Conversation loss changes by +0.002657 nats. Its 95% bootstrap upper bound is
+0.004065, below the frozen +0.02 limit, using 10,000 samples and seed
2026091805. A bounded loss-retention pass is not proof of unchanged conversational
answer quality. These final examples are now exposed and cannot independently
approve a later candidate.

## Execution and reproducibility

Three A10G hosts in separate availability zones ran under one operator. Boundaries
were `[0, 6, 15, 24]`; owners held 503,343,104 / 604,016,640 / 604,016,640
parameters. Each held only its own model, reference and optimizer partitions.
Peak allocated GPU memory was 12.68 / 13.54 / 13.54 GiB. All owners agreed on
checkpoint commitments, generated responses and response losses.

The 64 updates took 1,337.31 seconds in total. The complete training process,
including development baselines, selection evaluation and startup, took
3,289.76 seconds. Numerical final evaluations took 701.23 seconds for phase A
and 1,893.22 seconds for the candidate. Primary generation used 6,292 versus
22,511 tokens, taking 413.58 versus 1,606.57 seconds. Longer calculation traces
are part of the method's inference cost. Both final processes completed without
retries; owned inputs were verified and loaded before the network group started.

- Numerical source and plan freeze: `3414b1e7c1cdd62dc1ca1a2286a531ff2c98daf4`.
- Prepared-input commit: `4297e1b51d067c8e593cab7ab2045a5e18e2e405`.
- Prepared identity: `acbc51a4587446af11ae159ec2babb68af7ccf518833a7cb44b7e1d87fde1dc9`.
- Actual selection commit: `c0fea490d7a224e88d26be3b0ee586cd4119a439`.
- Selected checkpoint: `d57b32e38cbf416119487c3f33d6e211dfcc8d47e95c80144c1886595e779a56`.
- Learned-state root: `f10c25c8795f40432819f3f67b7a273ba5374e374eb1645f7c356d65eac815ff`.

Validation passed 445 tests, repository checks and distribution checks. Both CI
checks passed at the actual selection commit. Both checkpoints include full
Adam state and have verified S3 readbacks. Reproduction uses the pinned numerical
sources and [execution instructions](BALANCED_CONTINUATION.md). Large state files
and full execution evidence remain in operator-controlled storage.

The full evidence archive has SHA-256
`e465bb6456c3dd771b93a7dcb7abc700382827586447905ce014e63a6b4b8bc1`
and a verified readback. All three temporary instances, root disks and the
dedicated security group were confirmed removed at 21:26:12 UTC; both CPU
network hosts remained running. Compute is estimated at $6.76.
The conservative planning total, including lifetime network traffic and a $10
allowance, is $27.00, below the $100 cap. This is not an AWS invoice;
retained S3 storage accrues separately.

## Consequence for the network

This supplies a measured learning recipe for the next native integration. It
does not retroactively pay the research updates or change public serving. The
[portable native adapter](NATIVE_SHARD_REPLAY.md) still needs fresh-job activation,
funded reservations and exact replay of the windows that produce an accepted
state, followed by a separate quality transaction that may change the serving
root. Computation settlement must not itself approve model quality.

No NEURO was issued and no serving promotion occurred. Broader assistant tasks,
further independently evaluated cohorts, useful growth, economical verification
and independent providers remain separate requirements.
