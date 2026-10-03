# Verified divergence preferences, round 3: training and integration

**Finished September 28, 2026 at 08:27 UTC; training and gate fitting completed,
but the declared pair source yielded almost nothing.** This runs the
[third round](ASSISTANT_EXPERIENCE_LEARNING.md#third-round-verified-divergence-preferences),
declared after the [cached development result](ASSISTANT_EXPERIENCE_DEVELOPMENT_CACHED_RESULTS.md).
Development and confirmation goals were not opened here.

## Pair yield

The round-2 addition sampled six rollouts per training case on one L40S: 1,536
rollouts, 1,397 fully successful. The declared rule pairs a success and a failure
at the first message where they differ, after identical earlier messages. It
produced **2 pairs**: under sampling, two rollouts almost always first differ in
the wording of round one's confirmation reply, which is not a tool decision, so
the rule skips the pair even when the error comes later.

Comparing decisions instead of wording (the `aligned` option, added after this run
and not used by it) finds only 9 pairs in the same rollouts. The model's remaining
errors are concentrated where it fails every sample: 113 of 139 failures are in the
difference family, which reads both revisions correctly and then copies a wrong
quantity into the subtraction, or exhausts the call budget. With no success in
those cases, divergence pairs cannot target them. The follow-up date error appears
in only 18 of 192 sampled "latest" rollouts.

## Training and integration

Both round-2 arms continued for 64 steps on the round-1 experience and replay plus
the 2 pairs (DPO against the round-2 arm); experience loss fell to about 0.015.

| Integration, greedy | Round 2 update / addition | Round 3 update / addition |
| --- | ---: | ---: |
| All 64 episodes | 55 / 55 | 55 / **58** |
| Latest family | 7 / 6 of 8 | 7 / 8 of 8 |
| Difference family | 3 / 3 of 8 | 4 / 3 of 8 |

The round-3 addition improves anyway, most plausibly from the additional
experience steps rather than two pairs; the rollouts cannot separate those causes.
Its gate is better on 30 integration episodes and worse on none. The run cost
**$5.38** on a g6e.2xlarge. The [report](../config/experiments/assistant-experience-round3-report.json)
records the pair inventory, rollout counts, margins, gates and receipts.

## Lessons

Divergence preferences need failures that share decisions with successes. Sampling
the stronger model mostly produces cases that always succeed or always fail. The
remaining errors, copying numbers across multi-step calculations, call for targeted
exploration on the failing cases, such as more samples, coaching or tool-level
arithmetic, rather than more pairs from easy ones.
