# Verified decision preferences, round 2: training and integration

**Finished September 28, 2026 at 01:26 UTC; training and gate fitting completed.**
This runs the [conditional second round](ASSISTANT_EXPERIENCE_LEARNING.md#conditional-second-round-verified-decision-preferences),
declared before the [round-1 development result](ASSISTANT_EXPERIENCE_DEVELOPMENT_RESULTS.md)
was read. Development and confirmation goals were not opened here.

## Method as run

No new rollouts. From the pinned round-1 natural training rollouts, every rollout
was rescored through the frozen scorer. A pair shares all messages up to the first
`read_document` call: the chosen continuation opens the latest approved revision in
a rollout whose first round passed, the rejected one an older approved revision in
a rollout whose first round failed. Identical decision texts deduplicate, leaving
**37 pairs from 37 training cases**. Both round-1 arms continued for 64 steps on the
same experience, replay and pairs, adding DPO (β = 0.1) against the round-1 arm.

Preference margins started near 0 and ranged from 3.5 to 8.2 over the last
updates; experience loss fell to 0.017 for both arms (253 seconds each on one A10G).
Parent sampled counts differ slightly between runs because batched sampling is not
reproducible; greedy counts are identical.

## Integration

Each model ran the 64 integration episodes once greedily and four times sampled.

| Integration split | Parent | Round 1 update / addition | Round 2 update / addition |
| --- | ---: | ---: | ---: |
| Greedy successes | 34/64 | 42 / 42 | 55 / 55 |
| Sampled successes | 130/256 | 166 / 168 | 220 / 223 |
| Older approved revision listed first | 22% | 36% / 35% | 76% / 79% |
| Latest approved revision listed first | 83% | 95% / 98% | 97% / 95% |

Thirty-seven verified decisions changed version choice on new cases from about
one in three to about four in five, without loss where the listing order already
helped. The model learned to compare revision numbers rather than memorize
document IDs, which differ in every case. The gates found the round-2 addition
better on 32 integration episodes and worse on none; the update better on 30 and
worse on 1. Integration episodes fit the gates; they are not admission evidence.

## Attempts and cost

The first attempt (`d836fef`) built the 37 pairs and then stopped: the worker passed
runtime parameters where the pinned round-1 digests were read ($0.22). The corrected
attempt (`9cd1c70`) completed ($1.84). The
[report](../config/experiments/assistant-experience-round2-report.json) records both
attempts, the pair inventory, training margins, gates and receipts. The development
gate for the round-2 systems runs separately on the CPU parent runtime; it still
faces the structural latency limit described in the round-1 result.
