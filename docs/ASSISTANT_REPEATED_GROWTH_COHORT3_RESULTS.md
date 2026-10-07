# A3 cohort 3's GPU job: L3 solves every drafting integration case

**L3, the new low-rank module on top of U1, trained on 1,248 verified drafting
demonstrations and then solved all 64 drafting integration cases on its first, greedy
attempt, against 58 for U1: it gained 6 and lost none. Sampled at temperature 0.8, it solved
127 of 128 runs against U1's 119. Development runs next, with L3 pinned.** Integration is
reported, not a gate: it uses the training grammar's corrections on the drafting route alone,
while development routes whole conversations on cases with held-out corrections.

Declaration: [cohort 3](ASSISTANT_REPEATED_GROWTH_COHORT3.md). Evidence:
[result](../config/experiments/assistant-growth-cohort3-result.json) and
[report](../config/experiments/assistant-growth-cohort3-report.json), commit `60f6793`.

## Experience and training

The drafting solver wrote one demonstration for each of the 1,248 training cases, and every one
passed the scorer within the drafting policy's limits without reading an unapproved revision:
8,422 replies, 1,058 of them with two calls. L3 trained for 512 steps on six demonstrations and
two items of A2's parent replay per update, 1,048,576 parameters, in 1,070 s on one A10G. Its
final loss was 0.0018; the demonstrations are regular, which is why integration and development
decide, not the loss.

## Integration (drafting route alone, greedy)

| Family | U1 | L3 |
| --- | --- | --- |
| Copy | 7/8 | 8/8 |
| Date | 8/8 | 8/8 |
| Sum | 8/8 | 8/8 |
| Difference | 6/8 | 8/8 |
| Latest revision | 6/8 | 8/8 |
| Recipient | 8/8 | 8/8 |
| Reschedule | 8/8 | 8/8 |
| Scope | 7/8 | 8/8 |
| Total | 58/64 | 64/64 |

The gains are in the families where the sealed failures were: the change from revision 1, the
switch to revision 1, and corrections that need several calls.

## Cost

One A10G host, $1.09. The first allocation found no A10G capacity in the controller's zone and
started no instance ($0.004); the retry placed one. A3 has spent $92.27 of its $150 ceiling, plus
stage 0's $2.15.
