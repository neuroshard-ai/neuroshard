# A3 stage-1 development results: drafting kept, scheduling not reached

**Development failed on scheduling and cross, and the sealed confirmation stays
closed.** The candidate, the separate low-rank module, completed 6/24 scheduling
and 1/8 cross development cases, against a gate of 18/24 and 6/8. It kept all 19
drafting successes of the accepted version, within the p95 limit (113 s against
180 s). The turn selectors were a bottleneck. The separate update's selector
sent every scheduling turn to the drafting route, which cannot schedule. The
candidate's sent a third of its first scheduling turns there. When the candidate's
first scheduling turn did reach L2, it passed 12 times in 16.

Declarations: [stage 1](ASSISTANT_REPEATED_GROWTH_STAGE1.md) for the gate and
[round 3](ASSISTANT_REPEATED_GROWTH_ROUND3.md) for the units and selectors. Evidence:
[report](../config/experiments/assistant-growth-development-report.json) and the three
results ([separate update](../config/experiments/assistant-growth-development-separate-update-result.json),
[separate module](../config/experiments/assistant-growth-development-separate-module-result.json),
[shared](../config/experiments/assistant-growth-development-shared-result.json)), commit `5c17157`.

## Outcomes (opened development cases, routed turn by turn)

| Version | Scheduling | Cross | Drafting | Drafting successes lost | p95 |
| --- | --- | --- | --- | --- | --- |
| Separate update (U1, U2) | 0/24 | 0/8 | 19/24 | none | 115.7 s |
| Separate module (U1, L2), candidate | 6/24 | 1/8 | 19/24 | none | 113.1 s |
| Shared (U2) | 3/24 | 0/8 | 18/24 | 2 | 102.9 s |

The candidate is the separate version with more scheduling and cross successes:
7 against 0.

## Routing

| Version | Scheduling turns to the scheduling route | First scheduling turns passed when routed there | Drafting turns misrouted |
| --- | --- | --- | --- |
| Separate update | 0 of 28 | none routed | 0 of 40 |
| Separate module | 22 of 36 | 12 of 16 | 0 of 40 |
| Shared | 16 of 30 | 6 of 10 | 0 of 40 |

Every first turn routed to the drafting route failed, since it lacks the calendar
tools. Two things pushed the selectors toward drafting. Most integration turns were
ties, and a tie counts for drafting by the declared rule, including the turns where
both routes failed. The selectors were also fitted on features computed on the GPU
and served on features computed on the CPU. Even with perfect routing, the
scheduling units completed only 41% to 44% of integration episodes alone, below
the gate's 75%.

## What the comparison shows so far

Both separate versions kept every drafting success, and the shared version lost
two. That is the direction A3's comparison requires, but it counts only on the
sealed confirmation, which did not open.

## Cost

Three r7i.4xlarge hosts ran in parallel for about 1.1 hours each, $3.53 in total.
Every instance is terminated, with volumes deleted and security groups retired.
A3 has spent $25.53 of its $100 ceiling, plus stage 0's $2.15.
