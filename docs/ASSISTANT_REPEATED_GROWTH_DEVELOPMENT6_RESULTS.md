# A3 development with the calibrated router: the candidate passes

**With the calibrated router pinned, the candidate, the separate low-rank module,
completed 18/24 scheduling and 6/8 cross development cases and kept every accepted
drafting success, so it passes the development gate. The sealed confirmation opens
once.** It met the scheduling and cross levels exactly. Routing was exact: every
scheduling turn went to the scheduling route and every drafting turn stayed on drafting.

Declaration: [calibrated router](ASSISTANT_REPEATED_GROWTH_ROUTER4.md). Evidence:
[report](../config/experiments/assistant-growth-development6-report.json) and the three
results ([separate update](../config/experiments/assistant-growth-development6-separate-update-result.json),
[separate module](../config/experiments/assistant-growth-development6-separate-module-result.json),
[shared](../config/experiments/assistant-growth-development6-shared-result.json)), commit `551ca4f`.

## Outcomes (opened development cases, routed turn by turn)

| Version | Scheduling | Cross | Drafting | Accepted successes lost | p95 |
| --- | --- | --- | --- | --- | --- |
| Separate update (U1, U2) | 15/24 | 7/8 | 19/24 | none | 101.8 s |
| Separate module (U1, L2), candidate | 18/24 | 6/8 | 19/24 | none | 98.9 s |
| Shared (U2) | 15/24 | 7/8 | 19/24 | 1 | 105.5 s |

The gate asks the candidate for 18/24 scheduling, 6/8 cross, no lost accepted drafting
success and a p95 of at most 180 s. The candidate is the separate version with more
scheduling and cross successes: 24 against 22.

## Routing (every version)

| Turns | To the scheduling route | To the drafting route |
| --- | --- | --- |
| Scheduling (36) | 36 | 0 |
| Cross (12) | 8 | 4: each handoff's drafting turn, as intended |
| Drafting (40) | 0 | 40 |

One router serves all three versions, so they routed identically.

## What this shows

When round 4's scheduling unit served every turn, it met the scheduling and cross levels
but lost an accepted drafting success. The calibrated router keeps drafting turns on the
drafting route, and no separate version lost one. The shared version lost one
(`workflow-c6b1500447f3`); that is the retention difference A3's comparison measures on
the confirmation data. The candidate met the development levels exactly, and the
confirmation's are higher: at least 154 of 192 scheduling (80%), at least 16 of 24 in each
family, 36 of 48 cross, a net gain of at least 20 over the previous version with a lower
95% bound above zero, and no lost drafting success.

## Cost

Three r7i.4xlarge hosts ran in parallel, $3.40 in total, and every instance is terminated.
A3 has spent $42.83 of its $110 ceiling, plus stage 0's $2.15. The confirmation allows at
most $47 on five CPU hosts.
