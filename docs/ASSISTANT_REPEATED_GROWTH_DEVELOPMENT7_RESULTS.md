# A3 development with the free-slot tool: the candidate passes

**With round 5's units and the free-slot tool, the candidate, the separate low-rank
module, completed all 24 scheduling and all 8 cross development cases and kept every
accepted drafting success, so it passes the development gate. The fresh sealed
confirmation opens once.** Both separate versions solved every scheduling and cross case;
the shared version did too but lost one accepted drafting success. Routing was exact.

Declaration: [round 5](ASSISTANT_REPEATED_GROWTH_ROUND5.md). Evidence:
[report](../config/experiments/assistant-growth-development7-report.json) and the three
results ([separate update](../config/experiments/assistant-growth-development7-separate-update-result.json),
[separate module](../config/experiments/assistant-growth-development7-separate-module-result.json),
[shared](../config/experiments/assistant-growth-development7-shared-result.json)), commit `b536265`.

## Outcomes (opened development cases, routed turn by turn)

| Version | Scheduling | Cross | Drafting | Accepted successes lost | p95 |
| --- | --- | --- | --- | --- | --- |
| Separate update (U1, U2) | 24/24 | 8/8 | 19/24 | none | 91.8 s |
| Separate module (U1, L2), candidate | 24/24 | 8/8 | 19/24 | none | 99.4 s |
| Shared (U2) | 24/24 | 8/8 | 18/24 | 1 | 99.6 s |

Without the tool, round 4's candidate completed 18/24 and 6/8 on the same cases
([development 6](ASSISTANT_REPEATED_GROWTH_DEVELOPMENT6_RESULTS.md)). The gate asks the
candidate for 18/24 scheduling, 6/8 cross, no lost accepted drafting success and a p95 of
at most 180 s. The separate versions tied on scheduling and cross successes, and a tie
goes to the module.

## Routing (every version)

| Turns | To the scheduling route | To the drafting route |
| --- | --- | --- |
| Scheduling (36) | 36 | 0 |
| Cross (12) | 8 | 4: each handoff's drafting turn, as intended |
| Drafting (40) | 0 | 40 |

The pinned router serves all three versions, so they routed identically.

## What this shows

The tool removed the interval arithmetic that failed the first confirmation, and the
retrained units used it on every development case. The separate units kept all 19
accepted drafting successes; the shared update lost one (`workflow-c5a251cb9186`), the
retention difference A3's comparison measures on the confirmation data. Development does
not show learning by itself, because the previous version also has the tool; the
confirmation measures that gain. Its levels are stage 1's, on fresh sealed sets: at least
154 of 192 scheduling (80%), at least 16 of 24 in each family, 36 of 48 cross, a net gain
of at least 20 over the previous version with a lower 95% bound above zero, and no lost
drafting success.

## Cost

Three r7i.4xlarge hosts ran in parallel, $3.20 in total, and every instance is terminated.
A3 has spent $70.53 of its $150 ceiling, plus stage 0's $2.15. The confirmation allows at
most $47 on five CPU hosts.
