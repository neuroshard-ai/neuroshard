# A3 round-4 development results: scheduling and cross reached, one drafting success lost

**The candidate met the scheduling and cross gates but lost one drafting success,
so development failed and the sealed confirmation stays closed.** The separate
low-rank module completed 18/24 scheduling and 6/8 cross development cases, exactly
the gate. It also completed 19/24 drafting cases, but one of them was not among the
accepted version's 19, so it lost `workflow-21c0be04a7be`. The refitted selectors
sent every turn of every case to the scheduling route, drafting turns included. In
round 3's development, drafting turns served by the drafting route reproduced all
19 accepted successes.

Declarations: [stage 1](ASSISTANT_REPEATED_GROWTH_STAGE1.md) for the gate and
[round 4](ASSISTANT_REPEATED_GROWTH_ROUND4.md) for the units and the refit. Evidence:
[report](../config/experiments/assistant-growth-development4-report.json) and the three
results ([separate update](../config/experiments/assistant-growth-development4-separate-update-result.json),
[separate module](../config/experiments/assistant-growth-development4-separate-module-result.json),
[shared](../config/experiments/assistant-growth-development4-shared-result.json)), commit `7e739b7`.

## Outcomes (opened development cases, routed turn by turn)

| Version | Scheduling | Cross | Drafting | Accepted successes lost | p95 |
| --- | --- | --- | --- | --- | --- |
| Separate update (U1, U2) | 15/24 | 7/8 | 19/24 | 1 | 109.9 s |
| Separate module (U1, L2), candidate | 18/24 | 6/8 | 19/24 | 1 | 121.1 s |
| Shared (U2) | 15/24 | 7/8 | 19/24 | 1 | 114.7 s |

The candidate is the separate version with more scheduling and cross successes,
24 against 22. It passed the scheduling, cross and latency checks and failed only
the drafting check.

## Routing

Each host refitted its version's selector on its own CPU runtime in 8 to 9 minutes,
from 178 to 185 integration turns, leaving out the turns both routes failed. All
three selectors then chose the scheduling route for every development turn. The
drafting cases were therefore served by the scheduling units with the calendar
tools instead of by U1 with the drafting tools, as accepted. They still completed
19/24, but each version lost one of the accepted version's successes.

The A2 logistic recipe does not separate the two kinds of turn on these features.
Fitted on GPU features with every tie counted for drafting, it sent every
scheduling turn of the separate update to the drafting route. Refitted on CPU
features without the failed ties, it sent every turn to the scheduling route.

## Cost

Three r7i.4xlarge hosts ran in parallel, $4.05 in total. Every instance is
terminated, with volumes deleted and security groups retired. A3 has spent $33.12
of its $100 ceiling, plus stage 0's $2.15.
