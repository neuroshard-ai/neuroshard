# A3 cohort 3 development: the upgraded system passes

**With L3 on its drafting route, the upgraded system solved 23 of the 24 opened development
drafting cases against the previous system's 19, gaining 4 and losing none. It kept all 24
scheduling and all 8 cross cases, and its p95 was 88.3 s. It passes the development gate, so
the fresh sealed confirmation opens once.** It also used fewer replies: 183 model calls on
drafting against the previous system's 205.

Declaration: [cohort 3](ASSISTANT_REPEATED_GROWTH_COHORT3.md). Evidence:
[report](../config/experiments/assistant-growth-cohort3-development-report.json) and
[result](../config/experiments/assistant-growth-cohort3-development-result.json), commit `1ff1789`.
The previous system's episodes are the pinned separate-module episodes of
[development 7](ASSISTANT_REPEATED_GROWTH_DEVELOPMENT7_RESULTS.md); both systems were rescored.

## Outcomes (opened development cases, routed turn by turn)

| Set | Upgraded system | Previous system | Gained | Lost |
| --- | --- | --- | --- | --- |
| Drafting | 23/24 | 19/24 | 4 | 0 |
| Scheduling | 24/24 | 24/24 | 0 | 0 |
| Cross | 8/8 | 8/8 | 0 | 0 |

The gate asks for at least 21 of 24 drafting cases, no lost success on any set and a p95 of at
most 180 s. The previous system's p95 on the same cases was 99.4 s.

## What changed

All five of the previous system's failures were corrections that switch to another plan and
then move the resulting due date a day. In four `scope` corrections it saved the right draft and
then ran out of model turns, making one call per reply; in one `latest` correction it saved the
wrong due date. The upgraded system solved three of the `scope` cases and the `latest` case,
making independent calls together and finishing within the budget.
A2's gate chose the drafting unit in all 56 episodes, and the router sent every turn where it
belongs: 36 scheduling turns to scheduling, 40 drafting turns to drafting, and each cross case's
turns to the route each one needs.

The one remaining failure, a `scope` correction that also moves the date, is new in kind. The
model put three calls in one reply, over the policy's limit of two, so the reply was rejected;
it then fell back to one call per reply, shifted the earlier due date, and ran out of turns. The
previous system failed this case too.

## Cost

One r7i.4xlarge host, $1.02, terminated. A3 has spent $93.29 of its $150 ceiling, plus stage
0's $2.15. The confirmation allows at most $37.60 on four CPU hosts.
