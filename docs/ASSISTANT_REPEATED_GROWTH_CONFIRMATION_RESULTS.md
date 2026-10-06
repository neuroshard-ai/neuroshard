# A3 sealed confirmation: routing and retention hold, scheduling falls short

**On the sealed confirmation sets, opened once, the candidate completed 117 of 192
scheduling episodes against a bar of 154, and four of the eight scheduling families fell
below 16 of 24. Cohort 2, meeting scheduling, is therefore not accepted.** Every other
check passed: 37 of 48 cross episodes, a net gain of 152 over the previous version with a
lower 95% bound of +0.53, no lost drafting success, and p95 102.5 s. Routing was exact on
every sealed turn. The separate candidate lost none of the accepted version's drafting
successes and the shared update lost one, so the declared retention comparison holds,
narrowly.

Gate: [stage 1](ASSISTANT_REPEATED_GROWTH_STAGE1.md), served with the
[calibrated router](ASSISTANT_REPEATED_GROWTH_ROUTER4.md) behind the
[development pass](ASSISTANT_REPEATED_GROWTH_DEVELOPMENT6_RESULTS.md). Evidence:
[report](../config/experiments/assistant-growth-confirmation-report.json) and the five
results ([candidate, calendar](../config/experiments/assistant-growth-confirmation-candidate-calendar-result.json),
[candidate, drafting](../config/experiments/assistant-growth-confirmation-candidate-drafting-result.json),
[shared, drafting](../config/experiments/assistant-growth-confirmation-shared-drafting-result.json),
[accepted, drafting](../config/experiments/assistant-growth-confirmation-accepted-drafting-result.json),
[accepted, calendar](../config/experiments/assistant-growth-confirmation-accepted-calendar-result.json)),
commit `c5e3bc2`.

## Gate

| Check | Bar | Candidate (separate module) | |
| --- | --- | --- | --- |
| Scheduling | at least 154 of 192 | 117 | fail |
| Each scheduling family | at least 16 of 24 | lowest 9 | fail |
| Cross | at least 36 of 48 | 37 | pass |
| Net gain over the previous version | at least 20 | +152 | pass |
| Lower 95% gain over the previous version | above 0 | +0.53 | pass |
| Accepted drafting successes lost | none | none | pass |
| p95 | at most 180 s | 102.5 s | pass |

The previous version, the accepted version under the calendar interface, completed 1 of
192 scheduling and 1 of 48 cross episodes.

| Family | slot | after | three | day | longer | invite | move | swap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Completed of 24 | 19 | 20 | 17 | 9 | 11 | 10 | 16 | 15 |

## Drafting retention (confirmation4, 192 episodes)

| System | Completed | Accepted successes gained | Accepted successes lost |
| --- | --- | --- | --- |
| Accepted version | 181 | | |
| Candidate (separate module) | 181 | 0 | 0 |
| Shared update | 190 | 10 | 1 (`workflow-edab4b70531e`) |

The declared comparison counts losses only: the separate unit must lose fewer accepted
drafting successes than the shared update. It does, 0 against 1, a margin of one episode.
The shared update also gained ten.

## Routing

All 288 scheduling turns went to the scheduling route. Of 72 cross turns, 48 went to
scheduling and 24, each handoff's drafting turn, stayed on drafting as intended. All 288
of the candidate's drafting turns stayed on drafting, as did all 287 of the shared
version's; one of its episodes ended a turn early.

## What this shows

The router is no longer the limit: every one of the candidate's 648 routed sealed turns
went to the route it needs. The scheduling unit is. It completed 61% of the sealed scheduling episodes,
close to its 59% (38/64) on integration; development's 18/24 was the favourable end of a
small sample. The "day", "invite" and "longer" families are weakest. Drafting is retained
exactly. The sealed scheduling and cross confirmation sets are now spent, so another attempt
needs fresh sealed splits.

## Cost

Five r7i.4xlarge hosts ran in parallel for 3.6 to 4.0 hours each, $20.04 in total. Their
finish receipts record every instance terminated, no volume left and every security group
retired; EC2 had already dropped the instances from its listing when the report was written.
A3 has spent $62.87 of its $110 ceiling, plus stage 0's $2.15.
