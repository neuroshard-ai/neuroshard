# A3 development with the centroid router: drafting kept, scheduling turns misrouted

**The centroid router kept every drafting success but misrouted enough scheduling
and cross turns that development failed; the sealed confirmation stays closed.**
The candidate, the separate low-rank module, completed 15/24 scheduling and 2/8
cross development cases, against a gate of 18/24 and 6/8. It completed 22/24
drafting cases, gaining three of the accepted version's failures and losing none.
The router sent all four "review" cross turns to the drafting route, which cannot
schedule. A review is a meeting on a plan's due date. It also sent three scheduling
requests there. With every turn sent to the scheduling unit, the same candidate had
reached 18/24 and 6/8 in [round 4's development](ASSISTANT_REPEATED_GROWTH_DEVELOPMENT4_RESULTS.md).

Declaration: [router](ASSISTANT_REPEATED_GROWTH_ROUTER.md). Evidence:
[report](../config/experiments/assistant-growth-development5-report.json) and the three
results ([separate update](../config/experiments/assistant-growth-development5-separate-update-result.json),
[separate module](../config/experiments/assistant-growth-development5-separate-module-result.json),
[shared](../config/experiments/assistant-growth-development5-shared-result.json)), commit `483e6a5`.

## Outcomes (opened development cases, routed turn by turn)

| Version | Scheduling | Cross | Drafting | Accepted successes lost | p95 | Router held-out accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| Separate update (U1, U2) | 12/24 | 3/8 | 19/24 | none | 118.7 s | 0.85 |
| Separate module (U1, L2), candidate | 15/24 | 2/8 | 22/24 | none | 112.9 s | 0.84 |
| Shared (U2) | 11/24 | 3/8 | 19/24 | none | 116.9 s | 0.86 |

Held-out accuracy is the router's weighted accuracy on integration turns over four
folds of cases, reported and not gated.

## Routing (candidate)

| Turns | To the scheduling route | To the drafting route |
| --- | --- | --- |
| Scheduling (36) | 33 | 3: one "after" and two "day" requests, all failed |
| Cross (12) | 4 | 8: all four reviews, all failed, and each handoff's drafting turn, as intended |
| Drafting (40) | 12 | 28 |

The other two versions routed in exactly the same pattern. The drafting turns sent
to the scheduling route lost no accepted success this time: the scheduling units
also draft well with the calendar tools.

## What the five developments show

The units now learn scheduling. Round 4's scheduling unit met the scheduling and
cross levels when it served every turn. The router decides the outcome. Logistic
selectors sent every turn one way, to drafting in round 3 and to scheduling in
round 4. The centroid router is about 85% accurate. Its errors fall on turns that
mention plans and dates, which read like drafting but need the calendar. Every
router so far learned from about 180 integration turns, labelled by which route
happened to succeed.

## Cost

Three r7i.4xlarge hosts ran in parallel, $4.05 in total, and every instance is
terminated. A3 has spent $37.16 of its $100 ceiling, plus stage 0's $2.15.
Another development rerun and the confirmation together would need $65 in
allowances, more than the $62.84 left.
