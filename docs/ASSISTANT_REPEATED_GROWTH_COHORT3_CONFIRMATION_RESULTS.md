# A3 cohort 3's fresh sealed confirmation: the drafting upgrade is accepted

**On fresh sealed episodes, the upgraded system solved all 192 drafting episodes against the
previous system's 183: a net gain of 9 with none lost, and the lower 95% bound on the gain per
episode above zero. It kept every scheduling and cross success of the previous system (192/192
and 47/48 for both), and its p95 was 82.8 s against 100.4 s. Every check passes within the
declared resource budget, so cohort 3 is accepted.** With drafting and scheduling, A3 has its
three accepted cohorts; the [closure review](A3_REPEATED_GROWTH_REVIEW.md) maps every clause to
evidence.

Declarations: [cohort 3](ASSISTANT_REPEATED_GROWTH_COHORT3.md) and its
[development pass](ASSISTANT_REPEATED_GROWTH_COHORT3_DEVELOPMENT_RESULTS.md). Evidence:
[report](../config/experiments/assistant-growth-cohort3-confirmation-report.json) and the four
results ([upgraded, drafting](../config/experiments/assistant-growth-cohort3-confirmation-upgrade-drafting-result.json),
[upgraded, calendar](../config/experiments/assistant-growth-cohort3-confirmation-upgrade-calendar-result.json),
[previous, drafting](../config/experiments/assistant-growth-cohort3-confirmation-previous-drafting-result.json),
[previous, calendar](../config/experiments/assistant-growth-cohort3-confirmation-previous-calendar-result.json)),
commit `fbb76e4`. The sealed sets were `confirmation6` for drafting and `confirmation3` and
`cross-confirmation3` for scheduling, each opened once.

## The gate

| Check | Required | Result |
| --- | --- | --- |
| Net drafting gain over the previous system | at least 5 | +9 (9 gained, none lost) |
| Lower 95% bound on the drafting gain per episode | above 0 | 0.010 (bootstrap over the eight families) |
| Previous successes lost | none on drafting, scheduling or cross | none |
| p95 latency | at most 180 s | 82.8 s |
| p95 against the previous system | at most 10% above, 110.4 s | 18% below |

The upgraded system is cohort 2's accepted system with L3 on its drafting route; the previous
system is cohort 2's accepted system unchanged. Both ran routed turn by turn by the same pinned
router, with the same scheduling route.

## By drafting family

| Family | Upgraded | Previous |
| --- | --- | --- |
| Copy | 24/24 | 23/24 |
| Date | 24/24 | 24/24 |
| Sum | 24/24 | 24/24 |
| Difference | 24/24 | 20/24 |
| Latest revision | 24/24 | 21/24 |
| Recipient | 24/24 | 24/24 |
| Reschedule | 24/24 | 24/24 |
| Scope | 24/24 | 23/24 |

The gains are where the previous system's sealed failures were: the change from approved revision
1, the switch to it, and corrections that need several calls. On scheduling and cross the two
systems had the same outcome on every episode; the one cross failure is the same case for both.

## The resource budget

L3 adds one rank-16 module, 1,048,576 parameters (4.2 MB beside U1's 251.7 MB), and trained on
one A10G for $1.09 of its $10 allowance. Served, the upgraded system was faster, not slower: p95
82.8 s against 100.4 s, 1,299 drafting model calls against 1,491, and its drafting host finished
in 10,944 s against the previous system's 13,044. It is better than keeping the previous system
on every measured axis, within the declared budget.

## Cost

Four r7i.4xlarge hosts, $15.35, and every instance is terminated. A first attempt stopped on
every host before any sealed case was generated, because the sealed manifests were missing from
the execution's sources ($0.15); the execution was corrected and pinned before the retry. Cohort 3
cost $17.62 in all. A3 has spent $108.79 of its $150 ceiling, plus stage 0's $2.15.

## What this means

A3's three cohorts are accepted: drafting (cohort 1), scheduling as a new capability with tasks
that need both (cohort 2), and an upgrade of drafting (cohort 3). Each acceptance re-checked
every earlier cohort on fresh sealed episodes. Acceptance is not native promotion or a public
launch.
