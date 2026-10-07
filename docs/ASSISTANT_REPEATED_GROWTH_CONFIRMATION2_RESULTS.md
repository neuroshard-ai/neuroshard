# A3's fresh sealed confirmation: cohort 2 is accepted

**On fresh sealed episodes, the candidate completed all 192 scheduling episodes, 24/24 in
every family, and 44 of 48 cross episodes. The previous version, given the same free-slot
tool, completed 68 and 2: a net gain of 166 with none lost. The candidate kept every
accepted drafting success on 192 fresh sealed drafting episodes, at p95 95.6 s. Every check
passes, so cohort 2, meeting scheduling, is accepted.** The shared update lost five accepted
drafting successes, so the separate unit retained drafting better, as A3 requires.

Declarations: [round 5](ASSISTANT_REPEATED_GROWTH_ROUND5.md) and its
[development pass](ASSISTANT_REPEATED_GROWTH_DEVELOPMENT7_RESULTS.md). Evidence:
[report](../config/experiments/assistant-growth-confirmation2-report.json) and the five results
([candidate, calendar](../config/experiments/assistant-growth-confirmation2-candidate-calendar-result.json),
[candidate, drafting](../config/experiments/assistant-growth-confirmation2-candidate-drafting-result.json),
[shared, drafting](../config/experiments/assistant-growth-confirmation2-shared-drafting-result.json),
[accepted, drafting](../config/experiments/assistant-growth-confirmation2-accepted-drafting-result.json),
[accepted, calendar](../config/experiments/assistant-growth-confirmation2-accepted-calendar-result.json)),
commit `7c870ba`. The sealed sets were `confirmation2` and `cross-confirmation2` for
scheduling and `confirmation5` for drafting, each opened once.

## The gate

| Check | Required | Result |
| --- | --- | --- |
| Scheduling | at least 154/192 | 192/192 |
| Each scheduling family | at least 16/24 | 24/24 in all eight |
| Cross | at least 36/48 | 44/48 |
| Net gain over the previous version | at least 20 | +166 (166 gained, none lost) |
| Lower 95% bound on the gain per episode | above 0 | 0.596 (family bootstrap) |
| Accepted drafting successes lost | none | none (177/192, as accepted) |
| p95 latency | at most 180 s | 95.6 s |

The candidate is the separate low-rank module L2 on the scheduling route and the accepted
update U1 on the drafting route, chosen turn by turn by the pinned router. The previous
version is the accepted version with the same tool, so the gain over it is learning, not
the tool.

## By family

| Family | Candidate | Previous version, same tool |
| --- | --- | --- |
| Earliest slot on a date | 24/24 | 7/24 |
| No earlier than a given time | 24/24 | 12/24 |
| Three attendees | 24/24 | 12/24 |
| First date with a common slot | 24/24 | 5/24 |
| Follow-up: make it longer | 24/24 | 3/24 |
| Follow-up: invite a third team | 24/24 | 9/24 |
| Follow-up: move to the next day | 24/24 | 8/24 |
| Follow-up: replace an attendee | 24/24 | 12/24 |
| Cross: review on a plan's due date | 21/24 | 2/24 |
| Cross: handoff after a saved draft | 23/24 | 0/24 |

Without the tool, round 4's candidate completed 117/192 on the spent sealed set, with four
families below 16/24 ([first confirmation](ASSISTANT_REPEATED_GROWTH_CONFIRMATION_RESULTS.md)).

## Retention and A3's comparison

| System on the drafting set | Successes | Gained | Lost |
| --- | --- | --- | --- |
| Accepted version | 177/192 | | |
| Candidate (separate units) | 177/192 | 0 | 0 |
| Shared update | 178/192 | 6 | 5 |

The separate units lost no accepted drafting success; the shared update lost five while
gaining six. A3 requires the separate units to retain cohort 1 better than repeated
shared-weight updates under the same budget, and they do. In development the low-rank module
matched the separate update on every case.

## Routing

The router sent all 288 scheduling turns to the scheduling route and all 288 drafting turns
in each drafting system to the drafting route. Of the 72 cross turns, 48 went to scheduling
and 24 to drafting, the handoff tasks' drafting turns, as intended. Routing was exact on
every sealed turn.

## The four cross failures

Two review episodes used all six model turns, one tool call per reply, before saving the
meeting. One review and one handoff read an older approved revision of the plan, so the
meeting landed on that revision's due date. These are the same two patterns behind the
remaining drafting failures: in development, four of the five drafting failures saved the
right draft but ran out of model turns before the closing confirmation.

## Cost

Five r7i.4xlarge hosts ran in parallel, $20.64 in total, and every instance is terminated.
The previous version's calendar host took the longest, 16,644 s of its 18,000 s limit,
because failing episodes use the whole turn budget. A3 has spent $91.17 of its $150 ceiling,
plus stage 0's $2.15.

## What this means

A3 now has two accepted cohorts with cumulative retention: drafting (cohort 1) and
scheduling (cohort 2), including tasks that need both. The separate-unit comparison holds.
A3 still needs cohort 3, an upgrade of an existing capability, and an update or
consolidation that proves better than keeping the previous system under a declared resource
budget. Acceptance here is not native promotion or a public launch.
