# A3 closure review: repeated useful growth and an upgrade

October 7, 2026. This review checks [A3](../TODO_ASSISTANT.md) against published results only.
It runs nothing new and changes no result.

## The criterion

> Three successive accepted cohorts, cumulative retention and cross-capability tasks; at least
> one new capability and one upgrade. Separate modules must retain earlier cohorts better than
> repeated shared-weight updates under the same budget. Demonstrate a beneficial update or
> consolidation against keeping the previous system under a declared resource budget. This
> establishes bounded growth, not unlimited intelligence or a no-forgetting theorem.

## Clause by clause

Every result below served the pinned Granite 4.1 3B parent on the canonical CPU runtime, and
every acceptance was decided once on fresh sealed episodes behind a declared development pass.

| Clause | Evidence | Met |
| --- | --- | --- |
| Cohort 1 accepted | Drafting: the round-4 update U1 [passed A2's sealed third confirmation](ASSISTANT_EXPERIENCE_THIRD_RESULTS.md), 183/192 against the parent's 119, every family at least 21/24. | Yes |
| Cohort 2 accepted | Scheduling: the separate module L2, routed with U1, [solved 192/192 sealed scheduling episodes and 44/48 cross](ASSISTANT_REPEATED_GROWTH_CONFIRMATION2_RESULTS.md), against 68 and 2 for the previous version given the same tool; +166, none lost. | Yes |
| Cohort 3 accepted | An upgrade of drafting: L3 on U1 [solved 192/192 sealed drafting episodes](ASSISTANT_REPEATED_GROWTH_COHORT3_CONFIRMATION_RESULTS.md) against the previous system's 183, net +9 with none lost and a lower 95% bound above zero. | Yes |
| Successive | Each cohort was declared after the previous one was decided: [cohort 2](ASSISTANT_REPEATED_GROWTH.md) after A2, [cohort 3](ASSISTANT_REPEATED_GROWTH_COHORT3.md) after cohort 2's acceptance. | Yes |
| Cumulative retention | Cohort 2's confirmation re-checked drafting on 192 fresh episodes: no accepted success lost (177/192 for both). Cohort 3's re-checked drafting, scheduling and cross on fresh episodes: no success of the previous system lost on any of them. | Yes |
| Cross-capability tasks | Two cross families need both capabilities: a review on a plan's due date, and a handoff after a saved draft. Sealed: 44/48 at cohort 2's acceptance and 47/48 at cohort 3's, none lost. | Yes |
| At least one new capability | Scheduling was new: before cohort 2, the parent solved 1 and the accepted version 0 of 24 development cases ([stage 0](ASSISTANT_REPEATED_GROWTH_STAGE0_RESULTS.md)). | Yes |
| At least one upgrade | Cohort 3 upgraded drafting, cohort 1's capability, from 183 to 192 of 192 sealed episodes. | Yes |
| Separate modules retain better than shared updates, same budget | Cohort 2 trained a separate update, a separate module and the shared update on the same experience, replay and optimizer steps. On the sealed drafting set the separate unit lost no accepted success and the shared update lost five (gaining six); at the [first confirmation](ASSISTANT_REPEATED_GROWTH_CONFIRMATION_RESULTS.md), none against one. | Yes |
| Low-rank module against update (reported, per A2's amendment) | In [development](ASSISTANT_REPEATED_GROWTH_DEVELOPMENT7_RESULTS.md) the separate module and the separate update had identical outcomes on every case; the module was the accepted candidate. | Reported |
| Beneficial update against keeping the previous system, declared budget | Declared before training: one rank-16 module (1,048,576 parameters), one GPU host within $10, p95 at most 10% above the previous system's. L3 trained for $1.09 and its system was faster: p95 82.8 s against 100.4 s, with 13% fewer drafting model calls, while gaining 9 sealed drafting episodes and losing none. | Yes |

## Failed work

A3 published every attempt with its cost. Stage 1 learned little ($14.22); round 2 stopped by its
rule ($4.82); round 3 learned in part ($2.96) and its development failed ($3.53); round 4's
development failed on one drafting success ($4.05) and a router rerun failed on routing ($4.05);
the first learned router was 74% accurate ($1.82). The first sealed confirmation failed on
scheduling ($20.04): a [diagnostic](ASSISTANT_REPEATED_GROWTH_CONFIRMATION_DIAGNOSTIC.md) found
the arithmetic over busy times failing, and round 5 gave every system a free-slot tool. In cohort
3, a GPU placement found no capacity ($0.004) and the first confirmation attempt stopped before
any sealed case was generated ($0.15). A3 cost $108.79 within its $150 ceiling, plus stage 0's
$2.15.

## Verdict

**Every clause is met by published evidence, so A3 is complete.** The assistant gained a new
capability and then an upgrade, each learned as a separate unit and accepted on fresh sealed
episodes that re-checked everything accepted before, and the separate units retained earlier
cohorts better than a shared update trained on the same budget.

## What A3 does not establish

- **Unbounded growth.** Three cohorts, as the criterion says: bounded growth, not unlimited
  intelligence or a no-forgetting theorem.
- **Open-ended tasks.** All tasks come from fictional workspace grammars; held-out cases vary
  values and corrections within them.
- **Arithmetic in the model.** Free windows are computed by a deterministic tool that every
  system, the previous one included, was given; the learned part is using it in conversation.
- **Self-discovered skills.** Both later cohorts learned from solver-written demonstrations of
  training cases, verified by the scorer, rather than from the assistant's own exploration.
- **The comparison at other budgets.** Separate against shared was measured once, at cohort 2's
  budget.
- **Independent operation and a public assistant.** One operator ran every host in one region;
  those are A5 and A6.
