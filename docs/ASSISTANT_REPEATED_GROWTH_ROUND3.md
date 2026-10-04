# A3 stage 1, round 3: learning scheduling from correct solutions

Declared on October 4, 2026, after the [round-2 results](ASSISTANT_REPEATED_GROWTH_ROUND2_RESULTS.md)
and before any round-3 execution. Contract:
[assistant-growth-round3.json](../config/experiments/assistant-growth-round3.json).
The project owner chose this round from the options in the round-2 report.

## Why

Two collections produced no complete coached scheduling success. The accepted
version could not start: it guessed team names that the calendar does not accept,
made one call per reply, and booked times earlier than the first free one. Its own
experience therefore contains almost nothing to learn from.

## What changes: the source of scheduling experience

- **One demonstration per training case.** A goal-directed solver writes a correct
  solution for each of the 256 scheduling and 32 cross training cases, as the
  accepted version's own replies: one or two tool calls in its native envelope per
  reply, or a one-sentence confirmation.
- **What the solutions show.** Team names exactly as the user writes them. Two
  `list_busy` calls per reply. A plan's due date from `shift_date` on the plan's start
  date and review interval. The expected draft and meeting saved once each, with no
  extra writes.
- **Verified before use.** Each demonstration is executed in the calendar workspace
  under the calendar policy and must pass the scorer within the policy's limits.
  The solver refuses every split except training.
- **Nothing is sampled.** U2 and L2 train from the accepted version on the declared
  mixture, with the demonstrations as its four scheduling sequences per step.

Cohort 2 is thus learned from demonstrations rather than from the accepted version's
own experience. A3's criterion asks for accepted cohorts, not for a source of
experience. Cohort 1 was learned from verified own experience in A2.

## What stays

The calendar interface and its limits, routing, the three versions, training
schedule, steps and learning rates, integration, and every development and
confirmation gate are stage 1's.

## Budget

Stage 1 and round 2 cost $19.04. Round 3 allows at most $11 for one GPU host
(3.5-hour expiry), $18 for development on three CPU hosts and $47 for confirmation
on five, $95.04 in total within the unchanged $100 ceiling.
