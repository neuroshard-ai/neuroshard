# Repeated growth (A3): plan and stage 0

Declared on October 3, 2026, after [A2 was established](ASSISTANT_EXPERIENCE_THIRD_RESULTS.md)
and before any scheduling measurement. Authorized by the project owner, who chose
meeting scheduling as the second cohort and counted A2's capability as the first.
The [contract](../config/experiments/assistant-growth.json) pins every rule below.

## What A3 asks

[A3](../TODO_ASSISTANT.md) asks for three successive accepted cohorts with cumulative
retention and tasks that combine them, at least one new capability and one
upgrade. Separate learning units must retain earlier cohorts better than repeated
updates of shared weights under the same budget. One update or consolidation must
also prove better than keeping the previous system, under a declared resource
budget.

## The cohorts

- **Cohort 1: drafting.** The capability accepted in A2: the parent plus the
  round-4 update, selected per episode.
- **Cohort 2: scheduling, a new capability.** The assistant reads team calendars,
  finds the earliest time every attendee is free, books a meeting draft and
  applies follow-ups. Some tasks need both capabilities: book a review on a plan's
  due date, or schedule a handoff after saving a draft.
- **Cohort 3: an upgrade** of an existing capability, declared once cohort 2 is
  decided.

A cohort is accepted only by a sealed confirmation that also re-checks every
earlier cohort, so retention is cumulative.

## The calendar interface

The [calendar workspace](../src/neuroshard/evolution/assistant_calendar.py) keeps every
drafting tool and rule unchanged and adds three tools: a team's busy times on a
date, exact clock arithmetic, and a meeting draft saved locally. Nothing is booked
or sent. The [calendar policy](../config/experiments/assistant-workflow-policy-calendar.json)
keeps the drafting instruction word for word and adds the meeting rules: book the
earliest quarter-hour start at which every attendee is free for the whole
meeting, within working hours.

Every prompt lists the tools, so the larger tool list changes what drafting
conversations look like too. Drafting tasks therefore run unchanged under the
calendar interface, and retention is measured there.

## The tasks

The [scheduling grammar](../src/neuroshard/evolution/assistant_schedule_data.py) gives
each case its project's delivery plans, as in drafting, and the busy times of four
teams over two weeks.

- **Single-turn families:** the earliest common slot on a date; the same, no
  earlier than a given time; three attendees; and the first date from a given day
  with a common slot.
- **Families with a follow-up:** make the meeting longer, invite a third team, move
  it to the next day, or replace an attendee.
- **Cross families:** book a review on the latest approved plan's due date, citing
  the plan; or save a draft, then schedule a handoff on its due date.

Expected meetings are computed from the calendars and checked against an
independent minute-by-minute search; they never enter a request. Every case is
solvable within the conversation budgets. Splits use their own seeds, disjoint in
ID and project from every drafting split. The
[manifest](../config/experiments/assistant-schedule-data.json) freezes them all now,
including the sealed scheduling and cross confirmations. A new sealed drafting
split, [`confirmation4`](../config/experiments/assistant-workflow-data-confirmation4.json),
holds 192 cases for retention.

## The required comparison

Under the same experience, replay and optimizer steps, cohort 2 trains three ways:

- **Separate update:** a new update of the cohort-1 projections, trained on top of
  the frozen accepted version and kept apart from it, so selection can fall back
  to the accepted version.
- **Separate module:** a new rank-16 low-rank module on the same projections, also
  on top of the accepted version.
- **Shared update:** the cohort-1 update itself trained further on the scheduling
  experience with replay of verified drafting experience, replacing it.

A3 requires the separate units to retain cohort 1 better than the shared update.
Whether the low-rank module matches the separate update is reported, as the A2
amendment moved that question here.

## Stage 0: room to learn and the interface effect (CPU, opened data)

Before any training is declared, one CPU host serves the opened development cases
under the calendar interface: 24 scheduling, 8 cross and 24 drafting. It runs the
unchanged parent first, then the accepted version, each in a fresh worker with
prefix-cache serving.

- **Room to learn.** Both systems must solve at most 20 of the 24 scheduling cases;
  otherwise a harder grammar is declared.
- **Interface effect.** Drafting outcomes are compared case by case with the pinned
  results of the same systems under the drafting interface. Any drafting success
  lost to the interface must be regained before cohort 2 can be accepted.

Stage 1 is declared after this report, and only with room to learn. It covers
collection, training, selection and the sealed confirmation. Stage 0 opens no
sealed split, trains nothing and earns no credit. One r7i.4xlarge host, four-hour
expiry, $8 allowance.
