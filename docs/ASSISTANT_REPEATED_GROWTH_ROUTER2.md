# A3 stage 1: a router that learns what each turn needs

Declared on October 5, 2026, after the [centroid-router development results](ASSISTANT_REPEATED_GROWTH_DEVELOPMENT5_RESULTS.md)
and before any router fit. Contract:
[assistant-growth-router2.json](../config/experiments/assistant-growth-router2.json).
The project owner chose this router and raised A3's stage-1 ceiling from $100 to $110.

## Why

The centroid router kept every drafting success but was about 85% accurate. It sent
every review turn and three scheduling requests to the drafting route, which cannot
schedule. When every turn went to the scheduling unit instead, the same candidate
reached the scheduling and cross levels. Every router so far learned from about 180
integration turns, labelled by which route happened to succeed.

## The router

- **Label: what the turn needs.** A turn is labelled for the scheduling route if its
  verified correct solution uses the calendar. Each training scheduling case is
  solved by its round-3 demonstration, and a turn's label is whether that turn calls
  `list_busy`, `add_minutes` or `save_meeting`. Every drafting training turn is
  labelled for drafting: its goals are drafts, and drafting has no calendar.
- **Data.** The 1,152 training scheduling cases of round 4, 1,728 turns, and the 256
  drafting training cases, 384 turns. No integration, development or confirmation
  goal is a label.
- **Features and rule.** As before: the parent's state at each user message, computed
  on the CPU runtime that serves development. The centroid rule decides, with every
  example weighted equally.
- **Reported, not gated.** Held-out accuracy over four folds of training cases, and
  accuracy on the 204 integration turns.
- **One router, pinned.** What a turn needs does not depend on the unit, so one router
  serves all three versions. It is fitted once on a CPU host and pinned by digest
  before development starts.

## What stays

Round 4's units, routing, the three versions, and every development and
confirmation gate. Nothing is trained. If the candidate passes, the sealed
confirmation opens once with the same router.

## Budget

A3 has spent $37.16. The router allows at most $7 on one CPU host, development $18
on three and confirmation $47 on five. That is $109.16 in total, within the raised
$110 ceiling.
