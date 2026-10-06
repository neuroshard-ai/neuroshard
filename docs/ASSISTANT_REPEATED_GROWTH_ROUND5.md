# A3 round 5: scheduling with a free-slot tool, the control given the same tool

Declared on October 6, 2026, after the [sealed confirmation](ASSISTANT_REPEATED_GROWTH_CONFIRMATION_RESULTS.md)
and its [diagnostic](ASSISTANT_REPEATED_GROWTH_CONFIRMATION_DIAGNOSTIC.md), and before any
round-5 execution. Contract: [assistant-growth-round5.json](../config/experiments/assistant-growth-round5.json).
The project owner chose this round and raised A3's ceiling to $150.

## Why

The candidate completed 117/192 sealed scheduling episodes against a bar of 154. In 76 of
its 86 failures it saved the meeting on the right date, with the right teams and length,
at the wrong start time, having read every attendee's busy times. Routing, tool choice and
team choice worked; the computation over busy times did not.

## The change: one deterministic tool, for every system

A new calendar interface adds `free_slots(teams, date, duration_minutes, not_before)`. It
returns the windows on one date in which every listed team is free for at least the length,
earliest first, starting on the quarter hour no earlier than the bound. It computes from the
public calendars only. The model still chooses the teams, the date, the length and the
bound, and saves the meeting itself. Every other tool, limit and scoring rule is unchanged,
and the drafting route keeps its policy.

Every system that serves scheduling gets the tool: the candidate, the separate update, the
shared update, and the accepted version served as the previous system. The confirmation's
gain over that system is therefore learning, not the tool.

## What runs

- **Experience.** Round 4's 1,152 training cases, each solved once under the new interface.
  A demonstration must pass the scorer, and every saved meeting must start at the first
  window the tool returned for its date; otherwise the round stops before training.
- **Training.** Round 4's steps, mixture and settings. U2 and L2 again start from the
  accepted version.
- **Integration.** Stage 1's runs, plus the accepted version on scheduling: the control's
  rate with the tool.
- **Routing.** The calibrated router, pinned and unchanged. Its features use the drafting
  policy, which this round does not change.
- **Development, then a fresh confirmation.** Stage 1's gates and thresholds. The spent
  confirmation sets are replaced by fresh ones frozen in this declaration: 192 scheduling
  and 48 cross episodes, and 192 drafting episodes for retention. They open once, only after
  the candidate passes development.

## Budget

A3 has spent $62.87. Round 5 allows at most $10 on one A10G host, $18 for development on
three CPU hosts and $47 for the confirmation on five. That is $137.87 in total, within the
$150 ceiling.
