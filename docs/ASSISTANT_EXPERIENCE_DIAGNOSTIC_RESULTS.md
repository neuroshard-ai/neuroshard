# Development diagnostic results

The [diagnostic](ASSISTANT_EXPERIENCE_DIAGNOSTIC.md) served the compositional
study's update control and small module on the 24 opened development cases, on
two canonical CPU hosts, from commit `16d2bad`. No gate; the third confirmation
split stays sealed. Evidence: [report](../config/experiments/assistant-experience-diagnostic-report.json).

## Latency and outcomes

| System | Development successes | p95 episode seconds |
|---|---|---|
| Update control | 19 | 96.6 |
| Small module | 16 | 98.5 |
| Canonical parent | 9 | — |

The small module's serving would meet the second confirmation gate's latency
limits: 98.5 s against 180 s, and 1.02 times the update against 2.0.

## How the small module fails

- **latest (4 cases, 3 of them parent successes).** The module reads approved
  revision 1, computes the total, and applies the added "one calendar day
  later" correctly. But it starts the review-interval shift from the start date
  plus the interval instead of the start date, so it applies the interval twice.
  For example, with revision 1 starting on 2027-05-11 and a 4-day interval, it
  calls `shift_date("2027-05-15", 4)` and then adds the day, saving 2027-05-20.
  The update calls `shift_date("2027-05-11", 4)` and saves 2027-05-16. The lost
  parent successes are `workflow-1124e7a2f25c`, `workflow-c6b1500447f3` and
  `workflow-fddb97d8bc7d`.
- **scope (4 cases).** The module repeats calls until the model-turn budget runs
  out. The parent and the update fail these cases too.

The module does not ignore the added instruction. It miscomputes an argument
while chaining two date shifts.

## A grounding check

An argument that no document, tool result or user message ever showed is
suspect. Treating a date argument as ungrounded when that exact date appears
nowhere earlier in the conversation:

- flags 3 of the small module's 8 failures;
- flags none of the 35 successful episodes of either system;
- misses `workflow-fddb97d8bc7d`, where the miscomputed start date happens to
  equal the other revision's start date.

## Resources

Two CPU allocations, retired with nothing remaining ($1.07).
