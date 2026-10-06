# A3 confirmation diagnostic: meetings misplaced within the day

**76 of the candidate's 86 failed sealed episodes saved the meeting on the right date,
with the right teams and length, at the wrong start time.** In every one it had read
each attendee's busy times in that turn, so it had the information and got the interval
arithmetic wrong. 33 meetings started in a shared free gap too short for them, 25 skipped
an earlier free slot, and 18 started inside a busy block. Of the other 10, 6 landed on the
wrong date and 4 ran out of model turns.

Source: [script](../scripts/diagnose_assistant_growth_confirmation.py) and
[output](../config/experiments/assistant-growth-confirmation-diagnostic.json), computed
from the published [candidate result](../config/experiments/assistant-growth-confirmation-candidate-calendar-result.json)
of the [sealed confirmation](ASSISTANT_REPEATED_GROWTH_CONFIRMATION_RESULTS.md). No model
ran and nothing was spent. The sealed sets were spent by that confirmation, so another
attempt needs fresh splits in any case.

## Failures

| Kind | Episodes |
| --- | --- |
| Wrong start: free at the start but runs into a busy block | 33 |
| Wrong start: later than the earliest free slot | 25 |
| Wrong start: starts inside a busy block | 18 |
| Wrong date ("day" 4, "review" 2) | 6 |
| Ran out of model turns | 4 |

63 failures fell on a conversation's first booking and 23 on a follow-up. Most "longer"
and "invite" failures (9 of 13 and 10 of 14) are first bookings, before the follow-up
that defines the family. In "day", which must find the first date with a free slot, 11 of
15 failures chose the right date and then the wrong time.

For example, a 90-minute "slot" meeting: the design team is busy 10:30–11:30, 12:00–13:00
and 15:00–15:30, and the service team 09:30–11:30 and 13:30–14:30. The candidate booked
11:30, where the shared gap lasts 30 minutes. The earliest 90-minute slot is 15:30.

## What it implies

Routing, tool choice and team choice work; the computation over busy times does not.
More examples in the current format look weak: four times the verified solutions took the
scheduling units from 26–28 to 38–39 of 64 on integration, and the confirmation bar is 80%. Two
changes target the failure directly. A calendar tool could return the teams' common free
slots of a given length on a date, so the interval computation is executed rather than
estimated; the unit would still choose the teams, date, length and constraints.
Alternatively, the scheduling route could write out its working, the shared free gaps and
their lengths, before it saves. The interface does not allow that today: a reply that
calls a tool may contain nothing else.
