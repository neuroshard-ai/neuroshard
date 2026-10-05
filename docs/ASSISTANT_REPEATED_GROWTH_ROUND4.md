# A3 stage 1, round 4: more correct solutions and routing fitted where it serves

Declared on October 4, 2026, after the [development results](ASSISTANT_REPEATED_GROWTH_DEVELOPMENT_RESULTS.md)
and before any round-4 execution. Contract:
[assistant-growth-round4.json](../config/experiments/assistant-growth-round4.json).
The project owner chose this round from the options after development.

## Why

Round 3's scheduling units completed 41% to 44% of the integration scheduling
episodes alone, short of the gate's 75%. The selectors made it worse. They counted
every tie for drafting, including turns where both routes failed, so they sent most
scheduling turns to the drafting route, which cannot schedule. They were also
fitted on features computed on the GPU and served on features computed on the CPU.
When the candidate's first scheduling turn did reach L2, it passed 12 times in 16.

## What changes

- **Four times the demonstrations.** Six further training splits of the same
  grammar, 864 cases with new seeds, are frozen in
  [their own manifest](../config/experiments/assistant-schedule-data-growth.json).
  With round 3's 288 cases, 1,152 verified correct solutions are the scheduling
  experience, written by round 3's solver.
- **Steps scale with the data.** U2 and L2 train for 512 steps instead of 128, so
  each demonstration is seen about as often as in round 3. The mixture and every
  other training setting are unchanged, and the same for both arms.
- **Selectors refitted where they serve.** Each development host fits its version's
  selector before serving any development case. It uses the pinned integration
  outcomes and the parent's features computed on that host's CPU runtime. A turn
  that both routes always failed is no example; other ties still count for drafting.

## What stays

The calendar interface and its limits, routing, the three versions, integration,
and every development and confirmation gate are stage 1's.

## Budget

Stage 1, rounds 2 and 3 and development cost $25.53. Round 4 allows at most $9 for
one A10G GPU host, $18 for development on three CPU hosts and $47 for confirmation
on five. That is $99.53 in total, within the unchanged $100 ceiling, leaving no room
for another round.
