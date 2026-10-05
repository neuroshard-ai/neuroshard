# A3 round 3 results: correct solutions teach scheduling, not yet enough

**Trained on one verified correct solution per training case, the scheduling units
learned the capability, but not yet to the development gate's level.** On the
integration cases U2 with the calendar tools completed 28/64 scheduling and 6/8
cross episodes greedily, and L2 completed 26/64 and 4/8. In stage 1 the figures
were 10/64 and 7/64 scheduling, and 0/8 cross. The development gate asks for 18/24
and 6/8. Development runs as declared.

Declaration: [round 3](ASSISTANT_REPEATED_GROWTH_ROUND3.md). Evidence:
[result](../config/experiments/assistant-growth-round3-result.json) and
[report](../config/experiments/assistant-growth-round3-report.json), commit `c567d79`.

## Experience and training

All 288 demonstrations (256 scheduling, 32 cross) passed the scorer in the calendar
workspace, 1,488 replies in total, in 8 seconds. U2 continued U1 and L2 trained on
top of U1, each for the declared 128 steps on the declared mixture, with the
demonstrations as the four scheduling sequences per step. Final losses were 0.0135
and 0.0138.

## Integration

| Route (integration cases) | Scheduling | Cross | Drafting |
| --- | --- | --- | --- |
| U1 with drafting tools | 0/64 | 0/8 | 59/64 |
| U2 with drafting tools | 0/64 | 0/8 | 54/64 |
| U2 with calendar tools | 28/64 (45% sampled) | 6/8 (50%) | 51/64 |
| L2 with calendar tools | 26/64 (38%) | 4/8 (56%) | 55/64 |

Greedy episodes are shown, with the sampled success rate in parentheses.

- **Scheduling is learned in part.** Both scheduling units went from almost
  nothing to about two in five scheduling episodes, and to half or more of the
  cross episodes.
- **Drafting.** Continuing U1 into U2 cost drafting five integration successes
  (59 to 54), which is the shared version's cost. L2 under the calendar tools kept
  more drafting than U2 (55 against 51).
- **Selectors.** Per version, 72 or 73 of the 204 integration turns favoured the
  scheduling route, 9 to 18 the drafting route, and the rest were ties, which count
  for drafting.

## Cost

One g5.2xlarge (NVIDIA A10G) host ran for 2.4 hours, $2.96 conservatively, after
both L40S types had no capacity. Training took 10 minutes and integration 2.1 hours.
The instance is terminated, its volume deleted and its security group retired.
A3 has spent $22.00 of its $100 ceiling, plus stage 0's $2.15.
