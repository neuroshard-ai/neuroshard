# A3 round 4 results: four times the correct solutions, about 60% scheduling

**With four times the verified correct solutions and steps scaled to match, the
scheduling units completed about 60% of the integration scheduling episodes
alone.** U2 with the calendar tools completed 39/64 scheduling and 6/8 cross
episodes greedily, and L2 completed 38/64 and 3/8. In round 3 the figures were
28/64 and 26/64. Continuing U1 into U2 now also improved drafting, 63/64 against
U1's 59/64. The development gate asks for 18/24 scheduling and 6/8 cross.
Development runs as declared, with each host refitting its selector.

Declaration: [round 4](ASSISTANT_REPEATED_GROWTH_ROUND4.md). Evidence:
[result](../config/experiments/assistant-growth-round4-result.json) and
[report](../config/experiments/assistant-growth-round4-report.json), commit `6dedb88`.

## Experience and training

All 1,152 demonstrations passed the scorer in the calendar workspace: round 3's
288 cases and 864 from the six further splits. U2 continued U1 and L2 trained on
top of U1, each for 512 steps on the declared mixture, in 39 minutes together.
Final losses were 0.0125 and 0.0117.

## Integration

| Route (integration cases) | Scheduling | Cross | Drafting |
| --- | --- | --- | --- |
| U1 with drafting tools | 0/64 | 0/8 | 59/64 |
| U2 with drafting tools | 0/64 | 0/8 | 63/64 |
| U2 with calendar tools | 39/64 (59% sampled) | 6/8 (75%) | 61/64 |
| L2 with calendar tools | 38/64 (56%) | 3/8 (56%) | 56/64 |

Greedy episodes are shown, with the sampled success rate in parentheses.

- **Scheduling grew with the data.** Four times the correct solutions took both
  scheduling units from about two in five scheduling episodes to about three in five.
- **Drafting.** With four times the steps, each step also replays drafting
  experience, so U2's drafting rose from 54/64 in round 3 to 63/64.
- **Selectors.** The selectors fitted on the GPU are reported only. Per version,
  85 to 90 of the 204 turns favoured the scheduling route. Development refits each
  version's selector on its own CPU runtime, leaving out turns both routes failed.

## Cost

One g5.2xlarge (NVIDIA A10G) host ran for 2.9 hours, $3.54 conservatively.
Training took 39 minutes and integration 2.0 hours. The instance is terminated,
its volume deleted and its security group retired. A3 has spent $29.07 of its
$100 ceiling, plus stage 0's $2.15.
