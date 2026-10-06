# A3 round 5 results: with the free-slot tool, the scheduling units solve every integration case

**Under the free-slot interface, both scheduling units completed all 64 integration
scheduling episodes greedily, against 38 and 39 without the tool in round 4. The control,
the accepted version given the same tool but no scheduling training, completed 21.** The
tool removed the interval arithmetic the units failed at, and the difference from the
control is learning. Development runs next, with the calibrated router pinned.

Declaration: [round 5](ASSISTANT_REPEATED_GROWTH_ROUND5.md). Evidence:
[result](../config/experiments/assistant-growth-round5-result.json) and
[report](../config/experiments/assistant-growth-round5-report.json), commit `c3b8b34`.

## Integration (greedy; sampled success rate in brackets)

| Route | Scheduling (64) | Cross (8) | Drafting (64) |
| --- | --- | --- | --- |
| U2 on the scheduling route | 64 (1.00) | 8 (1.00) | 61 (0.94) |
| L2 on the scheduling route | 64 (0.99) | 6 (0.75) | 57 (0.90) |
| Control: U1 on the scheduling route, untrained, with the tool | 21 (0.38) | 0 (0.00) | 53 (0.82) |
| U1 on the drafting route | 0 | 0 | 60 (0.91) |
| U2 on the drafting route | 0 | 0 | 62 (0.96) |

The drafting route has no calendar, so it schedules nothing; routing decides which turns
reach it. Integration cases are not sealed, and earlier routers were fitted on them.
Development and the fresh sealed confirmation decide.

## Collection and training

All 1,152 demonstrations passed the scorer, and every saved meeting started at the first
window the tool returned; 5,568 replies in all. U2 and L2 each trained for 512 steps,
about 21 minutes apiece, to a final loss of 0.011. Integration took 2.8 hours on one A10G.

## Cost

One g5.2xlarge host cost $4.46; the instance is terminated and its security group retired.
A3 has spent $67.33 of its $150 ceiling, plus stage 0's $2.15.
