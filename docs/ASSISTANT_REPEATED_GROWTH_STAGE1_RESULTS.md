# A3 stage 1 results: the pipeline ran, but scheduling was barely learned

**Stage 1 ran end to end on one GPU host, yet the trained units learned little
scheduling, because almost no verified scheduling experience reached training.**
The accepted version made exactly one tool call per reply, while the calendar
policy allows six replies per user message and two calls per reply. It ran out of
replies in most rollouts, coached ones included. The near-policy filter then
removed every coached partial success. U2 and L2 trained on 33 trajectories from
11 of 288 training cases. On the integration cases their scheduling routes
completed 10/64 and 7/64 scheduling episodes and 0/8 cross episodes. The
development gate needs 18/24 and 6/8, so development was not run. A
[second round](ASSISTANT_REPEATED_GROWTH_ROUND2.md) changes only the collection.

Declaration: [stage 1](ASSISTANT_REPEATED_GROWTH_STAGE1.md). Evidence:
[result](../config/experiments/assistant-growth-stage1-result.json) and
[report](../config/experiments/assistant-growth-stage1-report.json), commit `326f125`.

## Collection

| | Natural (8 per case) | Coached (8 per triggered case) |
| --- | --- | --- |
| Rollouts | 2,304 | 2,288 (286 cases) |
| Every round verified | 6 | 0 |
| Only the first round verified | 46 | 86 (all handoff, whose first round is drafting) |
| Ran out of replies | 1,649 | 2,045 |
| Completed with a wrong or missing meeting | 499 | 156 |
| Generation did not terminate | 104 | 1 |

- **One call per reply.** Every reply that called a tool called exactly one: 13,634
  natural and 13,934 coached replies, against a limit of two. The goal-directed
  fixture solves every family in three to seven replies, so the budget is not
  infeasible, but one call at a time leaves no reply for the confirmation.
- **The card made it worse.** The stage-1 card asked for one `add_minutes` call per
  candidate start, adding steps: 89% of coached rollouts ran out of replies,
  against 72% of natural ones.
- **Wrong times are too early.** In 208 of the 211 natural rollouts whose only error
  was the start time, the saved start was earlier than the correct one: the
  conflict check fails, not the arithmetic.
- **The filter removed every coached success.** Its ceiling, 0.067 mean token loss,
  is the least likely natural success. Those were almost all drafting reference
  episodes, which the accepted version already does well, so no coached scheduling
  trajectory was likely enough. 138 verified scheduling trajectories became 33, none
  coached.

## Training and integration

U2 continued U1 and L2 trained on top of U1, each for 128 steps on the declared
mixture, in about 100 seconds each. Final losses were 0.014 and 0.015, which is
memorization of 33 sequences repeated about 15 times each.

| Route (integration cases) | Scheduling | Cross | Drafting |
| --- | --- | --- | --- |
| U1 with drafting tools | 0/64 | 0/8 | 57/64 |
| U2 with drafting tools | 0/64 | 0/8 | 53/64 |
| U2 with calendar tools | 10/64 | 0/8 | 47/64 |
| L2 with calendar tools | 7/64 | 0/8 | 40/64 |

Greedy episodes are shown. Continuing U1 cost drafting four integration successes
(57 to 53). With so few scheduling successes, most of the 204 integration turns were
ties between the routes (151 for the separate update), and a tie counts for the
drafting route. Only 20 to 29 turns per version targeted the scheduling route.

## Cost and resources

One g6e.2xlarge (NVIDIA L40S) host ran for 6.3 hours, $14.22 conservatively,
after the cheaper g6e.xlarge had no capacity. Collection took 4.7 hours, training
3 minutes and integration 1.4 hours. The worker finished inside its seven-hour
limit, so the declared [integration-only resumption](../config/experiments/assistant-growth-stage1-resume.json)
was not needed. The instance is terminated, its volume deleted and its security
group retired.

## What this means

A new capability that the model almost never completes alone cannot be learned
from experience filtered to what the model already does. The coaching has to fit
the interface's budget, and the experience it produces has to reach training.
Round 2 keeps everything else fixed: the interface and its limits, routing, the
versions, training, integration and every gate.
