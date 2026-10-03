# Second confirmation results (192 fresh episodes)

**Failed on 2 of 9 declared checks, so A2 is not established for this
capability.** On 192 fresh sealed episodes, the added module is far better than
the unchanged parent. It still loses two parent successes and trails the
60× larger update control by more than the declared margin. This split is now
spent, and the first confirmation also stays failed.

Declaration: [second attempt](ASSISTANT_EXPERIENCE_LEARNING.md#second-attempt-goal-guided-repairs-and-fresh-confirmation).
Evidence: [report](../config/experiments/assistant-experience-confirmation2-report.json) and per-system
results ([parent](../config/experiments/assistant-experience-confirmation2-parent-result.json),
[update](../config/experiments/assistant-experience-confirmation2-update-result.json),
[addition](../config/experiments/assistant-experience-confirmation2-addition-result.json)), commit `0c31b3e`.

## Outcome (all three systems under prefix-cache serving)

| System | Correct | p95 |
| --- | --- | --- |
| Parent | 106/192 (55.2%) | 94.4 s |
| Update (62.9M trained parameters) | 182/192 (94.8%) | 106.1 s |
| Addition (1.05M trained parameters) | 174/192 (90.6%) | 104.3 s |

| Check | Outcome |
| --- | --- |
| Total ≥154 | Pass (174) |
| Each family ≥16 | Pass (lowest 20: difference and reschedule) |
| Net vs parent ≥+20 | Pass (+68: 70 gained, 2 lost) |
| Lost parent successes = 0 | **Fail**: 2 lost |
| Protected successes | **Fail**: the same 2 |
| Lower 95% gain vs parent >0 | Pass (+24.5 points) |
| Lower 95% gain vs update ≥−5 points | **Fail** (−7.3 points) |
| p95 ≤180 s | Pass (104.3 s) |
| p95 ≤2× update | Pass |

## Where it falls short

- **Lost parent successes.** The addition lost two episodes the parent solved: a
  recipient case and a reschedule case. The update solved both. The refitted
  gate routed all 192 episodes to the arm; with one parent-better integration
  case, it could not learn where to fall back.
- **Update parity.** The addition trails the update on 9 episodes and leads on 1.
  The 9 are spread across families (date 3, latest 2, and one each in copy,
  recipient, reschedule and scope), so no single behavior failed. The small
  module is uniformly a little weaker than updating the same projections
  directly.

## What both attempts established

- **Learning works.** Verified self-generated experience, coached practice,
  verified preferences and goal-guided repairs lift the assistant from 55% to
  91–95% on fresh episodes. Both learned systems are about 2× more reliable than
  the parent at the same latency.
- **Modular learning falls just short of full updating.** A 1M-parameter module
  came within 4–8 episodes of a 63M-parameter update on both confirmations, but
  never inside the declared margin.
- **Serving works.** The addition system is also served across machines exactly
  as on one host ([shard serving](GRANITE_SHARD_SERVING_RESULTS.md)).

## Resources

Three r7i.4xlarge hosts cost $10.85, and everything is retired. Total spend
across all assistant and shard executions so far is $54.44 of the $1,000
ceiling.

## Next

A third attempt would need a new declaration and fresh confirmation data, plus
a choice of what to change. Candidates include a larger module capacity (a small
fraction of the backbone but closer to the update's size) or a selector that can
learn to fall back to the parent. That choice is open.
