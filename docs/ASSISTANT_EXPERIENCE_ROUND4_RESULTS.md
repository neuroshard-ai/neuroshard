# Round 4 results: goal-guided repairs

**Round 4 ran as declared, with a mixed result.** The update arm improved on
integration and the addition arm declined. Development under prefix-cache
serving, with the A1 served-system check, runs next as declared. Integration
refits the gates and does not decide anything.

Declaration: [second attempt](ASSISTANT_EXPERIENCE_LEARNING.md#second-attempt-goal-guided-repairs-and-fresh-confirmation).
Evidence: [report](../config/experiments/assistant-experience-round4-report.json), commit `ec57db9`.

## Repairs

The round-3 addition sampled four rollouts per training case (1,024 in all) and
passed 969 (94.6%).

- 37 of the 55 failures opened a document outside their round's training goal.
- Each of these got two repair attempts, 74 in all. 49 continued to a rollout
  that the frozen scorer passed in every round.
- The repairs produced 15 distinct preference pairs across 9 cases (at most 4 per
  case) and 18 repaired trajectories (at most 2 per case). The trajectories joined
  the round-1 experience.

Both round-3 arms then continued for 64 steps with DPO against themselves.
Preference margins rose from 0 to 4.5–9.3 for the addition and 5.7–10.0 for the
update.

## Integration (64 cases, one greedy and four sampled episodes each)

| System | Greedy | Sampled | Round 3 greedy / sampled |
| --- | --- | --- | --- |
| Parent | 53.1% | 52.7% | 53.1% / 52.3% |
| Update | 90.6% | 90.6% | 85.9% / 87.1% |
| Addition | 89.1% | 84.8% | 90.6% / 91.4% |

Both refitted gates are logistic, with one parent-better case each. The update
arm gained on the cases it had missed. The addition lost sampled reliability: 15
pairs moved its 2 MiB of LoRA more than they fixed. That makes update parity on
confirmation harder than after round 3.

## Resources

One g5.2xlarge (A10G) host: collection 87 minutes, training 10, integration 72.
It cost $3.67, and everything is retired. Total experience compute so far is $30.04.

## Next

Development runs with the round-4 arms under the declared prefix cache, plus the
A1 served-system check. The fresh 192-episode confirmation opens once, only if
both pass.
