# A3 stage 0 results: room to learn, and the cost of a larger tool list

**Scheduling is a genuinely new capability, and adding its tools alone costs
drafting four successes.** Under the calendar interface, neither the parent nor
the accepted A2 version can schedule: 1 and 0 of 24 development cases. Stage 1
may therefore be declared. The same interface also lowers drafting for both
systems, from 9 to 5 for the parent and from 19 to 15 for the accepted version,
before any scheduling training.

Declaration: [plan and stage 0](ASSISTANT_REPEATED_GROWTH.md). Evidence:
[result](../config/experiments/assistant-growth-baseline-result.json) and
[report](../config/experiments/assistant-growth-baseline-report.json), commit `9c1abdb`.

## Outcomes under the calendar interface (opened development cases)

| Set | Parent | Accepted version |
| --- | --- | --- |
| Scheduling | 1/24 | 0/24 |
| Cross | 0/8 | 0/8 |
| Drafting | 5/24 (9 under the drafting interface) | 15/24 (19 under the drafting interface) |

- **Room to learn.** Both systems solve at most 20 of the 24 scheduling cases, as
  declared, so stage 1 may be declared.
- **How scheduling fails.** Both systems call `list_busy` and `save_meeting`, so they
  find the new tools. The accepted version ran out of model turns in 20 of 24
  episodes, still listing busy times or saving meetings. The parent finished 17
  episodes but booked the wrong time; three of its generations did not terminate.
- **The interface effect.** The drafting cases and goals are unchanged; only the
  tool list and instruction grew. The parent lost 5 drafting successes and gained 1.
  The accepted version lost 4
  (`workflow-1124e7a2f25c`, `workflow-21c0be04a7be`, `workflow-9e59abaad961` and
  `workflow-e47fa617df9c`) and gained none. As declared, cohort 2 cannot be
  accepted until these are regained.
- **Selection.** The round-4 gate chose the update for every episode, scheduling
  and cross included.
- **Latency.** p95 was 98.3 s for the parent on drafting and 116.9 s for the
  accepted version, both within 180 s.

One r7i.4xlarge host cost $2.15, and everything is retired.

## What this means for stage 1

- **Collection starts from almost nothing.** With 1 natural success in 24, verified
  scheduling experience will mostly come from coached practice, as in A2's first
  round.
- **Every prompt lists every tool, and that has a cost.** Three more tool
  definitions changed drafting behaviour enough to lose a sixth of its successes.
  An assistant that keeps adding capabilities cannot show every tool in every
  prompt. Stage 1 therefore has to decide how drafting keeps its accepted behaviour
  under a growing interface.
