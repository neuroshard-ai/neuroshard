# A3 round 2 results: stopped by its rule, and the cause found

**Round 2 ended by its declared stop rule, before any training.** Only 2 of the 288
training cases had a complete verified trajectory, both from stage 1's natural
rollouts. The budget-aware card produced 36 verified first rounds and no complete
success. This round's evidence shows why coaching alone does not get started.
The `list_busy` tool accepts a team only under its exact name, such as "operations
team". The model usually shortened it, and the generic error gave no hint.
Even when its first call named a real team, no coached first round passed.

Declaration: [round 2](ASSISTANT_REPEATED_GROWTH_ROUND2.md). Evidence:
[result](../config/experiments/assistant-growth-round2-result.json) and
[report](../config/experiments/assistant-growth-round2-report.json), commit `bec4d08`.

## What the coached rollouts did

| Ending (2,288 coached rollouts, 286 cases) | Count |
| --- | --- |
| Ran out of replies | 1,777 |
| Completed with a wrong or missing meeting | 465 |
| Only the first round verified (handoff, whose first round is drafting) | 36 |
| Generation did not terminate | 10 |
| Every round verified | 0 |

- **Still one call per reply.** All 14,275 replies that called a tool called exactly
  one, although the card asked for two independent calls per reply.
- **Team names.** The `team` argument of `list_busy` is a free string, valid only as
  the user writes it. 5,317 of 6,871 coached `list_busy` calls named a team that does
  not exist ("operations", "Operations", "ops"), and so did 3,998 of 4,615 stage-1
  natural calls. Each returned "invalid tool arguments or unavailable workspace
  object", and the model guessed again until its replies ran out. It also opened
  2,171 of the 2,288 episodes with `list_documents` and called `save_draft` in 1,187,
  although only the 32 cross cases use documents and only handoff needs a draft.
- **Valid names were not enough.** In 503 coached rollouts the first `list_busy`
  call named a real team. None of their first rounds passed: 306 saved a wrong or
  no meeting and 197 ran out of replies. Among stage-1 natural rollouts with a valid
  first call, 10 of 251 passed.

## Cost

One g6e.xlarge (NVIDIA L40S) host ran for 2.6 hours, $4.82 conservatively. It
re-verified stage 1's natural rollouts, ran the coached collection in 2.4 hours and
stopped as declared. The instance is terminated, its volume deleted and its
security group retired. With stage 1, A3 has spent $19.04 of its $100 ceiling,
plus stage 0's $2.15.

## What this means

Scheduling as this interface defines it does not get started from the model's own
experience. The model would have to discover the exact team names from
uninformative errors. It would also have to find the earliest common free time
without writing any reasoning, since a reply that calls tools may contain nothing
else. Coaching text changed neither its one-call style nor its outcomes. Another
round like this one would very likely stop the same way, so the next step is a
design decision rather than another collection.
