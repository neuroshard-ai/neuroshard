# A3 stage 1, round 2: coaching that fits the budget

Declared on October 4, 2026, after the [stage-1 results](ASSISTANT_REPEATED_GROWTH_STAGE1_RESULTS.md)
and before any round-2 rollout. Contract:
[assistant-growth-round2.json](../config/experiments/assistant-growth-round2.json).
Stage 1's outcome rule allows a second round within the remaining budget. Stage 1
ran no development or confirmation case.

## Why

Stage 1's units learned little scheduling because almost no verified scheduling
experience reached training. The accepted version made one tool call per reply and
ran out of replies, and the stage-1 card made that worse by adding steps. When it
did save a meeting, the start was usually too early, because the conflict check
failed. The near-policy filter then removed every coached partial success: its
ceiling came from drafting episodes the accepted version already does well.

## What changes: only the collection

- **No new natural sampling.** Stage 1's 2,304 natural rollouts came from the same
  sampler under the same policy. They are uploaded, checked against the digest of
  the rows the stage-1 report records, and re-verified. Their verified rounds are
  experience, and they decide which cases are coached: those without a complete
  natural success.
- **A budget-aware card.** It tells the model that each user message allows six
  replies and ten calls, so it should put two independent calls in one reply and
  make no call it does not need. It gives the conflict rule exactly, and one worked
  example with invented teams. It names no case and no answer, and is stored
  nowhere in the training conversations.
- **No near-policy filter.** Six complete natural successes are too few to
  calibrate it. Verified trajectories are kept up to four per case, preferring more
  verified rounds and fewer model calls, as before.
- **A stop rule.** If fewer than 32 of the 288 training cases have a complete
  verified trajectory, training and integration do not run and the round ends.

## What stays

The calendar interface and its limits, routing, the three versions, the training
mixture, schedule, steps and learning rates, the integration sets, runs, seeds and
selector recipe, and every development and confirmation gate are stage 1's. U2 and
L2 again start from the accepted version; round 1's units are not reused.

## Budget

Stage 1's GPU host cost $14.22. Round 2 allows at most $17 for one GPU host
(six-hour expiry), $18 for development on three CPU hosts and $47 for confirmation
on five. In total that is $96.22, within the unchanged $100 ceiling.
