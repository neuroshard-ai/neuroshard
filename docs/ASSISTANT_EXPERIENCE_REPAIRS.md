# Verified date-base repairs before the third attempt

Declared on September 30, 2026, after the [development diagnostic](ASSISTANT_EXPERIENCE_DIAGNOSTIC_RESULTS.md),
before any repair rollout. Authorized by the project owner. The
[contract](../config/experiments/assistant-experience-repairs.json) pins every
rule below.

## Why

In every failed latest development case, the compositional small module read
approved revision 1, computed the total and applied the added day correctly. But
it started the review-interval shift from the start date plus the interval,
applying the interval twice. The update control started from the start date.
Round 4 fixed a similar, narrow mistake (reading the wrong revision) with
goal-guided repairs. This study applies that method to the date base.

## Method

- **Sampling.** The pinned compositional small module samples 4 episodes at
  temperature 0.8 on every training case of the date, latest and scope families
  in all five training splits, 528 cases. No development or confirmation case.
- **Locating the error.** In a failed rollout, find the first `shift_date` call
  whose start date its round cannot justify. Justified starts are the start date
  of one of the round's goal sources, a date an earlier `shift_date` returned, or
  an earlier round's goal due date. Goals locate the error; they never enter a
  prompt. Run on the diagnostic's development transcripts, this rule located all
  four wrong-base failures and flagged none of the 35 successful episodes. That
  check used opened data only to validate the rule; the study applies it only to
  training cases.
- **Repair.** Replace that start date with the round's goal-source start date,
  replay every earlier generation exactly, and let the same module continue by
  sampling, twice per failure.
- **Verification.** Keep only repaired rollouts the frozen scorer passes in every
  round. Each yields a preference pair, with the repaired message chosen over
  the original (at most 4 per case), and a verified trajectory (at most 2 per
  case).
- **Continuation.** Both compositional arms continue from their pinned
  checkpoints through one preference phase with round 4's settings, on identical
  data: the 3302-sequence pool plus the repaired trajectories, the replay items,
  and the 54 earlier pairs plus the repair pairs.
- **Evaluation.** Parent, update and small module on the 64 integration cases
  (one greedy and two sampled episodes) and the 24 opened development cases
  (greedy), on the GPU.

## Decision rule

- Scores are reported for both arms as before.
- The third attempt is declared with the small module only if it lost no parent
  success on either split. If no verified repair is found, the study fails.
- Otherwise there is no third attempt, and the result is reported.

The third attempt needs its own declaration on the frozen `confirmation3` split
under the second confirmation's gate. The diagnostic already measured this
serving path's latency within that gate's limits.

## Limits

Already-opened data decides only whether the third attempt runs. The study
earns no A2 credit. One GPU host with an eight-hour expiry and a $21
worst-case allowance; about $6 expected. One attempt.
