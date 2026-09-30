# Growing verified experience before the third attempt

Declared on September 29, 2026, after the [methodology study](ASSISTANT_EXPERIENCE_STUDY_RESULTS.md)
carried no method forward, and before any growth rollout. Authorized by the
project owner. The [contract](../config/experiments/assistant-experience-growth.json)
pins every rule below.

## Question

The study's committee was the only system that lost no parent success, but each
member had learned from only a third of the experience and was weak alone
(47–56 of 64 integration cases, against 61 for one small module on all of it).
With about three times the verified experience, each member gets about as much
as the small module had. Does a consensus of such members then score at least
as high as one small module trained on the whole pool, while still losing no
parent success?

## Collection

- **Fresh training cases.** Two new training splits, `train2` and `train3`, with
  256 cases each (32 per family), are [frozen](../config/experiments/assistant-workflow-data-growth.json).
  They use the training grammar with new values and share no case, project or
  document with any other split. No evaluation goal is among them.
- **The round-1 recipe, unchanged.** 8 parent samples per case at temperature
  0.8 and top-p 0.95, coached retries for cases without a verified trajectory,
  each collection's near-policy ceiling, and at most 4 distinct trajectories per
  case.
- **Two GPU hosts in parallel**, one per split. They collect and verify only; no
  training.

The collected files are then pinned, and the study re-verifies every trajectory
from its recorded rollout before training.

## Study

- **Pool.** Every verified sequence and preference of the methodology study (740
  sequences, 256 replay items, 54 pairs) plus both growth collections.
- **Arms.** The update control and one small module on the whole pool, and three
  committee members, each on the cases of one fixed third
  (`int(sha256(case_id)[:8], 16) mod 3`, over all three training splits).
- **One pass.** First-phase steps are `ceil(experience sequences / 6)`, so every
  arm draws each of its experience sequences once. Otherwise the whole-pool
  arms would see only about a third of the pool in round 1's 128 steps. All
  other settings are round 1's, and the preference phase keeps round 4's.
- **Committee.** The same vote as in the study: the three members and the parent
  each propose every message, the most common action wins, and a tie that
  includes the parent goes to the parent.
- **Evaluation.** Parent, every arm and the committee on the 64 integration
  cases (one greedy and two sampled episodes) and the 24 opened development
  cases (greedy), on the GPU.

## Decision rule

- Score = greedy integration successes + greedy development successes − 2 × the
  parent successes lost on those greedy episodes.
- The candidate is the committee if it scores at least as high as the small
  module; otherwise the small module.
- The third attempt is declared with the candidate only if it lost no parent
  success on either split. Otherwise there is no third attempt, and the result
  is reported.

The third attempt needs its own declaration on the frozen `confirmation3` split
under the second confirmation's gate. Its serving must meet that gate's latency
limits on the canonical CPU runtime, measured on opened development cases
before the split opens.

## Limits

Already-opened data decides only whether, and with which method, the third
attempt runs. The study earns no A2 credit.

Resources: two collection hosts and one study host, each with an eight-hour
expiry and a $21 worst-case allowance, under the
[resource contract](../config/experiments/assistant-experience-growth-resources.json).
About $15 is expected in total. One attempt.

## First collection attempt: stopped at verification

Both collection hosts finished their 2,048 uncoached rollouts, then stopped at
the first verification step
([report](../config/experiments/assistant-experience-growth-collection-report.json)).
The experience module accepted trajectories only from the split named `train`,
and the declaration's tests had substituted `train` cases for the growth split.
Rollouts are written only after verification, so nothing was kept. Three
earlier `train3` launches found no GPU capacity and started no instance. All
hosts were retired ($7.01).

## Amendment for the second collection attempt

Declared on September 30, 2026, after the first attempt and before the second.
The experience module now accepts the two growth splits as training goals. A
test runs real `train2` cases through collection and re-verification, and it
fails without this change. Everything else is unchanged. One attempt.

## Collections

The second attempt completed on both hosts
([report](../config/experiments/assistant-experience-growth-collection2-report.json)),
and both were retired with nothing remaining ($9.88).

| Split | GPU | Rollouts | Verified trajectories | Coached | Cases without experience |
|---|---|---|---|---|---|
| `train2` | A10G | 2720 | 702 | 36 | 68 of 256 |
| `train3` | L40S | 2728 | 672 | 45 | 66 of 256 |

Every trajectory was rebuilt from its pinned rollout through the frozen scorer
before the files were pinned. With round 1's 722 trajectories and the 18 repaired
ones, the pool holds 2114 sequences, 2.9 times the study's 740.
