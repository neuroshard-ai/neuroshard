# Compositional practice before the third attempt

Declared on September 30, 2026, after the [growth study](ASSISTANT_EXPERIENCE_GROWTH_RESULTS.md)
found no qualifying candidate, and before any compositional rollout. Authorized
by the project owner. The [contract](../config/experiments/assistant-experience-compose.json)
pins every rule below.

## Why

In both studies, the small module lost parent successes only on development
cases whose correction adds an instruction it never saw combined that way in
training, such as "Also move the resulting due date one calendar day later."
The update control solved those cases. The small module seems to have learned
the training corrections' phrasing rather than to apply every instruction.

## Practice cases

Two new training splits, `compose1` and `compose2`, hold 240 cases each: 40
per family for copy, date, sum, difference, latest and scope. They are
[frozen](../config/experiments/assistant-workflow-data-compose.json) and share
no case, project or document with any other split. Each case adds one extra
operation to its instructions:

- **Single-turn families** (copy, date, sum, difference) add, in rotation,
  "push the due date 3–5 calendar days later", "add 3–5 more units to the
  total", or "address it to the finance team instead".
- **latest and scope** add "add 3–5 more units to the total" to their
  correction.

The development and confirmation splits hold out particular pairs of family
and extra-operation kind:

| Family | Development adds | Confirmation adds | Practice adds |
|---|---|---|---|
| latest, scope | a date shift | a recipient change | a total change |
| recipient | a total change | a date shift | nothing (left out) |
| reschedule | a recipient change | a total change | nothing (left out) |
| copy, date, sum, difference | nothing | nothing | any of the three |

No held-out pair appears in practice, and the practice phrasing, amounts and
recipient differ from the held-out ones. recipient and reschedule are left out
because every extra kind is held out for them somewhere.

## Collection and study

- **Collection.** The round-1 recipe unchanged, on two GPU hosts in parallel, one
  per split: 8 parent samples per case, coached retries, each collection's
  near-policy ceiling, and at most 4 distinct trajectories per case. The files
  are then pinned.
- **Pool.** The growth study pool (2114 sequences, 256 replay items, 54 pairs)
  plus both compositional collections, every trajectory re-verified from its
  pinned rollout.
- **Arms.** The update control and the small module, each trained from the
  parent with one-pass first-phase steps and round 4's preference phase, as in
  the growth study. No committee.
- **Evaluation.** Parent, update and small module on the 64 integration cases
  (one greedy and two sampled episodes) and the 24 opened development cases
  (greedy), on the GPU.

## Decision rule

- Score = greedy integration successes + greedy development successes − 2 × the
  parent successes lost on those greedy episodes, reported for both arms.
- The third attempt is declared with the small module only if it lost no parent
  success on either split. Otherwise there is no third attempt, and the result
  is reported.

The third attempt needs its own declaration on the frozen `confirmation3` split
under the second confirmation's gate. Its serving must meet that gate's latency
limits on the canonical CPU runtime, measured on opened development cases
before the split opens.

## Limits

Already-opened data decides only whether the third attempt runs. The study
earns no A2 credit.

Resources: two collection hosts and one study host, each with an eight-hour
expiry and a $21 worst-case allowance, under the
[resource contract](../config/experiments/assistant-experience-compose-resources.json).
About $15 is expected in total. One attempt.

## Collections

Both collections completed on A10G hosts after one capacity refusal each
([report](../config/experiments/assistant-experience-compose-collection-report.json)),
and both hosts were retired with nothing remaining ($9.04).

| Split | Rollouts | Verified trajectories | Complete | Coached | Cases without experience |
|---|---|---|---|---|---|
| `compose1` | 2664 | 596 | 545 | 18 | 78 of 240 |
| `compose2` | 2640 | 592 | 547 | 14 | 81 of 240 |

Every trajectory was rebuilt from its pinned rollout through the frozen scorer
before the files were pinned. The pool now holds 3302 sequences, 36% of them
compositional practice.
