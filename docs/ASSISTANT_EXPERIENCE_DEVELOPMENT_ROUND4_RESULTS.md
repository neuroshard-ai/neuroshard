# Round-4 development results

**The development gate passed, and so did the A1 served-system check.** The fresh
192-episode confirmation therefore opens once, as declared.

Declaration: [second attempt](ASSISTANT_EXPERIENCE_LEARNING.md#second-attempt-goal-guided-repairs-and-fresh-confirmation).
Evidence: [result](../config/experiments/assistant-experience-development-round4-result.json) and
[report](../config/experiments/assistant-experience-development-round4-report.json), commit `f299e0f`.

## Development gate (24 episodes, prefix-cache serving, per-episode routing)

| System | Correct | p95 with selection |
| --- | --- | --- |
| Parent (canonical) | 9/24 | 177.1 s |
| Update | 19/24 | 93.2 s |
| Addition | 18/24 | 92.7 s |

The addition gained 9 parent failures and lost no parent success. It is one below
the update, exactly at the allowed −1: the update solves one follow-up
(`workflow-5b541f25dc67`) that the addition misses. Every check passed. Both
routed systems chose their arm for all 24 episodes.

When forced onto the anchors, both arms lose one protected anchor
(`granite-instruction-counts`). The served system routes anchors to the parent,
so that loss is never served.

## A1 served-system check

A1 judges the version that would actually be served: the pinned parent, plus the
verified addition under prefix-cache serving and selection.

| Check | Outcome |
| --- | --- |
| Primitive workflows | 7/8 (copy 2, date 1, sum 2, difference 2); needs ≥6 with every family |
| Anchors | Canonical anchor gate holds, 19/24, with every anchor routed to the parent |
| p95 latency | 92.7 s, including selection; needs ≤180 s |
| Fresh-process replay | Both declared episodes replayed exactly in a new process |

The bare parent still fails primitive qualification (2/8). A1's usable-foundation
requirement is met on development by the served system. It stays open until the
confirmation and the remaining A1 items (published shard estimates now exist from
[A4](GRANITE_SHARD_EXECUTION_RESULTS.md)) are recorded.

## Resources

One r7i.4xlarge host cost $1.04, and everything is retired.
