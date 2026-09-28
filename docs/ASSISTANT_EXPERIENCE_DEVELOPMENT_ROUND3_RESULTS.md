# Verified-experience development result, round 3: passed

**Finished September 28, 2026 at 09:59 UTC; development gate passed.** Execution
freeze `fdc41387411d43ee7bdd726b8da836f452377ba4` passed exact-commit CI. The
[round-3 systems](ASSISTANT_EXPERIENCE_ROUND3_RESULTS.md) ran the 24 development
workflows on the canonical parent's CPU runtime with the declared
[prefix-cache serving](ASSISTANT_EXPERIENCE_LEARNING.md#serving-runtime-amendment-prefix-cache),
one fresh worker per arm and routing timed per episode.

| Measurement | Parent | Update system | Addition system | Required of addition |
| --- | ---: | ---: | ---: | ---: |
| Complete workflows | 9/24 | 19/24 | **18/24** | At least 18/24 ✓ |
| Net versus parent | — | +10 | **+9** | At least +4 ✓ |
| Lost parent successes | — | 0 | **0** | 0 ✓ |
| Net versus update | — | — | **−1** | At least −1 ✓ |
| Episode p95, routing included | 177.1 s | 97.5 s | **96.5 s** | At most 180 s ✓ |
| p95 ratio versus update | — | — | 0.99 | At most 2.0 ✓ |

The added module, 1,048,576 trained parameters stored in 4.2 MB, gains nine of the
parent's failures and loses none. It is one workflow behind an update of 62.9
million existing parameters trained on the same data, inside the declared margin.
Both gates select the trained arm for every episode. With the arm forced on for
every original anchor, each arm answers 18 of 19 protected anchors
(`granite-instruction-counts` is lost); routed systems serve anchors with the
parent.

## What this does and does not establish

The development split has been used for four evaluations of successive systems
(round 1, round 2 under two serving runtimes, round 3), so this pass is weaker
evidence than a first look would be. The declared response is the
[confirmation](../config/experiments/assistant-experience-confirmation-execution.json)
split: 96 sealed episodes with different correction conjunctions, evaluated once
for the parent, update and addition systems, with the family-bootstrapped gates.
Only that result can establish A2 for this capability. A1 remains open, and no
native model, ledger or checklist state changes.

## Evidence and cost

The [raw result](../config/experiments/assistant-experience-development-round3-result.json)
and [report](../config/experiments/assistant-experience-development-round3-report.json)
record every episode, reused-prefix count, selection time, forced anchor, receipt
and CI binding. The CPU host cost **$1.06**; AWS verification found the instance
terminated, no volume and no security group.
