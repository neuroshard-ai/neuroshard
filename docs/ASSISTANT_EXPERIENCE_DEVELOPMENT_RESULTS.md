# Verified-experience development result, round 1

**Finished September 27, 2026 at 22:50 UTC; development gate failed.** Execution
freeze `6eed482130201bd3d247d2575d4976685a86e1d7` passed exact-commit CI. Both
routed systems from the [GPU result](ASSISTANT_EXPERIENCE_GPU_RESULTS.md) ran the
24 development workflows on the canonical parent's CPU runtime, one fresh worker
per arm. The parent control is the [canonical baseline](ASSISTANT_WORKFLOW_CANONICAL_RESULTS.md).
Confirmation remains sealed.

| Measurement | Parent | Update system | Addition system | Required of addition |
| --- | ---: | ---: | ---: | ---: |
| Complete workflows | 9/24 | 15/24 | 14/24 | At least 18/24 |
| Net versus parent | — | +6 | +5 | At least +4 |
| Net versus update | — | — | −1 | At least −1 |
| Lost parent successes | — | 1 | 1 | 0 |
| Episode p95, routing included | 177.1 s | 184.8 s | 189.2 s | At most 180 s and 2× update |

The addition gained six parent failures and lost one protected parent success; it
matches the update control within one episode while training 60 times fewer
parameters. It fails the absolute total, the protected-success rule and the
latency limit. Both gates selected the trained arm for all 24 episodes. With the
arm forced on for every original anchor, each arm answers 18 of the 19 protected
anchors, losing `granite-instruction-counts`; the routed systems serve anchors
with the parent, so this is a forgetting diagnostic, not a served loss.

## What the episodes show

- **Gains** come from the call budget (both difference workflows and two scope
  follow-ups now finish within six generations) and from some version choices.
- **Version choice by listing position** remains the largest failure: most
  remaining copy, date, recipient, reschedule and scope failures cite approved
  revision 1 when revision 2 was required.
- **The lost protected episode** (`workflow-c6b1500447f3`) is a follow-up date
  error: the arm shifted from an already shifted date, adding the review interval
  twice. The parent made the same error on a sibling episode in its baseline.
- **Latency is structural.** Compound workflows use twelve generations, six per
  round, and the parent already takes 160–181 s on them. Routing adds about 3 s
  per episode and the arms 5–10 s of generation. Meeting 180 s requires fewer
  generations, for example two independent tool calls per response, which the
  parent almost never produces and the collected experience therefore lacks.

## Decision

Keep this result failed. The declared [decision-preference round](ASSISTANT_EXPERIENCE_LEARNING.md#conditional-second-round-verified-decision-preferences)
follows: both arms continue on 37 verified version-choice pairs from the pinned
rollouts. It does not address latency; that needs its own declared change. No gate
is relaxed.

## Evidence and cost

The [raw result](../config/experiments/assistant-experience-development-result.json)
and [report](../config/experiments/assistant-experience-development-report.json)
record every episode, selection time, forced anchor, receipt and CI binding. The
frozen scorer rescored every episode inside the run. The CPU host cost **$1.82**;
AWS verification found the instance terminated, no volume and no security group.
