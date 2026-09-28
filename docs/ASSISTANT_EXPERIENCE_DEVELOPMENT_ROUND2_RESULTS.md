# Verified-experience development result, round 2

**Finished September 28, 2026 at 03:39 UTC; development gate failed on latency
only.** Execution freeze `aed806414d360fa362d01da0669bed30c389776d` passed
exact-commit CI. The [round-2 systems](ASSISTANT_EXPERIENCE_ROUND2_RESULTS.md) ran
the 24 development workflows on the canonical parent's CPU runtime, one fresh
worker per arm, with routing timed per episode. Confirmation remains sealed.

| Measurement | Parent | Update system | Addition system | Required of addition |
| --- | ---: | ---: | ---: | ---: |
| Complete workflows | 9/24 | 17/24 | **18/24** | At least 18/24 ✓ |
| Net versus parent | — | +8 | **+9** | At least +4 ✓ |
| Lost parent successes | — | 0 | **0** | 0 ✓ |
| Net versus update | — | — | **+1** | At least −1 ✓ |
| Episode p95, routing included | 177.1 s | 190.0 s | **192.0 s** | At most 180 s ✗ |
| p95 ratio versus update | — | — | 1.01 | At most 2.0 ✓ |

The addition system gains nine parent failures and loses none, and it now edges
the update control while training 60 times fewer parameters (4.2 MB of trained
tensors against 251.7 MB). Both gates selected the trained arm for every episode.
With the arm forced on for every original anchor, each arm answers 18 of the 19
protected anchors, losing `granite-instruction-counts`, as in round 1; the routed
systems serve anchors with the parent.

## Latency

Without routing the addition's p95 would still be 189.8 s: five compound workflows
exceed 180 s on generation time alone. Routing adds 2.8 s per episode on average.
Compound workflows use twelve generations, six per round, and the parent already
needs 160–181 s for them on this CPU runtime. The 180 s limit was fixed before the
canonical parent measured 177.1 s. No quality change can meet it without fewer or
faster generations: combining independent tool calls into one response, reusing the
cached conversation prefix between generations, or a faster serving host.

## Decision

Keep this result failed; the gate is not relaxed after the fact. Development shows
the learning effect clearly: 9/24 to 18/24 with every parent success preserved,
from verified self-generated experience plus 37 verified decisions. Confirmation
opens only after a development pass under a declared latency path.

## Evidence and cost

The [raw result](../config/experiments/assistant-experience-development-round2-result.json)
and [report](../config/experiments/assistant-experience-development-round2-report.json)
record every episode, selection time, forced anchor, receipt and CI binding. The CPU
host cost **$1.87**; AWS verification found the instance terminated, no volume and
no security group. The comparison has spent **$12.06** of compute so far across all
GPU and CPU attempts, within the $100 ceiling.
