# Round-2 development under prefix-cache serving

**Finished September 28, 2026 at 05:28 UTC; development gate failed.** Execution
freeze `2df1e3a9baf425fbf60484d9b05e0ff0a82df078` passed exact-commit CI. The same
round-2 arms and gates as the [recompute run](ASSISTANT_EXPERIENCE_DEVELOPMENT_ROUND2_RESULTS.md)
were served with the declared [prefix cache](ASSISTANT_EXPERIENCE_LEARNING.md#serving-runtime-amendment-prefix-cache).
Confirmation remains sealed.

| Measurement | Parent | Update system | Addition system | Required of addition |
| --- | ---: | ---: | ---: | ---: |
| Complete workflows | 9/24 | 18/24 | 17/24 | At least 18/24 ✗ |
| Net versus parent | — | +9 | +8 | At least +4 ✓ |
| Lost parent successes | — | 0 | 1 | 0 ✗ |
| Net versus update | — | — | −1 | At least −1 ✓ |
| Episode p95, routing included | 177.1 s | 109.0 s | **102.7 s** | At most 180 s ✓ |

The cache reused 87.9% of prompt tokens and cut the mean episode from 119 s to
68 s, so latency now passes with a wide margin. The addition's outcome changed on
exactly one episode relative to recompute serving: `workflow-c6b1500447f3`, the
follow-up where the model shifts from an already shifted date and adds the review
interval twice. It failed in round 1, passed under round-2 recompute and fails
here. Cached and recomputed BF16 prefill differ slightly, and this episode sits on
the decision boundary. The update system gained one episode under the cache.

## Interpretation

The capability is real and large: 9/24 to 17–18/24 on development, 34/64 to 55/64
on integration, with no parent success lost under recompute. It is not robust
enough to clear the declared gate independent of serving numerics. The remaining
brittle behavior is follow-up date arithmetic, which neither the collected
experience nor the version-choice pairs target. Keep this result failed; the
development split has now been used three times, so confirmation remains the
decisive fresh test once a complete development pass exists.

## Evidence and cost

The [raw result](../config/experiments/assistant-experience-development-cached-result.json)
and [report](../config/experiments/assistant-experience-development-cached-report.json)
record every episode, reused-prefix count, selection time, receipt and CI binding.
The CPU host cost **$1.12**; AWS verification found the instance terminated, no
volume and no security group. Total compute for the comparison is **$13.18**.
