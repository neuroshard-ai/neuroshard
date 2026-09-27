# Verified-experience collection, training and integration result

**Finished September 27, 2026 at 20:41 UTC; training and gate fitting completed.**
This executes the collection, training and integration stages of the
[verified-experience contract](ASSISTANT_EXPERIENCE_LEARNING.md). Development and
confirmation goals were not opened; the [development gate](ASSISTANT_EXPERIENCE_LEARNING.md#gates)
runs separately on the CPU parent runtime.

## Attempts

| Attempt | Commit | Outcome | Compute |
| --- | --- | --- | ---: |
| a | `1987750` | g6e.xlarge not offered in the controller's zone; nothing started | $0.00 |
| b | `5877b4d` | No g6e.xlarge capacity in any offering zone; nothing started | $0.00 |
| c | `fc4e192` | g5.2xlarge: collection, replay and update training completed; the addition arm failed | $4.42 |
| d | `621125d` | g5.2xlarge: re-verified the pinned collection, retrained both arms, fitted gates | $1.87 |

Attempt c's trainer drew adapter initial values with a CPU generator directly into
CUDA tensors. PyTorch rejected it with `Expected a 'cuda' device type for generator
but found 'cpu'`, as predicted before the phase ran. CPU-only tests could not
reach that path. The corrected trainer draws on the CPU and then moves the tensor,
so initialization is identical on every device. Rather than repeat 3.4 hours of
rollouts, attempt d uploaded attempt c's five collection files by digest and
rebuilt every selected trajectory from its recorded rollout through the frozen
scorer before training. All attempts total **$6.31** of conservative compute.

## Collection

The parent produced 2,048 sampled training episodes; 74 of 256 cases had no fully
successful one and received 592 coached retries. Of 1,355 verified successes, 722
complete conversations were selected (at most four per case; 48 coached), covering
200 of 256 training cases. Their mean assistant-token negative log-likelihood under
the parent has a median of 0.036 nats and a near-policy ceiling of 0.089. Coaching
unlocked the difference family: 16 cases have only coached trajectories, because
the parent alone always ran out of generations.

Version choice by listing position is the weakest point. When the older approved
revision is listed first, natural rollouts succeeded 284 of 920 times and coached
retries on the hardest of those cases 44 of 488 times. The selected data still
contains 197 trajectories that choose the latest revision against the listing order.
Collection used 19,381 model calls, 24.3 million input and 0.81 million output tokens.

## Training and integration

Both arms trained 128 steps on identical sequences and schedule (final loss:
update 0.0241, addition 0.0245; 278 seconds each on one A10G). The addition's
trainable tensors occupy 4.2 MB; the update's 251.7 MB.

On the 64 integration episodes, which are separate from training, development and
confirmation, each model ran once greedily and four times sampled:

| Integration split | Parent | Update | Addition |
| --- | ---: | ---: | ---: |
| Greedy successes | 34/64 | 42/64 | 42/64 |
| Sampled successes | 132/256 | 166/256 | 168/256 |
| Difference family, greedy | 0/8 | 5/8 | 6/8 |

Most of the gain is the difference family, the call-budget behavior taught through
coaching. Copy, recipient and scope did not improve, consistent with the remaining
version-choice error. The addition matches the update at one-sixtieth of its
trained parameters. Gates: the update is better on 17 episodes and worse on 2
(45 ties); the addition better on 16 and worse on 1 (47 ties). Both fitted the
declared logistic gate rather than a constant rule.

## Evidence

The [report](../config/experiments/assistant-experience-gpu-report.json) records
every attempt, receipt, collection digest, training manifest and gate. Integration
episodes are gate-fitting data, not an admission result; this result earns no
checklist credit and does not change A1. AWS verification after attempt d found no
remaining instance, volume or temporary security group.
