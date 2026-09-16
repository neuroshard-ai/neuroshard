# Shared-gradient method study

This research experiment asks whether a shared AdamW optimizer with compressed
gradients can retain useful learning while reducing distributed training cost.
It changes no released network, issuance, serving profile or application.

The previous local-window method changed both local optimization histories and
loss normalization. Its reduced traffic accompanied failed retention. Here the
single-GPU, two-GPU dense and two-GPU PowerSGD arms use the same seed, document
schedule, global target normalization, clipping and optimizer configuration.
PowerSGD changes the gradient communicated to that common optimizer. Its error
feedback remains local to each rank; this is not a proof against malicious work.

The [plan](../config/experiments/learning-method-study.json) prescribes 128
updates, 32 documents per global batch, 3,072 generated tasks and 1,024 public
conversation-replay documents. All 1.7B parameters train. The larger batch allows
more computation between reductions; comparisons therefore rerun every control.
PowerSGD uses rank 32, eight dense warmup updates, warm starts and CPU-offloaded
FP32 error feedback. The [upstream paper](https://arxiv.org/abs/1905.13727) and
[PyTorch hook](https://docs.pytorch.org/docs/2.9/ddp_comm_hooks.html) motivate this
candidate; neither guarantees its quality or speed on this workload.

There are 1,024 new generated final-test cases, with the same four public task
families. The 128 retention probes have already been exposed. This is one
screening seed, not broad assistant evaluation or repeated continual learning.
Preparation commits input hashes and numerical sources. Every final model is
fixed at step 128; all candidates must be committed before any test scoring.
Unavailable, failed or rejected arms must remain visible.

Report task accuracy by family and paired change against the seed and both
controls. Report retention change with the plan's approximate normal bound.
The quality screen requires at least ten percentage points of task gain, a
retention upper bound at most +0.02 nats, and accuracy noninferiority to both
controls within three percentage points using the declared normal multiplier.
Report speed and GPU cost separately: the proposed speed screen is at least
1.10 times the single-GPU active throughput. Faster execution with more total
GPU time is not a cost reduction. Include checkpoint time and whole-host wire
traffic separately; upstream logical compression excludes warmup and overhead.
No independent audit cost or permissionless verification claim is established.

AWS could not place a pair together. Two g6e.2xlarge hosts, each with one L40S,
were obtained individually in us-east-1a and us-east-1b. Cross-zone traffic is
billable and its cost must be retained. A six-hour stop deadline and a $100
allocation bound apply. All three arms run sequentially without overlapping
training jobs or artifact transfers. The controls share these hosts.

The final checkpoint stores model weights, not a resumable compression state.
Before any operational adoption, the selected method would also need a complete
optimizer/error/projection/RNG checkpoint, exact recovery testing, repeated
learning cohorts and a measured verification cost. The method study should
determine whether that additional work is justified.
