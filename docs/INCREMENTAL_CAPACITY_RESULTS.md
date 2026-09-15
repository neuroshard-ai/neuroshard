# Incremental capacity results

All four short-run candidates failed the declared useful-learning gate. Adding two trainable blocks to the established model did not reliably teach the new facts while preserving every previously correct answer. No candidate opened the independent final set, issued tokens or changed serving.

| Layout | Peak learning rate | New facts correct | Prior skill answers correct | Previously correct answers lost |
| --- | ---: | ---: | ---: | ---: |
| append | 5e-05 | 13/256 | 152/192 | 1 |
| tail-control | 5e-05 | 15/256 | 152/192 | 0 |
| append | 0.0002 | 11/256 | 150/192 | 3 |
| tail-control | 0.0002 | 33/256 | 153/192 | 0 |

The unchanged parent answered 0/256 new-fact questions and 151/192 prior-skill questions correctly. Passing required at least 75% new-fact accuracy, a positive gain beyond the frozen margin, no individual loss of a previously correct skill answer, and bounded conversation loss. A net skill-score increase cannot compensate for forgetting a previously correct answer.

Both layouts trained 134 million parameters for 128 updates with identical data and two predeclared learning rates. The growing layout appended two blocks to the 24-layer parent and froze the established model. The fixed-size control updated its existing final two blocks. Each fact had twelve training exposures. The training response loss fell substantially, but that did not produce reliable answers to the differently worded questions.

## Recovery and ownership

After all four candidates and their optimizer states were backed up and verified, the final shard owner was terminated and replaced by a new EC2 instance. Restoring the lower-rate growing candidate at update 64 and repeating its last 64 updates reproduced the entire terminal checkpoint, Adam state, development answers and losses exactly. The replacement recovered its owned tensors from storage; no worker held the whole model.

AWS briefly counted the terminated instance against the GPU quota. Retrying the identical idempotent request after that accounting delay succeeded. The interruption and reporting amendment remain in the evidence.

This establishes recovery of this frozen computation across an actual worker replacement. It does not establish useful capacity growth, heterogeneous hardware agreement or independent operators.

## Accounting and next intervention

The comparison performed 512 candidate updates and 64 recovery updates. A separately frozen feature diagnostic reused the allocation only after the full comparison, recovery and committed selection had completed. Its optimizer work and evidence are recorded separately; its resources are included in the allocation totals.

All five physical instances used over the allocation lifetime, their temporary volumes and the experiment security group were retired after verified preservation. The two existing network hosts were protected. All workers belonged to one operator.

The [machine-readable results](../config/experiments/incremental-capacity-results.json) include every candidate, the exact selection, recovery, resource accounting and immutable evidence receipts. The [frozen plan](../config/experiments/incremental-capacity.json) remains unchanged. A separate longer-run experiment tests repeated exposure while keeping the answer and retention gates fixed; it cannot turn this failed comparison into a pass.
