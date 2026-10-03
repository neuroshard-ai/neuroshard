# A4 closure review: actual sharding and value from additional peers

October 2, 2026. This review checks [A4](../TODO_ASSISTANT.md) against published
results only. It runs nothing new and changes no result.

## The criterion

> No execution worker holds the complete backbone. Measure forward/backward/generation
> agreement, per-owner memory and traffic, outage recovery and a benefit from extra
> machines (pooled memory, throughput, training capacity or availability). Include
> placement, communication and failed work; throughput is not single-request latency.

## Clause by clause

Every result below ran the pinned Granite 4.1 3B on three owner hosts with layer
boundaries 0/12/26/40.

| Clause | Evidence | Met |
| --- | --- | --- |
| No worker holds the backbone | Owners fetched only their own tensors, each checked against its inventory digest: 2.40, 2.20 and 2.20 of 6.81 GB for [generation](GRANITE_SHARD_EXECUTION_RESULTS.md); 2.24, 2.05 and 2.05 GB for [training](GRANITE_SHARD_TRAINING_RESULTS.md). The training reference host held the complete model only as the comparison. | Yes |
| Forward and generation agreement | All 230 canonical generations reproduced token for token ([execution](GRANITE_SHARD_EXECUTION_RESULTS.md)). The complete learned assistant (parent, round-4 addition on owner 2, gate, prefix cache) reproduced all 24 single-host development episodes token for token ([serving](GRANITE_SHARD_SERVING_RESULTS.md)). | Yes |
| Backward agreement | Training the addition arm across owners matched the single-host trainer bit for bit: six losses, twelve preference margins and all 16 LoRA tensors ([training](GRANITE_SHARD_TRAINING_RESULTS.md)). | Yes |
| Per-owner memory | Generation peaks 6.22, 6.05 and 6.09 GB against 11.34 GB on one host; serving peaks 5.22, 4.82 and 4.97 GB; training peaks 5.30, 4.93 and 17.05 GB against 22.43 GB for the reference. | Yes |
| Traffic and communication | Generation: owners 0 and 1 each sent 1.41 GB of hidden states, owner 2 sent 47 MB, and decoding costs 5 KiB per token per hop. Serving with the prefix cache: 304, 304 and 45 MB. Training: 0.95, 0.50 and 0.50 GB. Final hidden states return to owner 0 for the tied head. | Yes |
| Outage recovery | A generation owner lost at its 40th step: the relaunched group replayed the 39 committed tokens and finished the canonical 95. The training arm's owner lost at step 3: the group resumed from its checkpoint and finished with the reference's exact tensors ([second attempt](GRANITE_SHARD_TRAINING_RESULTS.md)). | Yes |
| Benefit from extra machines | Pooled memory: every owner stays below the single-host peak. Throughput: three episodes in flight ran 2.25× faster than one at a time on the same owners and 2.20× faster than the single host, every token identical ([throughput](GRANITE_SHARD_THROUGHPUT_RESULTS.md)). | Yes |
| Throughput is not single-request latency | Reported separately: one request across owners is 7.6% slower than on one host (2,946 s against 2,739 s for the 230 generations). The gain comes from concurrency. | Yes |
| Placement | Owner 0 holds the embedding, layers 0–11 and the tied head; owner 1 layers 12–25; owner 2 layers 26–39. Each owner ran on its own r7i.4xlarge allocation at eight threads. All owners ran in one subnet, in `us-east-1f`. | Yes, for one zone |
| Failed work | Recorded with costs: the first throughput attempt (0.97×, $3.09), the first training attempt (failed recovery, $1.89) and its [determinism diagnosis](GRANITE_SHARD_DETERMINISM.md) ($0.92 and $1.76), and a controller that left one owner running for an extra hour. | Yes |

Across all eight executions, A4 cost $18.48, failed attempts included.

## Verdict

**Every clause is met by published evidence, so A4 is complete.** The assistant
runs, trains and recovers across owner machines that never hold the backbone,
with exactly the single-host result. Extra machines add pooled memory, recovery
and 2.25× throughput.

## What A4 does not establish

- **Independent operators.** We administered every host. Independent operation is
  A5.
- **Wide-area links.** All owners shared one subnet in one availability zone.
  Cross-zone and internet links are unmeasured; per-token traffic is small (5 KiB
  per hop), but latency across the internet would add to every decode step.
- **Scale.** Three owners at one partition. More owners or other boundaries are
  not measured.
- **Execution class.** Exactness holds within the pinned runtime and CPU
  instruction class.
- **Auditing the arm.** Validators cannot yet replay the shard that carries the
  learned arm ([limits](OPTIMISTIC_SERVING.md#limits)).
