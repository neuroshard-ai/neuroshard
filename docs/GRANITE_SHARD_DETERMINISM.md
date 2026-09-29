# Granite shard ring determinism (A4 diagnostic)

Declared on September 29, 2026, before any repeated training-ring launch. The
[contract](../config/experiments/granite-shard-determinism.json) pins the job.
This is a diagnostic, not a gated execution.

## Why

The [shard training execution](GRANITE_SHARD_TRAINING_RESULTS.md) trained the
addition arm across three owners bit-identically to a single host. Its outage
run still computed its very first preference reference differently from the
uninterrupted run on the same owners, so recovery could not match.

The [serving execution's diagnostic](GRANITE_SHARD_SERVING_RESULTS.md) found
fresh processes reproducible when run alone. That leaves the live ring as the
suspect.

## Method

The same three owner hosts and boundaries run the shard training job six times,
each launch in fresh processes. The job uses the same sequences, arm and
settings, shortened to one step: 8 preference references, then 12 forward passes
and one update. Every owner records a SHA-256 digest of every tensor it sends,
including boundary gradients.

## Reported

- Per owner: the number of distinct traces across launches, and the first message
  that differs.
- Across launches: the number of distinct preference references, and the number
  of distinct trained tensors.

The ring counts as reproducible only if every count is one. If a difference
appears, its owner and message decide the fix. The shard training recovery is
repeated only after the ring is reproducible.

Resources: three r7i.4xlarge allocations, each with its own expiry and a $8
allowance (at most $24), under the
[resource contract](../config/experiments/granite-shard-determinism-resources.json).
