# Granite shard throughput (A4, fourth execution)

Declared on September 29, 2026, after the [shard serving execution](GRANITE_SHARD_SERVING_RESULTS.md)
passed and before any concurrent serving. The
[contract](../config/experiments/granite-shard-throughput.json) pins every rule below.

## Question

Do extra machines buy throughput without changing a single token?

A single host can serve faster by batching requests, but batching changes the
matrix shapes and therefore the numbers. This execution measures the gain an
owner ring provides while keeping every token identical to one-at-a-time
serving, and every intermediate result reproducible by anyone who recomputes it.

## Method

The owners, roles, arm, gate, fetch and target are those of the shard serving
execution: the learned assistant on three owner hosts, with the 24 round-4
development episodes as the target. The same owners serve the episodes twice:

1. **Sequential:** one episode at a time, as in the serving execution.
2. **Concurrent:** three episodes in flight.
   - Every message carries its stream.
   - Owners keep one cache and one arm setting per stream, and process messages
     in arrival order.
   - Owner 0 serializes its own computation and matches results in order.
   - Every step is the same batch-one computation as sequential serving; only
     owner idle time is filled.

Before this run, tests on a small Granite-shaped assistant showed that three
concurrent streams serve every episode, parent- and arm-selected, exactly as the
single-host evaluator does, on transformers 4.57 and 5.5.4.

## Checks

The execution passes only if all four hold:

1. The sequential pass passes every check of the shard serving execution.
2. The concurrent pass passes every check of the shard serving execution.
3. At least two episodes are in flight at once during the concurrent pass.
4. The concurrent pass is at least 1.5 times faster than the sequential pass, on
   the same owners and the same 24 episodes.

The run also measures, without gating, concurrent throughput against the
single-host development wall time, and traffic and memory per owner.

## Second attempt

Declared after the [first attempt](GRANITE_SHARD_THROUGHPUT_RESULTS.md) kept
every token identical but failed the throughput check (0.97×). The first attempt
stays failed.

Owner 0 now runs every tensor operation on one compute thread: the embedding,
its layers, the head, the logits check and argmax, and the parent feature.
Episode threads only render prompts, run tools and wait. Every owner records its
busy time in both passes. Owners, boundaries, target and all four checks are
unchanged.

## Limits

This makes no new quality claim, and independent operators remain A5. The run
gives no checklist credit and is not admission evidence.

Resources: three r7i.4xlarge allocations, each with its own expiry and a $8
allowance (at most $24), under the
[resource contract](../config/experiments/granite-shard-throughput-resources.json).
One attempt.
