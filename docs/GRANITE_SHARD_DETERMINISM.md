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

## First attempt: not reproducible, localized

Evidence: [result](../config/experiments/granite-shard-determinism-result.json) and
[report](../config/experiments/granite-shard-determinism-report.json), commit `89731bb`. Cost $0.92.

Five of six launches matched each other exactly, and they also match the
uninterrupted training run and the single-host reference.

In one launch (ring-5), owner 0's very first sent tensor differed: the output of
its first forward pass in the process. Its first preference reference became
−0.154545 instead of −0.155187. Everything else was identical:

- Every later forward pass, including ones with new shapes.
- On owners 1 and 2, every message except the one computed from owner 0's
  first tensor.
- On owner 0, every message except the two boundary gradients of the preference
  pair that uses that reference.

The training outage run showed the same effect, with a third value, −0.155652.
The effect is a first-call rounding difference in a fresh owner process: not
protocol, not shapes, not transport.

## Second attempt: warm-up

Declared after the first attempt, before its launches. Every owner now joins the
ring, then runs a declared warm-up before any work:

- discarded 1,024-token and 1-token passes through every owned module, with
  owner 0's head;
- one discarded backward pass through the arm on its owner, with its gradients
  cleared.

Tests on small checkpoints show the warm-up changes no output and no trained
tensor. Twelve launches replace six. If all twelve are identical, the shard
training recovery is repeated with the warm-up as a new declared execution.

Resources: three r7i.4xlarge allocations, each with its own expiry and a $8
allowance (at most $24), under the
[resource contract](../config/experiments/granite-shard-determinism-resources.json).
