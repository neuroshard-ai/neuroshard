# Granite shard throughput results (A4, fourth execution)

**Failed the throughput check. Every token stayed identical.** Both passes
reproduced all 24 single-host development episodes token for token: one episode
at a time, and three in flight. The concurrent protocol is therefore correct. It
brought no speedup: the three-stream pass took 1,671 s against 1,617 s for one
stream, a ratio of 0.97 against the declared 1.5.

Declaration: [GRANITE_SHARD_THROUGHPUT.md](GRANITE_SHARD_THROUGHPUT.md).
Evidence: [result](../config/experiments/granite-shard-throughput-result.json) and
[report](../config/experiments/granite-shard-throughput-report.json), commit `d572cdf`.

## Checks

| Check | Outcome |
| --- | --- |
| Sequential agreement | Pass: 24/24 episodes identical to the single host (18/24 solved), p95 95.3 s. |
| Concurrent agreement | Pass: 24/24 identical, same selections, tokens, prefixes and scores. |
| Overlap | Pass: three episodes in flight; mean concurrency 2.92. |
| Throughput | **Fail**: 0.97× (needs ≥1.5×). |

Owner peaks stayed at 4.5–4.9 GB in both passes.

## What happened

Three episodes really were in flight the whole time, but each decode step took
545 ms instead of 176 ms, so the ring finished steps at the same rate as before.

On one host a step takes about 176 ms, mostly streaming 6.8 GB of weights, so
each of three owner stages should need about 55–60 ms. A local ring with a fixed
50 ms delay standing in for each owner's compute ran three streams 1.5× faster
than one, so the protocol does overlap.

The likely cause is thread contention on owner 0. Owner 0 ran tensor work from
one Python thread per episode. Each calling thread gets its own OpenMP team of
eight, so three or four teams competed for 16 vCPUs while idle teams spun.
Owners 1 and 2 run a single thread and are unaffected. This is a hypothesis: the
execution recorded no per-owner busy time to confirm it.

## Next

A new declared execution will run every tensor operation on owner 0 on one
compute thread, record each owner's busy time, and keep the same gate. This
execution stays failed.

Resources: three r7i.4xlarge hosts cost a conservative $3.09. Every host is
retired.
