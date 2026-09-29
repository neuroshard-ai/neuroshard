# Granite shard throughput results (A4, fourth execution)

## Second attempt: passed

**All four checks passed, 2.25× throughput with every token identical.** With
every tensor operation on owner 0 moved onto one compute thread, three episodes
in flight finished the 24 development episodes in 705 s. One at a time on the
same owners took 1,585 s, and the single host took 1,547 s: 2.25× the sequential
ring and 2.20× the single host. Both passes reproduced the single-host episodes
token for token (18/24 solved, same selections, prefixes and scores).

Evidence: [result](../config/experiments/granite-shard-throughput2-result.json) and
[report](../config/experiments/granite-shard-throughput2-report.json), commit `b143cf5`.

| Check | Outcome |
| --- | --- |
| Sequential agreement | Pass: 24/24 identical, p95 92.7 s. |
| Concurrent agreement | Pass: 24/24 identical, p95 115.9 s. |
| Overlap | Pass: three episodes in flight. |
| Throughput | Pass: 2.25× (needs ≥1.5×). |

Owner busy time confirms the first attempt's diagnosis. Each owner computed for
510–525 s in the sequential pass and 525–545 s in the concurrent pass, so the
concurrent pass no longer inflates any owner's work, and the three stages are
balanced within 3%. The remaining gap to 3× comes from owner 0's per-episode
Python work and from the tail, where fewer than three episodes remain.

This is the benefit extra machines give while keeping every result exactly
reproducible. A single host could only serve faster by batching, which changes
the computation and makes independent audit by recomputation harder. The run
cost $2.20, and every host is retired.

## First attempt: failed

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
