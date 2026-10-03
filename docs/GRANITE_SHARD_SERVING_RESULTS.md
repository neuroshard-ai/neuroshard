# Granite shard serving results (A4, third execution)

**Passed all six declared checks.** The complete learned assistant ran across
three owner hosts, and none of them held the backbone. It served all 24
development episodes exactly as the single host served them in the round-4
development evaluation. That covers every selection, prompt, output token,
reused cache prefix and score, including the 18 of 24 workflows solved. This is
the capability improved by training, served across machines, with the parent's
earlier successes preserved. Its quality on fresh data is decided by the A2
confirmation.

Declaration: [GRANITE_SHARD_SERVING.md](GRANITE_SHARD_SERVING.md).
Evidence: [result](../config/experiments/granite-shard-serving-result.json) and
[report](../config/experiments/granite-shard-serving-report.json), commit `b80139c`.

## Checks

| Check | Outcome |
| --- | --- |
| Complete | All three owners finished; owner 0 served 24 episodes (8,806 generated tokens). |
| Tokenizer | Owner 0's checked tokenizer matches the pinned pipeline and fixture digests. |
| Fetched only owned | Owners fetched only their byte ranges (32–39 s each) and the pinned small files. |
| Arm on its owner | Only owner 2 held the round-4 addition, with the digest pinned by development. |
| Agreement | 24/24 episodes identical to the single-host result: same selections (all arm), generations and scores (18/24). |
| Memory | Serving peak RSS 5.22, 4.82 and 4.97 GB, under the 8 GiB limit. |

## Measurements (not gated)

- **Latency.** p95 with selection was 101.0 s, against 92.7 s on the single host
  (+9%).
- **Traffic.** Owners 0 and 1 each sent 304 MB and owner 2 sent 45 MB. The prefix
  cache reused 87.9% of all prompt tokens, so hidden-state traffic was about 4.6×
  lower than in the [first shard execution](GRANITE_SHARD_EXECUTION_RESULTS.md).
- **Determinism diagnostic.** On each owner, six fresh processes each pushed one
  fixed input through its layers three times. All 18 digests per owner were
  identical. A fresh process's first pass alone does not explain the
  [shard training](GRANITE_SHARD_TRAINING_RESULTS.md) recovery failure. That
  difference arose with the process group live and the ring carrying traffic,
  which this diagnostic did not include, so the question stays open.

## Resources

Three r7i.4xlarge hosts cost a conservative $1.75. Every host is retired.

## Meaning for A4

A4 now has four pieces of evidence:

- Generation across owners equals the complete model: 230/230 canonical
  generations.
- Training across owners equals the complete model, bit for bit on the real
  model.
- A lost inference owner recovers to the canonical tokens.
- The complete learned assistant is served across owners exactly as on one host.

Three things are still open:

- Bit-exact training recovery.
- A benefit beyond pooled memory, such as throughput with concurrent requests.
- Owners run by independent operators (A5).

No checklist credit or admission evidence.
