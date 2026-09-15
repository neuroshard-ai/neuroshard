# Request-local caches for sharded inference

The native continuation demonstration takes 61.25 seconds to produce its
140-token response, and its complete quality evidence exceeds 126 GB per
original shard. The original generator recomputes every preceding token at
each decode step. This research executor instead prefills the prompt once,
then processes one new token at a time using key/value state held by each
layer's owner. The underlying cache semantics follow the pinned
[Transformers 4.57.3 documentation](https://huggingface.co/docs/transformers/v4.57.3/en/cache_explanation).

Each owner retains only its own layers' cache tensors. Global decoder indices
map to a local contiguous cache; no unowned cache tensors or model parameters
are allocated. The tied head remains on rank zero. Intermediate owners forward
the full prompt during prefill and one hidden vector during decode. The final
owner sends only the last position needed by the head.

The cache belongs to one unpadded request. A new request reconstructs it from
the prompt. Changed weights, changed execution mode, stale positions,
incomplete local layers or a partially failed operation invalidate the session.
Cache tensors are never imported from a peer. This prototype does not implement
live cache migration, concurrent request scheduling or multi-turn cache reuse.

## Verification and numerical identity

Caching changes matrix shapes and can change floating-point rounding and greedy
answers. It therefore uses a separate executor; the existing native generator
and frozen learning/growth profiles continue to use their original functions.
No new native execution profile or token issuance is activated here.

The existing boundary recorder captures the cached execution. An auditor starts
with an empty local cache and replays every operation from the committed prompt,
checking every outgoing boundary and generated token. All partitions must be
replayed. Closing the graph by matching sender and receiver hashes alone is
insufficient: a self-consistent forged activation is rejected by numerical
replay. Cache reuse reduces work within that replay; it does not remove the
honest-quorum assumption or prove independent ownership.

CPU tests compare three- and four-owner execution against the independent full
Hugging Face model under eager and SDPA attention. They check logits, generated
tokens, fresh-request repeatability, EOS handling, owned cache memory, actual
boundary bytes and complete witness replay. They also reject changed weights,
bad cursors, partial failures and a forged but closed boundary graph. GPU
performance and fidelity require the following frozen comparison.

## GPU comparison contract

The [plan](../config/experiments/shard-cache-probe.json) binds the existing R4
1.7B checkpoint and all 768 already-exposed generated-answer cases. It trains
nothing and supplies no new independent learning evidence.

1. Commit the executor and plan, prepare the exact requests, then commit
   `config/experiments/shard-cache-prepared.json` before GPU execution.
2. On three GPU owners, alternate which generator runs first for each request.
   Compare unrecorded generation times with identical transport. Separately
   repeat each cached response with complete witness recording; its output
   must match the first cached execution exactly.
3. Every one of three audit hosts sequentially replays all three partitions
   and all requests. Audits bind both transcript roots and actual output tokens.
4. Recompute scores from decoded responses. The original generator must still
   reproduce every published R4 answer. Require zero individual correct-answer
   losses, at least 95% identical token sequences, at least 2× aggregate speedup
   on identical sequences and at least 90% less boundary tensor traffic.

The comparison records per-owner memory, cache bytes, boundary bytes, generation
times, witness production and complete replay costs. Output changes remain
visible even when both answers score correctly. Report transport counters as
tensor traffic, not total network bandwidth. Report the cache's additional
resident memory alongside its savings.

`scripts/run_shard_cache_probe.py` exposes `prepare`, `compare`, `replay` and
`score`. The operated GPU allocation may start only after the incremental
capacity trial retires its resources, with three g5.2xlarge instances, a
three-hour limit and a $50 planning cap. Preserve full evidence before cleanup.
A passing comparison is an inference-executor result; native adoption still
requires an explicitly bound profile and its serving-quality decision.
