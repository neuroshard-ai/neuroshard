# Checked chunks for owned-shard inference

`sharded.canonical_stream.stream` proposes a small chunk with cached generation,
checks it with the distinct fixed-context causal program, and yields tokens only
after the complete chunk passes. A disagreement replaces the first incorrect
token and repeats the check. Original source weights, the gate and every expert
interface stay pinned for the entire request.

For a deterministic fixed-shape causal computation, later input tokens cannot
change an earlier prediction. Each correction therefore fixes a strictly longer
prefix. A chunk of at most K tokens needs at most K+1 checks; the implementation
enforces that bound and aborts if a previously established prefix changes.
Records bind every draft, check, accepted chunk, model version and token offset.
No unverified draft token is released.

The five-process check reproduced the valid response, repaired deliberately
incorrect drafts, bounded correction work, and rejected switching model weights
between chunks. These are CPU integration results. GPU quality, throughput,
latency, concurrent clients, provider discovery and native billing remain
separate requirements. The ongoing GPU learning trial uses its original frozen
executor and does not include this later streaming implementation.

This is an execution primitive, not a public HTTP service. Every model owner
must consume the stream fully; a client disconnect must only detach delivery,
so that it cannot leave the other owners waiting in collective communication.
A service must separately bound its queue and decide refund/cancellation rules.
Prompt and response confidentiality is not supplied by this primitive.

The numerical distinction and research sources are described in the
[batched-audit prescription](EXPERT_INTERFACE_CONTINUATION.md). Native inference
acceptance has not been switched to this program.
