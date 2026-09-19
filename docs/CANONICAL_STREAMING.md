# Checked chunks for owned-shard inference

The subsequent GPU trial passed all eight complete-response checks, corrected
both observed cached-decoding disagreements, rejected both forgeries and
preserved output across both alternate chunk sizes. The method's measured
streaming cost remains high: 28.09–29.26 seconds and 1.46–1.61 GB of tensor
traffic for 128-token responses. See the [complete result](../config/experiments/checked-streaming-results.json)
and the next [fixed-block implementation and frozen comparison](BLOCKED_STREAMING.md).
These execution results do not promote the rejected learning candidate.

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

The existing local graph service now optionally installs `FusedService` through
its `fused_service` and `fused_weights` configuration paths. Its immutable
specification binds the graph and executor, gate checkpoint, all expert interface
checkpoints (or none), context, output limit and chunk size. Each owner loads
only its own adapter; the small gate is synchronized. This does not authorize
the specification for native settlement.

The local `stream_fused` request supplies `service`, complete alternating
`messages`, and `max_tokens`, alongside the usual fresh request identifier. It
rejects excess context instead of truncating history. Checked events appear in
`streams/<request>/events/`; each includes token offsets and the complete decoded
text prefix to avoid corrupting Unicode at individual token boundaries. A final
record commits the complete messages, tokenizer, prompt tokens and response.
Requests carry their own history and do not share conversation memory.

The queue consumes execution even if nobody reads those event files. Five CPU
processes checked complete multi-turn delivery, the same response with delivery
discarded, excess-context rejection before execution, and collective rejection
when one owner changes its installed adapter. Full completion identities agree
across owners. These checks establish behavior, not multi-turn answer quality.

This remains an operator-local queue with one executing request at a time. It
is not a public HTTP service. Queue admission, concurrent-client scheduling,
native authorization, price/refund rules and GPU load targets still need their
own integration. Operators and their auditors receive the complete prompt and
response. Local records retain those bytes; confidentiality and automatic
deletion are not provided. Public clients must not be given this filesystem
queue directly.

The numerical distinction and research sources are described in the
[batched-audit prescription](EXPERT_INTERFACE_CONTINUATION.md). Native inference
acceptance has not been switched to this program.
