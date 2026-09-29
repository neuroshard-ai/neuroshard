# Granite shard execution results (A4, first execution)

**Passed all six declared checks.** The pinned Granite 4.1 3B ran across three
owner hosts, and none of them downloaded, stored or loaded the complete backbone.
The owners reproduced all 230 recorded generations of the canonical single-host
runtime token for token. When one owner was lost mid-generation, the relaunched
group finished with the canonical tokens. This is the first A4 evidence; A4 stays
open.

Declaration: [GRANITE_SHARD_EXECUTION.md](GRANITE_SHARD_EXECUTION.md).
Evidence: [result](../config/experiments/granite-shard-result.json) and
[report](../config/experiments/granite-shard-report.json), commit `cb06bf1`.

## Checks

| Check | Outcome |
| --- | --- |
| Complete | All three owners finished; owner 0 returned 230 generations (9,170 tokens). |
| Agreement | 230/230 generations token-identical to the canonical recording. |
| Ownership | Each owner fetched exactly its tensors: 2.40, 2.20 and 2.20 GB of 6.81 GB. |
| Memory | Serving peak RSS 6.22, 6.05 and 6.09 GB, under the 8 GiB limit and below the 11.34 GB single-host peak. |
| Outage | Owner 2 exited at its 40th step; both peers saw the connection reset at once, with 39 tokens committed. |
| Recovery | The relaunched group replayed the 39 committed tokens and finished all 95 canonical tokens. |

## Measurements (not gated)

- **Fetch.** Each owner took 31–38 seconds, using two to four merged HTTP range
  requests against the pinned Hugging Face revision. Every tensor was checked
  against its inventory digest.
- **Time.** Generation took 2,946 seconds, against 2,739 seconds on the canonical
  single host (7.6% slower). The layer ranges run in sequence, so a single request
  is not faster.
- **Traffic.** Owners 0 and 1 each sent 1.41 GB of hidden states, mostly prompt
  prefill across about 267,000 prompt tokens. Owner 2 sent 47 MB, because it returns
  only the last position. Decoding costs 5 KiB per token per hop.
- **Pooled memory.** Each owner holds about a third of the weights. The model runs
  on hosts that each stay below the single-host peak, and adding owners shrinks
  every share further.

## Resources

Three r7i.4xlarge allocations cost a conservative $4.93 in compute. Every
instance, volume and security group is retired.

The controller finished every phase and collected all evidence, then stopped
while retiring hosts. Owner 1's delete-on-termination volume outlived its
terminated instance by a few seconds. The volume check raised before owner 2 was
retired and before `result.json` was written. Owner 2 ran about one extra hour,
until it was retired manually. The result above was assembled from the saved
phase and fetch files with the same `assess` function. Retirement now waits for
lingering volumes, and the shard controller retires each owner independently
after saving its result.

## What this does and does not show

It shows that sharding the assistant costs nothing in output: the pipeline is
numerically the canonical runtime. Owners fetch and verify only their own bytes,
memory pools across machines, and an outage costs a replay rather than a
different answer.

Other A4 evidence is still to come:

- Throughput with concurrent requests.
- Backward and training across owners on the real model. Tests on small
  checkpoints show both arms trained across owners are bit-identical to the
  single-host trainer, including preference steps and a resumed outage.
- Serving the learned module on shards.

Owners run by independent operators belong to A5. No checklist credit or
admission evidence.
