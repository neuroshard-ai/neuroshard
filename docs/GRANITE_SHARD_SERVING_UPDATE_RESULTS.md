# Serving the accepted assistant across owners: results

**The assistant A2 accepted, with its round-4 update held by one owner, served across three
Granite owner hosts and reproduced all 24 single-host development episodes token for
token.** Every selection, generation and score equals the development result of A2's third
attempt: 19 of 24 episodes succeed, as on one host, with no mismatch. The second attempt
passed all six checks.

Declaration: [serving the accepted update](GRANITE_SHARD_SERVING_UPDATE.md). Evidence:
[first attempt](../config/experiments/granite-shard-serving-update-result.json) (commit
`4d55822`) and [second attempt](../config/experiments/granite-shard-serving-update2-result.json)
(commit `5c345b3`).

## Checks (second attempt)

| Check | Result |
| --- | --- |
| All three owners finish; owner 0 serves all 24 episodes | pass |
| Owner 0's tokenizer matches the pinned pipeline and fixtures | pass |
| Every owner fetched only its own byte ranges and the pinned small files | pass |
| Only owner 2 holds the update, with the pinned digest | pass |
| Every episode equals the single-host result | pass, 0 mismatches |
| Every owner's serving peak stays under 8 GiB | pass, 4.8–5.2 GB |

## Measurements

The gate selected the update for all 24 episodes, as on one host. p95 latency with selection
was 94.4 s against 94.8 s on one host. Owners 0 and 1 each sent 304 MB and owner 2 sent
45 MB. In six fresh processes per owner, every first forward pass equalled the later ones.

## Attempts and cost

The first attempt failed at serving and served no episode ($0.28): owners 1 and 2 could not
import the auditing module that the owner runtime imports, because the plan's source list,
copied from A4's, predated it. The second attempt listed the runtime's whole import closure,
which a test now checks, and cost $1.68. Every instance is terminated and its security group
retired. This is not independent operation: one operator ran all three hosts.
