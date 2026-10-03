# CometBFT chain settlement results

The [second attempt](GRANITE_SHARD_CHAIN.md#amendment-for-the-second-attempt)
**passed all eight declared checks** (commit `2ff26b6`). On the real 3B
assistant, four CometBFT validators on separate hosts, each holding only
shard 1, settled audited sharded serving through consensus: the honest job
paid after its window, a framing was refused at mempool admission, and a
cheating owner was slashed on a real fraud proof. Every party signed its own
transactions on its own host. The controller only submitted them to validator
mempools. Evidence:
[report](../config/experiments/granite-shard-chain2-report.json). The first
attempt failed when the challenge reply was lost and stays failed.

## Checks

| Check | Result |
|---|---|
| Honest pass matches the single-host development result | 18/24 correct, no token mismatch, p95 97.8 s |
| Validators agree | Same state root (`4f99cd0e…`) at height 3157 on all four hosts |
| Honest work committed | Both bonds, both job openings and all four log commitments |
| Framing refused | Validator 4 refused it at admission in 1.0 s: "Fraud proof does not verify"; never committed |
| Fault named | Owner 1's wrong output at forward pass 200, as declared |
| Fault proven | Challenge committed at height 3154; the cheating job recorded as fraud by owner 1; owner 1 slashed |
| Honest job settled first | By consensus at height 3130, before the challenge: 0.5 NEURO to each owner; owner 2 still bonded |
| Exact balances | Every final balance equal to the declared expectation |

## Final ledger

| Account | NEURO | Why |
|---|---|---|
| User | 18.998 | Paid 1 for the honest job; refunded the cheated job; two fees |
| Owner 1 | 15.497 | Paid 0.5; 5 bond slashed; three fees |
| Owner 2 | 15.497 | Paid 0.5; 5 bond still locked; three fees |
| Auditor | 22.499 | Half the slashed bond (2.5); one fee |
| Accuser | 20.000 | The rejected framing changed no state |

Burned: 2.509 NEURO (half the slashed bond and nine fees).

## Consensus

| Transaction | Height |
|---|---|
| Owner 1 bond | 18 |
| Owner 2 bond | 21 |
| Honest job opened | 24 |
| Honest commitments | 1326, 1329 |
| Framing | refused at admission |
| Cheating job opened | 1727 |
| Cheating commitments | 3042, 3046 |
| Honest job settled | 3130 |
| Proven challenge | 3154 |

The honest window was 1800 one-second blocks from the last honest commitment.
The challenge arrived 24 blocks after settlement, still inside the cheating
job's window.

## Measurements

- **Validator cost.** Refusing the framing took validator 4 1.0 s, replaying
  the single message it claimed. Accepting the real proof took 12.3–13.1 s on
  each validator, replaying 201 messages of shard 1, outside the state lock.
  Honest work cost validators no replay at all. No admission reply was lost.
- **Audits.** The honest audit replayed all 8830 of owner 1's logged forward
  passes exactly in 495 s. The faulty log's first wrong output was found after
  11.8 s.
- **Serving.** Each pass reproduced the 24 development episodes; honest wall
  time 1660 s, p95 97.8 s; peak RSS 4.5–5.3 GB per owner.
- **Data moved.** The two proof bundles totaled 18.3 MB (5.6 MB framing, 12.7 MB
  proven).
- **Evidence.** Every host's evidence was archived (0.6–28 MB compressed),
  including both proof bundles, signed log records, validator logs and the
  CometBFT data directories. Bulk retained inputs were moved out once no one
  needed them.

## Limits

One operator ran every host and every validator. Validators held the
challenged shard; proof availability was assumed; challenges carried no
deposit; and audits that find nothing earned nothing. This is not independent
operation and gives no checklist credit.

## Resources

Eight r7i.4xlarge allocations, all retired with nothing remaining ($10.81).
With the failed first attempt ($10.72), the execution cost $21.54.
