# Bonded settlement results

The [third attempt](GRANITE_SHARD_SETTLEMENT.md#amendment-for-the-third-attempt)
**passed all eight declared checks** (commit `4c2c2ea`). On the real 3B
assistant, the [bonded optimistic serving ledger](OPTIMISTIC_SERVING.md) paid
honest sharded serving after its challenge window, rejected an attempt to frame
the honest owner, and slashed the cheating owner on a real fraud proof. Two
validators, each holding only shard 1, replayed the same 11 blocks to the same
state root. Every party signed its own transactions on its own host. Evidence:
[report](../config/experiments/granite-shard-settlement3-report.json). The
first two attempts failed before any serving and stay failed.

## Checks

| Check | Result |
|---|---|
| Honest pass matches the single-host development result | 18/24 correct, no token mismatch, p95 97.5 s |
| Validators agree | Same state root (`15bdd0bb…`) and the same 10 outcomes |
| Honest work accepted | Both bonds, both job openings and all four log commitments |
| Framing rejected | Its proof replayed to the honest output: "Fraud proof does not verify" |
| Fault named | Owner 1's wrong output at forward pass 200, as declared |
| Fault proven | The cheating job recorded as fraud by owner 1; owner 1 slashed |
| Honest job settled | At block 8: 0.5 NEURO to each owner; owner 2 still bonded |
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

## Measurements

- **Validator cost.** Rejecting the framing took each validator about 1 s,
  replaying the single message it claimed. Accepting the real proof took 11–12 s,
  replaying 201 messages of shard 1. Loading the shard took 8–9 s. Honest work
  cost validators no replay at all.
- **Audits.** The honest audit replayed all 8830 of owner 1's logged forward passes
  exactly in 526 s. The faulty log's first wrong output was found after 12.6 s.
- **Data moved.** Each owner-1 log sent to the auditor was about 242 MB
  compressed. The two proof bundles sent to the validators totaled 13.4 MB.
- **Evidence.** Every host's evidence was archived, from 0.6 MB to 28 MB,
  including both proof bundles, the signed log records, the blocks and the
  validator reports. Bulk retained inputs were moved out once no one needed
  them.

## Limits

One operator ran every host and ordered the blocks; validators replayed them
rather than reaching consensus through CometBFT. Validators held the challenged
shard; proof availability was assumed; challenges carried no deposit; and audits
that find nothing earned nothing. No checklist credit.

## Resources

Six r7i.4xlarge allocations, all retired with nothing remaining ($7.57). With
the two failed attempts ($0.33 and $0.44), the execution cost $8.34.
