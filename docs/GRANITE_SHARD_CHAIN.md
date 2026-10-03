# Settlement of real-model serving through CometBFT consensus (toward A5)

Declared on October 2, 2026, before any chain run. Authorized by the project
owner. The [contract](../config/experiments/granite-shard-chain.json) pins every
rule below.

## Question

The [settlement execution](GRANITE_SHARD_SETTLEMENT_RESULTS.md) passed with the
controller ordering blocks and two validators replaying them. Does the same
settlement hold when blocks come from consensus? Four CometBFT validators would
each admit transactions to their own mempool, check challenged proofs with their
own shard, and agree on every block.

## Method

The owners, arm, target, runtime and declared fault are those of the settlement
execution.

- **Hosts.** Eight canonical CPU hosts: three owners, one light auditor holding
  only shard 1, and four validators, each holding only shard 1.
- **Consensus.** Each validator creates its own consensus key. The controller
  assembles the genesis from their public keys, with equal voting power and the
  ledger terms as application state. Each validator runs CometBFT 0.38.26, from
  a binary pinned by digest, and the
  [settlement application](../src/neuroshard/inference/optimistic_app.py) in the
  owners' numerical environment. The validators connect to each other over
  their private addresses.
- **Transactions.** Every party signs its own transactions on its own host, as in
  the settlement execution. The controller only submits them to validator
  mempools and never orders anything.
- **Sequence:**
  1. both bonds;
  2. the honest job opened;
  3. the honest pass served;
  4. both honest commitments;
  5. the honest audit;
  6. the framing challenge, submitted inside the honest job's window;
  7. the cheating job opened;
  8. the cheating pass served;
  9. both commitments from the cheating pass;
  10. the cheating audit;
  11. once the honest window has closed, the auditor's challenge, submitted
      inside the cheating job's window.

  Proof bundles reach every validator before their challenge is submitted.
- **Ledger terms.** As in the settlement execution, except for the windows: a
  1800-block challenge window and 6000 blocks to commit, at one-second blocks.

## Checks

The execution passes only if all eight hold:

1. The honest pass matches the single-host development result under every check
   of the shard serving execution.
2. All four validators report the same state root at one common height.
3. Both bonds, both job openings and all four log commitments are admitted and
   committed.
4. The framing challenge is refused at mempool admission because its proof does
   not verify, and is never committed.
5. The auditor names owner 1's fault at the declared message.
6. The auditor's challenge is committed. The cheating job is recorded as fraud by
   owner 1, and owner 1 is slashed.
7. The honest job settles by consensus before the auditor's challenge is
   committed, paying each owner half the price, and owner 2 stays bonded.
8. Every final balance equals the declared expectation exactly.

The run also measures, without gating, block heights and intervals, each
transaction's admission and commit latency, and log, proof, transfer and
evidence sizes.

## Limits

One operator runs every host and every validator. Proof availability is
assumed, challenges carry no deposit, and audits that find nothing earn nothing.
No checklist credit.

Resources: eight r7i.4xlarge allocations, each with its own expiry and a $12
allowance (at most $96), under the
[resource contract](../config/experiments/granite-shard-chain-resources.json).
About two hours and $17 are expected. One attempt.

## First attempt: the challenge reply was lost

The [first attempt](../config/experiments/granite-shard-chain-report.json)
(commit `69d5b47`) went through consensus as declared up to the last step. Both
bonds, both job openings and all four log commitments were committed. The
framing challenge was refused at admission with "Fraud proof does not verify",
and the honest job settled by consensus at height 3109.

Submitting the auditor's challenge then failed. Admitting it replays the proof,
about 12 s on the real model. CometBFT's RPC server closes a response after its
write timeout, about 11 s by default, so the admission reply was lost. The
controller treated that as fatal, and its cleanup stopped every validator. The
application had also replayed under its state lock, which left one validator
four blocks behind.

Locally, four validators with a 15-second stand-in proof check reproduced the
lost reply. All hosts were retired with nothing remaining ($10.72). The attempt
stays failed.

## Amendment for the second attempt

Declared on October 2, 2026, after the first attempt and before the second.

- Each node allows 120 s for an admission reply.
- The application replays proofs outside its state lock, on one dedicated thread
  with a fixed thread count, and caches each verdict. Its handlers log and answer
  safely rather than fail.
- Nodes keep info-level logs, and the application keeps Python tracebacks and
  crash dumps, all in the evidence.
- The controller treats a lost admission reply as unknown and waits for
  inclusion.

With these changes, the local experiment admitted the slow challenge after 15 s
and committed it, and all four validators advanced together. A unit test checks
that a replay runs while the state lock is free. Every check, phase and
resource is unchanged. One attempt.

## Second attempt: passed

The [second attempt](GRANITE_SHARD_CHAIN_RESULTS.md) (commit `2ff26b6`)
**passed all eight declared checks**. The honest job settled by consensus at
height 3130; the auditor's challenge was admitted after a 12–13 s proof replay
and committed at height 3154. All four validators reported the same state root
at height 3157. All hosts were retired with nothing remaining ($10.81). The
first attempt stays failed.
