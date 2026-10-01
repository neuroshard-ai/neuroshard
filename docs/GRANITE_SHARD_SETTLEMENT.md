# Bonded settlement of audited serving (toward A5)

Declared on October 1, 2026, before any settlement run. Authorized by the project
owner. The [contract](../config/experiments/granite-shard-settlement.json) pins
every rule below.

## Question

Can the [bonded optimistic serving ledger](OPTIMISTIC_SERVING.md) settle real
serving of the 3B assistant? Honest work should be paid after its challenge
window, a framing attempt should fail, and a cheating owner should be slashed by
a real fraud proof. Independent validators must replay the same blocks to the
same state.

## Method

The owners, arm, target and runtime are those of the
[audited serving execution](GRANITE_SHARD_AUDIT_RESULTS.md), including the same
declared fault: owner 1 flips the lowest bit of the first element of its 201st
sent tensor in the second pass.

- **Hosts.** Six canonical CPU hosts: three owners, one light auditor holding
  only shard 1, and two validators, each holding only shard 1.
- **Signing.** Every party signs its own transactions on its own host. Owners 1
  and 2 sign their bonds and log commitments with an account key and the log key
  they serve with. The auditor signs the challenge. A separate accuser account
  on the auditor host signs a framing challenge, whose proof claims the honest
  log's first output was wrong. The controller holds the user account.
- **Blocks.** The controller orders the transactions:
  1. both owner bonds;
  2. the honest job opened;
  3. both honest log commitments;
  4. the cheating job opened;
  5. both log commitments from the cheating pass;
  6. the framing challenge, inside the honest job's window;
  7. and 8. empty; the honest job settles at block 8;
  9. the auditor's challenge, in the last block of the cheating job's window;
  10. and 11. empty.

  A proven cheater is paid for no job still in its window, so the honest job
  must settle before the proof against its owner arrives.

  Each pass is served, and its owner-1 log goes to the auditor, between the job's
  opening and its commitments.
- **Ledger terms.** Each party starts with 20 NEURO. Owner bonds are 5 NEURO, the
  job price is 1 NEURO, and the challenge window is 4 blocks; the other
  parameters are the ledger's defaults.
- **Validation.** Both validators receive the blocks and the proof bundles and
  replay everything from genesis, checking each challenge by replaying its proof
  with shard 1.
- **Evidence.** Each host moves its retained-input files out of the evidence
  directory once they are no longer needed. The signed log records, proof
  bundles, blocks and validator reports stay within the evidence cap.

## Checks

The execution passes only if all eight hold:

1. The honest pass matches the single-host development result under every check
   of the shard serving execution.
2. Both validators complete and reach the same state root and the same
   transaction outcomes.
3. Both bonds, both job openings and all four log commitments are accepted.
4. The framing challenge is rejected because its proof does not verify.
5. The auditor names owner 1's fault at the declared message.
6. The auditor's challenge is accepted. The cheating job is recorded as fraud by
   owner 1, and owner 1 is slashed.
7. The honest job settles after its window, paying each owner half the price,
   and owner 2 stays bonded.
8. Every final balance equals the declared expectation exactly.

The run also measures, without gating, the validators' proof-replay and shard
load times, and the log, proof and transaction sizes.

## Limits

One operator runs every host, and the controller sequences blocks instead of
CometBFT; validators only replay them. Proof availability is assumed, challenges
carry no deposit, and audits that find nothing earn nothing. No checklist credit.

Resources: six r7i.4xlarge allocations, each with its own expiry and a $10
allowance (at most $60), under the
[resource contract](../config/experiments/granite-shard-settlement-resources.json).
About $10 is expected. One attempt.
