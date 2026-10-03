# Audited sharded serving (A4, sixth execution)

Declared on September 29, 2026, before any audited serving run. The
[contract](../config/experiments/granite-shard-audit.json) pins every rule below.
It is the first real-model test of the optimistic verification layer in the
[verifiable network design](VERIFIABLE_NETWORK_DESIGN.md).

## Question

Can a light auditor, holding only one owner's shard, verify that owner's
serving work by replaying its signed log? And when the owner cheats by a single
bit, can the auditor name the exact message and hand a fraud proof to anyone
else who holds the shard, without blaming an honest owner?

## Method

The owners, arm, target and runtime are those of the
[shard serving execution](GRANITE_SHARD_SERVING_RESULTS.md): three owner hosts
serve the 24 round-4 development episodes, all with the declared warm-up.

- **Signed logs.** Owners 1 and 2 log every command with the digests of what
  they received and sent, retain the received tensors, and sign the log. Their
  signing keys are created at fetch, and the public keys are published in their
  fetch receipts.
- **Light auditors.** Two separate auditor hosts join; they never join the ring.
  Auditor 1 fetches only owner 1's verified byte ranges, and auditor 2 only
  owner 2's plus the uploaded arm. Each receives its owner's log through the
  controller and replays it from an empty cache after the warm-up.
- **Two serving passes.** The first is honest. In the second, owner 1 flips the
  lowest bit of the first element of its 201st sent tensor.
- **Fraud proof.** Auditor 1 turns the first mismatch into a proof: owner 1's
  unchanged signed log plus the inputs up to the mismatch. A fresh verifier
  process, given only the proof and owner 1's published key, checks it.

## Checks

The execution passes only if all six hold:

1. The honest pass matches the single-host development result under every check
   of the shard serving execution.
2. Both auditors replay their owner's complete honest log without a mismatch,
   and each log is signed by its owner's published key.
3. Auditor 1 finds owner 1's first wrong output at exactly the declared message.
4. The fresh verifier accepts the fraud proof.
5. Auditor 2 finds no mismatch in owner 2's log from the faulty pass, so no
   honest owner is blamed.
6. Each auditor fetched exactly its owner's tensor bytes, less than the whole
   checkpoint.

The run also measures, without gating: audit replay time against the audited
owner's busy time, log and proof sizes, and transfer bytes.

## Limits

One operator runs every host. Bonds, challenge windows and slashing are not yet
on-chain, and independent operation belongs to A5. The run gives no checklist
credit and is not admission evidence.

Resources: five r7i.4xlarge allocations, each with its own expiry and a $10
allowance (at most $50), under the
[resource contract](../config/experiments/granite-shard-audit-resources.json).
One attempt.

## First attempt: failed at fetch

The [first attempt](../config/experiments/granite-shard-audit-report.json)
(commit `295b34f`) failed before any serving pass. Owner 0 and both auditors
fetched their shards, but owners 1 and 2 wrote no fetch receipt. Those two
create their signing keys at fetch with the `cryptography` package, which the
pinned remote runtime did not install. The local tests had passed because the
development environment has that package. All five hosts were retired with
nothing left running ($0.32). The attempt produced no audit evidence and stays
failed.

## Amendment for the second attempt

Declared on September 29, 2026, after the first attempt and before the second.
The audit profile now installs a
[runtime](granite-shard-audit-requirements.txt) that adds only `cffi` 2.0.0,
`cryptography` 50.0.1 and `pycparser` 2.23 to the pinned reference runtime. The
[contract](../config/experiments/granite-shard-audit.json) pins these versions,
and every owner and auditor role checks them before doing any work. A role
that fails now saves its error where the evidence copy collects it. The six
checks, the fault, the phases and the resources are unchanged. One attempt.
