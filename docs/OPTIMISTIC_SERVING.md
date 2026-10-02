# Bonded optimistic serving ledger

Status: implemented, tested on small checkpoints, settled once on the real
assistant through four CometBFT validator hosts, October 2, 2026. Not a public
chain and not independent operation. This is the settlement layer of the
[verifiable network design](VERIFIABLE_NETWORK_DESIGN.md): serving work is
accepted unless an auditor proves fraud within a challenge window. Validators
replay a fraud proof only when a challenge arrives, so honest serving costs them
no neural recomputation.

The state machine is [`neuroshard.inference.optimistic`](../src/neuroshard/inference/optimistic.py).
It imports no model code. The validator's proof checker is
`granite_audit.challenge_checker`.

## On CometBFT

[`neuroshard.inference.optimistic_app`](../src/neuroshard/inference/optimistic_app.py)
runs the ledger as an ABCI application behind native CometBFT 0.38.26
validators, and [`optimistic_network`](../src/neuroshard/inference/optimistic_network.py)
launches a local validator set.

- Each block first advances the ledger, settling jobs whose window closed, then
  applies its transactions.
- Mempool admission and proposals evaluate a transaction against the next block's
  state, so only transactions that apply reach a block. A challenge whose proof
  does not verify is refused at admission and never enters a block.
- Each validator replays a challenged proof once, with the shard it holds, and
  caches the verdict. A bundle that cannot be read or replayed counts as not
  verifying, on every validator alike.

## Roles

- **Users** escrow payment for a serving job. Owner 0 (embedding, first layers
  and output head) runs on the user's own device and is never bonded.
- **Owners** of every other shard bond NEURO and register the Ed25519 key that
  signs their logs.
- **Auditors** hold one shard, replay an owner's signed log, and submit a fraud
  proof when an output does not reproduce.
- **Validators** order transactions and, for a challenge, replay the proof with
  the challenged shard.

## Transactions

All transactions are secp256k1-signed envelopes with chain ID and account nonce,
and each burns the fee.

| Kind | Effect |
|---|---|
| `owner_bond` | Locks at least the minimum bond for one shard of the current model and registers a log key. Requires a possession signature by that key over the account, shard, amount and nonce. A key can bond once. |
| `owner_unbond` | Starts withdrawal. Refused while the owner is named in an unsettled job. |
| `owner_withdraw` | Returns the bond once the challenge window has passed since unbonding. |
| `serve_open` | Escrows the price and names one active owner per bonded shard, in order, with the request digest. |
| `log_commit` | A named owner commits the digest of its signed log statement for the job, signed by its log key over the chain, job and digest. When every owner has committed, the challenge window opens. |
| `challenge` | Names a committed owner and the content address of a fraud-proof bundle. Validators accept it only if the bundle's log hashes to the owner's committed statement, carries the owner's key and signature, and its replay disagrees at exactly the claimed message. |

## Settlement rules

- **Settled.** At the first block after the challenge window closes, the price is
  split equally among the job's owners; any remainder is burned.
- **Fraud.** A verified challenge slashes the owner's whole bond: half goes to the
  challenger and half is burned. The user is refunded, and every other unsettled
  job naming that owner is voided and refunded too.
- **Rejected challenge.** A proof that does not verify, an owner who has not
  committed, or a closed window leaves the state unchanged.
- **Expired.** If the job's owners have not all committed by the job deadline,
  the user is refunded.
- **Conservation.** Initial supply equals liquid balances plus owner bonds plus
  job escrow plus burned fees and penalties, checked after every transition.

Parameters: fee 0.001 NEURO, minimum owner bond 5 NEURO, minimum price 0.1 NEURO,
a 20-block challenge window, 60 blocks to commit, and a 50% challenger share of
the slashed bond.

## Evidence

Tests on the small Granite-shaped checkpoints
([ledger tests](../tests/evolution/test_optimistic_serving.py) and the
[end-to-end test](../tests/evolution/test_granite_serving.py)):

- Three owners serve an honest pass and a pass in which owner 1 flips one bit,
  both with signed logs whose keys are bonded on the ledger. An auditor saves a
  real fraud proof against owner 1.
- Two separate validator processes, each holding shard 1, replay the same blocks
  and reach the same state root.
- A forged proof claiming the honest owner's first output was wrong is rejected.
- The real proof is accepted: owner 1 is slashed, the auditor receives half the
  bond, and the user is refunded. The honest job settles and pays both owners.
- Unit tests cover possession proofs, bond exposure while named or within the
  window, commitment signatures, voiding, late and failed challenges, expiry,
  conservation, and deterministic replay.
- Four local CometBFT validators, each holding shard 1 of the small checkpoint,
  settled the same scenario through consensus. The framing challenge was refused
  at admission, the honest job settled, and the real proof slashed owner 1. All
  four reached the same state root with exactly the expected balances.
- [On the real 3B assistant](GRANITE_SHARD_SETTLEMENT_RESULTS.md), with every
  party signing on its own host, the same sequence passed all eight declared
  checks. Two validators reached the same state root, and the final balances
  matched the declared expectation exactly.
- [Through CometBFT consensus](GRANITE_SHARD_CHAIN_RESULTS.md), four validator
  hosts holding only shard 1 admitted the same transactions to their mempools
  and agreed on every block. The honest job settled at height 3130, the framing
  was refused at admission, and the real proof slashed owner 1 at height 3154.

## Limits

- **Validators hold every audited shard.** A validator missing the challenged
  shard cannot decide, and the checker raises rather than guessing. Committees of
  shard holders are future work.
- **Proof availability.** Bundles are content-addressed files. Data availability
  is assumed, not provided.
- **Challenge cost.** A rejected challenge changes no state, so it costs the
  challenger nothing while costing validators a replay. Rate limits or a
  challenge deposit are needed before public use.
- **Clean audits earn nothing.** Auditors are paid only for proven fraud. A fee
  for audits that find nothing would need a way to show the replay was actually
  done.
- **Execution class.** Bit-exact replay requires validators and owners to share
  the pinned runtime and CPU instruction class.
- **Not yet on a public chain.** It has run on four CometBFT hosts under one
  operator, and no independent operator has run it.
