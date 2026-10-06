# Bonded optimistic serving ledger

Status: implemented, tested on small checkpoints, settled once on the real
assistant through four CometBFT validator hosts, October 2, 2026. Revised the same
day, before any untrusted participant joins: owner logs are bound to the paid
request, a proof bundle a validator lacks gives no verdict instead of a rejection,
and every cheap admission check runs before any proof replay. A job then buys a
budget of token positions and pays only for the
[positions its logs cover](#metered-settlement), and a challenge
[locks a deposit](#challenges-and-deposits) that it forfeits unless its proof
verifies. The revisions are tested on small checkpoints and four local CometBFT
validators. The published real-model runs used the earlier protocol (commit
`fc53411`). Not a public chain and not independent operation.

This is the settlement layer of the [verifiable network design](VERIFIABLE_NETWORK_DESIGN.md):
serving work is accepted unless an auditor proves fraud within a challenge window.
Validators replay a fraud proof only for a challenge that has locked a deposit, so
honest serving costs them no neural recomputation.

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
  state, so only transactions that apply reach a block. A proof that does not
  verify is refused at admission and never enters a block.
- After each block, every validator replays the proofs of open challenges in the
  background, once each, with the shard it holds, and caches the verdicts. A proof
  is then judged from the cache. A validator that has not finished the replay
  finishes it outside the state lock at admission, or while judging a proposal
  that holds the proof. [Proof availability](#proof-availability) says what
  happens when it cannot.

## Roles

- **Users** escrow payment for a serving job and register a session key with it.
  Owner 0 (embedding, first layers and output head) runs on the user's own device,
  signs everything it sends under that key, and is never bonded.
- **Owners** of every other shard bond NEURO and register the Ed25519 key that
  signs their logs and everything they send on.
- **Auditors** hold one shard, replay an owner's signed log, and challenge it with
  a fraud proof when an output does not reproduce or the log departs from what was
  signed.
- **Validators** order transactions and check the proof of each open challenge,
  replaying it with the challenged shard when the claim needs it.

## Transactions

All transactions are secp256k1-signed envelopes with chain ID and account nonce,
and each burns the fee.

| Kind | Effect |
|---|---|
| `owner_bond` | Locks at least the minimum bond for one shard of the current model and registers a log key. Requires a possession signature by that key over the account, shard, amount and nonce. A key can bond once. |
| `owner_unbond` | Starts withdrawal. Refused while the owner is named in an unsettled job. |
| `owner_withdraw` | Returns the bond once the challenge window has passed since unbonding. |
| `serve_open` | Escrows the price for a budget of token positions and names one active owner per bonded shard, in order, the request digest and the user's Ed25519 session key. The job's ID is the digest of the transaction. |
| `log_commit` | A named owner commits a log bound to the job: its header, the digest of its entries and the statement they produce, signed by its log key over the chain, job and statement. The header must name this chain, job and request and carry its upstream sender's signature over the log's inputs and the positions they carried. When every owner has committed, the challenge window opens. |
| `challenge` | Opens a challenge: names an owner whose log the job has committed, while its challenge window is open, and the content address of a fraud-proof bundle, and locks the challenge deposit. Nothing is replayed. |
| `prove` | Lands the proof of an open challenge before its proof window closes. Validators accept it only if the bundle's log is the committed one and proves one of the claims below. |

## Request-bound logs

Every serving link carries a running transcript: one step per command (reset,
crop, arm switch) and one per message, with the digest of the sent tensor. The
transcript starts from the chain, the job and the link. With every message the
sender signs the transcript's head and the number of token positions the link has
carried so far: the user's device under the job's session key, each owner under
its log key. A receiver checks that signature against the transcript and count it
computed itself before using the message, and keeps the latest one.

An owner's log ends at the last message it was sent; commands after it change no
output. Its header names the chain, job and request and carries that latest
upstream signature. The ledger refuses a commitment whose header names another
job, request or chain, or whose upstream signature is not by the job's session key
(for shard 1) or the previous shard's owner, over this job's link. An internally
correct log from another session, reused as it is or relabelled, is refused at
commitment.

A challenge's bundle holds the committed log and one claim. Each slashes:

- **Replay.** The output at the claimed message is not what the shard computes from
  the logged inputs. Checking it needs the shard and a replay up to that message.
- **Unattested.** The log's entries are not exactly the transcript, and the
  positions, its upstream sender signed. An owner that commits other entries under
  this session's signature is caught by hashing alone.
- **Equivocation.** While passing results on, the owner signed a transcript or a
  position count its log contradicts. The evidence is the owner's own signature,
  held in the next owner's committed header or, for the last owner, by the user's
  device. This catches an owner that sent wrong outputs but commits a log of
  correct computation, and one that signed more positions than it sent.

Honest owners cannot be framed: their logs are exactly what they were sent and
what they signed. A proof does not need the log record's own signature, because
the commitment already signs its statement. An owner therefore cannot escape a
proof by publishing its log with a broken signature, which the earlier checker
allowed.

## Metered settlement

Committed logs show that work was done correctly, not that the request was
finished: the earlier ledger paid the full price to owners who committed only the
first message of a session. A job therefore buys a budget of token positions, the
way a transaction buys gas, and pays for the positions served out of it.

- A position is one token row of a forward or feature message. Commands carry
  none.
- Every link signature covers the positions the link has carried, so each
  commitment states, under its upstream sender's signature, how much work its log
  covers. The ledger records that count when it accepts the commitment.
- At settlement, each owner is paid for its own count, but never for more than the
  user's device sent to owner 1 under the session key, nor more than the budget.
  An owner's share of the price is `price × positions ÷ (budget × owners)`. What is
  not paid, including rounding, returns to the user.

Neither side can take much from the other.

- **Owners are paid for what they received.** A receiver holds its sender's
  signature over every position it was sent, whether or not the session goes on.
  A user that stops sending or withholds a final receipt cannot unpay work already
  sent.
- **Users pay only for what was sent.** An owner that commits a prefix is paid for
  the prefix. An owner cannot claim positions it was not sent: its count must carry
  its upstream sender's signature. An inflated count it signed downstream is
  provable as equivocation, and the ledger caps payment at what the user sent.
- **The bounded loss is what is in flight.** A user's device sends a stream's next
  message only after the previous result returns, so if an owner stops, the user
  has paid for at most one message per stream whose result never returned.

## Challenges and deposits

The earlier ledger replayed a proof when its challenge arrived and refused it if
it failed, so a failed proof cost its sender nothing while every validator paid for
the replay. On the real assistant that replay took 12.3–13.1 s per validator
([measurements](GRANITE_SHARD_CHAIN_RESULTS.md#measurements)), longer than the
3-second proposal timeout. A challenge now takes two transactions.

- **Open.** `challenge` names the owner and the bundle's content address and locks
  a deposit. It passes only cheap checks and replays nothing. While a job has an
  open challenge it neither settles nor expires.
- **Replay.** After each block, validators replay the proofs of open challenges in
  the background, one at a time, and cache the verdicts. Only challenges whose
  deposits are locked on chain are replayed.
- **Prove.** `prove` lands the proof within the proof window, `proof_blocks` after
  the opening, and only in a later block than the challenge. A proof in the
  challenge's own block is refused before any replay, because its deposit is not yet
  committed: a proposer could otherwise make every validator replay a proof whose
  deposit would never be locked. A proof that verifies slashes the owner and returns the deposit with
  the challenger's reward; the deposits of the job's other open challenges return
  to their challengers. A proof that does not verify is refused, and its challenge
  stays open.
- **Lapse.** A challenge not proven within its window closes, and its deposit is
  burned. The job then settles or expires as usual.

Every replay a validator runs is therefore paid for by a locked deposit, which the
challenger forfeits unless the proof verifies. Forfeited deposits are burned rather
than paid to the owner, so an owner colluding with a challenger cannot buy replays
for nothing. A job may have any number of open challenges: a cap would let an
attacker fill it and crowd out a real proof. Block size and the deposit bound them
instead. An unproven challenge delays an honest owner's payment by at most one
proof window after the challenge window.

## Proof availability

Holding a bundle is local to each validator and never a ledger outcome. A
validator holds a bundle when its store has all three files and they hash to the
challenged content address.

- A validator that does not hold a challenged bundle, or cannot replay it (it lacks
  the shard, or memory), gives no verdict. It still admits the challenge, which
  needs no bundle. It refuses the proof at mempool admission with a retryable code
  (3), leaves it out of its own proposals and rejects proposals containing it. It
  caches nothing and retries after each block, so it judges the proof once the
  bundle arrives.
- Only verdicts computed from bytes matching the content address are cached. A
  malformed bundle at its address fails the same way on every validator: replay
  checks each logged command against what serving could have sent.
- A block commits only with prevotes from more than two thirds of the voting
  power, and an honest validator prevotes for a block only after judging every
  proof in it. Execution therefore applies a committed proof without replaying it
  and needs no bundle. Validators with and without the bundle reach the same
  state, and a validator replaying old blocks needs none of them.
- The challenger delivers its bundle to validators' stores, best before opening
  the challenge so their replays start at once. A proof commits only once
  validators with more than two thirds of the voting power hold its bundle and have
  judged it, so delivery and replay time count against the proof window.

## Admission order

Mempool admission and every block check run the same checks in order: the
envelope's signature, the schema, the chain, the account nonce and the fee
balance. A challenge must then name a committed log in an open window and a
canonical proof address, and its sender must hold the deposit. A proof must name
an open challenge. Only a proof that passes all of them is judged, and its replay
has usually run already, in the background after the block that opened its
challenge. Admission waits for any unfinished replay outside the state lock. Every
replay runs on one dedicated thread.

## Settlement rules

- **Settled.** At the first block after the challenge window closes in which the
  job has no open challenge, each owner is paid for the positions its log covers
  ([metered](#metered-settlement)) and the rest of the price is refunded to the
  user.
- **Fraud.** A verified proof slashes the owner's whole bond: half goes to the
  challenger, with its deposit back, and half is burned. The user is refunded, the
  job's other open challenges return their deposits, and every other unsettled job
  naming that owner is voided and refunded the same way.
- **Forfeited.** A challenge not proven within its proof window closes and its
  deposit is burned.
- **Rejected.** A challenge naming an owner who has not committed, a closed window
  or a deposit its sender lacks leaves the state unchanged, as does a proof that
  does not verify or names no open challenge. A validator lacking the bundle gives
  no verdict, which is not a rejection.
- **Expired.** If the job's owners have not all committed by the job deadline,
  the user is refunded once the job has no open challenge.
- **Conservation.** Initial supply equals liquid balances plus owner bonds plus
  job escrow and challenge deposits plus burned fees and penalties, checked after
  every transition.

Parameters: fee 0.001 NEURO, minimum owner bond 5 NEURO, minimum price 0.1 NEURO,
a 20-block challenge window, 60 blocks to commit, a 50% challenger share of the
slashed bond, a 1 NEURO challenge deposit and a 20-block proof window.

## Evidence

Tests on the small Granite-shaped checkpoints
([ledger tests](../tests/evolution/test_optimistic_serving.py),
[bound-log tests](../tests/evolution/test_bound_owner_logs.py) and the
[end-to-end tests](../tests/evolution/test_granite_serving.py)):

- Three owners serve through signed links an honest pass and a pass in which
  owner 1 flips one bit, reproducing the single-host tokens exactly when honest.
  Their logs are bound to their jobs and their keys are bonded on the ledger. An
  auditor saves a real fraud proof against owner 1.
- Two separate validator processes, each holding shard 1, replay the same blocks
  and reach the same state root. Owner 1's honest log offered for the cheating
  job is refused at commitment.
- A challenge naming a forged proof, claiming the honest owner's first output was
  wrong, opens with its deposit. The proof is refused, the challenge lapses and its
  deposit is burned, and the honest job then settles.
- The real proof is accepted: owner 1 is slashed, the auditor receives half the
  bond and its deposit back, and the user is refunded. The honest job pays both
  owners for the positions served, refunding the rest of the budget.
- Owners that serve a whole session but commit only the prefix their first
  upstream signature covers (4 of 14 positions) are paid 4/14 of their share, and
  the user gets the rest back. An owner that signed a count its log contradicts
  is proven by its own signature, and a receiver refuses a message whose signed
  count is not what it was sent. Ledger tests cover the caps at the user's count
  and the budget, and refund of rounding.
- Old entries committed under a new session's signature are proven unattested
  without the shard. An owner that sent a wrong output and committed a corrected
  log is proven by its own signature, for a middle owner and for the last owner.
  Missing, incomplete or altered bundles give no verdict; malformed ones fail.
- Admission refuses challenges and proofs with a bad signature, schema, chain,
  nonce, fee, job, log, proof address or challenge, and challenges without their
  deposit, before any replay. Opening a challenge replays nothing, and validators
  replay only challenges whose deposits are locked on chain. Unit tests also cover
  possession proofs, bond exposure while named or within the window, commitment
  signatures, voiding, late, failed and lapsed challenges, deposits returned when
  another challenge proves fraud, expiry, conservation and deterministic replay.
- Four local CometBFT validators, each holding shard 1 of the small checkpoint and
  its own bundle store, settled the scenario through consensus. The framing
  challenge opened, its proof was refused at admission, and once the challenge
  lapsed the honest job settled. The real proof reached three validators. The
  fourth admitted its challenge, which needs no bundle, but refused the proof with
  no verdict, and the proof committed through the other three. All four reached
  the same state root with exactly the expected balances.
- With the earlier protocol, [on the real 3B assistant](GRANITE_SHARD_SETTLEMENT_RESULTS.md),
  every party signing on its own host, the same sequence passed all eight declared
  checks: two validators reached the same state root, and the final balances
  matched the declared expectation exactly.
- With the earlier protocol, [through CometBFT consensus](GRANITE_SHARD_CHAIN_RESULTS.md),
  four validator hosts holding only shard 1 admitted the same transactions to their
  mempools and agreed on every block. The honest job settled at height 3130, the
  framing was refused at admission, and the real proof slashed owner 1 at height
  3154.

## Limits

- **Validators hold every audited shard.** A validator missing the challenged
  shard cannot judge a replay claim and gives no verdict. Replaying the shard that
  carries the learned arm also needs the arm, which validator configuration does
  not load yet; the published runs audited shard 1 only. Committees of shard
  holders are future work.
- **Proof delivery.** Bundles reach validators' stores out of band; validators do
  not fetch them from one another or from challengers. A proof commits only if
  validators with more than two thirds of the voting power hold its bundle within
  its proof window.
- **Commit trust.** Execution trusts that the validators that prevoted for a
  committed block, more than two thirds of the voting power, judged every proof in
  it. A validator whose own verdict disagrees logs it but follows the chain. Safety
  rests on less than one third of the voting power being faulty, and on the
  execution class below.
- **Challenge cost.** Every replay is paid for by a deposit forfeited unless the
  proof verifies, but the 1 NEURO deposit is not calibrated against what a replay
  costs. Forfeited deposits are burned: they neither pay validators for replays nor
  compensate an honest owner, whose payment an unproven challenge delays by up to
  one proof window.
- **Replay capacity.** Each validator replays one proof at a time, so challenges
  with slow but well-formed bundles queue ahead of a real proof. The deposit must
  cost more than delaying a real proof past its windows could gain, and the windows
  must outlast the backlog it can buy, at about 13 s per real replay. Today only
  bundles in a validator's own store are replayed; accepting bundles from anyone
  needs this settled first.
- **Job checks before serving.** An owner should confirm that the job is open,
  names it and registers the session key it was given. The serving code takes the
  session from its job file.
- **Positions, not answers.** Metering pays for work carried, not for a finished
  or useful answer, and prices a prefill position like a decoded one. A session's
  positions are not known when the job opens, so the user sets the budget and
  receives the unused part back.
- **Withheld commitments.** An owner that never commits leaves its job to expire
  and refund the user. It goes unpaid but unpunished, since no challenge can name a
  log that was never committed, and the owners that did commit go unpaid too.
- **Clean audits earn nothing.** Auditors are paid only for proven fraud. A fee
  for audits that find nothing would need a way to show the replay was actually
  done.
- **Execution class.** Bit-exact replay requires validators and owners to share
  the pinned runtime and CPU instruction class.
- **Not yet on a public chain.** It has run on four CometBFT hosts under one
  operator, and no independent operator has run it. The revised protocol has not
  run on the real model. The published plans declare no position budget or
  challenge deposit, so a new run needs a new plan.
