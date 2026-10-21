# Design: admission across shards, bounded waiting, safe retry (revision 3)

Status: **approved (October 21 review of revision 3.1); implemented in the admission PR, under review.** Targets the
release after `pilot-1`; the frozen release and the running soak are unchanged. Where the implementation departs
from this text, it is listed under [Implementation notes](#implementation-notes).

Revision 3.1 makes clear that escrow does not exempt idle occupancy from the time budget, and that the
funding credit is applied once (test 13).

Revision 3 (after the review of revision 2) corrects five contracts:
- owners may be substituted only before funding;
- an advanced nonce is never taken as proof that funding failed;
- a hold is charged against the budget while it is live;
- pre-funding messages are signed with the account key;
- a funded retry uses its existing escrow, and expiry refunds the full escrow.

Revision 2 addresses five review points:
- leases can lapse while funding confirms (A1);
- funding is resolved from a persisted operation, not by scanning open jobs (A2);
- the owner keeps a durable state machine with connection generations (A3);
- reservations have a resource budget (A4);
- capabilities are negotiated explicitly, never inferred from silence (A5).

The bounded wait is also tied to the real service time (A6). It drops revision 1's two absolute claims, that no
escrow is ever opened without being served and that a client that gives up has funded nothing. An owner
can always fail after funding, so neither is true.

## The problem, in today's code

- **Funding comes first.** A client funds a job (`serve_open`) before contacting any owner.
- **A busy owner gives no reply.** It closes the socket silently (`serving_lock.acquire(blocking=False)`), so
  the client can't tell busy from failure.
- **A job is marked served too early.** A free owner adds the job to `served_jobs` before sending READY, and
  `job_of` refuses that job forever after. So a retry after a lost READY is impossible.
- **Partial admission holds a slot.** If shard 1 admits and shard 2 is busy, shard 1 holds its only serving
  slot until its peer timeout.
- **Measured in the soak:** in every two-request burst, one request failed with "the peer closed the
  serving link". Its job expired about an hour later and was refunded in full.

## Guarantees this design gives, and does not give

| Gives | Does not give |
| --- | --- |
| A busy network is reported as busy before funding in the common case | That a funded job is always served: an owner can crash after funding |
| At most one funded job per user turn, across lost replies and client restarts | Fair or FIFO admission: concurrent clients race, as today |
| A funded job that cannot be served is either served later or refunded by expiry | That refunds are immediate: expiry takes `job_blocks` (3,600 blocks, about an hour) |
| An owner never executes work for a job twice, or from a superseded connection | Protection against an adversary with unlimited funded accounts (A4) |
| A funded job is never reported unfunded | That every funding outcome resolves at once: it can stay `unknown` until the index shows it |
| A replacement owner can be chosen before funding | Moving a funded job to a different owner: the ledger fixes its owner list |

## 1. Exactly-once funding (A2)

The job ID is the funding transaction's own ID: `transaction_id(envelope)`, a digest of the signed body,
including its nonce. Account nonces are strictly ordered, and a transaction with a used nonce can never be
included. These two facts give exactly-once funding without reading open jobs.

**Before broadcasting**, the client writes a durable **operation record** for the turn. It lives under the
client home, is written atomically, and is never deleted before resolution. It holds:
- `turn_id`, a random value chosen by the client;
- the exact signed `serve_open` envelope, and its `nonce`;
- `job_id = transaction_id(envelope)`, the ledger's ID for the job. This is distinct from `tx_hash`, the
  SHA-256 of the canonical envelope that CometBFT indexes. Both are recorded and never confused;
- `request_root`, the session key, the selected owners, and the reservation IDs;
- `state ∈ {prepared, broadcast, funded, unfunded, served, abandoned}`.

**One funding operation in flight per account.** A second turn on the same account waits until the first
resolves, so concurrent turns never share a nonce or guess at each other's jobs.

**Resolution** uses only the persisted envelope:
1. Query the transaction by `tx_hash` (CometBFT `tx`, which `Chain.submit` already uses). If it is included, it is
   `funded` or failed with its code. This holds even after the job is terminal and evicted from recent results,
   **provided validators keep their transaction index**. They do today, and `Chain.submit` depends on it. The
   design makes that index a stated requirement of the validator role.
2. If it is not included, read the account's nonce:
   - if the nonce is still ≤ the envelope's, the transaction may be pending, so re-broadcast **the same
     envelope**, which is idempotent;
   - if the nonce is now greater, the nonce has been consumed, but not necessarily by another transaction. The
     transaction may have committed between the two reads, or the index may lag. So **query `tx_hash` again**.
     If it is now found, the turn is `funded`. If it is still absent, the turn stays **unknown** until a
     different transaction that consumed that nonce is identified: the account's transactions at that nonce,
     from the index, at a height at which `tx_hash` is consistently absent. Only then is the turn `unfunded`.
3. If the transaction index is unavailable or the answer is ambiguous, the state stays **unknown**. The
   client never signs a new `serve_open` for that turn; it shows the user "funding outcome unknown" and
   retries resolution. Absence of evidence is not permission to fund again.

A client restart reloads every unresolved operation and resolves it before starting any new turn.

## 2. Reservations and leases (A1)

**A lease has three states on each owner:**
- `offered`: a short hold, 20 s.
- `funding`: entered when the client sends `HOLD` naming the exact `job_id` it is about to fund. It is
  extended to at most `funding_bound` = 240 s, which exceeds `Chain.submit`'s 180 s confirmation wait plus
  margin. While a `funding` lease is live, the owner may check the chain for the job. It drops the lease once
  the job is visible and HELLO has not arrived within 30 s.
- `admitted`: see §3.

**The client sequence:**
1. RESERVE every selected shard, in parallel.
2. Once all are offered, persist the operation record, send `HOLD` with the `job_id` to all shards, and then
   broadcast.
3. On confirmation, HELLO every shard with the `job_id`, signed with the job's session key.

**If any shard is busy or refused before step 2,** the client RELEASEs the others. Nothing is funded, and it
backs off (§5).

**Owners are chosen only before funding.** The ledger fixes a funded job's ordered owner list, and
`Owner.job_of` refuses a job that names another owner. So a busy or failed owner can be replaced by another
owner of its shard only before `serve_open`. Reassigning a funded job would need a separate ledger, proof and
payment design, which is out of scope.

**If a lease is lost after funding** (it expired, the owner restarted, or confirmation outlasted it), the job
is funded and keeps its identity. The client then:
- re-RESERVEs **the same named owners** for that `job_id`;
- retries with backoff until the job's own expiry, or the user's deadline if earlier;
- if any named owner never admits it, lets the job expire. Expiry returns the **full escrow**; the
  transaction fee was charged separately when the job was opened, and is not deducted again.

A post-funding RESERVE names the `job_id` and is signed by the job's session key. The owner checks on chain
that the job is open, names this owner at this rank, and is the account's own. It does **not** require a
second spendable balance: the job's escrow is its eligibility.

The user is told "funded, waiting for shard N" or "not served, refund at block H". The client never opens a
second job for the turn.

## 3. Owner state machine and connection generations (A3)

**Per job, an owner moves through:**

```
free → offered → funding → admitted(gen g) → started → committed
                    ↓           ↓                ↘ (crash) recovered-started
                 expired     admitted(gen g+1) on a later HELLO, only while not started
```

- **`admitted(gen g)`.** A durable record is written (job, session key, gen g) before READY is sent. A new
  HELLO for the same job while it is `admitted`, signed by the job's session key, increments the generation.
  The owner closes the gen-g socket, discards any frame on it, and only then sends READY on gen g+1. The
  owner accepts work frames only from the socket of the current generation.
- **`started`.** A write-ahead record is fsynced **before** the first work frame is executed. A crash before
  that record leaves the job `admitted`, with no work done, so re-admission is safe. A crash after it means
  the owner may have done some work. On restart it becomes `recovered-started`:
  - the owner never re-admits the job;
  - it commits whatever its log covers, possibly nothing;
  - the client sees the job refused as started and resolves it by its outcome.
- **All shards before work.** The client sends no work frame until it holds READY from every shard in the
  current generation. One shard started while another failed can therefore only happen through a crash
  mid-session, and that is handled by the job's normal settlement or expiry.
- **Every control message** after funding carries `job_id`, `rank`, the owner's log key, `gen`, an expiry and
  a nonce, signed by the job's session key. These are HOLD, HELLO and RELEASE, plus BUSY and READY from the
  owner, signed by its log key. Before funding, RESERVE is signed by the client's **account key** (§4), not an
  ephemeral key.

## 4. A resource budget for reservations (A4)

**The abuse model.** An attacker reserves repeatedly and never funds, to keep owners unavailable. Ephemeral
keys and socket limits do not stop this.

**Chosen rule: only accounts that can pay may reserve, within an unproductive-time budget.**

**Authentication before funding.** There is no job on chain yet, so every pre-funding message (RESERVE, HOLD
and RELEASE) is signed by the client's **account key**. Each message carries:
- the reservation ID;
- the operation's `tx_hash` (from HOLD onward);
- the operation's session key;
- the rank and owner key, an expiry and a nonce.

After funding, the owner checks that the funded job's account, session key and owner list match what was
held. From HELLO onward, messages are signed by the job's session key.

**Eligibility.** For a new job, the owner checks on chain that the account's spendable balance covers the
price plus fees. For an existing funded job, the job's escrow is its eligibility (§2).

**Accounting.** Per account and per source address, the budget is 60 s of unproductive slot time in any 10
minutes.
- **Reservation is charged as it is held.** Every second a lease is `offered`, or `funding` while the job is not
  yet visible on chain, is debited from the budget as it elapses.
- **A hold is bounded by what remains.** Whatever the funding bound, a hold lasts at most the remaining budget.
  An honest client whose confirmation outlasts that allowance loses the lease, then re-reserves its funded job
  (§2) when the budget allows.
- **Escrow is not an exemption.** Existing escrow removes the need for a second spendable balance; it does
  not exempt idle occupancy from the time budget. Reservations, and admitted connections without work,
  consume that budget across releases and reconnects.
- **Funding is credited once.** When the owner first sees the held `tx_hash`'s job on chain naming it, the
  elapsed time of that **original pre-funding hold** is credited back, once per job. Later idle time for the
  same job is never credited: post-funding re-reservations, and admissions without a first work frame. A hold
  for a transaction that was never broadcast, or for an invented job, never appears and stays charged.
- **Reconnects don't reset the clock.** `admitted` without a first work frame is unproductive and charged. A
  reconnect or a release continues the same accounting. Only a first work frame (`started`) ends idle
  occupancy.
- **Refused once exhausted.** When the budget is exhausted, RESERVE gets `REFUSED limit` until it recovers.

**What this does not stop.**
- **Capital isn't locked.** A balance check does not lock funds: the same funds can be moved between accounts
  while earlier holds are live. So the budget bounds each account, not an adversary with many accounts.
- **The faucet bounds account creation** on the test network, through its per-address limit.
- **A refundable on-chain reservation deposit,** the unselected option, would add a cost per attempt and lock
  capital for the hold. It would make a many-account attack more expensive, but would not eliminate it.

**Tests:**
- an adversary rotating many funded accounts and keys, measuring honest clients' admission rate;
- HOLDs naming never-broadcast transactions, which must stay charged;
- a slow-confirmation honest client, which must lose its lease and then be re-admitted for its funded job.

## 5. Capabilities, waiting and compatibility (A5, A6)

**Explicit capability negotiation.** Each owner answers a signed `CAPS` request with its protocol versions,
signed by its log key. The client caches the answer per owner key. Silence or a timeout is **unknown
availability**: the client retries, and it never falls back to legacy behaviour on silence.

**Modes:**
- **Reserved mode** (default): every selected shard must have advertised reservation support. Owners that
  haven't are not selected, so a shard is served only by capable owners.
- **Legacy mode**: used only when some shard has no capable owner at all, and named explicitly to the user.
  It funds before HELLO, so its failures can happen after funding, exactly as today. This mode is reported,
  never silent.

**Old client, new owner.** The old client understands only an exact READY. A BUSY frame makes it fail as it
does today, with a different message. The friendly explanation needs the new client, and this limit is tested.

**Bounded waiting.**
- The client retries with jittered backoff (1, 2, 4, 8 s, then a 15 s cap, each ±25%), honouring `retry_after`.
- It waits until the user's deadline, which defaults to the expected time for the conversation ahead plus one
  conversation, at least 300 s. The soak's turns take 62–253 s.
- Jitter only spreads out retries. It guarantees nothing about fairness.

## Failure cases

| Case | Outcome |
| --- | --- |
| Shard 2 busy after shard 1 offered | RELEASE shard 1, back off; nothing funded |
| Confirmation arrives after the leases expired | Job funded, so re-reserve the same named owners for that `job_id` until its expiry; else full escrow refunded at expiry |
| Confirmation lost | Resolve from the persisted envelope (§1); never a second job |
| Client restarts after broadcast | Reload the operation and resolve it before any new turn |
| Owner restarts after funding, before HELLO | Lease lost, so re-reserve **that** owner for the same job (the ledger fixes the owner list); else expiry and refund |
| READY lost | Re-HELLO the same job, gen+1; the old socket is closed first |
| Old socket delivers work after reconnect | Discarded: wrong generation |
| Owner crashes around the first frame | Before the write-ahead record: re-admit. After it: `recovered-started`, so commit, settle or expire |
| Deadline passes with the funding outcome unknown | Report unknown and keep resolving; never fund again for the turn |
| The funding commits between "not found" and "nonce advanced" | Re-query `tx_hash` and find it: funded, never wrongly reported unfunded |
| Two turns on one account | Serialized: the second waits for the first's resolution |

## Acceptance tests

1. **Two clients, one owner per shard, deadline ≥ 600 s:** the second is served after the first, or reports
   busy before funding. No job is funded while it waits.
2. **Partial busy:** shard 1 offered and shard 2 busy. Shard 1 is released, and nothing is funded.
3. **Confirmation after lease expiry:** the same job is admitted later, or refunded at expiry. Never two jobs.
4. **Lost confirmation, then a client restart; a commit between the "not found" and "nonce advanced" reads;
   delayed indexing:** resolution from the persisted envelope gives one job or none. A funded job is never
   reported unfunded.
5. **Lost READY:** a re-HELLO at gen+1 is served. A frame injected on the gen-g socket is discarded.
6. **Owner crash just before and just after the write-ahead record:** respectively re-admitted, or
   recovered-started and settled.
7. **Two concurrent turns on one account:** serialized, and nonces never conflict.
8. **The reservation adversary:** rotating funded accounts, and HOLDs naming never-broadcast transactions.
   Measured: honest admission rate, and holds that stay charged.
11. **Owner substitution after funding is refused:** a different owner of the same shard rejects the funded job
    (as `Owner.job_of` does today), and the client resolves the job by its named owners or by expiry.
12. **A post-funding re-reservation needs no second balance:** an account holding only the escrowed job is
    re-admitted.
13. **One funded job cannot monopolize its owners:** the same funded job repeatedly RESERVEs and HELLOs
    without sending work. Its total idle occupancy stays within the budget; RESERVE then gets
    `REFUSED limit`, and other clients are admitted. The funding credit is applied once and cannot be replayed
    by releasing and re-reserving.
9. **Capability matrix**, with a dropped CAPS reply: unknown is never treated as legacy.
10. **The soak burst workload rerun with this protocol:** every request is either served or reports busy before
    funding, and the ledger's conservation invariant holds after every transition.

## Implementation notes

Code: `assistant/funding.py` (§1), `assistant/admission.py` (the owner's gate: §2–§4),
`assistant/admission_messages.py` (signed control messages), `assistant/admission_client.py` (the client sequence,
§2 and §5), wired into `Owner` and `Conversation` in `assistant/network.py`. Tests: `test_admission.py` and
`test_admission_recovery.py`, named by acceptance-test number, run the real ledger transition and the real owner
admission and serving handshake, with the model work stubbed out.

Where the implementation differs from the text above:

1. **READY keeps its exact legacy form.** An old client accepts only an exact READY, so READY carries no generation
   and no signature. Generations are enforced on the owner, which closes (shuts down) the superseded connection
   before sending READY on the new one, and accepts work only on the current generation's connection. The BUSY and
   REFUSED replies to a HELLO are signed by the log key.
2. **Legacy mode is explicit.** Silence is never taken as "legacy", and nothing else proves an owner is legacy, so
   legacy mode runs only when the user asks for it (`neuroshard assistant chat --legacy-admission`) and is reported.
   In the default reserved mode, a shard with no capable owner stays busy until the deadline, and nothing is funded.
3. **The nonce's consumer is found in committed blocks.** The ledger emits no events to search an account's
   transactions by nonce, so resolution scans the blocks from the height recorded before broadcast, and counts only a
   transaction that executed (a failed one leaves the nonce unchanged). If that scan finds the turn's own
   transaction while the index lags, the turn is `funded`.
4. **An earlier turn that was funded but not served is left to expire.** A restarted client resolves it, reports it,
   and marks it `abandoned`; its escrow is refunded in full at expiry. The session key is not persisted, so the job
   is not resumed.
5. **A lapsed hold is still credited once.** If a hold lapses before its job appears on chain (confirmation
   outlasted the budget), the owner remembers the hold for one budget window and credits it, once, when the funded
   job is first seen naming it.
6. **A charge stops at its lease's bound,** not when the owner next notices the lease has lapsed.
7. **Test 10** is run locally (four concurrent clients, one owner per shard): each is served or reports busy with
   nothing funded, and the conservation invariant is asserted on every block. The rerun of the soak's burst workload
   on the live network still has to be done with a release that includes this change.
