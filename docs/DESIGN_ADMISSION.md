# Design: admission across shards, bounded waiting, safe retry (revision 2)

Status: **proposal for review, revised after the October 21 design review. Not implemented.** Targets the release
after `pilot-1`; the frozen release and the running soak are unchanged.

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

## 1. Exactly-once funding (A2)

The job ID is the funding transaction's own ID: `transaction_id(envelope)`, a digest of the signed body,
including its nonce. Account nonces are strictly ordered, and a transaction with a used nonce can never be
included. These two facts give exactly-once funding without reading open jobs.

**Before broadcasting**, the client writes a durable **operation record** for the turn. It lives under the
client home, is written atomically, and is never deleted before resolution. It holds:
- `turn_id`, a random value chosen by the client;
- the exact signed `serve_open` envelope, and its `nonce`;
- `job_id = transaction_id(envelope)`;
- `request_root`, the session key, the selected owners, and the reservation IDs;
- `state ∈ {prepared, broadcast, funded, unfunded, served, abandoned}`.

**One funding operation in flight per account.** A second turn on the same account waits until the first
resolves, so concurrent turns never share a nonce or guess at each other's jobs.

**Resolution** uses only the persisted envelope:
1. Query the transaction by hash (CometBFT `tx`, which `Chain.submit` already uses). If it is included, it is
   `funded` or failed with its code. This holds even after the job is terminal and evicted from recent results,
   **provided validators keep their transaction index**. They do today, and `Chain.submit` depends on it. The
   design makes that index a stated requirement of the validator role.
2. If it is not included, read the account's nonce:
   - if the nonce is still ≤ the envelope's, the transaction may be pending, so re-broadcast **the same
     envelope**, which is idempotent;
   - if the nonce is now greater and step 1 found nothing, the nonce went to another transaction, so this
     envelope can never be included and the turn is `unfunded`. Under the one-in-flight rule this happens only
     if the user signed something else from that account, and it is recorded.
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

**If a lease is lost after funding** (it expired, the owner restarted, or confirmation outlasted it), the job
is funded and keeps its identity. The client then:
- re-RESERVEs that owner for **that `job_id`**, or tries another owner of the same shard;
- retries with backoff until the job's own expiry, or the user's deadline if earlier;
- if it never gets admission, lets the job expire, which refunds the escrow minus the transaction fee.

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

**Chosen rule: only funded accounts may reserve, within an unproductive-time budget.**
- RESERVE is signed by an account key, and the owner checks on chain that the account's balance covers the
  price plus fees. Every reservation is tied to scarce funds, not a free key.
- Each owner keeps a budget per account and per source address. Unproductive lease time (offered or funding
  that ended without HELLO) is limited to 60 s in any 10 minutes. Beyond that, RESERVE gets `REFUSED limit`
  until the budget recovers. A lease that leads to service costs nothing from the budget.
- `offered` leases are not renewable. Only the `funding` state extends a lease, and it requires a signed HOLD
  naming a job, which the owner can later check on chain.
- An owner admits at most one live lease per slot. Leases are cheap to refuse, and the owner keeps no queue.

**What this does not stop:** an adversary with many funded accounts. On the test network the faucet's
per-address rate limit bounds this. A value-bearing network makes it costly, since every account must hold
the job's price. A refundable reservation deposit on chain would be stronger but adds a transaction per
attempt. That option is recorded here, not chosen.

**Test:** an adversary rotating many funded accounts and new keys, reserving and abandoning repeatedly. It
must measure honest clients' admission rate against the budget.

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
| Confirmation arrives after the leases expired | Job funded, so re-reserve for that `job_id` until its expiry; else refund at expiry |
| Confirmation lost | Resolve from the persisted envelope (§1); never a second job |
| Client restarts after broadcast | Reload the operation and resolve it before any new turn |
| Owner restarts after funding, before HELLO | Lease lost, so re-reserve that owner or another for the same job |
| READY lost | Re-HELLO the same job, gen+1; the old socket is closed first |
| Old socket delivers work after reconnect | Discarded: wrong generation |
| Owner crashes around the first frame | Before the write-ahead record: re-admit. After it: `recovered-started`, so commit, settle or expire |
| Deadline passes with the funding outcome unknown | Report unknown and keep resolving; never fund again for the turn |
| Two turns on one account | Serialized: the second waits for the first's resolution |

## Acceptance tests

1. **Two clients, one owner per shard, deadline ≥ 600 s:** the second is served after the first, or reports
   busy before funding. No job is funded while it waits.
2. **Partial busy:** shard 1 offered and shard 2 busy. Shard 1 is released, and nothing is funded.
3. **Confirmation after lease expiry:** the same job is admitted later, or refunded at expiry. Never two jobs.
4. **Lost confirmation, then a client restart:** resolution from the persisted envelope gives one job, or none.
5. **Lost READY:** a re-HELLO at gen+1 is served. A frame injected on the gen-g socket is discarded.
6. **Owner crash just before and just after the write-ahead record:** respectively re-admitted, or
   recovered-started and settled.
7. **Two concurrent turns on one account:** serialized, and nonces never conflict.
8. **The rotating-account reservation adversary:** honest admission rate and owner budget enforcement are
   measured.
9. **Capability matrix**, with a dropped CAPS reply: unknown is never treated as legacy.
10. **The soak burst workload rerun with this protocol:** every request is either served or reports busy before
    funding, and the ledger's conservation invariant holds after every transition.
