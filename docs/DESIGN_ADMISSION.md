# Design: explicit admission across shards, bounded waiting, safe retry

Status: **proposal for review (October 20, 2026). Not implemented.** It targets the release after `pilot-1`;
the frozen release and the running soak are unchanged.

## The problem, in today's code

- A client opens and **funds** a job (`serve_open`, escrow of the price) before it contacts any owner.
- It then dials each owner in turn. An owner that is already serving does `serving_lock.acquire(blocking=False)`
  and **closes the socket with no reply**. The client cannot tell busy from crashed, or from a network failure.
- An owner that is free calls `serve_authenticated`, which **adds the job to `served_jobs` before it sends
  READY**. `job_of` then refuses that job forever. So a retry of the same funded job to the same owner is
  refused, even if READY was lost or the client never used the link.
- Partial admission: if shard 1 admits and shard 2 is busy, shard 1 has marked the job served and is
  holding its single serving lock for a client that will never send work. It holds that lock until its peer
  timeout.
- Measured effect in the soak: in every two-request burst, one request failed with "the peer closed the
  serving link", and its job expired an hour later with a full refund. No money was lost, but the user got
  no answer and no explanation.

## Design

### 1. Reserve every shard before funding anything

A new **reservation** step, before `serve_open`, takes no escrow and changes no ledger state:

```
client → owner:  RESERVE {format, chain_id, model_root, rank, client_key, reservation_id, until, signature}
owner  → client: RESERVED {reservation_id, rank, lease_until, owner_signature}
               | BUSY {reservation_id, rank, retry_after_ms, queue_position?}
               | REFUSED {reservation_id, rank, reason: model|chain|version|limit|auth}
```

- `reservation_id` is random, chosen by the client, and signed with an ephemeral `client_key`. The signature
  over `chain_id`, `model_root`, `rank` and `until` lets an owner reject replays and wrong-model requests.
- `RESERVED` holds the owner's serving slot for a short **lease** (proposed 20 s, never beyond `until`). The
  owner refuses other `RESERVE`s and HELLOs while the lease is live, answering `BUSY` with a `retry_after`
  hint. When the lease expires unclaimed, the slot is free again. Nothing is marked served.
- The client reserves **every** shard at once. If any shard answers `BUSY` or `REFUSED`, it sends
  `RELEASE {reservation_id}` to the shards it did reserve, then backs off. It never holds one shard while
  waiting on another.
- Only when every shard has answered `RESERVED` does the client submit `serve_open`. It then HELLOs each
  owner with the `job_id` and the `reservation_id`. An owner admits a HELLO only for the job and client that
  hold its lease.

**Effect:** a busy network costs the client a few small messages, not a funded job. No escrow is opened that
cannot be served.

### 2. An owner marks a job served when work starts, not at READY

- At HELLO, the owner records the job as **admitted** (in memory and in its log directory), sends READY, and
  waits for the first work frame.
- The job becomes **served** (and so un-retryable) only when the owner receives and executes the first
  signed work frame for it. That is the moment its log has content to commit.
- If READY is lost or the client drops before the first frame, the job is admitted but not served. The same
  client may HELLO the same `job_id` again under a fresh reservation, and the owner re-admits it. Its log
  starts empty both times, so nothing is replayed or double-counted.
- Once served, the job is never re-admitted; a reconnect after work began fails explicitly with
  `REFUSED {reason: served}`. Positions already carried stay billable, exactly as today: an owner is paid
  for what its log covers.

### 3. Bounded waiting

- The client waits with jittered exponential backoff (proposed 1, 2, 4, 8 s ± 25%), honouring each
  `retry_after`. It stops at `min(user deadline, a cap of 120 s)`.
- On giving up, it has funded nothing and tells the user "every owner of shard N is busy".
- Owners keep **no unauthenticated queue**. `queue_position` is advisory only. Per-address and global limits
  on `RESERVE` stay as today (`MAX_PER_ADDRESS`, `MAX_HANDSHAKES`), plus a per-`client_key` rate limit.
  Fairness comes from the client's jitter, not from owner state.
- When several owners serve one shard, the client tries them in random order before backing off. A free
  second owner is used at once, as happened in the soak during the owner outage.

### 4. Failure cases

| Case | Outcome |
| --- | --- |
| Shard 2 busy after shard 1 reserved | Client releases shard 1 (or its lease expires), backs off, and has funded nothing |
| Lost RESERVED reply | The lease expires unclaimed; the client's retry uses a new `reservation_id` |
| Lost READY | Job admitted, not served; the client re-HELLOs the same job under a new reservation |
| Client crashes after funding, before work | Job expires and is refunded in full (today's rule); the lease frees the owner within 20 s |
| Disconnect after work began | The job is served; positions carried are paid; no retry of that job; a new job is needed |
| Owner restarts mid-lease | The lease is lost and the client sees a refused connection, so it backs off and re-reserves |
| Old client, new owner | No RESERVE is sent, so the owner answers HELLO as today, sending BUSY instead of a silent close when locked |
| New client, old owner | RESERVE gets no RESERVED reply, so the client falls back to direct HELLO (today's behaviour) |
| `serve_open` lands but its reply is lost | The client reads its account's open jobs (as the soak harness does) before funding again: never two jobs for one turn |

### 5. Payment conservation

Nothing in the ledger changes. Reservations are off-chain. A job is funded only after every shard is
reserved. An unserved job expires and is refunded exactly as today. A served job pays per committed
position as today. The client's "lost `serve_open` reply" rule prevents duplicate funding.

## Compatibility and rollout

- This is a relay protocol change, `relay.FORMAT` bumped to a new version. Owners speak both versions
  during transition, and clients fall back as in the table above.
- It needs no consensus change and no activation height.

## Acceptance tests

1. Two clients at once with one owner per shard: one is served, the other waits and is served after, and no
   job is funded for the waiting client until its reservation succeeds.
2. Shard 1 free and shard 2 busy: shard 1 is released within one message, or by its lease, and nothing is
   funded.
3. READY dropped by a test proxy: the client re-HELLOs the same job and is served; the owner's log and
   payment are as for one clean session.
4. Disconnect after the first work frame: no re-admission; the positions carried are paid.
5. The client's give-up deadline passes: nothing is funded, and the message names the busy shard.
6. Lost `serve_open` reply: no second job.
7. Old/new client × old/new owner matrix: every pairing either serves or fails with an explicit reason.
8. The soak's burst workload rerun: no "peer closed" failures; each request is either served or reports
   busy before funding. Conservation holds after every transition.
