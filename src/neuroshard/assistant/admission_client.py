"""The client's side of admission (DESIGN_ADMISSION.md §2, §5): reserve every shard, then fund, then HELLO.

1. CAPS: owners that advertise ``admission/1`` are selectable; silence is *unknown*, never legacy.
2. RESERVE one capable owner per shard, in parallel. A busy shard is offered to another owner of that shard;
   if a shard still has none, every offer is released and the client backs off. Nothing is funded.
3. HOLD every shard with the *unsigned* ``serve_open`` (its body and signer: enough to derive the job ID, not to
   broadcast it). The signed envelope is persisted first (``funding.Operations``), and only this client broadcasts
   it, after every HOLD succeeded; it is resolved from that record (``funding.fund``).
4. HELLO every shard with a fresh, session-signed admission attempt bound to the job, rank, owner and reservation;
   the owner's READY echoes it, signed. If a lease was lost after funding, the *same* named owners are re-reserved
   for that job until the job expires or the user's deadline passes; its escrow is refunded at expiry. A turn
   never opens a second job.

Legacy mode (fund first, as before) is used only when the caller asks for it, and it is reported.
"""
import random
import threading
import time

from neuroshard.assistant import admission_messages as messages
from neuroshard.assistant import funding
from neuroshard.inference import optimistic as ledger
from neuroshard.inference import relay

DEFAULT_WAIT = 600
SCHEDULE = (1, 2, 4, 8, 15)


class NetworkBusy(RuntimeError):
    """Every attempt found a shard busy before funding; nothing was funded."""


class NotServed(RuntimeError):
    """A funded job could not be admitted by its named owners before the deadline; expiry refunds its escrow."""

    def __init__(self, job_id, expires, detail):
        super().__init__(f'job {job_id[:16]} is funded but was not served ({detail}); '
                         f'its full escrow is refunded at block {expires}')
        self.job_id, self.expires = job_id, expires


class Backoff:
    """Jittered exponential waits (1, 2, 4, 8, then 15 s, each ±25%), at least ``retry_after``, within a deadline."""

    def __init__(self, deadline, rng=None, clock=time.monotonic, sleep=time.sleep):
        self.deadline, self.rng, self.clock, self.sleep, self.attempt = deadline, rng or random.Random(), clock, sleep, 0

    def wait(self, retry_after=0):
        base = SCHEDULE[min(self.attempt, len(SCHEDULE) - 1)]
        self.attempt += 1
        delay = max(base * self.rng.uniform(0.75, 1.25), retry_after)
        if self.clock() + delay > self.deadline:
            return False
        self.sleep(delay)
        return True


def parallel(calls):
    results = [None] * len(calls)

    def run(index, call):
        try:
            results[index] = ('ok', call())
        except Exception as error:  # noqa: BLE001 - each failure is reported per shard
            results[index] = ('error', error)

    threads = [threading.Thread(target=run, args=(i, call), daemon=True) for i, call in enumerate(calls)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


class Admitter:
    def __init__(self, chain, account, served, home, price, rng=None, clock=time.monotonic, sleep=time.sleep,
                 request=messages.request, dial=relay.dial, index=None, progress=print, wall=time.time):
        from neuroshard.inference import optimistic_network as chain_network

        self.chain, self.account, self.served, self.price = chain, account, served, price
        self.operations = funding.Operations(home)
        self.index = index or funding.LedgerIndex(chain, chain_network.rpc)
        self.rng, self.clock, self.sleep, self.wall = rng or random.Random(), clock, sleep, wall
        self.request, self.dial, self.progress = request, dial, progress
        self.capable = {}  # owner key -> True once its signed CAPS advertised admission/1

    # -- requests ----------------------------------------------------------------------------------------------
    def ask(self, owner, endpoint, message):
        return messages.answer(self.request(endpoint, message), owner, self.chain.chain_id, self.wall)

    def caps(self, owner, endpoint):
        if owner not in self.capable:
            probe = {'body': {'domain': messages.DOMAIN, 'kind': 'caps', 'chain_id': self.chain.chain_id,
                              'expires': int(self.wall()) + messages.LIFETIME,
                              'challenge': ledger.digest({'caps': self.rng.random()})[:32]}, 'signature': ''}
            reply = self.ask(owner, endpoint, probe)
            if reply['kind'] != 'caps' or reply.get('challenge') != probe['body']['challenge']:
                raise ValueError('the owner answered another CAPS request')
            self.capable[owner] = messages.VERSION in reply.get('versions', [])
        return self.capable[owner]

    def candidates(self, state, shard):
        options = sorted((key, owner['endpoint']) for key, owner in state['owners'].items()
                         if owner['shard'] == shard and owner['status'] == 'active' and owner.get('endpoint'))
        self.rng.shuffle(options)
        capable = []
        for key, endpoint in options:
            try:
                if self.caps(key, endpoint):
                    capable.append((key, endpoint))
            except (OSError, ValueError, KeyError):
                continue  # unknown availability: asked again on the next attempt, never taken as legacy
        return capable

    def reserve_shards(self, state, session_public, deadline):
        """One offered owner per shard, or None after releasing everything (the network is busy)."""
        offers, waits, missing = {}, [0], []
        for shard in range(1, self.served.world):
            for owner, endpoint in self.candidates(state, shard):
                message = messages.by_account(self.account, self.chain.chain_id, 'reserve', self.wall, rank=shard,
                                              owner=owner, session_key=session_public, price=self.price)
                try:
                    reply = self.ask(owner, endpoint, message)
                except (OSError, ValueError, KeyError):
                    continue
                if reply['kind'] == 'offer':
                    offers[shard] = (owner, endpoint, reply['reservation'])
                    break
                if reply['kind'] == 'busy':
                    waits.append(int(reply.get('retry_after', 0)))
            else:
                missing.append(shard)
        if missing:
            self.release(offers)
            return None, max(waits), missing
        return offers, 0, []

    def release(self, offers, job_id=None, session_key=None):
        calls = []
        for shard, (owner, endpoint, reservation) in offers.items():
            if session_key is None:
                message = messages.by_account(self.account, self.chain.chain_id, 'release', self.wall, rank=shard,
                                              owner=owner, reservation=reservation)
            else:
                message = messages.by_key(session_key, 'release', self.chain.chain_id, self.wall, rank=shard,
                                          owner=owner, reservation=reservation, job_id=job_id)
            calls.append(lambda o=owner, e=endpoint, m=message: self.ask(o, e, m))
        parallel(calls)

    # -- one turn ----------------------------------------------------------------------------------------------
    def recover(self):
        """Resolve earlier turns first; a funded one that was not served is reported, then refunded at expiry."""
        for record in funding.recover(self.index, self.operations, self.account.public, sleep=self.sleep,
                                      clock=self.clock, supersede=self.supersede):
            if record['state'] == funding.FUNDED:
                self.operations.mark(record, funding.ABANDONED)
                self.progress(f"An earlier turn's job {record['job_id'][:16]} was funded but not served; "
                              f'its escrow is refunded when it expires')

    def open(self, request_root, session_key, budget, hello, wait=DEFAULT_WAIT):
        """Reserve, fund and admit one job: (job_id, owners [(key, endpoint)], sockets {rank: socket})."""
        session_public = session_key.public_key().public_bytes_raw().hex()
        deadline = self.clock() + wait
        backoff = Backoff(deadline, self.rng, self.clock, self.sleep)
        with self.operations.exclusive(self.account.public):
            self.recover()
            while True:
                state = self.chain.state()
                if state['model_root'] != self.served.root:
                    raise ValueError('the ledger serves another version than this device holds')
                offers, retry_after, missing = self.reserve_shards(state, session_public, deadline)
                if offers is not None:
                    record = self.held(offers, state, request_root, session_public, budget)
                    if record is not None:
                        break
                else:
                    self.progress(f"Shard{'s' if len(missing) > 1 else ''} {', '.join(map(str, missing))} busy; "
                                  'nothing funded, waiting')
                if not backoff.wait(retry_after):
                    raise NetworkBusy('the network stayed busy until the deadline; nothing was funded')
            record = funding.fund(self.index, self.operations, record, sleep=self.sleep, clock=self.clock,
                                  supersede=self.supersede)
            if record['state'] != funding.FUNDED:
                raise RuntimeError(f"the job was not funded ({record['state']}: {record.get('log')})")
        owners = [(offers[s][0], offers[s][1]) for s in sorted(offers)]
        sockets = self.admit(record, offers, session_key, hello, deadline)
        self.operations.mark(record, funding.SERVED)
        return record['job_id'], owners, sockets

    def held(self, offers, state, request_root, session_public, budget):
        """Persist the exact envelope, then HOLD every shard with its *unsigned* transaction; None on a refusal.

        An owner learns the job ID and the fields it checks (signer, owners, session key) but never the signature,
        so it cannot broadcast the job's funding. Only the client broadcasts it, after every HOLD succeeded. A
        refused or lost HOLD therefore leaves nothing to fund; that is still checked against verified state (the
        nonce is unconsumed) before the record is abandoned, and any other answer resolves the same operation."""
        since = self.index.height()
        envelope = self.account.sign(self.chain.chain_id, self.index.account_nonce(self.account.public), 'serve_open',
                                     model_root=self.served.root, owners=[offers[s][0] for s in sorted(offers)],
                                     request_root=request_root, session_key=session_public, price=self.price,
                                     positions=budget)
        record = self.operations.prepare(envelope, session_public, request_root, [offers[s][0] for s in sorted(offers)],
                                         {str(s): offers[s][2] for s in offers})
        record = self.operations.mark(record, funding.PREPARED, since_height=since)
        transaction = messages.unsigned(envelope)
        calls = [lambda s=s, o=o, e=e, r=r: self.ask(o, e, messages.by_account(
            self.account, self.chain.chain_id, 'hold', self.wall, rank=s, owner=o, reservation=r,
            transaction=transaction))
            for s, (o, e, r) in sorted(offers.items())]
        results = parallel(calls)
        if all(status == 'ok' and reply['kind'] == 'held' and reply.get('job_id') == record['job_id']
               for status, reply in results):
            return record
        self.release(offers)
        try:
            unconsumed = self.index.account_nonce(self.account.public) <= record['nonce']
        except (OSError, ValueError, KeyError):
            unconsumed = False
        if not unconsumed:  # cannot show the envelope is dead: resolve this same operation, never fund another
            self.progress('A shard did not confirm its hold and the account nonce moved; resolving this turn')
            return record
        self.operations.mark(record, funding.ABANDONED,
                             log='a shard refused the hold; the signed envelope never left this device')
        return None

    def supersede(self, nonce):
        """Consume ``nonce`` with a 1-unit self-transfer, so a reportedly failed funding becomes provably dead."""
        envelope = self.account.sign(self.chain.chain_id, nonce, 'transfer', to=self.account.public, amount=1)
        admitted = self.index.rebroadcast(envelope)
        if admitted.get('code'):
            raise ValueError(admitted.get('log') or 'the superseding transfer was refused')

    def admit(self, record, offers, session_key, hello, deadline):
        """READY from every named owner. After a lost lease, the *same* owners are re-reserved for this job, all
        of them before any HELLO; a round that misses a shard releases the rest, so no owner idles for it."""
        job_id = record['job_id']
        backoff = Backoff(deadline, self.rng, self.clock, self.sleep)
        first = True
        while True:
            retry_after, missing = 0, []
            if not first:
                retry_after, missing = self.rereserve_all(job_id, offers, session_key)
            sockets = {}
            if not missing:
                for shard in sorted(offers):
                    owner, endpoint, reservation = offers[shard]
                    try:
                        sockets[shard] = self.dial(endpoint, hello(job_id, shard), session_key,
                                                   **self.attempt(session_key, job_id, shard, owner, reservation))
                    except relay.OwnerBusy as busy:
                        body = messages.answer(busy.reply, owner, self.chain.chain_id, self.wall)
                        if body['kind'] == 'refused' and body.get('reason') == 'started':
                            self.hang_up(sockets)
                            raise NotServed(job_id, self.expiry(job_id), f'shard {shard} already started it')
                        retry_after = max(retry_after, int(body.get('retry_after', 0)))
                        missing.append(shard)
                    except (OSError, ValueError):
                        missing.append(shard)
                if not missing:
                    return sockets
            self.hang_up(sockets)
            self.release(offers, job_id, session_key)
            first = False
            expires = self.expiry(job_id)
            if expires is None:
                raise NotServed(job_id, '(already expired)', 'the job is no longer open')
            self.progress(f"Funded, waiting for shard{'s' if len(missing) > 1 else ''} {', '.join(map(str, missing))}")
            if not backoff.wait(retry_after):
                raise NotServed(job_id, expires, 'the deadline passed')

    def attempt(self, session_key, job_id, shard, owner, reservation):
        """A fresh admission attempt for one HELLO, and the check that the owner's READY answers *this* attempt."""
        admission = messages.by_key(session_key, 'hello', self.chain.chain_id, self.wall, rank=shard, owner=owner,
                                    job_id=job_id, reservation=reservation)

        def accept_ready(message):
            body = messages.answer(message, owner, self.chain.chain_id, self.wall, kinds=('ready',))
            if (body.get('attempt') != admission['body']['challenge'] or body.get('job_id') != job_id
                    or body.get('rank') != shard):
                raise ValueError('the READY answers another admission attempt')
            return body

        return {'admission': admission, 'accept_ready': accept_ready}

    @staticmethod
    def hang_up(sockets):
        for sock in sockets.values():
            sock.close()

    def rereserve_all(self, job_id, offers, session_key):
        """Re-reserve each named owner for the funded job (session-signed; no second balance needed)."""
        calls = [lambda s=s, o=o, e=e: self.ask(o, e, messages.by_key(
            session_key, 'reserve', self.chain.chain_id, self.wall, rank=s, owner=o, job_id=job_id))
            for s, (o, e, _) in sorted(offers.items())]
        retry_after, missing = 0, []
        for (shard, (owner, endpoint, _)), (status, reply) in zip(sorted(offers.items()), parallel(calls)):
            if status == 'ok' and reply['kind'] == 'offer':
                offers[shard] = (owner, endpoint, reply['reservation'])
                continue
            missing.append(shard)
            if status == 'ok' and reply['kind'] == 'busy':
                retry_after = max(retry_after, int(reply.get('retry_after', 0)))
            elif status == 'ok' and reply.get('reason') == 'started':
                raise NotServed(job_id, self.expiry(job_id), f'shard {shard} already started it')
        return retry_after, missing

    def expiry(self, job_id):
        job = self.chain.state()['jobs'].get(job_id)
        return None if job is None else job['expires']
