"""An owner's admission gate: leases, generations, and a budget of unproductive slot time (DESIGN_ADMISSION.md §2–§4).

An owner has one serving slot. A lease on it moves through ``offered`` (20 s) → ``funding`` (at most 240 s, after
HOLD names the exact signed ``serve_open``) → ``admitted(gen)`` (after a HELLO, recorded durably before READY) →
``started`` (a write-ahead record, fsynced before the first work frame is executed).

Unproductive occupancy is charged per account and per source address, 60 s in any 10 minutes: offered time, funding
time before the job is visible on chain, and admitted time without a first work frame. A hold never outlasts what
remains of that budget. When the held job first appears on chain naming this owner, the original pre-funding hold is
credited back, once per job; later idle time for the job never is.

The gate has no clock or chain of its own: ``clock()`` and ``view`` (``job(job_id)``: the open job or None;
``spendable(account)``; ``fee()``) are injected, so every transition is testable without a network.
"""
import json
import secrets
import threading
from pathlib import Path

from neuroshard.assistant.funding import write_atomic
from neuroshard.inference import optimistic as ledger

OFFER_SECONDS = 20
FUNDING_BOUND = 240  # Chain.submit's 180 s confirmation wait plus margin
VISIBLE_GRACE = 30   # a funding lease whose job is visible lapses if no HELLO arrives within this
BUDGET_SECONDS = 60
BUDGET_WINDOW = 600
STARTED_RETRY = 15   # retry_after while the slot serves a started job


class Busy(Exception):
    def __init__(self, retry_after):
        super().__init__(f'busy; retry after {retry_after} s')
        self.retry_after = retry_after


class Refused(Exception):
    """A request this owner will not grant: ``reason`` is 'limit', 'started', 'invalid', 'eligibility' or 'unknown'."""

    def __init__(self, reason, detail=''):
        super().__init__(f'{reason}: {detail}' if detail else reason)
        self.reason = reason


class Budget:
    """Charged seconds per key in a sliding window; entries are tagged so one hold can be credited back."""

    def __init__(self, seconds=BUDGET_SECONDS, window=BUDGET_WINDOW):
        self.seconds, self.window, self.entries = seconds, window, {}

    def charge(self, keys, seconds, tag, now):
        if seconds > 0:
            for key in keys:
                self.entries.setdefault(key, []).append((now, seconds, tag))

    def used(self, key, now):
        recent = [entry for entry in self.entries.get(key, []) if entry[0] >= now - self.window]
        self.entries[key] = recent
        return sum(seconds for _, seconds, _ in recent)

    def remaining(self, keys, now):
        return min(self.seconds - self.used(key, now) for key in keys)

    def credit(self, tag):
        for key, entries in self.entries.items():
            self.entries[key] = [entry for entry in entries if entry[2] != tag]


class Gate:
    def __init__(self, rank, public, home, view, clock, budget=None):
        self.rank, self.public, self.view, self.clock = rank, public, view, clock
        self.directory = Path(home) / 'admission'
        self.budget = budget or Budget()
        self.lock = threading.RLock()
        self.lease = None
        self.gens, self.started, self.credited = {}, set(), set()
        self.holds = {}  # job_id -> the pre-funding hold: credited when the job appears, even after its lease lapsed
        self.closers = {}  # job_id -> callable closing the current generation's connection
        if self.directory.exists():
            for path in self.directory.glob('*.json'):
                record = json.loads(path.read_text())
                if path.name == 'credited.json':
                    self.credited = set(record)
                    continue
                self.gens[record['job_id']] = record['gen']
                if record['state'] == 'started':
                    self.started.add(record['job_id'])  # recovered-started: never admitted again

    # -- accounting --------------------------------------------------------------------------------------------
    def settle(self, now):
        """Charge the live lease up to ``now``, and lapse it when its bound or its budget runs out."""
        lease = self.lease
        if lease is None:
            return
        if lease['charging']:
            until = now if lease['state'] == 'started' else min(now, lease['expires'])
            self.budget.charge(lease['keys'], until - lease['charged_until'], lease['tag'], until)
            lease['charged_until'] = max(lease['charged_until'], until)
        if lease['state'] == 'started':
            return
        lapsed = now >= lease['expires'] or (lease['charging'] and self.budget.remaining(lease['keys'], now) <= 0)
        if lease['state'] == 'funding' and lease.get('visible_at') is not None:
            lapsed = now - lease['visible_at'] > VISIBLE_GRACE
        if lapsed:
            self.drop(lease)

    def drop(self, lease):
        if self.lease is lease:
            self.lease = None
            closer = self.closers.pop(lease.get('job_id'), None)
            if closer is not None:
                closer()

    def bounded(self, keys, now, seconds):
        remaining = self.budget.remaining(keys, now)
        if remaining <= 0:
            raise Refused('limit', 'unproductive reservation time exhausted; retry later')
        return now + min(seconds, remaining)

    def busy(self, now):
        lease = self.lease
        return Busy(STARTED_RETRY if lease['state'] == 'started' else max(1, int(lease['expires'] - now) + 1))

    def record(self, job_id, gen, state):
        write_atomic(self.directory / f'{job_id}.json', {'job_id': job_id, 'gen': gen, 'state': state})

    # -- requests ----------------------------------------------------------------------------------------------
    def reserve_new(self, account, source, session_key, price, now=None):
        """A pre-funding RESERVE, already verified as signed by ``account``."""
        with self.lock:
            now = self.clock() if now is None else now
            self.settle(now)
            ledger.ed25519_key(session_key)
            if self.view.spendable(account) < price + 2 * self.view.fee():
                raise Refused('eligibility', 'the account cannot pay for the job it would open')
            if self.lease is not None:
                raise self.busy(now)
            keys = (('account', account), ('address', source))
            expires, reservation = self.bounded(keys, now, OFFER_SECONDS), secrets.token_hex(16)
            self.lease = {'reservation': reservation, 'state': 'offered', 'keys': keys, 'account': account,
                          'session_key': session_key, 'job_id': None, 'funded': False, 'charging': True,
                          'charged_until': now, 'tag': ('hold', reservation), 'expires': expires, 'gen': None}
            return {'reservation': reservation, 'expires_in': self.lease['expires'] - now}

    def reserve_funded(self, job_id, job, source, now=None):
        """A post-funding RESERVE, already verified as signed by the job's session key; ``job`` is the open job."""
        with self.lock:
            now = self.clock() if now is None else now
            self.settle(now)
            if job_id in self.started:
                raise Refused('started', 'this owner started that job before')
            self.funded(job_id, job)
            lease = self.lease
            if lease is not None and lease.get('job_id') == job_id:
                return {'reservation': lease['reservation'], 'expires_in': lease['expires'] - now}
            if lease is not None:
                raise self.busy(now)
            keys = (('account', job['user']), ('address', source))
            expires, reservation = self.bounded(keys, now, OFFER_SECONDS), secrets.token_hex(16)
            self.lease = {'reservation': reservation, 'state': 'offered', 'keys': keys, 'account': job['user'],
                          'session_key': job['session_key'], 'job_id': job_id, 'funded': True, 'charging': True,
                          'charged_until': now, 'tag': ('idle', job_id), 'expires': expires, 'gen': None}
            return {'reservation': reservation, 'expires_in': self.lease['expires'] - now}

    def hold(self, account, reservation, transaction, now=None):
        """HOLD: the unsigned ``serve_open`` (body and signer) the client is about to sign-and-broadcast.

        It carries no signature, so the owner can derive the job ID but never broadcast the funding itself."""
        with self.lock:
            now = self.clock() if now is None else now
            self.settle(now)
            lease = self.lease
            if lease is None or lease['reservation'] != reservation or lease['account'] != account or lease['funded']:
                raise Refused('unknown', 'no live pre-funding reservation by that id')
            if not isinstance(transaction, dict) or set(transaction) != {'body', 'public_key'} \
                    or not isinstance(transaction['body'], dict):
                raise Refused('invalid', 'a hold carries the unsigned transaction: its body and signer only')
            body, owners = transaction['body'], transaction['body'].get('owners')
            if (transaction['public_key'] != account or body.get('kind') != 'serve_open'
                    or body.get('session_key') != lease['session_key'] or type(body.get('nonce')) is not int
                    or not isinstance(owners, list) or len(owners) < self.rank or owners[self.rank - 1] != self.public):
                raise Refused('invalid', 'the held transaction is not this reservation\'s job for this owner')
            job_id = ledger.transaction_id(transaction)
            if job_id in self.credited or self.view.job(job_id) is not None:
                raise Refused('invalid', 'that job is already funded: re-reserve it as a funded job')
            lease.update(state='funding', job_id=job_id, visible_at=None)
            self.holds = {j: h for j, h in self.holds.items() if h['at'] >= now - self.budget.window}
            self.holds[job_id] = {'tag': lease['tag'], 'account': account, 'session_key': lease['session_key'], 'at': now}
            lease['expires'] = self.bounded(lease['keys'], now, FUNDING_BOUND)
            return {'reservation': reservation, 'job_id': job_id, 'expires_in': lease['expires'] - now}

    def release(self, reservation, signer_account=None, job_id=None, now=None):
        with self.lock:
            now = self.clock() if now is None else now
            self.settle(now)
            lease = self.lease
            if lease is None or lease['reservation'] != reservation or lease['state'] == 'started':
                return {'reservation': reservation}
            if (signer_account is not None and lease['account'] != signer_account) or \
                    (job_id is not None and lease.get('job_id') != job_id):
                raise Refused('unknown', 'that reservation belongs to someone else')
            self.drop(lease)
            return {'reservation': reservation}

    def tick(self, now=None):
        """Lapse leases; notice a held job on chain (crediting its original hold once)."""
        with self.lock:
            now = self.clock() if now is None else now
            lease = self.lease
            if lease is not None and lease['state'] == 'funding' and lease.get('visible_at') is None:
                job = self.view.job(lease['job_id'])
                if job is not None:
                    self.visible(lease, job, now)
            self.settle(now)

    def visible(self, lease, job, now):
        self.settle(now)
        if job['user'] != lease['account'] or job['session_key'] != lease['session_key']:
            self.drop(lease)
            raise Refused('invalid', 'the funded job differs from the one held')
        lease.update(visible_at=now, charging=False, funded=True)
        self.funded(lease['job_id'], job)

    def funded(self, job_id, job):
        """Credit the job's original pre-funding hold, once, when the job is first seen on chain naming this owner."""
        hold = self.holds.pop(job_id, None)
        if hold is None or job_id in self.credited:
            return
        if job['user'] == hold['account'] and job['session_key'] == hold['session_key']:
            self.budget.credit(hold['tag'])
            self.credited.add(job_id)
            write_atomic(self.directory / 'credited.json', sorted(self.credited))

    # -- serving -----------------------------------------------------------------------------------------------
    def admit(self, job_id, job, source, closer, reservation=None, now=None):
        """A HELLO for ``job`` (open on chain, session signature verified): the generation it is admitted at.

        ``reservation`` is the reservation a reserved client's fresh, session-signed admission attempt names (the
        caller has verified it and checked it against replay). Only such an attempt for the live lease admits a
        reserved job, and only it supersedes an admitted generation, whose connection is closed first.

        A legacy HELLO (``reservation=None``, an older client that does not reserve) carries nothing fresh, so a
        captured one could be replayed: it is admitted only directly, with no lease and no earlier generation of
        the job here, still within the budget. It never replaces a connection. A job this owner started is never
        admitted again."""
        with self.lock:
            now = self.clock() if now is None else now
            self.settle(now)
            if job_id in self.started:
                raise Refused('started', 'this owner started that job before')
            lease = self.lease
            if reservation is None:
                if lease is not None and lease.get('job_id') == job_id or job_id in self.gens:
                    raise Refused('invalid', 'this job is reserved: a fresh signed admission attempt is required')
                if lease is not None:
                    raise self.busy(now)
            elif lease is None or lease.get('job_id') != job_id:
                if lease is not None:
                    raise self.busy(now)
                raise Refused('unknown', 'no live reservation for that job: re-reserve it')
            elif lease['reservation'] != reservation:
                raise Refused('unknown', 'the attempt names another reservation')
            if self.lease is None or self.lease.get('job_id') != job_id or self.lease['state'] != 'funding':
                self.funded(job_id, job)
            if lease is None:
                keys = (('account', job['user']), ('address', source))
                self.bounded(keys, now, 1)
                lease = self.lease = {'reservation': None, 'state': 'offered', 'keys': keys, 'account': job['user'],
                                      'session_key': job['session_key'], 'job_id': job_id, 'funded': True,
                                      'charging': True, 'charged_until': now, 'tag': ('idle', job_id),
                                      'expires': now, 'gen': None}
            if lease['state'] == 'funding' and lease.get('visible_at') is None:
                self.visible(lease, job, now)
            elif lease['account'] != job['user'] or lease['session_key'] != job['session_key']:
                raise Refused('invalid', 'the funded job differs from the one reserved')
            previous = self.closers.pop(job_id, None)
            if previous is not None:
                previous()  # the superseded generation's socket is closed before READY goes out on the new one
            gen = self.gens.get(job_id, 0) + 1
            self.record(job_id, gen, 'admitted')
            self.gens[job_id] = gen
            lease.update(state='admitted', gen=gen, charging=True, charged_until=now, tag=('idle', job_id))
            lease['expires'] = self.bounded(lease['keys'], now, BUDGET_SECONDS)
            self.closers[job_id] = closer
            return gen

    def idle_seconds(self, job_id, gen, now=None):
        """How long the admitted generation may wait for its first work frame."""
        with self.lock:
            now = self.clock() if now is None else now
            self.settle(now)
            lease = self.lease
            if lease is None or lease.get('job_id') != job_id or lease['gen'] != gen:
                return 0
            return max(0, lease['expires'] - now)

    def start(self, job_id, gen, now=None):
        """The write-ahead record before the first work frame; False if this generation was superseded."""
        with self.lock:
            now = self.clock() if now is None else now
            self.settle(now)
            lease = self.lease
            if lease is None or lease.get('job_id') != job_id or lease['gen'] != gen or lease['state'] != 'admitted':
                return False
            self.record(job_id, gen, 'started')
            self.started.add(job_id)
            lease.update(state='started', charging=False)
            return True

    def finish(self, job_id, gen):
        with self.lock:
            lease = self.lease
            if lease is not None and lease.get('job_id') == job_id and lease['gen'] == gen:
                if lease['state'] != 'started':
                    self.settle(self.clock())
                self.lease = None
                self.closers.pop(job_id, None)
