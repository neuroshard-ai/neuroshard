#!/usr/bin/env python3
"""Controlled live burst rerun of admission (pilot-2-rc1): measurement 3, after the reviews of PR #128 (8ccb1aa, 7bb2124).

Runs from the main host against the live pilot network with pilot-2-rc1 clients. Its evidence is kept apart from the
frozen pilot-1 soak: its own --out, run ID and release label. It never touches the soak's files or services.

The plan is persisted before any work: every phase instance (phase, repeat) with its expected slots and each slot's
role (``contender``, or ``holder`` for the conversation that keeps the owners busy). Every attempt is recorded as
``planned`` before its client starts, its output is written as it streams, and every exception becomes an explicit
``error`` result. Nothing that was planned can silently vanish.

Outcomes (from the client's output; funding from the attempt's own signed records and the chain):
  served          clean exit, a real (nonempty, not "(no reply)") reply, and exactly one funded job bound to it
  busy-unfunded   the client reported busy before funding, and no job bound to its records is visible (judged
                  again over the whole run: a bound job appearing later fails it)
  funded-failed   a job was funded but no real reply was delivered (followed to its refund)
  unfunded-error  no job and no busy report: an error before funding
  violation       busy reported while a job was opened, or more than one job opened by one attempt
  error           the harness could not run or observe the attempt; never a pass

Attribution: before each client starts, its home's funding-operation records are snapshotted; after it is confirmed
stopped, the new records are this attempt's. Each names its job (``job_id`` is the transaction ID of the signed body,
recomputed by the harness), so a job is bound to the attempt that signed it however late it executes. Accounts are
rescanned continuously and through the drain; a job of a run account bound to no attempt's record fails the run.
A client not confirmed stopped quarantines its home. Every nonce the run consumed must be explained exactly once,
by a job of the run or by a superseding transfer verified to have taken its place (``superseded`` alone means
submitted, not executed).

Busy-owner: the holder records when it had a real reply, when its release was commanded and when it exited. The
phase is valid only if it stayed open for the planned hold, released on command, and every contender started
while it held; an early exit invalidates it.

Settlement: every job of the run is tracked from the moment it is first seen, with its opening record (price,
owners, opened height). Accepted terminal outcomes are ``settled`` (checked: paid + pooled + refunded == price) and
``expired`` (the full price returned by the ledger's refund rule; the record has no amount, so the refund is stated
as derived from the price, not measured). ``fraud``, ``voided`` or anything else fails the clean-run criterion.

    python scripts/admission_burst.py run --out .neuroshard/admission-burst-pilot-2-rc1 --release pilot-2-rc1 \
        --client "..." --client-home '.../burst-client-{slot}' --client-account 0=KEY ... (four accounts)
    python scripts/admission_burst.py verify --out .neuroshard/admission-burst-pilot-2-rc1   # exit 0 only on pass
"""
import argparse
import hashlib
import json
import os
import re
import selectors
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import soak  # noqa: E402 - reuse the soak's scoring, probe and chain helpers; its own state is untouched

MEASUREMENT = 3
CONTEXT = {}
LOCK = threading.Lock()
PHASES = ('baseline', 'burst-2', 'burst-4', 'busy-owner')
SLOTS = {'baseline': 1, 'burst-2': 2, 'burst-4': 4, 'busy-owner': 3}
NO_REPLY = ('', '(no reply)')
ACCEPTED = ('settled', 'expired')


def log(out, kind, **fields):
    record = {'unix': round(time.time(), 3), 'kind': kind, **CONTEXT, **fields}
    with LOCK, (out / 'events.jsonl').open('a') as handle:
        handle.write(json.dumps(record, sort_keys=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    if kind in ('incident', 'phase', 'attempt', 'verdict'):
        print(json.dumps({k: record.get(k) for k in ('kind', 'instance', 'slot', 'role', 'outcome', 'what')}),
              flush=True)
    return record


def reply_of(text):
    """The first assistant reply line's text, or '' (a '(no reply)' line is no reply)."""
    for line in text.splitlines():
        if 'assistant (' in line and '): ' in line:
            reply = line.split('): ', 1)[1].strip()
            return '' if reply in NO_REPLY else reply
    return ''


def classify(text, code, opened, job=None):
    """The attempt's outcome from its output and the jobs reconciled to it (``opened``). ``job``, a printed ID, is a
    claim only: it never substitutes for the chain."""
    busy = 'nothing was funded' in text or 'stayed busy until the deadline' in text
    reply = reply_of(text)
    clean = code == 0 and 'NeuroShard:' not in text and 'stopped: ' not in text
    if len(opened) > 1 or (busy and opened):
        return 'violation'
    if busy:
        return 'busy-unfunded'
    if opened and clean and reply:
        return 'served'
    if opened:
        return 'funded-failed'
    return 'unfunded-error'


class Ledger:
    """Every job the run's accounts opened, tracked from first sight in verified state, with its opening record
    (price, owners, opened height) and its terminal result.

    Attribution is not by timing: serializing client processes does not serialize the execution of transactions
    they already broadcast. A job belongs to the attempt whose operation snapshot first contains a record bound to
    it (``all_operations``: ``job_id`` recomputed from the signed body). ``judge`` does the matching."""

    def __init__(self, out, chain, accounts):
        self.out, self.chain, self.accounts = out, chain, set(accounts.values())
        self.jobs = {}  # job_id -> {'account', 'opening', 'terminal'}
        self.lock = threading.Lock()
        self.since = None

    def begin(self):
        """The run's starting point: account nonces and the height from which jobs count as the run's."""
        state = self.chain.state()
        self.since = state['height']
        return state

    def scan(self, state=None):
        state = state or self.chain.state()
        found, closed = [], []
        with self.lock:
            for job_id, job in state['jobs'].items():
                if job['user'] in self.accounts and job_id not in self.jobs and job['opened'] >= (self.since or 0):
                    self.jobs[job_id] = {'account': job['user'], 'opening': dict(job), 'terminal': None}
                    found.append((job_id, job['user'], dict(job)))
            for job_id, result in state['results'].items():
                if result.get('user') in self.accounts and job_id not in self.jobs \
                        and result.get('height', 0) >= (self.since or 0):
                    self.jobs[job_id] = {'account': result['user'], 'opening': None, 'terminal': None}
                    found.append((job_id, result['user'], None))
            for job_id, entry in self.jobs.items():
                result = state['results'].get(job_id)
                if entry['terminal'] is None and result is not None and job_id not in state['jobs']:
                    entry['terminal'] = result
                    closed.append((job_id, result))
        for job_id, account, opening in found:
            log(self.out, 'job-seen', job=job_id, account=account, opened_height=(opening or {}).get('opened'),
                price=(opening or {}).get('price'), owners=(opening or {}).get('owners'), seen_height=state['height'],
                first_seen_closed=opening is None)
        for job_id, result in closed:
            log(self.out, 'settlement', job=job_id, status=result.get('status'), paid=result.get('paid'),
                pooled=result.get('pooled', 0), refunded=result.get('refunded'), positions=result.get('positions'),
                height=state['height'])
        return state

    def nonces(self, state=None):
        state = state or self.chain.state()
        return {a: state['accounts'].get(a, {'nonce': 0})['nonce'] for a in self.accounts}

    def visible(self, job_ids):
        with self.lock:
            return sorted(j for j in job_ids if j in self.jobs)

    def open_jobs(self):
        with self.lock:
            return sorted(j for j, e in self.jobs.items() if e['terminal'] is None)


SHELL = []  # command prefix that runs a shell command where the client homes are (``--client-shell``); [] = here


def remote(command, timeout=60):
    """Run ``command`` (a shell string) where the client homes are; its stdout."""
    argv = SHELL + [command] if SHELL else ['sh', '-c', command]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=True).stdout


def all_operations(home):
    """Every funding-operation record in a client home (``funding.Operations``: one JSON object per file).

    Each record names its job: ``job_id`` is ``transaction_id`` of the signed body, so recomputing it binds the
    record's nonce, price and account to that job (``bound``). A record whose digest does not match is not
    evidence of anything."""
    from neuroshard.inference import optimistic as ledger_rules

    listing = remote(f'for f in {home}/operations/*.json; do [ -f "$f" ] && cat "$f" && echo; done; true')
    items = []
    for line in listing.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            envelope = record['envelope']
            bound = ledger_rules.transaction_id(envelope) == record.get('job_id')
            body = envelope['body']
        except (ValueError, KeyError, TypeError):
            items.append({'unreadable': line[:200]})
            continue
        items.append({'turn_id': record.get('turn_id'), 'job_id': record.get('job_id'), 'nonce': body.get('nonce'),
                      'account': envelope.get('public_key'), 'price': body.get('price'), 'state': record.get('state'),
                      'exposed': record.get('exposed'), 'superseded': record.get('superseded', False),
                      'bound': bound})
    return items


def unresolved_operations(home):
    """Funding operations the client left unresolved: a possibly submitted transaction of unknown outcome."""
    return [r for r in all_operations(home) if r.get('unreadable') or
            r.get('state') in ('prepared', 'held', 'broadcast', 'unknown')]


def kill_remote(home):
    """Stop a timed-out client where it runs, then confirm no process uses this home. Returns whether its exit was
    confirmed: a local ssh ending is not evidence that the remote client stopped."""
    try:
        remote(f"pkill -f -- '[-]-home {home}' || true", timeout=30)  # [-]: never matches this shell's own line
    except Exception:  # noqa: BLE001 - confirmation below decides
        pass
    return client_gone(home)


def client_gone(home, attempts=6):
    for _ in range(attempts):
        try:
            if not remote(f"pgrep -f -- '[-]-home {home}' || true", timeout=30).strip():
                return True
        except Exception:  # noqa: BLE001 - unconfirmed
            pass
        time.sleep(5)
    return False


QUARANTINED = set()  # client homes whose last client was not confirmed stopped: no further attempt may use them


def attempt(out, ledger, instance, slot, role, index, command, home, account, deadline_s, hold=None, release=None):
    """One client conversation, always ending in exactly one ``attempt`` record (``outcome: error`` on any failure).

    Before the client starts, its home's operation records are snapshotted; after it is confirmed stopped, they are
    read again. The records that appeared are this attempt's (``operations``), each bound to its job, so a job is
    attributed to the attempt that signed it however late it executes. Output streams to its file as it arrives;
    ``deadline_s`` bounds the whole attempt, stream reads included. A holder signals ``hold`` once it has a real
    reply, keeps the conversation open until ``release``, and records when it was ready, when release was
    commanded, and when it exited (an exit before release is ``early_exit``)."""
    attempt_id = f"{CONTEXT['run']}-{index}"
    question, kind, expected = soak.QUESTIONS[index % len(soak.QUESTIONS)]
    outputs = out / 'outputs' / CONTEXT['run']
    outputs.mkdir(parents=True, exist_ok=True)
    path = outputs / f'{index}.txt'
    log(out, 'planned-attempt', attempt=attempt_id, instance=instance, slot=slot, role=role, index=index,
        account=account, question=question, output_file=str(path))
    started, process, code, text, error = time.monotonic(), None, None, '', None
    started_unix = time.time()
    served_at = held_at = released_at = exited_at = None
    early_exit, before, after, stopped = False, None, None, True
    try:
        if str(home) in QUARANTINED:
            raise RuntimeError('not started: an earlier client of this home was not confirmed stopped')
        before = {r.get('turn_id') for r in all_operations(home)}
        process = subprocess.Popen(command + ['--home', str(home)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        process.stdin.write(question + '\n')
        process.stdin.flush()
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        quit_sent, pending = False, ''
        stream = process.stdout.fileno()
        with path.open('w') as sink:
            while True:
                remaining = deadline_s - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError(f'attempt exceeded its {deadline_s} s deadline')
                if hold is not None and held_at is not None and release.is_set() and not quit_sent:
                    process.stdin.write('/quit\n')  # the commanded release: observed here, not assumed
                    process.stdin.flush()
                    quit_sent, released_at = True, round(time.time(), 3)
                if not selector.select(timeout=min(remaining, 1)):
                    if process.poll() is not None:
                        break
                    continue
                chunk = os.read(stream, 65536).decode(errors='replace')  # raw: no buffered lines left unseen
                if not chunk:
                    break
                sink.write(chunk)
                sink.flush()
                text += chunk
                pending += chunk
                lines, pending = pending.split('\n')[:-1], pending.split('\n')[-1]
                for line in lines:
                    if served_at is None and 'assistant (' in line and '): ' in line:
                        served_at = round(time.monotonic() - started, 1)
                        if hold is not None and reply_of(line):
                            held_at = round(time.time(), 3)
                            hold.set()  # the owners are now serving this conversation; it stays open
                        else:
                            process.stdin.write('/quit\n')
                            process.stdin.flush()
                            quit_sent = True
        code = process.wait(timeout=max(1, deadline_s - (time.monotonic() - started)))
        exited_at = round(time.time(), 3)
        if hold is not None and held_at is not None and released_at is None:
            early_exit = True  # the holder's conversation ended before release: the owners were not held
    except Exception as exc:  # noqa: BLE001 - every failure becomes this attempt's explicit result
        error = f'{type(exc).__name__}: {exc}'[:500]
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=30)
            except Exception:  # noqa: BLE001
                pass
        if process is not None:
            stopped = client_gone(home) if code is not None else kill_remote(home)
            if not stopped:
                QUARANTINED.add(str(home))
                log(out, 'incident', what='client not confirmed stopped; its home is quarantined', attempt=attempt_id,
                    home=str(home))
            for stream_ in (process.stdin, process.stdout):
                try:
                    stream_.close()
                except Exception:  # noqa: BLE001
                    pass
    seconds = round(time.monotonic() - started, 1)
    operations = []
    try:
        after = all_operations(home)
        operations = [r for r in after if r.get('turn_id') not in (before or set())] if before is not None else []
    except Exception as exc:  # noqa: BLE001 - this attempt's records cannot be established
        error = error or f'operation records unreadable: {exc}'[:500]
    try:
        ledger.scan()
    except Exception as exc:  # noqa: BLE001 - the drain rescans; attribution is by record, not by this read
        error = error or f'post-attempt verified read failed: {exc}'[:500]
    signed = sorted({r['job_id'] for r in operations if r.get('bound')})
    opened = ledger.visible(signed)
    full = re.search(r'status --job ([0-9a-f]{64})', text)
    claimed = full.group(1) if full else None
    unresolved = [r for r in operations if r.get('unreadable') or not r.get('bound')
                  or r.get('state') in ('prepared', 'held', 'broadcast', 'unknown')]
    if not stopped:
        error = error or 'client not confirmed stopped'
    outcome = 'error' if error else classify(text, code, opened, claimed)
    reply = reply_of(text)
    record = log(out, 'attempt', attempt=attempt_id, instance=instance, slot=slot, role=role, index=index,
                 account=account, started_unix=round(started_unix, 3), total_s=seconds, first_reply_s=served_at,
                 held_from_unix=held_at, release_commanded_unix=released_at, exited_unix=exited_at,
                 early_exit=early_exit, client_stopped=stopped, exit=code, error=error, claimed_job=claimed,
                 claim_reconciled=claimed is None or claimed in opened, operations=operations, signed_jobs=signed,
                 opened=opened, outcome=outcome, unresolved_operations=unresolved,
                 busy_reports=sum('busy' in l.lower() for l in text.splitlines()),
                 admission_waits=[l.strip() for l in text.splitlines() if 'busy' in l.lower() or 'waiting' in l.lower()][:40],
                 fact=soak.score(reply, kind, expected) if reply else 'none', reply=reply[:2000],
                 output_file=str(path), output_sha256=hashlib.sha256(text.encode()).hexdigest())
    if outcome == 'violation':
        log(out, 'incident', what='admission guarantee violated', attempt=attempt_id, opened=opened)
    return record


def instance_plan(phases, repeats):
    """Every phase instance, its slots and their roles, fixed before anything runs."""
    plan = []
    for repeat in range(repeats):
        for name in phases:
            roles = ['holder'] + ['contender'] * (SLOTS[name] - 1) if name == 'busy-owner' else \
                ['contender'] * SLOTS[name]
            plan.append({'instance': f'{name}#{repeat}', 'phase': name, 'repeat': repeat,
                         'slots': {str(s): role for s, role in enumerate(roles)}})
    return plan


def phase(out, ledger, item, args, accounts, index):
    """Run one phase instance. Returns the next index. Every slot ends in exactly one attempt record."""
    name, instance = item['phase'], item['instance']
    log(out, 'phase', instance=instance, phase=name, repeat=item['repeat'], slots=item['slots'])
    command = args.client.split() + ['assistant', 'chat', '--wait', str(args.wait)]
    home = lambda s: Path(args.client_home.format(slot=s))  # noqa: E731
    deadline = args.wait + args.attempt_margin  # admission wait, then the conversation itself
    threads, release_timer = [], None

    def run(slot, i, **kw):
        role = item['slots'][str(slot)]
        try:
            attempt(out, ledger, instance, slot, role, i, command, home(slot), accounts[str(slot)],
                    deadline + (args.hold_seconds if kw else 0), **kw)
        except BaseException as exc:  # noqa: BLE001 - a worker failure is an explicit result, never a gap
            log(out, 'attempt', attempt=f"{CONTEXT['run']}-{i}", instance=instance, slot=slot, role=role, index=i,
                account=accounts[str(slot)], outcome='error', opened=[], error=f'{type(exc).__name__}: {exc}'[:500])

    try:
        if name == 'busy-owner':
            hold, release = threading.Event(), threading.Event()
            holder = threading.Thread(target=run, args=(0, index), kwargs={'hold': hold, 'release': release},
                                      daemon=True)
            holder.start()
            ready = hold.wait(deadline)
            log(out, 'holder', instance=instance, ready=ready)
            if not ready:  # the owners were never shown busy: the phase is invalid, contenders are not started
                release.set()
                holder.join(deadline)
                log(out, 'phase-invalid', instance=instance, reason='the holder was never served')
                for slot in range(1, SLOTS[name]):
                    log(out, 'attempt', attempt=f"{CONTEXT['run']}-{index + slot}", instance=instance, slot=slot,
                        role='contender', index=index + slot, outcome='error', opened=[],
                        error='not started: the holder was never served', account=accounts[str(slot)])
                return index + SLOTS[name]
            release_timer = threading.Timer(args.hold_seconds, release.set)
            release_timer.start()  # the holder itself observes and records the release it carries out
            log(out, 'holding', instance=instance, hold_seconds=args.hold_seconds, from_unix=round(time.time(), 3))
            for slot in range(1, SLOTS[name]):
                threads.append(threading.Thread(target=run, args=(slot, index + slot), daemon=True))
            threads.append(holder)
            for thread in threads[:-1]:
                thread.start()
        else:
            for slot in range(SLOTS[name]):
                threads.append(threading.Thread(target=run, args=(slot, index + slot), daemon=True))
            for thread in threads:
                thread.start()
        for thread in threads:
            thread.join(deadline + args.hold_seconds + 120)
        stuck = [t for t in threads if t.is_alive()]
        if stuck:
            log(out, 'incident', what='attempt thread did not finish', instance=instance, count=len(stuck))
    finally:
        if release_timer is not None and release_timer.is_alive():
            release_timer.cancel()  # not a release: the holder's own record says whether one was observed
            log(out, 'hold-cancelled', instance=instance)
    log(out, 'phase-end', instance=instance)
    return index + SLOTS[name]


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(soak.REPO / 'src'))
    from neuroshard.assistant import network

    descriptor = network.descriptor()
    if not descriptor.get('trusted_validators'):
        raise SystemExit('verified funding outcomes require pinned validators')
    accounts = dict(spec.split('=', 1) for spec in args.client_account)
    if sorted(accounts) != ['0', '1', '2', '3']:
        raise SystemExit('four client accounts, slots 0-3, are required')
    CONTEXT.update(run=uuid.uuid4().hex[:12], measurement=MEASUREMENT, release=args.release,
                   harness=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
                   repo=soak.revision(soak.REPO))
    soak.CONTEXT.update(CONTEXT)  # the monitor's records carry this run's context from its first probe
    chain = network.chain(descriptor, out / 'probe-home')
    ledger = Ledger(out, chain, accounts)
    begun = ledger.begin()
    plan = instance_plan(args.phase, args.repeats)
    log(out, 'start', client=args.client, wait=args.wait, hold_seconds=args.hold_seconds, accounts=accounts,
        plan=plan, attempts_expected=sum(len(p['slots']) for p in plan),
        job_blocks=begun['params']['job_blocks'], nonces=ledger.nonces(begun), since_height=begun['height'],
        operations_before={slot: all_operations(Path(args.client_home.format(slot=slot))) for slot in sorted(accounts)})
    stop = threading.Event()
    threading.Thread(target=soak.monitor, args=(out, descriptor, network, 60, stop), daemon=True).start()
    watcher = threading.Thread(target=watch, args=(ledger, stop), daemon=True)
    watcher.start()  # reconciles accounts and tracks every job continuously, from the moment it appears
    index = args.first_index
    try:
        for item in plan:
            index = phase(out, ledger, item, args, accounts, index)
            time.sleep(args.gap)
        log(out, 'workload-end', jobs=sorted(ledger.jobs))
        drain(out, ledger)
    except BaseException as exc:
        log(out, 'interrupted', error=f'{type(exc).__name__}: {exc}'[:500])
        raise
    finally:
        stop.set()
    log(out, 'operations-final', homes={slot: all_operations(Path(args.client_home.format(slot=slot)))
                                        for slot in sorted(accounts)})
    log(out, 'end', jobs=sorted(ledger.jobs), open=ledger.open_jobs())


def watch(ledger, stop):
    while not stop.wait(30):
        try:
            ledger.scan()
        except Exception as exc:  # noqa: BLE001 - retried; recorded
            log(ledger.out, 'incident', what='verified read failed', error=str(exc)[:300])


def drain(out, ledger):
    """Keep reconciling every account until every job of the run is terminal and late-visible funding has had a
    full ``job_blocks`` to appear after the workload ended. The upper bound is ``job_blocks`` (expiry of a job never
    committed) + ``challenge_blocks`` (a committed job's window) + 4 × ``proof_blocks`` (open challenges), doubled."""
    state = ledger.chain.state()
    params, ended = state['params'], state['height']
    bound = 2 * (params['job_blocks'] + params.get('challenge_blocks', 0) + 4 * params.get('proof_blocks', 0))
    log(out, 'drain', from_height=ended, until_height=ended + bound, open=ledger.open_jobs())
    while True:
        try:
            state = ledger.scan()
        except Exception as exc:  # noqa: BLE001
            log(out, 'incident', what='verified read failed', error=str(exc)[:300])
            time.sleep(60)
            continue
        if not ledger.open_jobs() and state['height'] > ended + params['job_blocks']:
            break
        if state['height'] > ended + bound:
            break
        time.sleep(60)
    log(out, 'drained', height=state['height'], open=ledger.open_jobs(), nonces=ledger.nonces(state))


def settlement_check(entry, opening_price):
    """None if the terminal record is a clean, internally consistent outcome; else why not."""
    status = entry.get('status')
    if status not in ACCEPTED:
        return f'terminal status {status!r} is not an accepted clean outcome'
    if opening_price is None:
        return 'no verified opening record: the escrow (price) is not known, so the outcome cannot be checked'
    if status == 'settled':
        paid, pooled, refunded = entry.get('paid'), entry.get('pooled') or 0, entry.get('refunded')
        if not isinstance(paid, list) or not all(isinstance(a, int) and a >= 0 for a in paid) \
                or not isinstance(refunded, int) or refunded < 0 or not isinstance(pooled, int) or pooled < 0:
            return 'settled record without integer payments and refund'
        if sum(paid) + pooled + refunded != opening_price:
            return f'paid {sum(paid)} + pooled {pooled} + refunded {refunded} != escrow {opening_price}'
    elif entry.get('paid') is not None:
        return 'an expired record that claims payments'
    return None


def verify(args):
    """The run's verdict from its own events only: pass, fail (a required condition is contradicted) or
    inconclusive (one is not established). Exit status 0 only if every judged run passes; no run is not a pass."""
    path = Path(args.out) / 'events.jsonl'
    events = [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []
    runs = list(dict.fromkeys(e['run'] for e in events if e.get('run')))
    if args.run:
        runs = [args.run]
    if not runs:
        print(json.dumps({'verdict': 'inconclusive', 'reasons': [f'no run in {path}']}, indent=1))
        return 2
    report, worst = {}, 0
    for run_id in runs:
        items = [e for e in events if e.get('run') == run_id]
        result = judge(items)
        report[run_id] = result
        worst = max(worst, {'pass': 0, 'fail': 1, 'inconclusive': 2}[result['verdict']])
    print(json.dumps(report, indent=1, sort_keys=True))
    return worst


def judge(items):
    fails, gaps = [], []
    first = lambda kind: next((e for e in items if e['kind'] == kind), None)  # noqa: E731
    start = first('start')
    if start is None or not start.get('plan'):
        return {'verdict': 'inconclusive', 'reasons': ['no persisted plan: not a measurement run']}
    for kind in ('workload-end', 'drained', 'operations-final', 'end'):
        if first(kind) is None:
            gaps.append(f'the run has no {kind!r} record: interrupted or incomplete')
    if first('interrupted'):
        gaps.append('the run was interrupted')
    planned = {(p['instance'], s): role for p in start['plan'] for s, role in p['slots'].items()}
    attempts = {}
    for e in items:
        if e['kind'] == 'attempt':
            key = (e['instance'], str(e['slot']))
            if key in attempts:
                fails.append(f'two results for {key}')
            attempts[key] = e
    missing = sorted(set(planned) - set(attempts))
    if missing:
        gaps.append(f'{len(missing)} planned attempts have no result: {missing[:6]}')

    # Attribution by signed record: the attempt whose new operation records include a record bound to the job.
    seen = {e['job']: e for e in items if e['kind'] == 'job-seen'}
    settlements = {e['job']: e for e in items if e['kind'] == 'settlement'}
    signer = {}
    for key, e in attempts.items():
        for record in e.get('operations') or []:
            if record.get('bound') and record.get('job_id'):
                if record['job_id'] in signer and signer[record['job_id']] != e['attempt']:
                    fails.append(f"job {record['job_id'][:16]} is bound to two attempts")
                signer[record['job_id']] = e['attempt']
    by_attempt = {}
    for job, e in seen.items():
        origin = signer.get(job)
        if origin is None:
            fails.append(f'job {job[:16]} of a run account has no operation record of any attempt: '
                         'funding this run cannot account for')
        by_attempt.setdefault(origin, []).append(job)

    for key, e in attempts.items():
        attempt_id = e.get('attempt')
        final = sorted(by_attempt.get(attempt_id, []))  # every job its records name that ever became visible
        if e['outcome'] == 'error':
            gaps.append(f"attempt {attempt_id} could not be observed: {e.get('error')}")
        if e['outcome'] == 'violation':
            fails.append(f"attempt {attempt_id} violated admission: opened {e.get('opened')}")
        if e['outcome'] == 'busy-unfunded' and final:
            fails.append(f'attempt {attempt_id} reported busy, but its signed job(s) {final} were funded')
        if len(final) > 1:
            fails.append(f'attempt {attempt_id} funded {len(final)} jobs')
        if e['outcome'] in ('funded-failed', 'unfunded-error'):
            # DESIGN_ADMISSION test 10: every request is either served or reports busy before funding.
            fails.append(f"attempt {attempt_id} was neither served nor busy before funding: {e['outcome']}")
        if e['outcome'] == 'served' and not final:
            fails.append(f'attempt {attempt_id} is served without a funded job bound to its own record')
        if e.get('unresolved_operations'):
            gaps.append(f"attempt {attempt_id} left unresolved or unbound funding records")
        if not e.get('claim_reconciled', True):
            fails.append(f"attempt {attempt_id}: printed job {e.get('claimed_job')} is not its funded job")
        if e.get('client_stopped') is False:
            gaps.append(f'attempt {attempt_id}: its client was not confirmed stopped')

    funded = sorted(seen)
    for job in funded:
        if job not in settlements:
            gaps.append(f'funded job {job[:16]} has no terminal record')
            continue
        why = settlement_check(settlements[job], seen[job].get('price'))
        if why:
            (gaps if 'not known' in why else fails).append(f'job {job[:16]}: {why}')

    # Nonces, one outcome per unique nonce: each nonce the run consumed is explained by exactly one of its jobs or
    # by a superseding transfer that executed in its place. A job and the transfer that raced it share one nonce;
    # the flag ``superseded`` means submitted, never executed.
    final_ops = (first('operations-final') or {}).get('homes', {})
    before_ops = {slot: {r.get('turn_id') for r in records} for slot, records in (start.get('operations_before') or {}).items()}
    nonces_before, nonces_after = start.get('nonces') or {}, (first('drained') or {}).get('nonces') or {}
    for slot, account in (start.get('accounts') or {}).items():
        if account not in nonces_before or account not in nonces_after:
            gaps.append(f'no nonce record for {account[:12]}')
            continue
        consumed = set(range(nonces_before[account], nonces_after[account]))
        run_records = [r for r in final_ops.get(slot, []) if r.get('turn_id') not in before_ops.get(slot, set())]
        explained, unexplained = set(), set()
        for record in run_records:
            nonce = record.get('nonce')
            if nonce not in consumed:
                continue
            if record.get('job_id') in seen:
                explained.add(nonce)  # the funding itself executed at this nonce
            elif record.get('superseded') and record.get('state') in ('unfunded', 'abandoned'):
                explained.add(nonce)  # verified unfunded: the superseding transfer took its nonce
            else:
                unexplained.add(nonce)
        for nonce in sorted(consumed - explained):
            (gaps if nonce in unexplained else fails).append(
                f'account {account[:12]}: nonce {nonce} was consumed '
                + ('but its outcome is not established by verified records' if nonce in unexplained
                   else 'by no transaction of this run'))

    # Progress, and the busy-owner hold as observed.
    hold_seconds = start.get('hold_seconds')
    instances = {}
    for (instance, slot), role in planned.items():
        instances.setdefault(instance, []).append((role, attempts.get((instance, slot))))
    progress = {}
    for instance, slots in sorted(instances.items()):
        contenders = [a for role, a in slots if role == 'contender']
        served = [a for a in contenders if a and a['outcome'] == 'served'
                  and any(settlements.get(j, {}).get('status') in ACCEPTED for j in by_attempt.get(a['attempt'], []))]
        holder = [a for role, a in slots if role == 'holder']
        entry = {'contenders': len(contenders), 'contenders_served_and_terminal': len(served),
                 'outcomes': [a['outcome'] if a else 'missing' for _, a in slots]}
        if holder:
            why = hold_check(holder[0], contenders, hold_seconds)
            entry['hold'] = why or 'held for the planned interval while the contenders ran'
            if why:
                gaps.append(f'{instance}: the owners were not shown held busy ({why}); the phase is invalid')
        if not served:
            fails.append(f'{instance}: no contender was served (with a terminal job) within its deadline')
        progress[instance] = entry
    outcomes = {}
    for e in attempts.values():
        outcomes[e['outcome']] = outcomes.get(e['outcome'], 0) + 1
    incomplete = bool(missing) or any('interrupted' in g or 'incomplete' in g or 'phase is invalid' in g for g in gaps)
    # A safety or served-or-busy contradiction always fails. An incomplete or invalid run is inconclusive even if
    # progress looks absent: its missing attempts may be the ones that would have been served.
    safety = [f for f in fails if 'no contender was served' not in f]
    verdict = 'fail' if safety or (fails and not incomplete) else 'inconclusive' if gaps or fails else 'pass'
    return {'verdict': verdict, 'reasons': fails + gaps, 'attempts_expected': len(planned),
            'attempts_recorded': len(attempts), 'outcomes': outcomes, 'funded_jobs': len(funded),
            'settled': sum(s['status'] == 'settled' for s in settlements.values()),
            'expired_full_refund_by_rule': sum(s['status'] == 'expired' for s in settlements.values()),
            'progress': progress,
            'durations_s': {i: [a['total_s'] for _, a in sl if a and 'total_s' in a] for i, sl in instances.items()},
            'first_reply_s': {i: [a.get('first_reply_s') for _, a in sl if a] for i, sl in instances.items()},
            'incidents': sorted({e['what'] for e in items if e['kind'] == 'incident'})}


def hold_check(holder, contenders, hold_seconds):
    """None if the holder demonstrably kept the owners serving for the planned interval while the contenders ran;
    otherwise why not. Every time is observed by the holder's own attempt (unix seconds)."""
    if not holder:
        return 'no holder result'
    ready, released, exited = (holder.get(k) for k in ('held_from_unix', 'release_commanded_unix', 'exited_unix'))
    if ready is None:
        return 'never ready: no real reply'
    if holder.get('early_exit') or released is None:
        return 'the holder exited before its release was commanded'
    if exited is None or exited < released:
        return 'no observed exit after the commanded release'
    if not isinstance(hold_seconds, (int, float)) or released - ready < hold_seconds - 1:
        return f'held {released - ready:.1f} s of the planned {hold_seconds} s'
    starts = [c.get('started_unix') for c in contenders if c]
    if not starts or any(t is None or not ready <= t <= released for t in starts):
        return 'a contender did not start while the owners were held'
    return None


def gates(args):
    """The runbook's start gates, from evidence: the frozen soak ended, its pending jobs drained without an
    unresolved settlement, and no open job names a serving owner. Exit 0 only if every gate holds."""
    sys.path.insert(0, str(soak.REPO / 'src'))
    from neuroshard.assistant import network

    soak_dir = Path(args.soak_out)
    events = [json.loads(l) for l in (soak_dir / 'events.jsonl').read_text().splitlines() if l]
    run_events = [e for e in events if e.get('run') == args.soak_run]
    pending_path = soak_dir / f'pending-{args.soak_run}.json'
    pending = json.loads(pending_path.read_text()) if pending_path.exists() else None
    service = subprocess.run(['systemctl', 'is-active', args.soak_unit], capture_output=True, text=True).stdout.strip()
    state = network.chain(network.descriptor(), Path(args.probe_home)).state()
    owners = {k for k, o in state['owners'].items() if o['status'] == 'active'}
    naming = sorted(j for j, job in state['jobs'].items() if set(job['owners']) & owners)
    checks = {
        'soak_unit_inactive': service in ('inactive', 'failed') and service != '',
        'soak_run_ended': any(e['kind'] == 'end' for e in run_events),
        'soak_pending_empty': pending == [],
        'no_unresolved_settlement': not any(e['kind'] == 'incident' and e.get('what') == 'settlement unresolved at end'
                                            for e in run_events),
        'no_open_job_names_an_active_owner': not naming,
    }
    print(json.dumps({'checks': checks, 'soak_unit': service, 'pending': pending, 'open_jobs_naming_owners': naming,
                      'verified_height': state['height']}, indent=1, sort_keys=True))
    return 0 if all(checks.values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='action', required=True)
    go = sub.add_parser('run')
    go.add_argument('--out', required=True)
    go.add_argument('--release', required=True)
    go.add_argument('--client', required=True)
    go.add_argument('--client-home', required=True, help='one home per slot; {slot} is replaced')
    go.add_argument('--client-shell', default='', help='command prefix running a shell command on the client host '
                    '(e.g. "ssh -i KEY -o BatchMode=yes ubuntu@HOST"); empty: the homes are local')
    go.add_argument('--client-account', action='append', default=[], required=True)
    go.add_argument('--phase', action='append', choices=PHASES)
    go.add_argument('--repeats', type=int, default=3)
    go.add_argument('--wait', type=int, default=600, help='the client deadline (assistant chat --wait)')
    go.add_argument('--attempt-margin', type=int, default=900, help='seconds beyond --wait for the conversation')
    go.add_argument('--hold-seconds', type=int, default=240, help='how long the busy-owner holder stays open')
    go.add_argument('--gap', type=int, default=60, help='seconds between phase instances')
    go.add_argument('--first-index', type=int, default=0)
    check = sub.add_parser('verify')
    check.add_argument('--out', required=True)
    check.add_argument('--run')
    gate = sub.add_parser('gates', help='the start gates (read-only); exit 0 only if all hold')
    gate.add_argument('--soak-out', default=str(soak.REPO / '.neuroshard' / 'soak-pilot-1'))
    gate.add_argument('--soak-run', default='aa18ae92d013')
    gate.add_argument('--soak-unit', default='neuroshard-soak.service')
    gate.add_argument('--probe-home', default='/tmp/admission-burst-gates')
    args = parser.parse_args()
    if args.action == 'gates':
        return gates(args)
    if args.action == 'run':
        args.phase = args.phase or list(PHASES)
        SHELL[:] = args.client_shell.split()
        run(args)
        return 0
    return verify(args)


if __name__ == '__main__':
    sys.exit(main())
