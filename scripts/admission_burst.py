#!/usr/bin/env python3
"""Controlled live burst rerun of admission (pilot-2-rc1): measurement 1, after review of PR #127.

Runs on the soak host against the live pilot network, using pilot-2-rc1 clients. Its evidence is kept apart from the
frozen pilot-1 soak: its own --out, run ID and release label. It never touches the soak's files or services.

Phases (each recorded separately):
  baseline        one client, alone
  burst-2         two clients at once
  burst-4         four clients at once
  busy-owner      one client holds a real conversation open (an owner already serving), then two more start

For every attempt it records: the client's exit, waiting time, BUSY/progress lines, completion, the job it funded (from
the client's output and from the chain: every job the client account opened during the attempt), and the outcome
class:
  served          a job was funded and the conversation completed
  busy-unfunded   the client reported busy before funding and the chain shows no job opened by that attempt
  funded-failed   a job was funded but the conversation did not complete (followed to its refund)
  violation       busy reported while a job was opened, or more than one job opened by one attempt
Every funded job is followed to its terminal ledger record (settled or expired: paid and refunded amounts), and
`verify` checks the guarantees over the whole run.

    python scripts/admission_burst.py run --out .neuroshard/admission-burst-pilot-2-rc1 --client "..." \
        --client-account 0=KEY --client-account 1=KEY --client-account 2=KEY --client-account 3=KEY ...
    python scripts/admission_burst.py verify --out .neuroshard/admission-burst-pilot-2-rc1
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import soak  # noqa: E402 - reuse the soak's scoring, probe and chain helpers; its own state is untouched

MEASUREMENT = 1
CONTEXT = {}
LOCK = threading.Lock()
PHASES = ('baseline', 'burst-2', 'burst-4', 'busy-owner')


def log(out, kind, **fields):
    record = {'unix': round(time.time(), 3), 'kind': kind, **CONTEXT, **fields}
    with LOCK, (out / 'events.jsonl').open('a') as handle:
        handle.write(json.dumps(record, sort_keys=True) + '\n')
    if kind in ('incident', 'phase', 'attempt'):
        print(json.dumps({k: record.get(k) for k in ('kind', 'phase', 'slot', 'outcome', 'seconds', 'what')}),
              flush=True)
    return record


def classify(text, code, opened, job):
    busy = 'nothing was funded' in text or 'stayed busy until the deadline' in text
    transport = code == 0 and job is not None and 'NeuroShard:' not in text and 'stopped: ' not in text
    if len(opened) > 1 or (busy and opened):
        return 'violation'
    if busy:
        return 'busy-unfunded'
    if transport and opened:
        return 'served'
    if opened:
        return 'funded-failed'
    return 'unfunded-error'  # no job, no busy report: an error before funding (recorded in full)


def attempt(out, phase, slot, index, command, home, account, chain, hold=None, release=None):
    """One client conversation. ``hold``: keep it open (an owner busy serving) until ``release`` is set."""
    question, kind, expected = soak.QUESTIONS[index % len(soak.QUESTIONS)]
    known = soak.account_jobs(chain.state(), account)
    started = time.monotonic()
    process = subprocess.Popen(command + ['--home', str(home)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    process.stdin.write(question + '\n')
    process.stdin.flush()
    lines, served_at = [], None
    for line in process.stdout:
        lines.append(line)
        if served_at is None and 'assistant (' in line and '): ' in line:
            served_at = round(time.monotonic() - started, 1)
            if hold is not None:
                hold.set()  # the owners are now serving this conversation
                release.wait(1800)
            process.stdin.write('/quit\n')
            process.stdin.flush()
        if 'nothing was funded' in line or 'NeuroShard:' in line:
            break
    try:
        process.stdin.close()
    except OSError:
        pass
    code = process.wait(timeout=600)
    text = ''.join(lines) + (process.stdout.read() or '')
    seconds = round(time.monotonic() - started, 1)
    full = re.search(r'status --job ([0-9a-f]{64})', text) or re.search(r'Job ([0-9a-f]{16})', text)
    job = full.group(1) if full and len(full.group(1)) == 64 else None
    opened = sorted(soak.account_jobs(chain.state(), account) - known)
    if job and job not in opened:
        opened.append(job)
    reply = next((l.split('): ', 1)[1].strip() for l in lines if 'assistant (' in l and '): ' in l), '')
    outcome = classify(text, code, opened, job)
    outputs = out / 'outputs' / CONTEXT['run']
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / f'{index}.txt').write_text(text)
    busy_lines = [l.strip() for l in lines if 'busy' in l.lower() or 'waiting' in l.lower()]
    record = log(out, 'attempt', phase=phase, slot=slot, index=index, account=account, question=question,
                 seconds=seconds, served_after_s=served_at, exit=code, job=job, opened=opened, outcome=outcome,
                 busy_lines=busy_lines[:40], busy_reports=len(busy_lines),
                 fact=soak.score(reply, kind, expected) if reply else 'none', reply=reply[:2000],
                 output_file=str(outputs / f'{index}.txt'), output_sha256=hashlib.sha256(text.encode()).hexdigest())
    if outcome == 'violation':
        log(out, 'incident', what='admission guarantee violated', phase=phase, slot=slot, index=index, opened=opened)
    return record


def phase(out, name, slots, args, accounts, chain, index):
    log(out, 'phase', phase=name, slots=slots)
    command = args.client.split() + ['assistant', 'chat', '--wait', str(args.wait)]
    home = lambda s: Path(args.client_home.format(slot=s))  # noqa: E731
    threads, results = [], {}

    def run(slot, i, **kw):
        results[slot] = attempt(out, name, slot, i, command, home(slot), accounts[str(slot)], chain, **kw)

    if name == 'busy-owner':
        hold, release = threading.Event(), threading.Event()
        holder = threading.Thread(target=run, args=(0, index), kwargs={'hold': hold, 'release': release})
        holder.start()
        if not hold.wait(args.wait + 600):
            log(out, 'incident', what='busy-owner holder was never served', phase=name)
        release_timer = threading.Timer(args.hold_seconds, release.set)
        release_timer.start()  # the holder ends its conversation after hold_seconds: capacity returns
        log(out, 'holding', phase=name, hold_seconds=args.hold_seconds)
        for slot in range(1, slots):
            threads.append(threading.Thread(target=run, args=(slot, index + slot)))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        release_timer.join()  # the holder keeps its owners busy for the whole hold, however the others fared
        holder.join()
    else:
        for slot in range(slots):
            threads.append(threading.Thread(target=run, args=(slot, index + slot)))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    return index + slots, results


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(soak.REPO / 'src'))
    from neuroshard.assistant import network

    descriptor = network.descriptor()
    assert descriptor.get('trusted_validators'), 'verified funding outcomes require pinned validators'
    CONTEXT.update(run=uuid.uuid4().hex[:12], measurement=MEASUREMENT, release=args.release,
                   harness=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
                   repo=soak.revision(soak.REPO))
    accounts = dict(spec.split('=', 1) for spec in args.client_account)
    assert len(accounts) >= 4, 'four client accounts (slots 0-3) are needed for the four-client burst'
    chain = network.chain(descriptor, out / 'probe-home')
    log(out, 'start', client=args.client, wait=args.wait, phases=args.phase, repeats=args.repeats,
        hold_seconds=args.hold_seconds, accounts=accounts)
    stop = threading.Event()
    threading.Thread(target=soak.monitor, args=(out, descriptor, network, 60, stop), daemon=True).start()
    soak.CONTEXT.update(CONTEXT)
    index, funded = args.first_index, set()
    plan = {'baseline': 1, 'burst-2': 2, 'burst-4': 4, 'busy-owner': 3}
    for repeat in range(args.repeats):
        for name in args.phase:
            index, results = phase(out, name, plan[name], args, accounts, chain, index)
            for record in results.values():
                funded.update(record['opened'])
            log(out, 'phase-end', phase=name, repeat=repeat,
                outcomes={str(s): r['outcome'] for s, r in sorted(results.items())})
            time.sleep(args.gap)
    log(out, 'workload-end', funded=sorted(funded))
    drain(out, chain, sorted(funded), descriptor)
    stop.set()
    log(out, 'end')


def drain(out, chain, jobs, descriptor):
    """Follow every funded job to its terminal ledger record: settled (paid) or expired (refunded)."""
    left, deadline = set(jobs), time.time() + (descriptor.get('job_blocks') or 3600) * 2 + 3600
    while left and time.time() < deadline:
        try:
            state = chain.state()
        except Exception as error:  # noqa: BLE001 - retried
            log(out, 'incident', what='verified read failed', error=str(error)[:300])
            time.sleep(60)
            continue
        for job in sorted(left):
            result = state['results'].get(job)
            if result is not None:
                left.discard(job)
                log(out, 'settlement', job=job, status=result.get('status'), paid=result.get('paid'),
                    refunded=result.get('refunded'), positions=result.get('positions'), height=state.get('height'))
            elif job not in state['jobs']:
                left.discard(job)
                log(out, 'incident', what='funded job neither open nor resulted', job=job)
        time.sleep(60)
    for job in sorted(left):
        log(out, 'incident', what='settlement unresolved at end', job=job)


def verify(args):
    """The run's guarantees, from its own events only."""
    events = [json.loads(line) for line in (Path(args.out) / 'events.jsonl').read_text().splitlines()]
    report = {}
    for run_id in dict.fromkeys(e['run'] for e in events):
        items = [e for e in events if e['run'] == run_id]
        attempts = [e for e in items if e['kind'] == 'attempt']
        settlements = {e['job']: e for e in items if e['kind'] == 'settlement'}
        funded = sorted({job for a in attempts for job in a['opened']})
        by_phase = {}
        for a in attempts:
            by_phase.setdefault(a['phase'], []).append(a)
        report[run_id] = {
            'release': items[0].get('release'), 'measurement': items[0].get('measurement'),
            'attempts': len(attempts),
            'outcomes': {o: sum(a['outcome'] == o for a in attempts) for o in
                         ('served', 'busy-unfunded', 'funded-failed', 'unfunded-error', 'violation')},
            'at_most_one_job_per_attempt': all(len(a['opened']) <= 1 for a in attempts),
            'busy_funded_nothing': all(not a['opened'] for a in attempts if a['outcome'] == 'busy-unfunded'),
            'every_funded_job_terminal': all(job in settlements for job in funded),
            'funded_jobs': len(funded),
            'settled': sum(s['status'] == 'settled' for s in settlements.values()),
            'expired_refunded': sum(s['status'] == 'expired' for s in settlements.values()),
            'served_never_settled_unpaid': [j for j in funded if settlements.get(j, {}).get('status') not in
                                            ('settled', 'expired')],
            'progress_in_each_burst': {name: [a['outcome'] for a in group] for name, group in by_phase.items()},
            'waits_s': {name: sorted(a['seconds'] for a in group) for name, group in by_phase.items()},
            'incidents': sorted({e['what'] for e in items if e['kind'] == 'incident'}),
        }
    print(json.dumps(report, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='action', required=True)
    go = sub.add_parser('run')
    go.add_argument('--out', required=True)
    go.add_argument('--release', required=True)
    go.add_argument('--client', required=True)
    go.add_argument('--client-home', default='/home/ubuntu/.neuroshard/burst-client-{slot}')
    go.add_argument('--client-account', action='append', default=[], required=True)
    go.add_argument('--phase', action='append', choices=PHASES)
    go.add_argument('--repeats', type=int, default=3)
    go.add_argument('--wait', type=int, default=600, help='the client deadline (assistant chat --wait)')
    go.add_argument('--hold-seconds', type=int, default=240, help='how long the busy-owner holder stays open')
    go.add_argument('--gap', type=int, default=60, help='seconds between phases')
    go.add_argument('--first-index', type=int, default=0)
    check = sub.add_parser('verify')
    check.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.action == 'run':
        args.phase = args.phase or list(PHASES)
        run(args)
    else:
        verify(args)


if __name__ == '__main__':
    main()
