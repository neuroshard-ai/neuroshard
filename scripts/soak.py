"""Soak of a frozen release: independent monitoring, paid conversations with verified outcomes, verified faults.

Measurement version 2 (October 19, 2026), after review of version 1:
- Faults: every command's exit status is checked; the target service's state is observed after injection and
  after restoration; restoration always runs (``finally``); a remote timer restores the service even if this
  machine or its SSH connection is lost. Attempted and observed recovery are separate records.
- Conversations: transport completion (exit status, a full job ID, a non-empty reply, no failure), answer
  correctness (the reply contains the workspace's known fact), and settlement (the job's terminal record on chain
  after its deadline, with its payments and refund) are separate outcomes. Every attempt keeps its output.
- Monitoring runs in its own thread, never blocked by conversations or faults. Each probe reads, from every RPC,
  the signed commit at one common height two blocks behind the lowest tip, and compares block IDs and app hashes;
  signers are counted per RPC. Probe gaps are measured and reported.
- Every event carries a run ID, the runtime and harness revisions, and the configuration. ``summary`` reports each
  run separately. Earlier runs in the same log are kept and labelled with their measurement version.

    python scripts/soak.py run --out .neuroshard/soak-pilot-1 --hours 72
    python scripts/soak.py summary --out .neuroshard/soak-pilot-1
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MEASUREMENT_VERSION = 2
# The demo workspace (``assistant chat`` sample 0) and the fact a correct reply must state.
QUESTIONS = [
    ('What is the review interval in the latest approved Cedar 6581 delivery plan?', r'\b9\b'),
    ('What is the baseline quantity in the latest approved Cedar 6581 plan?', r'\b61\b'),
    ('When does the latest approved Cedar 6581 plan start?', r'2027-05-14|May 14'),
    ('What is the baseline quantity in the latest approved Cedar 6581 Annex plan?', r'\b70\b'),
]
EXPECT = {'max_seconds': 300, 'min_signers': 4, 'max_probe_gap': 180}
LOCK = threading.Lock()
CONTEXT = {}


def revision(path):
    try:
        return subprocess.check_output(['git', '-C', str(path), 'describe', '--tags', '--always', '--dirty'],
                                       text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def log(out, kind, **fields):
    record = {'unix': round(time.time(), 3), 'kind': kind, **CONTEXT, **fields}
    with LOCK, (out / 'events.jsonl').open('a') as handle:
        handle.write(json.dumps(record, sort_keys=True) + '\n')
    if kind == 'incident':
        print(json.dumps(record, sort_keys=True), flush=True)
    return record


def rpc(address, method, timeout=8):
    with urllib.request.urlopen(f'{address}/{method}', timeout=timeout) as response:
        return json.loads(response.read())['result']


def endpoints(descriptor):
    value = descriptor['rpc']
    return ([value] if isinstance(value, str) else list(value)) + list(descriptor.get('rpcs', []))


# Monitoring

def probe(out, descriptor, network):
    """Every RPC: its tip, then the signed commit at one common height (block ID, app hash, signers)."""
    tips, errors = {}, {}
    for address in endpoints(descriptor):
        try:
            tips[address] = int(rpc(address, 'status')['sync_info']['latest_block_height'])
        except (OSError, KeyError, ValueError) as error:
            errors[address] = str(error)[:200]
    for address, error in errors.items():
        log(out, 'incident', what='rpc unreachable', rpc=address, error=error)
    if not tips:
        return log(out, 'probe', tips={}, agree=None)
    common = min(tips.values()) - 2  # finalized everywhere: its commit is in the next block
    commits = {}
    for address in tips:
        try:
            signed = rpc(address, f'commit?height={common}')['signed_header']
            commits[address] = {'block': signed['commit']['block_id']['hash'], 'app': signed['header']['app_hash'],
                                'signers': sum(1 for s in signed['commit']['signatures'] if s['block_id_flag'] == 2)}
        except (OSError, KeyError, ValueError) as error:
            log(out, 'incident', what='rpc commit unreadable', rpc=address, height=common, error=str(error)[:200])
    blocks = {c['block'] for c in commits.values()}
    apps = {c['app'] for c in commits.values()}
    agree = len(blocks) == 1 and len(apps) == 1
    if not agree:
        log(out, 'incident', what='rpcs disagree', height=common, commits=commits)
    fewest = min((c['signers'] for c in commits.values()), default=None)
    if fewest is not None and fewest < EXPECT['min_signers']:
        log(out, 'incident', what='few signers', height=common, signers={a: c['signers'] for a, c in commits.items()})
    owners, verified = {}, None
    try:
        state = network.chain(descriptor, out / 'probe-home').state()
        verified = state['height']
        for key, owner in sorted(state['owners'].items()):
            if owner['status'] == 'active' and owner.get('endpoint'):
                host, port = owner['endpoint'].rsplit(':', 1)
                try:
                    __import__('socket').create_connection((host, int(port)), timeout=5).close()
                    owners[key[:10]] = True
                except OSError:
                    owners[key[:10]] = False
                    log(out, 'incident', what='owner unreachable', owner=key[:10], shard=owner['shard'])
    except Exception as error:  # noqa: BLE001 - a failed verified read is itself an incident
        log(out, 'incident', what='verified read failed', error=f'{type(error).__name__}: {error}'[:300])
    return log(out, 'probe', tips=tips, height=common, agree=agree, rpcs_compared=len(commits),
               signers={a[7:].split(':')[0]: c['signers'] for a, c in commits.items()}, owners=owners,
               verified=verified)


def monitor(out, descriptor, network, every, stop):
    last = None
    while not stop.is_set():
        started = time.time()
        if last is not None and started - last > EXPECT['max_probe_gap']:
            log(out, 'incident', what='probe gap', seconds=round(started - last, 1))
        last = started
        try:
            probe(out, descriptor, network)
        except Exception as error:  # noqa: BLE001
            log(out, 'incident', what='probe failed', error=f'{type(error).__name__}: {error}'[:300])
        stop.wait(max(1, every - (time.time() - started)))


# Conversations

def conversation(out, index, question, expect, command, home, pending):
    started = time.monotonic()
    try:
        result = subprocess.run(command + ['--home', str(home)], input=f'{question}\n/quit\n', text=True,
                                capture_output=True, timeout=EXPECT['max_seconds'] * 3)
        text, code = result.stdout + result.stderr, result.returncode
    except subprocess.TimeoutExpired as error:
        text, code = ((error.stdout or b'') + (error.stderr or b'')).decode(errors='replace') \
            if isinstance(error.stdout, bytes) else (error.stdout or '') + (error.stderr or ''), 'timeout'
    seconds = round(time.monotonic() - started, 1)
    full = re.search(r'status --job ([0-9a-f]{64})', text)
    job = full.group(1) if full else None
    reply = next((line.split('): ', 1)[1] for line in text.splitlines() if 'assistant (' in line and '): ' in line), '')
    stopped = next((line.split('stopped: ', 1)[1] for line in text.splitlines() if 'stopped: ' in line), None)
    transport = code == 0 and job is not None and reply not in ('', '(no reply)') and stopped is None \
        and 'NeuroShard:' not in text
    correct = bool(transport and re.search(expect, reply))
    record = log(out, 'conversation', index=index, question=question, job=job, seconds=seconds, exit=code,
                 transport_ok=transport, correct=correct, reply=reply[:300], stopped=stopped, output=text[-2000:])
    if job:
        pending.append({'job': job, 'index': index, 'transport_ok': transport, 'unix': time.time()})
    if not transport:
        log(out, 'incident', what='conversation failed', index=index, seconds=seconds, exit=code,
            reason=(stopped or next((l for l in text.splitlines() if 'NeuroShard:' in l), '') or 'no job or reply')[:300])
    elif seconds > EXPECT['max_seconds']:
        log(out, 'incident', what='conversation slow', index=index, seconds=seconds)
    return record


def settle(out, descriptor, network, pending, wait):
    """After a job's challenge window, record its terminal ledger outcome: paid, refunded, expired or slashed."""
    if not pending:
        return
    try:
        state = network.chain(descriptor, out / 'probe-home').state()
    except Exception:  # noqa: BLE001 - try again next pass
        return
    for item in list(pending):
        result = state['results'].get(item['job'])
        if result is None and item['job'] in state['jobs']:
            continue  # still open
        if result is None and time.time() - item['unix'] < wait:
            continue
        pending.remove(item)
        status = result.get('status') if result else 'unknown'
        log(out, 'settlement', index=item['index'], job=item['job'], status=status,
            paid=result.get('paid') if result else None, refunded=result.get('refunded') if result else None,
            positions=result.get('positions') if result else None, transport_ok=item['transport_ok'])
        if status not in ('settled', 'expired'):
            log(out, 'incident', what='settlement not terminal', index=item['index'], job=item['job'], status=status)


# Faults

FAULTS = {
    'stop-validator': ('172.31.91.19', 'neuroshard-validator'),
    'stop-owner-2b': ('172.31.41.136', 'neuroshard-owner'),
    'stop-auditor': ('172.31.30.33', 'neuroshard-auditor'),
    'restart-seed': ('172.31.73.85', 'neuroshard-seed'),
}


def ssh(host, command, timeout=60):
    try:
        result = subprocess.run(['ssh', '-i', os.path.expanduser('~/.ssh/id_ed25519'), '-o', 'BatchMode=yes',
                                 '-o', 'ConnectTimeout=15', f'ubuntu@{host}', command],
                                capture_output=True, text=True, timeout=timeout)
        return result.returncode, (result.stdout + result.stderr).strip()[-300:]
    except subprocess.TimeoutExpired:
        return 'timeout', ''


def service_state(host, unit):
    code, text = ssh(host, f'systemctl is-active {unit}')
    return text.splitlines()[-1] if code in (0, 3) and text else f'unknown ({code})'


def fault(out, name, hold):
    host, unit = FAULTS[name]
    restart = name.startswith('restart')
    try:
        before = service_state(host, unit)
        if restart:
            code, text = ssh(host, f'sudo systemctl restart {unit}')
        else:
            # A remote timer restarts the service even if this machine or its connection is lost meanwhile.
            code, text = ssh(host, f'sudo systemd-run --on-active={int(hold) + 120} --unit=soak-restore-{unit} '
                                   f'--collect systemctl start {unit} && sudo systemctl stop {unit}')
        after = service_state(host, unit)
        log(out, 'fault', name=name, phase='inject', host=host, unit=unit, exit=code, before=before, after=after,
            output=text)
        expected = 'active' if restart else 'inactive'
        if code != 0 or after != expected:
            log(out, 'incident', what='fault injection not observed', name=name, exit=code, state=after)
        if not restart:
            time.sleep(hold)
    finally:
        if not restart:
            code, text = ssh(host, f'sudo systemctl start {unit}; sudo systemctl stop soak-restore-{unit}.timer '
                                   f'2>/dev/null; true')
            log(out, 'fault', name=name, phase='restore-attempt', host=host, unit=unit, exit=code, held=hold)
        recovered, deadline = None, time.time() + 600
        while time.time() < deadline:
            state = service_state(host, unit)
            if state == 'active':
                recovered = round(time.time(), 1)
                break
            time.sleep(10)
        log(out, 'fault', name=name, phase='recovered' if recovered else 'not-recovered', host=host, unit=unit,
            state=service_state(host, unit))
        if not recovered:
            log(out, 'incident', what='fault not recovered', name=name, host=host, unit=unit)


# Runs

def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO / 'src'))
    from neuroshard.assistant import network

    descriptor = network.descriptor()
    harness = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    CONTEXT.update(run=uuid.uuid4().hex[:12], measurement=MEASUREMENT_VERSION, harness=harness,
                   runtime=args.release, repo=revision(REPO))
    command = args.client.split() + ['assistant', 'chat']
    log(out, 'start', hours=args.hours, chat_every=args.chat_every, probe_every=args.probe_every,
        concurrency=args.concurrency, burst_every=args.burst_every, faults=args.fault, client=args.client)
    stop = threading.Event()
    threading.Thread(target=monitor, args=(out, descriptor, network, args.probe_every, stop), daemon=True).start()
    began = time.time()
    schedule = sorted((began + float(at) * 3600, name, float(hold))
                      for name, at, hold in (spec.split(':') for spec in args.fault))
    end, next_chat, index, rounds, pending = began + args.hours * 3600, began, 0, 0, []
    wait = (descriptor.get('challenge_blocks') or 600) * 2 + 900
    while time.time() < end:
        now = time.time()
        while schedule and schedule[0][0] <= now:
            _, name, hold = schedule.pop(0)
            threading.Thread(target=fault, args=(out, name, hold), daemon=True).start()
        if now >= next_chat:
            rounds += 1
            width = args.concurrency if args.burst_every and rounds % args.burst_every == 0 else 1
            if width > 1:
                log(out, 'burst', width=width, round=rounds)
            threads = []
            for slot in range(width):
                index += 1
                question, expect = QUESTIONS[index % len(QUESTIONS)]
                home = Path(args.client_home.format(slot=slot))
                threads.append(threading.Thread(target=conversation, args=(
                    out, index, question, expect, command, home, pending), daemon=True))
            for thread in threads:
                thread.start()
            for thread in threads:  # conversations only; monitoring and faults run in their own threads
                thread.join()
            next_chat = time.time() + args.chat_every
        settle(out, descriptor, network, pending, wait)
        time.sleep(5)
    stop.set()
    log(out, 'end')


def summary(args):
    events = [json.loads(line) for line in (Path(args.out) / 'events.jsonl').read_text().splitlines()]
    runs = {}
    for event in events:
        # Version-1 events carry no run ID: label them by the start that preceded them.
        key = event.get('run') or f"v1-{event['unix'] if event['kind'] == 'start' else ''}"
        if not event.get('run'):
            key = runs.get('_last_v1', key) if event['kind'] != 'start' else key
            runs['_last_v1'] = key
        runs.setdefault(key, []).append(event)
    runs.pop('_last_v1', None)
    report = {}
    for key, items in runs.items():
        talks = [e for e in items if e['kind'] == 'conversation']
        probes = [e for e in items if e['kind'] == 'probe']
        stamps = [p['unix'] for p in probes]
        gaps = [round(b - a, 1) for a, b in zip(stamps, stamps[1:])]
        ok = sorted(e['seconds'] for e in talks if e.get('transport_ok', e.get('ok')))
        pick = lambda q: ok[min(len(ok) - 1, int(q * len(ok)))] if ok else None
        incidents = {}
        for e in items:
            if e['kind'] == 'incident':
                incidents[e['what']] = incidents.get(e['what'], 0) + 1
        settlements = [e for e in items if e['kind'] == 'settlement']
        report[key] = {
            'measurement': items[0].get('measurement', 1), 'runtime': items[0].get('runtime'),
            'harness': items[0].get('harness'), 'from_unix': items[0]['unix'], 'to_unix': items[-1]['unix'],
            'hours': round((items[-1]['unix'] - items[0]['unix']) / 3600, 2),
            'probes': len(probes), 'probe_gap_max_s': max(gaps) if gaps else None,
            'probes_rpcs_agree': sum(1 for p in probes if p.get('agree')) if probes and 'agree' in probes[0] else None,
            'conversations': len(talks),
            'transport_ok': sum(1 for e in talks if e.get('transport_ok', e.get('ok'))),
            'correct': sum(1 for e in talks if e.get('correct')) if talks and 'correct' in talks[0] else None,
            'settled': sum(1 for e in settlements if e['status'] == 'settled'),
            'expired': sum(1 for e in settlements if e['status'] == 'expired'),
            'latency_p50_s': pick(0.5), 'latency_p95_s': pick(0.95),
            'bursts': sum(1 for e in items if e['kind'] == 'burst'),
            'faults': [{k: e.get(k) for k in ('name', 'phase', 'exit', 'before', 'after', 'state')}
                       for e in items if e['kind'] == 'fault'],
            'incidents_by_kind': incidents}
    print(json.dumps(report, indent=1))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    go = sub.add_parser('run')
    go.add_argument('--out', required=True)
    go.add_argument('--hours', type=float, default=72)
    go.add_argument('--release', default='pilot-1')
    go.add_argument('--probe-every', type=int, default=60)
    go.add_argument('--chat-every', type=int, default=1200)
    go.add_argument('--concurrency', type=int, default=2, help='conversations at once in a burst')
    go.add_argument('--burst-every', type=int, default=6, help='rounds between bursts (0: never)')
    go.add_argument('--client', default=f'{sys.executable} -m neuroshard.client.cli')
    go.add_argument('--client-home', default='/home/ubuntu/.neuroshard/soak-client-{slot}',
                    help='one client home per concurrent slot; {slot} is replaced')
    go.add_argument('--fault', action='append', default=[], help='NAME:AT_HOURS:HOLD_SECONDS, e.g. stop-validator:6:300')
    look = sub.add_parser('summary')
    look.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args) if args.action == 'run' else summary(args)


if __name__ == '__main__':
    main()
