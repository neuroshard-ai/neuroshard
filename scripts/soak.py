"""Multi-day soak of a frozen release: external probes, concurrent paid conversations, an incident log.

Runs on a machine outside the fleet's roles (it needs the client's stage 0, about 6 GB of memory). Every
``--probe-every`` seconds it records, from every validator RPC, height, signers and agreement, and from every
owner endpoint, TCP reachability. Every ``--chat-every`` seconds it starts ``--concurrency`` conversations at
once, each a fresh paid job through the network, and records latency, outcome and settlement. Anything outside
the expected envelope becomes an incident line. ``--fault`` schedules controlled failures (see FAULTS).

Everything goes to JSON lines under ``--out``; ``summary`` prints the totals for the paper.

    python scripts/soak.py run --out .neuroshard/soak-pilot-1 --hours 72
    python scripts/soak.py summary --out .neuroshard/soak-pilot-1
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

QUESTIONS = [
    'What is the review interval in the latest approved Cedar 6581 delivery plan?',
    'What is the baseline quantity in the latest approved Cedar 6581 plan?',
    'Which Cedar 6581 plan revisions are approved?',
    'Is there an approved delivery plan for Cedar 6581 Annex?',
]
EXPECT = {'max_seconds': 300, 'min_signers': 4, 'max_height_lag': 10}


def rpc(address, method, timeout=8):
    with urllib.request.urlopen(f'{address}/{method}', timeout=timeout) as response:
        return json.loads(response.read())['result']


def log(out, kind, **fields):
    record = {'unix': round(time.time(), 3), 'kind': kind, **fields}
    with (out / 'events.jsonl').open('a') as handle:
        handle.write(json.dumps(record, sort_keys=True) + '\n')
    if kind == 'incident':
        print(json.dumps(record, sort_keys=True), flush=True)
    return record


def probe(out, descriptor):
    """One look at the network from outside: each RPC's height and signers, each owner endpoint's port."""
    rpcs = [descriptor['rpc'], *descriptor.get('rpcs', [])] if isinstance(descriptor['rpc'], str) else descriptor['rpc']
    heights, signers = {}, None
    for address in rpcs:
        try:
            status = rpc(address, 'status')
            heights[address] = int(status['sync_info']['latest_block_height'])
            if signers is None:
                commit = rpc(address, 'commit')['signed_header']['commit']
                signers = sum(1 for s in commit['signatures'] if s['block_id_flag'] == 2)
        except (OSError, KeyError, ValueError) as error:
            heights[address] = None
            log(out, 'incident', what='rpc unreachable', rpc=address, error=str(error)[:200])
    live = [h for h in heights.values() if h is not None]
    top = max(live) if live else None
    for address, height in heights.items():
        if height is not None and top - height > EXPECT['max_height_lag']:
            log(out, 'incident', what='rpc behind', rpc=address, behind=top - height)
    if signers is not None and signers < EXPECT['min_signers']:
        log(out, 'incident', what='few signers', signers=signers, height=top)
    owners = {}
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
        from neuroshard.assistant import network

        state = network.chain(descriptor, out / 'probe-home').state()
        for key, owner in sorted(state['owners'].items()):
            if owner['status'] != 'active' or not owner.get('endpoint'):
                continue
            host, port = owner['endpoint'].rsplit(':', 1)
            try:
                socket.create_connection((host, int(port)), timeout=5).close()
                owners[key[:10]] = True
            except OSError:
                owners[key[:10]] = False
                log(out, 'incident', what='owner unreachable', owner=key[:10], shard=owner['shard'],
                    endpoint=owner['endpoint'])
        verified = state['height']
    except Exception as error:  # noqa: BLE001 - a failed verified read is itself an incident
        verified = None
        log(out, 'incident', what='verified read failed', error=f'{type(error).__name__}: {error}'[:300])
    return log(out, 'probe', heights=heights, top=top, signers=signers, owners=owners, verified=verified)


def conversation(out, index, question, command, home):
    started = time.monotonic()
    try:
        result = subprocess.run(command + ['--home', str(home)], input=f'{question}\n/quit\n', text=True,
                                capture_output=True, timeout=EXPECT['max_seconds'] * 3)
        text = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        text, result = '', None
    seconds = round(time.monotonic() - started, 1)
    job = next((line.split()[1].rstrip(':') for line in text.splitlines() if line.startswith('Job ')), None)
    reply = next((line.split('): ', 1)[1] for line in text.splitlines() if 'assistant (' in line and '): ' in line), None)
    ok = bool(job and reply)
    record = log(out, 'conversation', index=index, question=question, job=job, seconds=seconds, ok=ok,
                 reply=(reply or '')[:300], tail=None if ok else text[-600:])
    if not ok:
        log(out, 'incident', what='conversation failed', index=index, seconds=seconds, tail=text[-300:])
    elif seconds > EXPECT['max_seconds']:
        log(out, 'incident', what='conversation slow', index=index, seconds=seconds)
    return record


FAULTS = {
    # name: (host, command to inject, command to restore); restore runs after ``hold`` seconds
    'stop-validator': ('172.31.91.19', 'sudo systemctl stop neuroshard-validator', 'sudo systemctl start neuroshard-validator'),
    'stop-owner-2b': ('172.31.41.136', 'sudo systemctl stop neuroshard-owner', 'sudo systemctl start neuroshard-owner'),
    'stop-auditor': ('172.31.30.33', 'sudo systemctl stop neuroshard-auditor', 'sudo systemctl start neuroshard-auditor'),
    'restart-seed': ('172.31.73.85', 'sudo systemctl restart neuroshard-seed', 'true'),
}


def fault(out, name, hold):
    host, inject, restore = FAULTS[name]
    ssh = ['ssh', '-i', os.path.expanduser('~/.ssh/id_ed25519'), '-o', 'BatchMode=yes', f'ubuntu@{host}']
    log(out, 'fault', name=name, phase='inject', host=host)
    subprocess.run(ssh + [inject], check=False, capture_output=True, timeout=120)
    time.sleep(hold)
    subprocess.run(ssh + [restore], check=False, capture_output=True, timeout=120)
    log(out, 'fault', name=name, phase='restore', host=host, held=hold)


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from neuroshard.assistant import network

    descriptor = network.descriptor()
    command = args.client.split() + ['assistant', 'chat']
    log(out, 'start', release=args.release, hours=args.hours, concurrency=args.concurrency, faults=args.fault)
    end = time.time() + args.hours * 3600
    next_probe = next_chat = time.time()
    schedule = sorted((time.time() + float(at) * 3600, name, float(hold))
                      for name, at, hold in (spec.split(':') for spec in args.fault))
    index = 0
    while time.time() < end:
        now = time.time()
        if now >= next_probe:
            probe(out, descriptor)
            next_probe = now + args.probe_every
        while schedule and schedule[0][0] <= now:
            _, name, hold = schedule.pop(0)
            threading.Thread(target=fault, args=(out, name, hold), daemon=True).start()
        if now >= next_chat:
            threads, rounds = [], getattr(run, 'rounds', 0) + 1
            run.rounds = rounds
            # One conversation at a time is the baseline; every ``--burst-every`` rounds, ``--concurrency`` at once
            # measures overload (an owner serves one conversation at a time; shard 1 has a single owner).
            width = args.concurrency if args.burst_every and rounds % args.burst_every == 0 else 1
            if width > 1:
                log(out, 'burst', width=width, round=rounds)
            for slot in range(width):
                index += 1
                home = out / 'clients' / str(slot)
                threads.append(threading.Thread(target=conversation, args=(
                    out, index, QUESTIONS[index % len(QUESTIONS)], command, home), daemon=True))
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            next_chat = time.time() + args.chat_every
        time.sleep(5)
    log(out, 'end')


def summary(args):
    events = [json.loads(line) for line in (Path(args.out) / 'events.jsonl').read_text().splitlines()]
    talks = [e for e in events if e['kind'] == 'conversation']
    probes = [e for e in events if e['kind'] == 'probe']
    incidents = [e for e in events if e['kind'] == 'incident']
    seconds = sorted(e['seconds'] for e in talks if e['ok'])
    pick = lambda q: seconds[min(len(seconds) - 1, int(q * len(seconds)))] if seconds else None
    span = (events[-1]['unix'] - events[0]['unix']) / 3600 if events else 0
    by_kind = {}
    for incident in incidents:
        by_kind[incident['what']] = by_kind.get(incident['what'], 0) + 1
    bursts = {e['unix'] for e in events if e['kind'] == 'burst'}
    print(json.dumps({
        'hours': round(span, 2), 'probes': len(probes), 'bursts': len(bursts),
        'probes_all_signers': sum(1 for p in probes if p.get('signers') == 5),
        'conversations': len(talks), 'completed': sum(e['ok'] for e in talks),
        'latency_p50_s': pick(0.5), 'latency_p95_s': pick(0.95),
        'faults': [e for e in events if e['kind'] == 'fault'],
        'incidents_by_kind': by_kind}, indent=1))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    go = sub.add_parser('run')
    go.add_argument('--out', required=True)
    go.add_argument('--hours', type=float, default=72)
    go.add_argument('--release', default='pilot-1')
    go.add_argument('--probe-every', type=int, default=60)
    go.add_argument('--chat-every', type=int, default=900)
    go.add_argument('--concurrency', type=int, default=2, help='conversations at once in a burst')
    go.add_argument('--burst-every', type=int, default=8, help='rounds between bursts (0: never)')
    go.add_argument('--client', default=f'{sys.executable} -m neuroshard.client.cli')
    go.add_argument('--fault', action='append', default=[], help='NAME:AT_HOURS:HOLD_SECONDS, e.g. stop-validator:6:300')
    look = sub.add_parser('summary')
    look.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args) if args.action == 'run' else summary(args)


if __name__ == '__main__':
    main()
