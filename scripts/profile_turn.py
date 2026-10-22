#!/usr/bin/env python3
"""Client-side timing of one paid conversation, by phase, with explicit span scopes.

Runs the unmodified ``assistant chat`` client in-process. Wrappers only record time; behaviour is unchanged.

**Spans form a tree.** Every span has an ID and a parent ID (the span open on the same thread when it began), and
each turn has its own ID. Categories are computed only from a span's own descendants, so nothing can leak from
one turn into the next, and every derived category is checked to be nonnegative and to sum to its parent.
Timestamps are unrounded monotonic seconds; rounding happens only when printing.

**What each category is, and is not:**
  setup.download            ``network.fetch_stage``: stage files downloaded (cold) or verified on disk (warm)
  setup.initialization      tokenizer load, shard-0 partition load and its warm-up pass
  setup.balance             the funding-balance check
  admission                 ``Admitter.open`` as a whole; its verified chain reads are the reads inside it
  turn                      one ``Conversation.say``
    turn.exchange_path      time inside ``ServingDriver.exchange``: sending to shard 1, transcript signing, the
                            client relaying each intermediate shard's tensor onward, receiving the last shard's
                            reply and verifying its signature. It includes owner computation but does NOT isolate
                            it: transport, client relaying, hashing, signatures and owner-side recording are inside.
    turn.local_partition    time inside this device's own partition/norm/logits calls (stage 0)
    turn.pass_other         the rest of each model-pass method: control messages, token selection, tensors
    turn.outside_passes     the rest of the turn: templating, tokenization, tools, bookkeeping
Owner computation is not measured here; that needs a separately reviewed owner-side timing change.

**Results survive failure.** The run's identity is written before the client starts, events are appended to a
JSON-lines file as they happen, and the report is written in ``finally`` with an explicit ``outcome`` (completed,
client-exit, client-exception, interrupted). A failing post-run diagnostic is recorded, never fatal to the result.
The exit status follows the recorded outcome: 0 only if the conversation completed.

    python scripts/profile_turn.py --home ~/.neuroshard/profile-client-0 --out /tmp/profile \\
        --label warm-quiet --question "..." --question "..."
"""
import argparse
import itertools
import json
import os
import platform
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

T0 = time.monotonic()
WALL0 = time.time()


def process_age():
    """Seconds since this process started (from /proc), so interpreter start-up is counted too."""
    try:
        ticks = int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
        boot = float(Path('/proc/uptime').read_text().split()[0])
        return boot - ticks / os.sysconf('SC_CLK_TCK')
    except (OSError, ValueError, IndexError):
        return None


STARTUP_BEFORE_PROFILER = process_age()


class Recorder:
    """Spans with IDs and parents, appended to a JSON-lines file as they close (crash-safe)."""

    def __init__(self, path=None):
        self.path, self.lock, self.ids = path, threading.Lock(), itertools.count(1)
        self.spans, self.events, self.local = [], [], threading.local()
        self.turn = None  # the open turn's ID; set only by the turn span

    def stack(self):
        if not hasattr(self.local, 'stack'):
            self.local.stack = []
        return self.local.stack

    def write(self, record):
        with self.lock:
            (self.spans if record['type'] == 'span' else self.events).append(record)
            if self.path is not None:
                with self.path.open('a') as handle:
                    handle.write(json.dumps(record) + '\n')

    def event(self, kind, **fields):
        self.write({'type': 'event', 'kind': kind, 't': time.monotonic() - T0, 'turn': self.turn, **fields})

    def span(self, name, **fields):
        return Span(self, name, fields)


class Span:
    def __init__(self, recorder, name, fields):
        self.recorder, self.name, self.fields = recorder, name, fields

    def __enter__(self):
        stack = self.recorder.stack()
        self.id = next(self.recorder.ids)
        self.parent = stack[-1].id if stack else None
        if self.name == 'turn':
            self.recorder.turn = self.id
        self.turn = self.recorder.turn
        self.start = time.monotonic() - T0
        stack.append(self)
        return self

    def __exit__(self, kind, value, tb):
        end = time.monotonic() - T0
        stack = self.recorder.stack()
        stack.pop()
        if self.name == 'turn':
            self.recorder.turn = None
        self.recorder.write({'type': 'span', 'id': self.id, 'parent': self.parent, 'turn': self.turn, 'name': self.name,
                             'start': self.start, 'end': end, 'seconds': end - self.start,
                             'error': kind.__name__ if kind else None, **self.fields})


REC = Recorder()


def timed(owner, attribute, name, **fields):
    """Wrap ``owner.attribute`` in a span; returns the original."""
    original = getattr(owner, attribute)

    def wrapper(*args, **kwargs):
        with REC.span(name, **fields):
            return original(*args, **kwargs)

    wrapper.__wrapped__ = original
    setattr(owner, attribute, wrapper)
    return original


def instrument():
    """Timing wrappers on the client's phases. Each only records; behaviour is unchanged.

    The pinned numerical environment is set first, exactly as ``assistant chat`` does, before anything imports
    torch; the CLI's own later call is then skipped (it would refuse once torch is loaded)."""
    with REC.span('setup.import'):
        from neuroshard.evolution import granite_shard_execution as runtime
        runtime.configure()
        runtime.configure = lambda: None  # applied identically above, before torch loaded
        from neuroshard.assistant import admission_client, network, public
        from neuroshard.client import cli
        from neuroshard.evolution import granite_tokenizer
        from neuroshard.evolution.sharded import granite, granite_audit, granite_serving as serving
        from neuroshard.inference import relay
    timed(network, 'fetch_stage', 'setup.download')
    timed(granite_tokenizer, 'load', 'setup.init.tokenizer')
    original_load = granite.load_partition

    def load_partition(*args, **kwargs):
        with REC.span('setup.init.partition_load'):
            loaded = original_load(*args, **kwargs)
        partition = loaded[0] if isinstance(loaded, tuple) else loaded  # (partition, manifest)
        timed(partition, 'warm_up', 'setup.init.warm_up')
        for method in ('embed', 'norm', 'logits'):
            timed(partition, method, 'local.partition')
        # The partition's forward (``partition(...)``) is its class's __call__: wrap it on this instance's class.
        cls = type(partition)
        if not getattr(cls.__call__, '__wrapped__', None):
            timed(cls, '__call__', 'local.partition')
        return loaded

    granite.load_partition = load_partition
    timed(network, 'ensure_balance', 'setup.balance')
    timed(network.Chain, 'state', 'chain.state')  # each verified read, attributed by its parent span
    timed(network.Conversation, '__init__', 'conversation.setup')
    timed(admission_client.Admitter, 'open', 'admission')
    for method, name in (('recover', 'admission.recover'), ('reserve_shards', 'admission.reserve_round'),
                         ('held', 'admission.hold'), ('admit', 'admission.hello')):
        timed(admission_client.Admitter, method, name)
    timed(admission_client.Backoff, 'wait', 'admission.backoff_wait')
    timed(network.Conversation, 'say', 'turn')
    timed(network.Conversation, 'close', 'close')
    timed(serving.ServingDriver, 'exchange', 'exchange')
    for method in ('feature', 'router_forward', 'next_token', 'verify'):
        timed(serving.ServingDriver, method, 'pass')
    timed(serving.ServingDriver, 'control', 'control')
    timed(granite_audit.Link, 'send', 'exchange.sign')
    timed(granite_audit.Link, 'receive', 'exchange.verify')
    timed(relay, 'send_frame', 'frame.send')
    timed(relay, 'receive_frame', 'frame.receive')  # blocked until an owner's frame arrives: includes owner work
    original_responder = serving.responder

    def responder(*args, **kwargs):
        respond = original_responder(*args, **kwargs)

        def timed_respond(messages, tools):
            with REC.span('generation'):
                result = respond(messages, tools)
            REC.event('generation', tokens=len(result.get('token_ids') or []), passes=result.get('passes'),
                      input_tokens=len(result.get('input_token_ids') or []), reused=result.get('reused_prefix_tokens'))
            return result
        return timed_respond

    serving.responder = responder
    original_workspace = public.Session.workspace

    def workspace(self, tools):
        world = original_workspace(self, tools)
        timed(world, 'execute', 'tools')
        return world

    public.Session.workspace = workspace
    return cli, network


def analyse(spans, events):
    """Per-phase and per-turn categories from the span tree; every category is checked nonnegative and every
    turn's categories are checked to sum to the turn."""
    by_id = {s['id']: s for s in spans}
    children = {}
    for s in spans:
        children.setdefault(s['parent'], []).append(s)

    def descendants(root):
        out, todo = [], list(children.get(root['id'], []))
        while todo:
            node = todo.pop()
            out.append(node)
            todo.extend(children.get(node['id'], []))
        return out

    def ancestors(span):
        node = by_id.get(span['parent'])
        while node is not None:
            yield node
            node = by_id.get(node['parent'])

    def outermost(nodes, name):
        return [n for n in nodes if n['name'] == name and not any(a['name'] == name for a in ancestors(n))]

    total = lambda nodes: sum(n['seconds'] for n in nodes)  # noqa: E731
    roots = [s for s in spans if s['parent'] is None]
    setup = {}
    for name in ('setup.import', 'setup.download', 'setup.balance'):
        setup[name] = total(outermost(spans, name))
    setup['setup.initialization'] = sum(total(outermost(spans, n)) for n in
                                        ('setup.init.tokenizer', 'setup.init.partition_load', 'setup.init.warm_up'))
    admission = outermost(spans, 'admission')
    inside = [d for a in admission for d in descendants(a)]
    setup['admission'] = total(admission)
    setup['admission.chain_reads'] = len(outermost(inside, 'chain.state'))
    setup['admission.chain_read_seconds'] = total(outermost(inside, 'chain.state'))
    setup['admission.reserve_rounds'] = len(outermost(inside, 'admission.reserve_round'))
    setup['admission.backoff_wait'] = total(outermost(inside, 'admission.backoff_wait'))
    other_reads = [s for s in outermost(spans, 'chain.state') if s not in inside]
    setup['chain_reads_outside_admission'] = len(other_reads)
    setup['chain_read_seconds_outside_admission'] = total(other_reads)
    turns, violations = [], []
    for turn in sorted(outermost(spans, 'turn'), key=lambda s: s['start']):
        nodes = descendants(turn)
        passes = outermost(nodes, 'pass')
        exchanges = outermost(nodes, 'exchange')
        exchange_nodes = [d for e in exchanges for d in descendants(e)]
        local = [n for n in outermost(nodes, 'local.partition') if not any(a['name'] == 'exchange' for a in ancestors(n))]
        tools = outermost(nodes, 'tools')
        row = {
            'turn_id': turn['id'], 'seconds': turn['seconds'], 'model_passes': len(passes),
            'exchange_path_s': total(exchanges),
            'exchange.blocked_on_owner_frames_s': total(outermost(exchange_nodes, 'frame.receive')),
            'exchange.send_s': total(outermost(exchange_nodes, 'frame.send')),
            'exchange.sign_s': total(outermost(exchange_nodes, 'exchange.sign')),
            'exchange.verify_s': total(outermost(exchange_nodes, 'exchange.verify')),
            'local_partition_s': total(local),
            'tools_s': total(tools),
        }
        row['exchange.other_s'] = row['exchange_path_s'] - sum(
            row[k] for k in ('exchange.blocked_on_owner_frames_s', 'exchange.send_s', 'exchange.sign_s',
                             'exchange.verify_s'))
        row['pass_other_s'] = total(passes) - row['exchange_path_s'] - row['local_partition_s']
        row['outside_passes_s'] = turn['seconds'] - total(passes)
        gens = [e for e in events if e['kind'] == 'generation' and e['turn'] == turn['id']]
        row.update(generations=len(gens), tokens_generated=sum(g['tokens'] for g in gens),
                   input_tokens=[g['input_tokens'] for g in gens], reused_prefix=[g['reused'] for g in gens])
        for key in ('exchange.other_s', 'pass_other_s', 'outside_passes_s'):
            if row[key] < -1e-6:
                violations.append(f"turn {turn['id']}: {key} = {row[key]:.6f} < 0")
        parts = row['exchange_path_s'] + row['local_partition_s'] + row['pass_other_s'] + row['outside_passes_s']
        if abs(parts - turn['seconds']) > 1e-6:
            violations.append(f"turn {turn['id']}: categories sum to {parts:.6f}, not {turn['seconds']:.6f}")
        if any(n['turn'] != turn['id'] for n in nodes):
            violations.append(f"turn {turn['id']}: a descendant span carries another turn's ID")
        turns.append(row)
    first_turn = min((t for t in outermost(spans, 'turn')), key=lambda s: s['start'], default=None)
    launch_to_first_reply = None
    if first_turn is not None:
        launch_to_first_reply = (STARTUP_BEFORE_PROFILER or 0) + first_turn['end']
    return {'setup': setup, 'turns': turns, 'launch_to_end_of_first_turn_s': launch_to_first_reply,
            'invariant_violations': violations, 'roots': sorted({r['name'] for r in roots})}


class CpuSampler(threading.Thread):
    """Every second: this host's CPU busy fraction and each other process's CPU use (in CPUs)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self.stop = [], threading.Event()
        self.tick = os.sysconf('SC_CLK_TCK')

    @staticmethod
    def totals():
        fields = [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:]]
        return sum(fields), fields[3] + fields[4]

    @staticmethod
    def processes():
        usage = {}
        for entry in Path('/proc').iterdir():
            if entry.name.isdigit():
                try:
                    parts = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
                    name = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')[:120]
                    usage[int(entry.name)] = (int(parts[11]) + int(parts[12]), name)
                except (OSError, ValueError, IndexError):
                    continue
        return usage

    def run(self):
        me = os.getpid()
        total, idle = self.totals()
        procs, last = self.processes(), time.monotonic()
        while not self.stop.wait(1.0):
            now_total, now_idle = self.totals()
            now_procs, now = self.processes(), time.monotonic()
            elapsed = now - last
            others = {}
            for pid, (ticks, name) in now_procs.items():
                if pid != me and pid in procs:
                    share = (ticks - procs[pid][0]) / self.tick / elapsed
                    if share >= 0.05:
                        others[name] = others.get(name, 0) + share
            mine = (now_procs.get(me, (0,))[0] - procs.get(me, (0,))[0]) / self.tick / elapsed
            self.samples.append({'t': now - T0, 'host_busy': 1 - (now_idle - idle) / max(1, now_total - total),
                                 'self_cpus': mine, 'others_cpus': others})
            total, idle, procs, last = now_total, now_idle, now_procs, now


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--home', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path, help='directory: <run>.events.jsonl and <run>.json')
    parser.add_argument('--label', required=True, help='the condition, e.g. warm-quiet, warm-synthetic, warm-growth')
    parser.add_argument('--question', action='append', required=True, help='one per user turn, in order')
    parser.add_argument('--wait', type=int, default=600)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(WALL0))}-{args.label}-{uuid.uuid4().hex[:6]}"
    REC.path = args.out / f'{run}.events.jsonl'
    identity = {'run': run, 'label': args.label, 'wall_started_unix': WALL0, 'host': platform.node(),
                'cpus': os.cpu_count(), 'omp_num_threads': os.environ.get('OMP_NUM_THREADS'),
                'home': str(args.home), 'questions': args.question, 'wait': args.wait, 'argv': sys.argv,
                'source': os.environ.get('PYTHONPATH'), 'process_age_at_start_s': STARTUP_BEFORE_PROFILER}
    REC.write({'type': 'event', 'kind': 'identity', 't': 0.0, 'turn': None, **identity})
    sampler = CpuSampler()
    sampler.start()
    outcome, code, failure, diagnostics = 'interrupted', None, None, {}
    stdout = None
    try:
        cli, network = instrument()
        import contextlib
        import io
        sys.stdin = io.StringIO(''.join(q + '\n' for q in args.question) + '/quit\n')
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            try:
                with REC.span('cli.total'):
                    cli.main(['assistant', 'chat', '--home', str(args.home), '--wait', str(args.wait)])
                outcome, code = 'completed', 0
            except SystemExit as stop:
                outcome, code = ('completed' if stop.code in (0, None) else 'client-exit'), stop.code
    except BaseException as exc:  # noqa: BLE001 - recorded, then the report is still written
        outcome, failure = 'client-exception', ''.join(traceback.format_exception(exc))[-4000:]
    finally:
        sampler.stop.set()
        sampler.join(5)
        try:
            from neuroshard.assistant import network as net
            state = net.chain(net.descriptor(), args.home).state()
            diagnostics['owners'] = {k[:12]: o.get('endpoint') for k, o in state['owners'].items() if o['status'] == 'active'}
            diagnostics['height'] = state['height']
        except Exception as exc:  # noqa: BLE001 - a diagnostic never erases the result
            diagnostics['error'] = f'{type(exc).__name__}: {exc}'[:300]
        try:
            analysis = analyse(REC.spans, REC.events)
        except Exception as exc:  # noqa: BLE001
            analysis = {'error': f'analysis failed: {type(exc).__name__}: {exc}'}
        report = {**identity, 'outcome': outcome, 'exit': code, 'failure': failure,
                  'client_output': stdout.getvalue() if stdout is not None else None,
                  'analysis': analysis, 'diagnostics': diagnostics, 'cpu_samples': sampler.samples}
        (args.out / f'{run}.json').write_text(json.dumps(report, indent=1))
        print(json.dumps({'run': run, 'outcome': outcome, 'analysis': analysis}, indent=1, default=str),
              file=sys.__stderr__)
    sys.exit(0 if outcome == 'completed' else 1)


if __name__ == '__main__':
    main()
