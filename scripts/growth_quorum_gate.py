#!/usr/bin/env python3
"""Gate 5b: growth proofs judged by mixed-version validators on a real isolated chain.

The chain is an isolated local CometBFT network with fresh test keys and its own chain ID. It never touches the
pilot fleet.
- Five validators run the real ABCI application (`neuroshard.inference.optimistic_app`), each from its own checkout
  and venv: pilot-1 or pilot-2-rc2.
- Each validator's real cached worker judges growth proofs through `growth_check`, with the pinned parent model or
  a tiny model.
- Every scenario submits real `growth_open`, `growth_commit`, `growth_challenge` and `growth_prove` transactions.
- It records:
  - each validator's own verdict, read from its app log;
  - whether the proof committed, and the resulting bonds and penalties, from chain state;
  - that ordinary blocks kept coming;
  - process memory and CPU.

    python scripts/growth_quorum_gate.py plan                          # the stages and cases, no side effects
    python scripts/growth_quorum_gate.py run --out DIR --stage all-rc2 --case honest \\
        --pilot1 ~/pilot1:~/.venvs/pilot1 --rc2 ~/rc2:~/.venvs/rc2 --recipe R.json --start-state S --models DIR

The chain is torn down after each scenario. Keys are generated per run under DIR and never reused.
"""
import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_rules  # noqa: E402

# Stages of the proposed rollout order (model holders first). Which validators run rc2, which hold the model.
# Index 0-4 stand for the live validators in rollout order: .8.86, .91.19, .30.33 (the 3 model holders), .47.8, seed.
STAGES = {
    'today': {'rc2': [], 'model': [0, 1, 2]},
    'step1': {'rc2': [0], 'model': [0, 1, 2]},
    'step2': {'rc2': [0, 1], 'model': [0, 1, 2]},
    'step3': {'rc2': [0, 1, 2], 'model': [0, 1, 2]},
    'all-rc2': {'rc2': [0, 1, 2, 3, 4], 'model': [0, 1, 2]},
    'all-equipped': {'rc2': [0, 1, 2, 3, 4], 'model': [0, 1, 2, 3, 4]},
    # Hypothetical, never to be configured live: shows why the provisioning restriction matters.
    'forbidden-4-legacy-holders': {'rc2': [], 'model': [0, 1, 2, 3]},
}
CASES = ('honest', 'forged', 'legacy8')
MEASUREMENT = 5  # 3: proof to all at once; 4: delivery thread; 5: mandatory late recovery, fixed-rate reconciled transfers, observed commit times, agent path
HONEST_ROOT_KEY, LEGACY_ROOT_KEY = 'honest_root', 'legacy8_root'


HARNESS = hashlib.sha256(Path(__file__).read_bytes() + (Path(__file__).parent / 'gate_rules.py').read_bytes()).hexdigest()[:16]


def run_cmd(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def versions(tree, venv):
    py = f'{venv}/bin/python'
    probe = ('import importlib.metadata as m, platform, json; '
             'print(json.dumps({"python": platform.python_version(), '
             '**{p: m.version(p) for p in ("torch", "transformers", "certifi")}}))')
    out = run_cmd([py, '-c', probe]).stdout.strip()
    rev = run_cmd(['git', '-C', tree, 'describe', '--tags', '--always', '--dirty']).stdout.strip()
    pinned = run_cmd([py, '-c', 'from neuroshard.evolution import growth_replay as g; print(getattr(g, "STEP_THREADS", None))'],
                     env={**os.environ, 'PYTHONPATH': f'{tree}/src'}).stdout.strip()
    return {'tree': tree, 'revision': rev, 'step_threads': pinned, **json.loads(out or '{}')}


def rss_mb(pid):
    total = 0
    for p in [pid] + [int(c) for c in run_cmd(['pgrep', '-P', str(pid)]).stdout.split()]:
        try:
            total += int(re.search(r'VmRSS:\s+(\d+)', Path(f'/proc/{p}/status').read_text()).group(1))
        except (OSError, AttributeError):
            pass
    return total // 1024


def log_verdict(text):
    """One validator's verdict from its app log: 'True' (fraud proven), 'False', 'no-verdict (...)' or 'not-judged'.

    The cached worker logs ``verification <proof> verdict True|False``; a checker that raises (a validator without
    the model, for one) logs ``verification failed for <proof>`` and gives no verdict, which rejects any proposal
    carrying the proof."""
    lines = re.findall(r'(verification \w+ verdict (True|False)|verification failed for|has no verdict on this validator)', text)
    if not lines:
        return 'not-judged'
    if lines[-1][1]:
        return lines[-1][1]
    cause = re.findall(r'^(\w+Error)\b', text, re.M)
    return f'no-verdict ({cause[-1] if cause else "error"})'


class Gate:
    def __init__(self, args):
        self.args, self.out = args, Path(args.out).resolve()
        self.trees = {'pilot1': args.pilot1.split(':'), 'rc2': args.rc2.split(':')}
        # The harness process (and with --agent the growth agent it drives) imports --agent-src if given, else the
        # rc2 validator tree. Validator apps always run their own tree and venv.
        sys.path.insert(0, args.agent_src or f'{self.trees["rc2"][0]}/src')
        from neuroshard.inference import optimistic as ledger, optimistic_network as net
        self.ledger, self.net = ledger, net
        self.events = self.out / 'events.jsonl'

    def log(self, kind, **fields):
        self.out.mkdir(parents=True, exist_ok=True)
        row = {'kind': kind, 'unix': round(time.time(), 3), 'measurement': MEASUREMENT, 'harness': HARNESS, **fields}
        with self.events.open('a') as f:
            f.write(json.dumps(row, sort_keys=True) + '\n')
        print(json.dumps(row, sort_keys=True), flush=True)  # complete: the console log is a second copy

    # -- chain ------------------------------------------------------------------------------------------------
    def accounts(self, run):
        """Fresh secp256k1 test keys for this run only, signing for this run's chain ID (as the ledger verifies)."""
        from neuroshard.core.crypto.ecdsa import derive_keypair_from_token, ecdsa_sign
        ledger, chain_id = self.ledger, self.chain_id

        class Account:
            def __init__(self, name):
                self.key = derive_keypair_from_token(hashlib.sha256((name + os.urandom(16).hex()).encode()).hexdigest())
                self.public, self.nonce = self.key.public_key_bytes.hex(), 0

            def sign(self, kind, **fields):
                body = {'kind': kind, 'chain_id': chain_id, 'nonce': self.nonce, **fields}
                self.nonce += 1
                return {'body': body, 'public_key': self.public,
                        'signature': ecdsa_sign(ledger.canonical(body).decode(), self.key.private_key_bytes)}
        return {n: Account(f'gate5b-{run}-{n}') for n in ('sponsor', 'contributor', 'challenger', 'bystander')}

    def node(self, i, stage, home):
        store = home / f'store{i}'
        for sub in ('bundles', 'units'):
            (store / sub).mkdir(parents=True, exist_ok=True)
        # Like a live validator: growth recipes and proofs share the bundle store; serving shards are not held
        # (this gate judges growth only). The parent model resolves through NEUROSHARD_MODELS, or is absent.
        # Live validators hold serving shards, and loading them is what sets their worker to the pinned 8 threads
        # (``shard_checker``); ``--shards`` is a live validator's own ``shards`` entry, with its paths moved here.
        shards = json.loads(Path(self.args.shards).read_text()) if self.args.shards else {}
        return {'promotion_authority': None, 'approved_models': {}, 'runtime': 'granite-shard', 'threads': 8,
                'bundles': str(store / 'bundles'), 'growth_recipes': str(store / 'bundles'),
                'growth_bases': [self.base], 'growth_units': str(store / 'units'), 'shards': shards,
                'held_model_root': 'ab' * 32}

    def start(self, stage, run):
        spec = STAGES[stage]
        # CometBFT limits chain IDs to 50 characters; the run name (stage, case, time) can be longer.
        self.chain_id = 'gate5b-' + hashlib.sha256(run.encode()).hexdigest()[:16]
        home = self.out / 'chains' / run
        for old in (home, self.out / 'stores' / run):
            if old.exists():
                shutil.rmtree(old)
        self.accts = self.accounts(run)
        params = dict(self.ledger.PARAMS)  # growth params default to the live ones (growth.PARAMS: 40 and 40 blocks)
        if self.args.window:
            params.update(growth_window_blocks=self.args.window, growth_proof_blocks=self.args.window)
        terms = {'chain_id': self.chain_id, 'model_root': 'ab' * 32, 'shards': 3,
                 'allocations': {a.public: 100_000_000 for a in self.accts.values()}, 'params': params}
        stores = self.out / 'stores' / run  # outside the chain home, which initialize() requires to be empty
        nodes = [self.node(i, stage, stores) for i in range(5)]
        self.config = self.net.initialize(home, terms, nodes, base_port=self.args.base_port, block_seconds=1.0)
        self.stores = stores
        self.home, self.procs = home, {}
        for i, v in enumerate(self.config['validators']):
            flavour = 'rc2' if i in spec['rc2'] else 'pilot1'
            tree, venv = self.trees[flavour]
            env = {**os.environ, 'PYTHONPATH': f'{tree}/src', 'NEUROSHARD_UNITS': str(self.stores / f'store{i}' / 'units'),
                   'PYTHONUNBUFFERED': '1'}  # the verdict lines are read from these logs while the app runs
            # A validator without the model: an empty model directory, so verify_artifacts fails (no verdict).
            env['NEUROSHARD_MODELS'] = self.args.models if i in spec['model'] else str(self.stores / f'no-model-{i}')
            if self.args.validator_threads:
                env['OMP_NUM_THREADS'] = str(self.args.validator_threads)
            log = open(self.stores / f'app{i}.log', 'ab')
            self.procs[f'app{i}'] = subprocess.Popen(
                [f'{venv}/bin/python', '-m', 'neuroshard.inference.optimistic_app', '--home', v['home'],
                 '--port', str(v['abci'])], stdout=log, stderr=log, env=env, cwd=tree, start_new_session=True)
            v['flavour'], v['model'] = flavour, i in spec['model']
        # Apps holding shards load and warm them before they listen; CometBFT gives up on an ABCI port that is not
        # yet open, so every app must be listening first.
        deadline = time.monotonic() + 900
        for i in range(5):
            while 'ABCI listening' not in (self.stores / f'app{i}.log').read_text(errors='replace'):
                if self.procs[f'app{i}'].poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f'app{i} did not start; see {self.stores}/app{i}.log')
                time.sleep(2)
        self.log('apps-listening', run=run, seconds=round(900 - (deadline - time.monotonic()), 1),
                 rss_mb={name: rss_mb(p.pid) for name, p in self.procs.items()})
        for i, v in enumerate(self.config['validators']):
            log = open(self.stores / f'node{i}.log', 'ab')
            self.procs[f'node{i}'] = subprocess.Popen([self.config['engine'], 'start', '--home', v['home']],
                                                      stdout=log, stderr=log, start_new_session=True)
        self.net.wait_height(self.config, 2, timeout=180)
        self.log('chain-started', run=run, stage=stage, chain_id=self.chain_id,  # chain_id derives from run
                 validators=[{'index': i, 'flavour': v['flavour'], 'model': v['model']}
                             for i, v in enumerate(self.config['validators'])])

    def stop(self):
        for name, proc in getattr(self, 'procs', {}).items():
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        time.sleep(3)
        for proc in getattr(self, 'procs', {}).values():
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def height(self, i=0):
        return self.net.state(self.net.url(self.config, i))['height']

    def submit(self, who, kind, rpc=0, wait=True, **fields):
        acct = self.accts[who]
        state = self.net.state(self.net.url(self.config, rpc))
        nonce = acct.nonce = state['accounts'].get(acct.public, {'nonce': 0})['nonce']
        envelope = acct.sign(kind, **fields)  # sign() advances acct.nonce; wait on the nonce it signed with
        admitted = self.net.broadcast(self.net.url(self.config, rpc), envelope)
        if not wait:
            return envelope, admitted
        if admitted['code'] != 0:
            raise RuntimeError(f'{kind} refused at admission: {admitted}')
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            now = self.net.state(self.net.url(self.config, rpc))
            if now['accounts'].get(acct.public, {'nonce': 0})['nonce'] > nonce:
                return envelope, admitted
            time.sleep(0.5)
        raise TimeoutError(kind)

    # -- scenario (measurement 2) ----------------------------------------------------------------------------
    def chain_view(self, i):
        """What a validator's own proof store reads: its node's current state (as ``network.Chain.state``)."""
        net, config = self.net, self.config

        class View:
            def state(self):
                return net.state(net.url(config, i))
        return View()

    def start_store(self, i):
        from neuroshard.assistant import network
        port = self.args.base_port + i * 10 + 5
        self.servers[i] = network.serve_store(self.chain_view(i), '127.0.0.1', port, self.stores / f'store{i}' / 'bundles')
        self.log('store-started', run=self.run, index=i, port=port)

    def stop_store(self, i):
        server = self.servers.pop(i, None)
        if server is not None:
            server.shutdown()
            server.server_close()

    def deliver(self, root, payload, holders):
        """Post content-addressed bytes to every running store that does not hold them yet, as an agent does."""
        from neuroshard.assistant import audit
        for i in range(5):
            if i in holders or i not in self.servers:
                continue
            endpoint = f'http://127.0.0.1:{self.args.base_port + i * 10 + 5}/proofs'
            try:
                audit.post_proof(endpoint, root, payload, timeout=60)
                holders[i] = time.time()
                self.log('delivered', run=self.run, index=i, root=root[:16], bytes=len(payload))
            except Exception as error:  # noqa: BLE001 - a refusing or absent store is retried
                self.log('delivery-refused', run=self.run, index=i, root=root[:16], error=str(error)[:160])

    def sampler(self, stop):
        """Peak memory of every app, sampled once a second from before any material arrives until the end."""
        while not stop.is_set():
            for name, proc in list(self.procs.items()):
                if name.startswith('app') and proc.poll() is None:
                    self.peak[name] = max(self.peak.get(name, 0), rss_mb(proc.pid))
            stop.wait(1)

    def bystander(self, stop, online):
        """Ordinary transactions on a fixed arrival schedule: one self-transfer every ``ordinary_every`` seconds.

        Each attempt is recorded *before* it is sent, with its nonce. Scheduling stops at ``stop``; sends already
        in flight finish. ``reconcile_ordinary`` then matches every attempt against the chain's committed nonce,
        so none can silently disappear.
        """
        acct = self.accts['bystander']
        nonce = 0
        start = time.time()
        k = 0
        while not stop.is_set():
            due = start + k * self.args.ordinary_every
            if time.time() < due:
                stop.wait(due - time.time())
                continue
            k += 1
            rpc = next(i for i in range(5) if online(i))
            acct.nonce = nonce
            envelope = acct.sign('transfer', to=acct.public, amount=1)
            attempt = {'nonce': nonce, 'scheduled': round(due, 3), 'sent': round(time.time(), 3), 'rpc': rpc}
            self.ordinary.append(attempt)  # recorded before the send, so a stop or error can never drop it
            try:
                admitted = self.net.broadcast(self.net.url(self.config, rpc), envelope)
                attempt.update(admission=admitted['code'], admitted_at=round(time.time(), 3), log=admitted['log'][:120])
                if admitted['code'] == 0:
                    nonce += 1  # the next transfer follows this one; the chain orders them by nonce
            except Exception as error:  # noqa: BLE001
                attempt.update(error=str(error)[:120])
        self.ordinary_rpc = next(i for i in range(5) if online(i))

    def watch_ordinary(self, stop, online):
        """Observed confirmation time per admitted transfer: the first poll at which the chain's nonce exceeds it."""
        acct = self.accts['bystander']
        while not stop.is_set():
            try:
                rpc = next(i for i in range(5) if online(i))
                state = self.net.state(self.net.url(self.config, rpc))
                committed = state['accounts'].get(acct.public, {'nonce': 0})['nonce']
                now = round(time.time(), 3)
                for attempt in self.ordinary:
                    if attempt.get('admission') == 0 and 'confirmed_at' not in attempt and attempt['nonce'] < committed:
                        attempt.update(confirmed_at=now, height=state['height'],
                                       latency=round(now - attempt['scheduled'], 2))
            except Exception:  # noqa: BLE001 - polled again
                pass
            stop.wait(0.5)

    def reconcile_ordinary(self, final_nonce):
        """Every admitted attempt below the committed nonce is committed; the rest are explicitly not."""
        for attempt in self.ordinary:
            if attempt.get('admission') == 0:
                attempt['committed'] = attempt['nonce'] < final_nonce
            elif 'error' in attempt:
                attempt['committed'] = attempt['nonce'] < final_nonce  # a lost reply may still have committed
            else:
                attempt['committed'] = False

    def block_times(self, rpc, first, last):
        times, unreadable = {}, []
        for h in range(max(1, first), last + 1):
            try:
                header = self.net.rpc(self.net.url(self.config, rpc), 'block', {'height': str(h)})['block']['header']
                times[h] = datetime.fromisoformat(header['time'][:26].rstrip('Z')).replace(tzinfo=timezone.utc).timestamp()
            except Exception:  # noqa: BLE001 - recorded as incomplete evidence
                unreadable.append(h)
        return times, unreadable

    def proof_blocks(self, rpc, first, last):
        carried, unreadable = [], []
        for h in range(max(1, first), last + 1):
            try:
                block = self.net.rpc(self.net.url(self.config, rpc), 'block', {'height': str(h)})['block']
            except Exception:  # noqa: BLE001
                unreadable.append(h)
                continue
            for raw in block['data'].get('txs') or []:
                if json.loads(base64.b64decode(raw)).get('body', {}).get('kind') == 'growth_prove':
                    carried.append(h)
        return carried, unreadable

    def scenario(self, stage, case, offline=None, recover=False, missing_store=(), store_delay=None):
        import threading
        run = f'{stage}-{case}' + (f'-off{offline}' if offline is not None else '') + (
            f'-nostore{"".join(map(str, missing_store))}' if missing_store else '') + (
            f'-late{store_delay[0]}' if store_delay else '') + ('-agent' if self.args.agent else '') + f'-{int(time.time())}'
        self.run, self.servers, self.peak, self.ordinary = run, {}, {}, []
        self.late_started = self.fault_seen = None
        claimed = {'honest': self.roots[HONEST_ROOT_KEY], 'forged': 'cd' * 32, 'legacy8': self.roots[LEGACY_ROOT_KEY]}[case]
        stop = threading.Event()
        down = {offline} if offline is not None else set()
        result = {'run': run, 'stage': stage, 'case': case, 'offline': offline, 'missing_store': list(missing_store),
                  'store_delay': store_delay, 'prover': ('agent:' + (self.args.agent_src or 'rc2-tree')) if self.args.agent else 'harness'}
        try:
            self.start(stage, run)
            validators = [{'flavour': v['flavour'], 'model': v['model']} for v in self.config['validators']]
            result['expected_verdicts'] = expected = gate_rules.expected_verdicts(
                validators, case, offline=down, missing_store=set(missing_store),
                late_store={store_delay[0]} if store_delay else set())
            result['expected_penalty'] = gate_rules.expect_penalty(expected)
            self.log('scenario-plan', **result)
            for i in range(5):  # stores, except the ones this scenario keeps away; units are provisioned in advance
                for sha, payload in self.units.items():
                    (self.stores / f'store{i}' / 'units' / f'{sha}.safetensors').write_bytes(payload)
                if i not in missing_store and not (store_delay and store_delay[0] == i):
                    self.start_store(i)
            threading.Thread(target=self.sampler, args=(stop,), daemon=True).start()
            ordinary_stop = threading.Event()  # stops scheduling; in-flight sends finish before the summary
            sender = threading.Thread(target=self.bystander, args=(ordinary_stop, lambda i: i not in down), daemon=True)
            watcher = threading.Thread(target=self.watch_ordinary, args=(stop, lambda i: i not in down), daemon=True)
            sender.start()
            watcher.start()

            opened, _ = self.submit('sponsor', 'growth_open', recipe_root=self.recipe_root,
                                    start_root=self.recipe['start_root'], steps=1, segment_steps=1, reward=1_000_000)
            job = self.ledger.transaction_id(opened)
            recipe_holders = {}
            self.deliver(self.recipe_root, json.dumps(self.recipe).encode(), recipe_holders)  # as replication would
            self.submit('contributor', 'growth_commit', job_id=job, segment=0, step_roots=[claimed])
            challenge, _ = self.submit('challenger', 'growth_challenge', job_id=job, segment=0, step=0,
                                       proof_root=self.proof_root)
            challenge_id = self.ledger.transaction_id(challenge)
            state = self.net.state(self.net.url(self.config, 0))
            opened_at = state['growth'][job]['challenges'][challenge_id]['opened']
            deadline = state['growth'][job]['challenges'][challenge_id]['deadline']
            challenged_unix = time.time()
            result.update(job=job, challenge_opened=opened_at, deadline=deadline)
            self.log('challenge-committed', run=run, opened=opened_at, deadline=deadline, window=deadline - opened_at)
            if offline is not None:
                self.stop_store(offline)
                for name in (f'node{offline}', f'app{offline}'):
                    os.killpg(self.procs[name].pid, signal.SIGTERM)
                self.log('validator-offline', run=run, index=offline)
            rpc = next(i for i in range(5) if i not in down)

            # Deliver and prove as an agent does, until the challenge closes:
            # - delivery runs on its own thread, every 2 s, to every store that lacks the recipe or the bundle (a store
            #   that comes up late gets both), and a late store starts on its own timer;
            # - so neither waits behind a proof submission, which returns only after that validator's replay.
            holders, admissions, outcome, last_send = {}, [], 'undecided', 0.0

            if store_delay:  # the late store starts on its own timer, never behind a blocking call
                def late_start():
                    if not stop.wait(max(0.0, challenged_unix + store_delay[1] - time.time())):
                        self.late_started = time.time()
                        self.start_store(store_delay[0])
                threading.Thread(target=late_start, daemon=True).start()

            def delivery():
                while not stop.is_set():
                    try:
                        if self.net.state(self.net.url(self.config, rpc))['height'] > opened_at:
                            self.deliver(self.recipe_root, json.dumps(self.recipe).encode(), recipe_holders)
                            self.deliver(self.proof_root, self.bundle, holders)
                    except Exception as error:  # noqa: BLE001 - retried
                        self.log('delivery-error', run=run, error=str(error)[:160])
                    stop.wait(2)
            if self.args.agent:
                # The released growth agent proves, not this harness: its own check() loop, its saved bundle, its
                # deliver() to every store and Chain.submit_everywhere, through a light-client-verified Chain.
                agent_thread = threading.Thread(target=self.run_agent, args=(stop, job, challenge_id, holders, recipe_holders),
                                                daemon=True)
                agent_thread.start()
            else:
                threading.Thread(target=delivery, daemon=True).start()
            give_up = time.monotonic() + self.args.judge_timeout
            while time.monotonic() < give_up:
                now = self.net.state(self.net.url(self.config, rpc))
                if now['height'] > opened_at and not self.args.agent:
                    if holders and time.time() - last_send >= self.args.prove_every:
                        last_send = time.time()
                        acct = self.accts['challenger']
                        acct.nonce = now['accounts'][acct.public]['nonce']
                        envelope = acct.sign('growth_prove', job_id=job, challenge_id=challenge_id)
                        # broadcast_tx_sync returns only after that validator's CheckTx, which runs the replay. Sent
                        # to every online validator at once, as a challenger can, so all judge within the window.
                        def send(i, sent=time.time(), height=now['height'], envelope=envelope):
                            try:
                                r = self.net.broadcast(self.net.url(self.config, i), envelope)
                                return {'index': i, 'sent': sent, 'seconds': round(time.time() - sent, 2),
                                        'code': r['code'], 'log': r['log'][:120], 'height': height}
                            except Exception as error:  # noqa: BLE001
                                return {'index': i, 'sent': sent, 'error': str(error)[:120]}
                        senders = [i for i in range(5) if i not in down]
                        with ThreadPoolExecutor(max_workers=len(senders)) as pool:
                            admissions += list(pool.map(send, senders))
                now = self.net.state(self.net.url(self.config, rpc))
                faults = [f for f in now.get('growth_faults', []) if f['job_id'] == job]
                if faults:
                    outcome, result['fault_height'] = 'penalty-committed', faults[-1]['height']
                    self.fault_seen = time.time()
                    break
                gjob = now.get('growth', {}).get(job)
                if gjob is None or challenge_id not in gjob.get('challenges', {}):
                    # The challenge closed without a fault: it lapsed. The full window has passed by construction.
                    deadline_wait = time.monotonic() + 120
                    while job in now.get('growth', {}) and time.monotonic() < deadline_wait:
                        time.sleep(1)  # the one-step job settles in the next block
                        now = self.net.state(self.net.url(self.config, rpc))
                    if job not in now.get('growth', {}):
                        outcome = 'lapsed-and-settled'
                    break
                time.sleep(1)
            ordinary_stop.set()
            sender.join(timeout=120)  # drain: the last in-flight transfer is recorded either way
            time.sleep(3)  # let the last confirmations be observed and every app flush its verdict line
            stop.set()
            watcher.join(timeout=10)
            after = self.net.state(self.net.url(self.config, rpc))
            times, unreadable_t = self.block_times(rpc, opened_at - 3, after['height'])
            carried, unreadable_b = self.proof_blocks(rpc, opened_at, after['height'])
            verdicts = {i: ('offline' if i in down else log_verdict((self.stores / f'app{i}.log').read_text(errors='replace')))
                        for i in range(5)}
            result.update(
                outcome=outcome, verdicts=verdicts, final_height=after['height'], proof_in_blocks=carried,
                blocks_complete=not (unreadable_t or unreadable_b), unreadable_heights=sorted(set(unreadable_t + unreadable_b)),
                balances={n: after['accounts'].get(a.public, {}).get('balance') for n, a in self.accts.items()},
                nonces={n: after['accounts'].get(a.public, {}).get('nonce', 0) for n, a in self.accts.items()},
                burned=after.get('burned'), gaps=gate_rules.block_gaps(times),
                timeline=self.timeline(times, opened_at, deadline, holders, recipe_holders, admissions, result.get('fault_height')),
                ordinary=self.ordinary_summary(after['accounts'].get(self.accts['bystander'].public, {}).get('nonce', 0)),
                peak_app_rss_mb=dict(self.peak), late_store=self.late_summary(store_delay, challenged_unix, holders, recipe_holders),
                rejected_proposals=self.rejections(), seconds=round(time.time() - challenged_unix, 1))
            if offline is not None and recover:
                result['recovery'] = self.recover(run, offline)
            result['gate'], result['reasons'] = gate_rules.judge_scenario(result)
            self.log('scenario-result', **result)
            return result
        except Exception as error:  # noqa: BLE001 - a harness failure is an inconclusive scenario, never a pass
            result.update(gate='inconclusive', reasons=[f'harness error: {type(error).__name__}: {error}'[:300]])
            self.log('scenario-result', **result)
            return result
        finally:
            stop.set()
            for i in list(getattr(self, 'servers', {})):
                self.stop_store(i)
            self.stop()

    def agent_chain(self):
        """A ``network.Chain`` over this isolated chain's RPCs, light-client verified against its genesis validators."""
        from neuroshard.assistant import network
        genesis = json.loads((Path(self.config['validators'][0]['home']) / 'config' / 'genesis.json').read_text())
        trusted = {base64.b64decode(v['pub_key']['value']).hex(): int(v['power']) for v in genesis['validators']}
        return network.Chain([self.net.url(self.config, i) for i in range(5)], self.chain_id, trusted)

    def run_agent(self, stop, job_id, challenge_id, holders, recipe_holders):
        """Drive the released ``growth_agent.Agent`` for an open challenge it made, until the challenge closes.

        The agent's memory holds the challenge (as after ``check`` opened it) and its bundle is on disk where
        ``check`` writes it. Each pass is the agent's own ``check``: deliver to every store, then prove. Its
        ``deliver`` is the CLI's: post to every store, True once one accepts.
        """
        from neuroshard.assistant import audit
        from neuroshard.evolution import growth_agent
        home = self.stores / 'agent'
        (home / 'proofs').mkdir(parents=True, exist_ok=True)
        (home / 'proofs' / self.proof_root).write_bytes(self.bundle)
        stores = [f'http://127.0.0.1:{self.args.base_port + i * 10 + 5}/proofs' for i in range(5)]

        def deliver(root, payload):
            accepted = 0
            for i, endpoint in enumerate(stores):
                try:
                    audit.post_proof(endpoint, root, payload)
                    holders.setdefault(i, time.time())
                    accepted += 1
                except Exception as error:  # noqa: BLE001 - one store being down does not stop the others
                    self.log('agent-delivery-refused', run=self.run, index=i, error=str(error)[:120])
            return accepted > 0

        class Key:  # the challenger's test key, behind network.Account's signing interface
            def __init__(self, acct):
                self.acct, self.public = acct, acct.public

            def sign(self, chain_id, nonce, kind, **fields):
                self.acct.nonce = nonce
                return self.acct.sign(kind, **fields)
        chain = self.agent_chain()
        agent = growth_agent.Agent(chain, Key(self.accts['challenger']), home, recipes={self.recipe_root: self.recipe},
                                   trainer=object(), deliver=deliver)
        agent.memory['actions'] = [{'kind': 'challenge', 'job_id': job_id, 'segment': 0, 'step': 0,
                                    'challenge_id': challenge_id, 'proof_root': self.proof_root}]
        agent.memory['checked'] = [f'{job_id}/0/' + hashlib.sha256(json.dumps(
            self.net.state(self.net.url(self.config, 0))['growth'][job_id]['commits'][0]['step_roots']).encode()).hexdigest()[:16]]
        # Recipes reach stores by replication from wherever they were published; the agent publishes the bundle.
        recipe = json.dumps(self.recipe).encode()
        while not stop.is_set():
            self.deliver(self.recipe_root, recipe, recipe_holders)
            try:
                state = chain.state()
                job = state.get('growth', {}).get(job_id)
                if job is None:
                    break
                for item in agent.check(state, job_id, job):
                    self.log('agent', run=self.run, action=item.get('kind'),
                             **{k: (str(v)[:200] if k == 'error' else v) for k, v in item.items() if k != 'kind'})
                    if item['kind'] in ('prove', 'challenge-proven', 'challenge-closed'):
                        self.agent_done = item['kind']
                        return
            except Exception as error:  # noqa: BLE001 - the agent's next pass retries, as in its own loop
                self.log('agent-pass-error', run=self.run, error=f'{type(error).__name__}: {error}'[:200])
            stop.wait(self.args.agent_interval)

    def late_summary(self, store_delay, challenged_unix, holders, recipe_holders):
        if not store_delay:
            return None
        index, delay = store_delay
        started = getattr(self, 'late_started', None)
        return {'index': index, 'configured_delay_s': delay, 'budget_s': self.args.late_budget,
                'started_after_s': round(started - challenged_unix, 1) if started else None,
                'recipe_received': index in recipe_holders, 'bundle_received': index in holders,
                'bundle_received_after_s': round(holders[index] - challenged_unix, 1) if index in holders else None}

    def timeline(self, times, opened_at, deadline, holders, recipe_holders, admissions, fault_height):
        """Challenge, material, replay and inclusion times: what the proof window is spent on."""
        replays = {}
        for i in range(5):
            text = (self.stores / f'app{i}.log').read_text(errors='replace')
            m = re.findall(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) INFO \S+ verification \w+ verdict (True|False) in ([\d.]+) s', text, re.M)
            if m:
                end = datetime.strptime(m[-1][0], '%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=timezone.utc).timestamp()
                replays[i] = {'end': round(end, 2), 'start': round(end - float(m[-1][2]), 2), 'seconds': float(m[-1][2])}
        return {'challenge_height': opened_at, 'challenge_time': times.get(opened_at), 'deadline_height': deadline,
                'deadline_blocks': deadline - opened_at, 'bundle_accepted': {i: round(t, 2) for i, t in holders.items()},
                'recipe_accepted': {i: round(t, 2) for i, t in recipe_holders.items()}, 'replays': replays,
                'admissions': admissions[:60], 'inclusion_height': fault_height,
                'inclusion_header_time': times.get(fault_height) if fault_height else None,
                'inclusion_observed_time': round(self.fault_seen, 3) if getattr(self, 'fault_seen', None) else None,
                'blocks_to_inclusion': (fault_height - opened_at) if fault_height else None}

    def ordinary_summary(self, final_nonce):
        """Offered load, outcome of every attempt, and confirmation latency (scheduled time to observed commit)."""
        self.reconcile_ordinary(final_nonce)
        committed = [a for a in self.ordinary if a.get('committed')]
        lat = sorted(a['latency'] for a in committed if a.get('latency') is not None)
        unobserved = [a for a in committed if a.get('latency') is None]  # committed, but confirmation not seen
        accounted = sum(1 for a in self.ordinary if a.get('committed')) == final_nonce
        return {'offered_every_s': self.args.ordinary_every, 'attempts': len(self.ordinary),
                'committed': len(committed), 'chain_nonce': final_nonce,
                'refused': sum(1 for a in self.ordinary if a.get('admission') not in (None, 0)),
                'errors': sum(1 for a in self.ordinary if 'error' in a),
                'confirmation_unobserved': len(unobserved),
                'unreconciled': 0 if accounted else abs(final_nonce - len(committed)),
                'median_s': lat[len(lat) // 2] if lat else None, 'p95_s': lat[int(len(lat) * 0.95)] if lat else None,
                'max_s': lat[-1] if lat else None, 'all': self.ordinary}

    def recover(self, run, index):
        """Restart a stopped validator with its own data and signing state; establish recovery against its peers."""
        v = self.config['validators'][index]
        tree, venv = self.trees[v['flavour']]
        env = {**os.environ, 'PYTHONPATH': f'{tree}/src', 'PYTHONUNBUFFERED': '1',
               'NEUROSHARD_MODELS': self.args.models if v['model'] else str(self.stores / f'no-model-{index}'),
               'NEUROSHARD_UNITS': str(self.stores / f'store{index}' / 'units')}
        restarted = time.time()
        log = open(self.stores / f'app{index}.log', 'ab')
        self.procs[f'app{index}'] = subprocess.Popen(
            [f'{venv}/bin/python', '-m', 'neuroshard.inference.optimistic_app', '--home', v['home'],
             '--port', str(v['abci'])], stdout=log, stderr=log, env=env, cwd=tree, start_new_session=True)
        deadline = time.monotonic() + 900
        while (self.stores / f'app{index}.log').read_text(errors='replace').count('ABCI listening') < 2:
            if time.monotonic() > deadline:
                return {'result': 'timeout', 'detail': 'app did not restart'}
            time.sleep(2)
        log = open(self.stores / f'node{index}.log', 'ab')
        self.procs[f'node{index}'] = subprocess.Popen([self.config['engine'], 'start', '--home', v['home']],
                                                      stdout=log, stderr=log, start_new_session=True)
        address = json.loads((Path(v['home']) / 'config' / 'priv_validator_key.json').read_text())['address']
        peers = [i for i in range(5) if i != index]
        rpc = lambda i, m, p=None: self.net.rpc(self.net.url(self.config, i), m, p or {})  # noqa: E731
        first_fresh = None
        decision = ('timeout', 'no observation')
        observations = []
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            try:
                status = rpc(index, 'status')['sync_info']
                mine = {'height': int(status['latest_block_height']), 'catching_up': status['catching_up']}
                peer_heights = [{'height': int(rpc(i, 'status')['sync_info']['latest_block_height'])} for i in peers]
                first_fresh = first_fresh or max(p['height'] for p in peer_heights)
                common_h = min([mine['height']] + [p['height'] for p in peer_heights]) - 1
                hashes = lambda i: (lambda b: (b['block_id']['hash'], b['block']['header']['app_hash']))(  # noqa: E731
                    rpc(i, 'block', {'height': str(common_h)}))
                common = {'height': common_h, 'mine': hashes(index), 'peers': [hashes(i) for i in peers]}
                signatures = {}
                for h in range(first_fresh + 1, common_h + 1):
                    commit = rpc(peers[0], 'commit', {'height': str(h)})['signed_header']['commit']
                    signatures[h] = [s['validator_address'] for s in commit['signatures'] if s.get('block_id_flag') == 2]
                sample = {'mine': mine, 'peers': peer_heights, 'common': common, 'signatures': signatures}
                decision = gate_rules.recovery_decision(sample, address)
                observations.append({'unix': round(time.time(), 1), 'mine': mine, 'peer_top': max(p['height'] for p in peer_heights),
                                     'common_height': common_h, 'decision': decision[0]})
                if decision[0] in ('recovered', 'disagreement'):
                    break
            except Exception as error:  # noqa: BLE001 - still starting
                decision = ('timeout', f'last error: {str(error)[:120]}')
            time.sleep(3)
        result = {'result': decision[0] if decision[0] != 'pending' else 'timeout', 'detail': decision[1],
                  'seconds': round(time.time() - restarted, 1), 'address': address, 'observations': observations[-20:]}
        self.log('validator-recovery', run=run, index=index, **{k: v for k, v in result.items() if k != 'observations'})
        return result

    def rejections(self):
        """Per validator: proposals it rejected because it could not judge a proof (ProcessProposal NoVerdict)."""
        return {i: (self.stores / f'app{i}.log').read_text(errors='replace').count('ProcessProposal rejected a block')
                for i in range(5)}

    def verdicts(self):
        """Each validator's own verdict on this proof, from its app log (the cached worker's lines)."""
        return {i: log_verdict((self.stores / f'app{i}.log').read_text(errors='replace')) for i in range(5)}

    def prepare(self):
        """The case material, once: the recipe, its start-state bundle, the honest and legacy 8-thread roots."""
        material = json.loads(Path(self.args.material).read_text())
        self.recipe, self.roots = material['recipe'], material['roots']
        self.bundle = base64.b64decode(material['bundle_b64'])
        self.units = {sha: base64.b64decode(b) for sha, b in material.get('units_b64', {}).items()}
        self.provenance = material.get('provenance', {})
        self.proof_root = hashlib.sha256(self.bundle).hexdigest()
        self.base = self.recipe.get('base', 'granite-parent')
        from neuroshard.evolution import growth_replay as g
        self.recipe_root = g.recipe_root(self.recipe, self.recipe['model_root'])
        # The material is evidence: copy it, with its hash, beside the results and the exact harness sources.
        evidence = self.out / 'evidence'
        evidence.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.args.material, evidence / 'material.json')
        for source in ('growth_quorum_gate.py', 'gate_rules.py', 'make_gate_material.py'):
            shutil.copyfile(Path(__file__).parent / source, evidence / source)
        from neuroshard.assistant import network as _network
        from neuroshard.evolution import growth_agent as _agent
        agent_code = {m.__name__: {'file': m.__file__, 'sha256': hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()[:16]}
                      for m in (_network, _agent)}
        agent_code['submit_everywhere'] = hasattr(_network.Chain, 'submit_everywhere')
        self.log('material', recipe_root=self.recipe_root, proof_root=self.proof_root, base=self.base, agent_code=agent_code,
                 material_sha256=hashlib.sha256(Path(self.args.material).read_bytes()).hexdigest(),
                 roots={k: v for k, v in self.roots.items()}, provenance=self.provenance,
                 versions={k: versions(*v) for k, v in self.trees.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('plan')
    go = sub.add_parser('run')
    go.add_argument('--out', required=True)
    go.add_argument('--pilot1', required=True, help='TREE:VENV of the pilot-1 checkout')
    go.add_argument('--rc2', required=True, help='TREE:VENV of the pilot-2-rc2 checkout')
    go.add_argument('--material', required=True, help='JSON from make_gate_material.py: recipe, bundle, roots')
    go.add_argument('--models', required=True, help='NEUROSHARD_MODELS for validators that hold the model')
    go.add_argument('--stage', action='append', required=True, choices=sorted(STAGES))
    go.add_argument('--case', action='append', choices=CASES)
    go.add_argument('--offline', type=int, help='stop this validator after the challenge, before the proof')
    go.add_argument('--recover', action='store_true', help='restart the offline validator; establish recovery vs peers')
    go.add_argument('--missing-store', type=int, action='append', default=[],
                    help='run no proof store for this validator (it never receives the bundle)')
    go.add_argument('--late-store', help='INDEX:SECONDS start this validator\'s store late, after the challenge')
    go.add_argument('--late-budget', type=float, default=30.0,
                    help='seconds after the challenge by which a late store must be up (late recovery must restore quorum)')
    go.add_argument('--window', type=int, help='override growth window and proof blocks (default: live, 40)')
    go.add_argument('--judge-timeout', type=int, default=1500)
    go.add_argument('--prove-every', type=float, default=5.0, help='seconds between proof resubmissions')
    go.add_argument('--agent', action='store_true', help='prove with the growth agent instead of the harness')
    go.add_argument('--agent-src', help='source tree the growth agent is imported from (default: the rc2 tree)')
    go.add_argument('--agent-interval', type=float, default=5.0, help='seconds between agent passes (live: --interval)')
    go.add_argument('--ordinary-every', type=float, default=5.0, help='seconds between ordinary transactions')
    go.add_argument('--validator-threads', type=int, help='OMP_NUM_THREADS for validator apps')
    go.add_argument('--shards', help="JSON file: a live validator's settlement 'shards' entry, paths rewritten")
    go.add_argument('--base-port', type=int, default=29650)
    args = parser.parse_args()
    if args.action == 'plan':
        print(json.dumps({'stages': STAGES, 'cases': CASES, 'measurement': MEASUREMENT, 'harness': HARNESS}, indent=1))
        return 0
    gate = Gate(args)
    gate.prepare()
    late = tuple(int(x) for x in args.late_store.split(':')) if args.late_store else None
    plan = [(stage, case) for stage in args.stage for case in (args.case or CASES)]
    gate.log('plan', scenarios=[{'stage': s, 'case': c} for s, c in plan], argv=sys.argv[1:])
    results = [gate.scenario(stage, case, args.offline, args.recover, tuple(args.missing_store), late)
               for stage, case in plan]
    verdicts = [r['gate'] for r in results]
    overall = 'fail' if 'fail' in verdicts else 'inconclusive' if 'inconclusive' in verdicts or len(results) != len(plan) else 'pass'
    gate.log('gate-result', overall=overall, scenarios=[{k: r.get(k) for k in ('run', 'stage', 'case', 'gate', 'reasons')}
                                                         for r in results])
    return {'pass': 0, 'fail': 1, 'inconclusive': 2}[overall]


if __name__ == '__main__':
    sys.exit(main())
