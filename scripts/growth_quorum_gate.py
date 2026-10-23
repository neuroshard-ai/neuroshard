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
from pathlib import Path

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
HONEST_ROOT_KEY, LEGACY_ROOT_KEY = 'honest_root', 'legacy8_root'


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
        sys.path.insert(0, f'{self.trees["rc2"][0]}/src')
        from neuroshard.inference import optimistic as ledger, optimistic_network as net
        self.ledger, self.net = ledger, net
        self.events = self.out / 'events.jsonl'

    def log(self, kind, **fields):
        self.out.mkdir(parents=True, exist_ok=True)
        row = {'kind': kind, 'unix': round(time.time(), 3), **fields}
        with self.events.open('a') as f:
            f.write(json.dumps(row, sort_keys=True) + '\n')
        print(json.dumps(row, sort_keys=True)[:400], flush=True)

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
        return {n: Account(f'gate5b-{run}-{n}') for n in ('sponsor', 'contributor', 'challenger')}

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
        params = {**self.ledger.PARAMS, 'growth_window_blocks': self.args.window, 'growth_proof_blocks': self.args.window}
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

    # -- scenario ---------------------------------------------------------------------------------------------
    def scenario(self, stage, case, offline=None, recover=False):
        run = f'{stage}-{case}' + (f'-off{offline}' if offline is not None else '') + f'-{int(time.time())}'
        recipe, roots = self.recipe, self.roots
        claimed = {'honest': roots[HONEST_ROOT_KEY], 'forged': 'cd' * 32, 'legacy8': roots[LEGACY_ROOT_KEY]}[case]
        try:
            self.start(stage, run)
            rroot = self.recipe_root
            for i in range(5):  # every validator holds the recipe, the proof bundle and units (delivery is not tested)
                bundles = self.stores / f'store{i}' / 'bundles'
                (bundles / f'{rroot}.json').write_text(json.dumps(recipe))
                (bundles / self.proof_root).write_bytes(self.bundle)
                for sha, payload in self.units.items():
                    (self.stores / f'store{i}' / 'units' / f'{sha}.safetensors').write_bytes(payload)
            opened, _ = self.submit('sponsor', 'growth_open', recipe_root=rroot, start_root=recipe['start_root'],
                                    steps=1, segment_steps=1, reward=1_000_000)
            job = self.ledger.transaction_id(opened)
            self.submit('contributor', 'growth_commit', job_id=job, segment=0, step_roots=[claimed])
            challenge, _ = self.submit('challenger', 'growth_challenge', job_id=job, segment=0, step=0,
                                       proof_root=self.proof_root)
            challenge_id = self.ledger.transaction_id(challenge)
            if offline is not None:
                os.killpg(self.procs[f'node{offline}'].pid, signal.SIGTERM)
                os.killpg(self.procs[f'app{offline}'].pid, signal.SIGTERM)
                self.log('validator-offline', run=run, index=offline)
            rpc = next(i for i in range(5) if i != offline)
            before = self.net.state(self.net.url(self.config, rpc))
            h0 = before['height']
            # The proof: every validator's worker judges it when it reaches CheckTx/PrepareProposal/ProcessProposal.
            started = time.monotonic()
            # A determined challenger sends the proof to every validator: one that refuses it at CheckTx does not
            # gossip it, so submitting to a single node would understate how far a proof can get.
            envelope, admitted = self.submit('challenger', 'growth_prove', rpc=rpc, wait=False, job_id=job,
                                             challenge_id=challenge_id)
            admissions = {rpc: admitted}
            for i in range(5):
                if i not in (rpc, offline):
                    try:
                        admissions[i] = self.net.broadcast(self.net.url(self.config, i), envelope)
                    except Exception as error:  # noqa: BLE001 - recorded
                        admissions[i] = {'error': str(error)[:200]}
            self.log('prove-submitted', run=run, admission={str(k): {'code': v.get('code'), 'log': v.get('log', v.get('error', ''))[:160]}
                                                              for k, v in admissions.items()})
            # Poll: did the proof commit (challenge popped with a fault), or did the challenge lapse? Blocks must
            # keep coming either way. Peak memory is sampled meanwhile.
            peak, outcome, judged_at = {}, None, None
            deadline = time.monotonic() + self.args.judge_timeout
            while time.monotonic() < deadline:
                for name, proc in self.procs.items():
                    if name.startswith('app') and proc.poll() is None:
                        peak[name] = max(peak.get(name, 0), rss_mb(proc.pid))
                now = self.net.state(self.net.url(self.config, rpc))
                gjob = now.get('growth', {}).get(job)
                faults = [f for f in now.get('growth_faults', []) if f['job_id'] == job]
                if faults:
                    outcome = 'penalty-committed'
                    break
                if gjob is None or challenge_id not in gjob.get('challenges', {}):
                    outcome = 'challenge-lapsed-or-settled'
                    break
                # Decided without a commit: every validator that can judge has judged, and SETTLE more blocks have
                # committed since, without the proof. (A proof is judged once per validator and cached; nothing
                # later changes a verdict, so no later block can carry it either.)
                verdicts = self.verdicts()
                holders = [i for i in range(5) if self.config['validators'][i]['model'] and i != offline]
                if all(verdicts[i] in ('True', 'False') for i in holders):
                    judged_at = judged_at or now['height']
                    if now['height'] >= judged_at + self.args.settle_blocks:
                        outcome = 'not-committed-after-all-verdicts'
                        break
                time.sleep(2)
            seconds = round(time.monotonic() - started, 1)
            time.sleep(3)  # let every app flush its verdict line
            after = self.net.state(self.net.url(self.config, rpc))
            carried = []
            for h in range(h0 + 1, after['height'] + 1):
                try:
                    block = self.net.rpc(self.net.url(self.config, rpc), 'block', {'height': str(h)})['block']
                except Exception:  # noqa: BLE001
                    continue
                for raw in block['data'].get('txs') or []:
                    if json.loads(base64.b64decode(raw)).get('body', {}).get('kind') == 'growth_prove':
                        carried.append(h)
            balances = {n: after['accounts'].get(a.public, {}).get('balance') for n, a in self.accts.items()}
            verdicts = self.verdicts()
            self.log('outcome', run=run, stage=stage, case=case, offline=offline, outcome=outcome or 'undecided',
                     seconds=seconds, heights=[h0, after['height']], blocks_during=after['height'] - h0,
                     balances=balances, burned=after.get('burned'), verdicts=verdicts, peak_app_rss_mb=peak,
                     proof_in_blocks=carried, rejected_proposals=self.rejections(),
                     claimed=claimed[:16])
            if offline is not None and recover:
                self.recover(run, offline)
            return outcome
        finally:
            self.stop()

    def recover(self, run, index):
        v = self.config['validators'][index]
        flavour = v['flavour']
        tree, venv = self.trees[flavour]
        env = {**os.environ, 'PYTHONPATH': f'{tree}/src', 'NEUROSHARD_MODELS': self.args.models, 'PYTHONUNBUFFERED': '1',
               'NEUROSHARD_UNITS': str(self.stores / f'store{index}' / 'units')}
        log = open(self.stores / f'app{index}.log', 'ab')
        self.procs[f'app{index}'] = subprocess.Popen(
            [f'{venv}/bin/python', '-m', 'neuroshard.inference.optimistic_app', '--home', v['home'],
             '--port', str(v['abci'])], stdout=log, stderr=log, env=env, cwd=tree, start_new_session=True)
        deadline = time.monotonic() + 900
        while (self.stores / f'app{index}.log').read_text(errors='replace').count('ABCI listening') < 2:
            if time.monotonic() > deadline:
                raise RuntimeError(f'app{index} did not restart')
            time.sleep(2)
        log = open(self.stores / f'node{index}.log', 'ab')
        self.procs[f'node{index}'] = subprocess.Popen([self.config['engine'], 'start', '--home', v['home']],
                                                      stdout=log, stderr=log, start_new_session=True)
        target = self.height((index + 1) % 5) + 3
        deadline, caught = time.monotonic() + 600, False
        while time.monotonic() < deadline:
            try:
                mine, theirs = (self.net.state(self.net.url(self.config, i)) for i in (index, (index + 1) % 5))
                if mine['height'] >= target and self.ledger.root(mine) == self.ledger.root(
                        self.net.state(self.net.url(self.config, index))):
                    caught = mine.get('growth_faults') == theirs.get('growth_faults')
                    break
            except Exception:  # noqa: BLE001 - still starting
                pass
            time.sleep(2)
        self.log('validator-recovered', run=run, index=index, caught_up=caught,
                 height=self.height(index) if caught else None)

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
        self.log('material', recipe_root=self.recipe_root, proof_root=self.proof_root, base=self.base,
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
    go.add_argument('--recover', action='store_true', help='restart the offline validator and check it catches up')
    go.add_argument('--window', type=int, default=400, help='growth window and proof blocks')
    go.add_argument('--judge-timeout', type=int, default=1500)
    go.add_argument('--settle-blocks', type=int, default=120, help='blocks after the last verdict before concluding')
    go.add_argument('--validator-threads', type=int, help='OMP_NUM_THREADS for validator apps')
    go.add_argument('--shards', help="JSON file: a live validator's settlement 'shards' entry, paths rewritten")
    go.add_argument('--base-port', type=int, default=29650)
    args = parser.parse_args()
    if args.action == 'plan':
        print(json.dumps({'stages': STAGES, 'cases': CASES}, indent=1))
        return
    gate = Gate(args)
    gate.prepare()
    for stage in args.stage:
        for case in args.case or CASES:
            gate.scenario(stage, case, args.offline, args.recover)


if __name__ == '__main__':
    main()
