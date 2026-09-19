#!/usr/bin/env python3
"""Run the frozen provider protocol preflight on an isolated native chain.

Requires a source checkout including numerical test fixtures, the research CPU
dependencies and CometBFT. All subprocesses are disposable and stop on exit.
"""
import argparse
import copy
from datetime import timedelta
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests/evolution'))

from neuroshard.demo import protocol, client
from neuroshard.evolution import answering, auditing, expert_lifecycle, expert_work, hosting, provider_assets
from neuroshard.evolution import provider_transport, settlement
from neuroshard.evolution.app import code_hash
from neuroshard.evolution.reference_data import identity, save, sha256
from neuroshard.evolution.transactions import Outbox
from neuroshard.lab.app import native_parameters
from portable_native_trial import Network


def until(check, seconds=120, interval=.05):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(interval)
    raise TimeoutError('Provider preflight did not satisfy its bounded condition')


def port():
    with socket.socket() as connection:
        connection.bind(('127.0.0.1', 0))
        return connection.getsockname()[1]


def audit_owner(rank, home):
    import torch.distributed as dist
    from neuroshard.evolution.sharded.graph_execution import GraphNetwork
    from neuroshard.evolution.sharded.graph_service import inference_report
    home = Path(home)
    config = json.loads((home/'audit-config.json').read_bytes())
    claim = json.loads((home/'claim.json').read_bytes())
    model = Path(config['model'])
    profile = json.loads((model/'profile.json').read_bytes())
    dist.init_process_group('gloo', init_method='file://'+str(home/'audit-rendezvous'),
        rank=rank, world_size=5, timeout=timedelta(seconds=90))
    try:
        net = GraphNetwork(claim['graph'], profile, objects=model/'objects', interpreter=model/'interpreter',
            seed=model/'seed', source_home=ROOT, rank=rank)
        reports = []
        for _index in range(3):
            report, result = inference_report(claim, net)
            if not expert_lifecycle.replay_report(claim, report)['valid']:
                raise ValueError('The actual native inference claim failed complete numerical replay')
            reports.append({'report': report, 'response': identity(result)})
        save(home/f'audit-{rank}.json', reports)
    finally:
        dist.destroy_process_group()


def prepare(home):
    from test_complete_answering import complete
    from neuroshard.evolution.sharded.peer_wire import SOURCES
    model = home/'model'
    model.mkdir()
    _, core, _, config, store, profile = complete.__wrapped__(model)
    profile['sources'].update({name: sha256(ROOT/name) for name in SOURCES})
    core['executor_root'] = identity(profile)
    config['graph'] = config['learned']['graph'] = identity(core)
    graph = answering.attach(core, config, store)
    save(model/'profile.json', profile)
    save(model/'graph.json', graph)
    template = json.loads((model/'prospective-policy.json').read_bytes())['candidate_template']
    template['executor_root'] = graph['executor_root']
    template['descriptor']['previous_graph'] = identity(graph['descriptor'])
    next_config = copy.deepcopy(config)
    next_config['graph'] = next_config['learned']['graph'] = identity(template)
    template = answering.attach(template, next_config, store)
    initial = template['experts']['protocol']
    params = {**settlement.PARAMS, 'max_claim_blocks': 1536}
    manifest = {'params': params, 'initial_model_root': graph['parent']['state_root'],
        'data_root': 'a'*64, 'code_hash': code_hash(),
        'native_consensus': {**native_parameters(params), 'block_max_bytes': 4*1024**2},
        'hosting': {**hosting.PROFILE, 'prepare_blocks': 128, 'execution_blocks': 128},
        'auditing': {**auditing.QUORUM_PROFILE, 'commit_blocks': 512, 'reveal_blocks': 64},
        'expert_work': {'format': expert_work.PROSPECTIVE, 'parent': graph['parent'], 'checkpoint': initial,
            'prepared': 'b'*64, 'feature_stages': 3, 'batch_count': 1, 'schedule': [0]*initial['recipe']['steps'],
            'numerical_profile': graph['numerical_profile']},
        'expert_lifecycle': {'format': expert_lifecycle.PROSPECTIVE, 'serving_graph': graph,
            'candidate_template': template, 'quality': {'policy_root': 'd'*64, 'prepared': 'b'*64, 'stages': 5},
            'price_per_token': 7, 'max_tokens': 4}}
    mirror = home/'mirror'
    mirror.mkdir()
    inventory = {}
    for rank in range(5):
        for name, spec in provider_assets.plan(graph, rank, config).items():
            source = model/name
            if sha256(source) != spec['sha256']:
                raise ValueError('Fixture asset differs from its declared digest')
            inventory[spec['sha256']] = source.stat().st_size
            shutil.copyfile(source, mirror/spec['sha256'])
    save(home/'object-lengths.json', inventory)
    return graph, manifest, mirror


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


def run(args):
    import torch.multiprocessing as mp
    home = args.home.resolve()
    home.mkdir(parents=True, exist_ok=False)
    freeze = json.loads((ROOT/'config/experiments/provider-native-preflight.json').read_bytes())
    def deadline(_signal, _frame):
        raise RuntimeError('The frozen provider preflight time limit expired')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(freeze['resources']['max_seconds'])
    save(home/'freeze.json', freeze)
    revision = args.source_commit or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Require the exact deployed source commit')
    save(home/'source.json', {'commit': revision, 'code_hash': code_hash()})
    began = time.monotonic()
    graph, manifest, mirror = prepare(home)
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Quiet, directory=str(mirror)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    network, children, providers = None, [], []
    try:
        network = Network.create(home/'native', manifest, engine=args.engine, base_port=args.base_port)
        network.start()
        until(lambda: all(not client.rpc(url, 'status')['sync_info']['catching_up'] for url in network.urls))
        model = home/'model'

        def newcomer(rank):
            folder = home/f'provider-{len(providers)}'
            owner = protocol.Identity.load_or_create(folder/'identity')
            _, fingerprint = provider_transport.certificate(folder, owner)
            provider_port = port()
            config = {'home': str(folder), 'node_rpc': network.urls[rank % 4],
                'chain_id': network.genesis['chain_id'], 'manifest_root': identity(manifest),
                'advertise': f'https://127.0.0.1:{provider_port}', 'bind': '127.0.0.1', 'port': provider_port,
                'rank': rank, 'profile': str(model/'profile.json'), 'inventory': str(home/'object-lengths.json'),
                'source_home': str(ROOT), 'mirrors': [f'http://127.0.0.1:{server.server_port}'],
                'max_bytes': 32*1024**2, 'prepare_seconds': 90, 'frame_seconds': 25,
                'run_seconds': freeze['resources']['max_seconds'], 'allow_private': True}
            save(folder/'config.json', config)
            index = len(providers)
            network.send(3, f'provider-{index}/fund', 'transfer', to=owner.public_key, amount=100_000_000)
            box = Outbox(folder/'transactions.sqlite', network.urls[rank % 4], network.genesis['chain_id'], owner)
            try:
                box.send('register', 'register_provider', endpoint=config['advertise'], certificate=fingerprint,
                         collateral=50_000_000)
                box.send('offer', 'offer_expert', graph=identity(graph), rank=rank, fee=100+rank,
                         capacity=1, expires_in=10000)
                offer = box.logical_id('offer')
            finally:
                box.close()
            value = {'home': folder, 'owner': owner, 'offer': offer}
            providers.append(value)
            return value

        active = [newcomer(rank) for rank in range(5)]
        cases = []
        for case in freeze['cases']:
            case_home = home/case
            case_home.mkdir()
            started = time.monotonic()
            maximum = freeze['max_tokens']
            request, _, price = expert_lifecycle.inference_terms(graph, freeze['question'], maximum, 7)
            stages = hosting.stage_limit(graph, request, 7)
            operation = case+'/audit'
            network.send(3, operation, 'fund_audit', publisher=active[0]['owner'].public_key,
                         auditors=[], stage_limit=stages, expires_in=10000)
            budget = network.outboxes[3].logical_id(operation)
            for index in range(3):
                network.send(index, case+'/accept-audit', 'accept_audit', budget_id=budget)
            offers = {str(rank): provider['offer'] for rank, provider in enumerate(active)}
            network.send(3, case+'/lease', 'lease_expert', graph=identity(graph), question=freeze['question'],
                max_tokens=maximum, offers=offers, max_price=price, max_provider_fee=1000,
                audit_budget=budget, expires_in=5000)
            job_id = network.outboxes[3].logical_id(case+'/lease')
            original = network.query('/hosting/job', {'job_id': job_id})

            def launch():
                processes = []
                for provider in active:
                    with (provider['home']/'process.log').open('ab') as log:
                        child = subprocess.Popen([sys.executable, '-m', 'neuroshard.evolution.provider_runtime',
                            '--config', str(provider['home']/'config.json'), '--job', job_id], stdout=log, stderr=log)
                    children.append(child)
                    processes.append(child)
                return processes

            current = launch()
            lost = freeze['fault_ranks'].get(case)
            failure = None
            if lost is not None:
                epoch = original['lease']['assignment_root']
                marker = active[lost]['home']/'jobs'/job_id/epoch/'started.json'
                until(marker.exists, 120, .01)
                current[lost].kill()
                current[lost].wait(timeout=10)
                failure = {'rank': lost, 'old_owner': active[lost]['owner'].public_key,
                           'old_assignment': epoch, 'killed_after_native_acceptance': True}
                until(lambda: (lambda s: s['lease']['status'] == 'ready' and s['height'] > s['lease']['work_deadline'])(
                    network.query('/hosting/job', {'job_id': job_id})), 150)
                for child in current:
                    if child.poll() is None:
                        child.terminate()
                        child.wait(timeout=10)
                active[lost] = newcomer(lost)
                offers = {str(rank): provider['offer'] for rank, provider in enumerate(active)}
                network.send(3, case+'/replace', 'replace_hosted_job', job_id=job_id, offers=offers, audit_budget=budget)
                replaced = network.query('/hosting/job', {'job_id': job_id})
                assert replaced['job']['request'] == original['job']['request']
                assert replaced['job']['graph'] == original['job']['graph']
                assert replaced['lease']['assignment_root'] != epoch and replaced['lease']['epoch'] == 1
                failure.update(new_owner=active[lost]['owner'].public_key,
                               new_assignment=replaced['lease']['assignment_root'])
                current = launch()
            claim = until(lambda: network.query('/candidate'), 150)
            if claim['job_id'] != job_id:
                raise ValueError('The provider submitted an unrelated native claim')
            for child in current:
                child.wait(timeout=30)
                if child.returncode:
                    raise ValueError('A provider process failed; inspect its local journal')
            save(case_home/'claim.json', claim)
            save(case_home/'audit-config.json', {'model': str(model)})
            auditors = mp.spawn(audit_owner, args=(str(case_home),), nprocs=5, join=False)
            try:
                until(lambda: auditors.join(timeout=1), 150)
            finally:
                for child in auditors.processes:
                    if child.is_alive():
                        child.terminate()
                    child.join(timeout=5)
            reports = json.loads((case_home/'audit-0.json').read_bytes())
            assert all(json.loads((case_home/f'audit-{rank}.json').read_bytes()) == reports for rank in range(5))
            coverage, salts = auditing.coverage(claim), [secrets.token_hex(32) for _ in range(3)]
            for index in range(3):
                commitment = auditing.verdict_commitment(network.genesis['chain_id'], claim['id'],
                    network.owners[index].public_key, coverage, salts[index], True)
                network.send(index, case+'/commit', 'audit_commit', claim_id=claim['id'], commitment=commitment)
            until(lambda: network.query()['height'] > network.query('/candidate')['audit_commit_end'])
            for index in range(3):
                network.send(index, case+'/reveal', 'audit_verdict', claim_id=claim['id'],
                             coverage_root=coverage, salt=salts[index], valid=True)
            settled = network.settled(claim['id'], seconds=120)
            assert settled['settlement']['accepted'] and settled['status']['issued'] == 0
            until(lambda: all(client.query(url, '/expert_lifecycle')['results'].get(job_id)
                              for url in network.urls))
            receipts = [client.query(url, '/expert_lifecycle')['results'][job_id] for url in network.urls]
            assert all(receipt == receipts[0] for receipt in receipts)
            assert receipts[0]['paid_atoms'] + receipts[0]['refunded_atoms'] == price
            history = [row for row in network.query('/hosting')['history'] if row['job_id'] == job_id]
            assert len(history) == 1 and history[0]['provider_paid_atoms'] == sum(100+rank for rank in range(5))
            case_result = {'case': case, 'job_id': job_id, 'claim_id': claim['id'], 'failure': failure,
                'complete_replays': len(reports), 'response': receipts[0], 'hosting': history[0],
                'seconds': time.monotonic()-started, 'validator_agreement': True}
            cases.append(case_result)
            save(case_home/'result.json', case_result)
            print(json.dumps({'case': case, 'passed': True, 'seconds': case_result['seconds']}), flush=True)
        save(home/'result.json', {'passed': True, 'freeze': identity(freeze), 'graph': identity(graph),
            'chain_id': network.genesis['chain_id'], 'cases': cases, 'issued': network.query()['issued'],
            'seconds': time.monotonic()-began, 'independent_operators': False, 'additional_instances': 0})
    finally:
        signal.alarm(0)
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        if network is not None:
            network.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        save(home/'resources-finished.json', {'trial_children_stopped': all(p.poll() is not None for p in children),
            'native_children_stopped': network is None or all(p.poll() is not None for p in network.processes),
            'new_instances': 0, 'gpus': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--engine')
    parser.add_argument('--source-commit', help='Pinned revision when executing a verified git archive')
    parser.add_argument('--base-port', type=int, default=28600)
    options = parser.parse_args()
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'test')
    run(options)
