#!/usr/bin/env python3
"""Exercise atomic admission and autonomous provider replacement before GPUs."""
import argparse
from functools import partial
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time

from probe_provider_native import ROOT, Quiet, audit_owner, port, prepare, until
from portable_native_trial import Network
from neuroshard.client.hosted import Customer
from neuroshard.client.local_node import LocalNode
from neuroshard.demo import protocol, client
from neuroshard.evolution import auditing, expert_lifecycle, provider_transport, service_admission
from neuroshard.evolution.app import code_hash
from neuroshard.evolution.reference_data import identity, save
from neuroshard.evolution.transactions import Outbox
from replay_provider_preflight import export, replay


def stop(processes):
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def run(args):
    import torch.multiprocessing as mp
    home = args.home.resolve()
    home.mkdir(parents=True, exist_ok=False)
    freeze = json.loads((ROOT/'config/experiments/operated-admission-preflight.json').read_bytes())
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=ROOT):
        raise ValueError('Commit the implementation and freeze before running this preflight')
    save(home/'source.json', {'commit': revision, 'code_hash': code_hash()})
    save(home/'freeze.json', freeze)
    def expired(*_args):
        raise TimeoutError('The operated admission preflight reached its declared deadline')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(freeze['resources']['max_seconds'])
    began = time.monotonic()
    graph, manifest, mirror = prepare(home)
    manifest['service_admission'] = {**service_admission.PROFILE, 'provider_heartbeat_blocks': 64}
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Quiet, directory=str(mirror)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    children, providers, dead = [], [], set()
    network, customer_box = None, None
    try:
        network = Network.create(home/'native', manifest, engine=args.engine, base_port=args.base_port)
        network.start()
        for index in range(3):
            network.send(index, 'standing-inference', 'offer_audit_service', purpose='expert_inference',
                scope=identity(graph), stage_limit=4096, capacity=2, expires_in=100000)
        for index, rank in enumerate([0, 1, 2, 3, 4, 0, 1]):
            folder = home/f'provider-{index}'
            owner = protocol.Identity.load_or_create(folder/'identity')
            _, certificate = provider_transport.certificate(folder, owner)
            provider_port = port()
            config = {'home': str(folder), 'node_rpc': network.urls[rank % 4],
                'chain_id': network.genesis['chain_id'], 'manifest_root': identity(manifest),
                'advertise': f'https://127.0.0.1:{provider_port}', 'bind': '127.0.0.1', 'port': provider_port,
                'rank': rank, 'profile': str(home/'model/profile.json'),
                'inventory': str(home/'object-lengths.json'), 'source_home': str(ROOT),
                'mirrors': [f'http://127.0.0.1:{server.server_port}'], 'max_bytes': 32*1024**2,
                'prepare_seconds': 90, 'frame_seconds': 25, 'offer_blocks': 10000,
                'run_seconds': freeze['resources']['max_seconds'], 'allow_private': True, 'maintain': True}
            save(folder/'config.json', config)
            network.send(3, f'fund-provider-{index}', 'transfer', to=owner.public_key, amount=100_000_000)
            box = Outbox(folder/'transactions.sqlite', config['node_rpc'], config['chain_id'], owner)
            try:
                box.send('register', 'register_provider', endpoint=config['advertise'],
                         certificate=certificate, collateral=50_000_000)
                box.send('offer', 'offer_expert', graph=identity(graph), rank=rank,
                         fee=100+rank, capacity=1, expires_in=10000)
            finally:
                box.close()
            providers.append({'home': folder, 'owner': owner.public_key, 'rank': rank})
        recovery_home = home/'recovery'
        recovery_owner = protocol.Identity.load_or_create(recovery_home/'identity')
        network.send(3, 'fund-recovery', 'transfer', to=recovery_owner.public_key, amount=10_000_000)
        save(recovery_home/'config.json', {'home': str(recovery_home), 'node_rpc': network.urls[0],
            'chain_id': network.genesis['chain_id'], 'manifest_root': identity(manifest),
            'run_seconds': freeze['resources']['max_seconds']})
        with (recovery_home/'process.log').open('ab') as log:
            recovery = subprocess.Popen([sys.executable, '-m', 'neuroshard.evolution.provider_control',
                '--config', str(recovery_home/'config.json')], stdout=log, stderr=log)
        children.append(recovery)
        customer_home = home/'customer'
        customer_owner = protocol.Identity.load_or_create(customer_home/'identity')
        network.send(3, 'fund-customer', 'transfer', to=customer_owner.public_key, amount=100_000_000)
        node = LocalNode(network.urls[0], network.genesis['chain_id'], identity(manifest))
        customer_box = Outbox(customer_home/'transactions.sqlite', network.urls[0], node.chain_id, customer_owner)
        customer = Customer(customer_home/'requests', node, customer_owner, customer_box)
        cases = []
        for name in freeze['cases']:
            case_home = home/name
            case_home.mkdir()
            processes = {}
            for provider in providers:
                if provider['owner'] in dead:
                    continue
                with (provider['home']/'process.log').open('ab') as log:
                    child = subprocess.Popen([sys.executable, '-m', 'neuroshard.evolution.provider_runtime',
                        '--config', str(provider['home']/'config.json')], stdout=log, stderr=log)
                processes[provider['owner']] = child
                children.append(child)
            starting_height = network.query()['height']
            until(lambda: all(row['last_seen'] >= starting_height for owner, row in
                network.query('/hosting')['providers'].items() if owner not in dead), 120)
            row = customer.prepare(freeze['question'], freeze['max_tokens'], 100_000_000)
            view = customer.tick(row)
            assert view['status'] == 'serving'
            job_id, budget_id = row['job_id'], row['budget_id']
            assert job_id == budget_id
            original = view['snapshot']
            lost = freeze['fault_ranks'].get(name)
            failure = None
            if lost is not None:
                epoch = original['lease']['assignment_root']
                owner = original['lease']['providers'][str(lost)]['owner']
                provider = next(p for p in providers if p['owner'] == owner)
                # Kill immediately after accepting the assignment, before the
                # group is able to finish. No replacement is submitted here.
                until(lambda: owner in node.snapshot(job_id, refresh=True)['lease']['accepted'], 120, .01)
                processes[owner].kill()
                processes[owner].wait(timeout=10)
                dead.add(owner)
                replacement = until(lambda: (lambda s: s if s['lease'] and
                    s['lease']['assignment_root'] != epoch else None)(node.snapshot(job_id, refresh=True)), 180)
                assert replacement['job']['graph'] == original['job']['graph']
                assert replacement['job']['request'] == original['job']['request']
                assert replacement['lease']['audit_budget'] == budget_id
                assert replacement['lease']['providers'][str(lost)]['owner'] != owner
                failure = {'rank': lost, 'old_owner': owner,
                    'new_owner': replacement['lease']['providers'][str(lost)]['owner'],
                    'old_assignment': epoch, 'new_assignment': replacement['lease']['assignment_root'],
                    'recovery_daemon': True, 'preexisting_standby': True}
            claim = until(lambda: network.query('/candidate'), 180)
            assert claim['job_id'] == job_id
            # Keep this CPU-only check below its memory cap while five fresh
            # numerical replay processes load. GPU soak retains warm providers.
            stop(list(processes.values()))
            save(case_home/'claim.json', claim)
            save(case_home/'audit-config.json', {'model': str(home/'model')})
            auditors = mp.spawn(audit_owner, args=(str(case_home),), nprocs=5, join=False)
            try:
                until(lambda: auditors.join(timeout=1), 180)
            finally:
                for child in auditors.processes:
                    if child.is_alive():
                        child.terminate()
                    child.join(timeout=5)
            reports = json.loads((case_home/'audit-0.json').read_bytes())
            assert all(json.loads((case_home/f'audit-{rank}.json').read_bytes()) == reports for rank in range(5))
            coverage = auditing.coverage(claim)
            salts = [secrets.token_hex(32) for _ in range(3)]
            for index in range(3):
                commitment = auditing.verdict_commitment(node.chain_id, claim['id'],
                    network.owners[index].public_key, coverage, salts[index], True)
                network.send(index, name+'/commit', 'audit_commit', claim_id=claim['id'], commitment=commitment)
            until(lambda: network.query()['height'] > network.query('/candidate')['audit_commit_end'], 180)
            for index in range(3):
                network.send(index, name+'/reveal', 'audit_verdict', claim_id=claim['id'],
                    coverage_root=coverage, salt=salts[index], valid=True)
            settled = network.settled(claim['id'], seconds=120)
            assert settled['settlement']['accepted'] and settled['status']['issued'] == 0
            final = customer.tick(row)
            assert final['status'] == 'finished' and final['result']['status'] == 'completed'
            until(lambda: all(client.query(url, '/expert_lifecycle')['results'].get(job_id)
                              for url in network.urls))
            results = [client.query(url, '/expert_lifecycle')['results'][job_id] for url in network.urls]
            assert all(result == results[0] for result in results)
            audit = next(r for r in network.query('/auditing')['history'] if r['id'] == budget_id)
            assert audit['occupancy_burned_atoms'] > 0
            assert audit['occupancy_burned_atoms'] + audit['occupancy_refunded_atoms'] == row['quote']['occupancy_atoms']
            host = [r for r in network.query('/hosting')['history'] if r['job_id'] == job_id]
            assert len(host) == 1 and host[0]['provider_paid_atoms'] == sum(100+rank for rank in range(5))
            case_result = {'case': name, 'failure': failure, 'job_id': job_id, 'complete_replays': len(reports),
                'response': results[0], 'audit': audit, 'hosting': host[0], 'validator_agreement': True}
            save(case_home/'result.json', case_result)
            cases.append(case_result)
            print(json.dumps({'case': name, 'passed': True}), flush=True)
        save(home/'result.json', {'passed': True, 'freeze': identity(freeze), 'source': revision,
            'cases': cases, 'seconds': time.monotonic()-began, 'independent_operators': False, 'gpus': 0})
    finally:
        signal.alarm(0)
        stop(children)
        if customer_box is not None:
            customer_box.close()
        if network is not None:
            network.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        save(home/'resources-finished.json', {'trial_children_stopped': all(p.poll() is not None for p in children),
            'native_children_stopped': network is None or all(p.poll() is not None for p in network.processes),
            'new_instances': 0, 'gpus': 0})
    export(home, args.engine, args.base_port+99)
    replay(home)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--base-port', type=int, default=29600)
    options = parser.parse_args()
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'test')
    run(options)
