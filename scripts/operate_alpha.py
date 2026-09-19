#!/usr/bin/env python3
"""Deploy and operate a bounded public alpha from a reviewed source freeze.

Consensus lives on dedicated CPU hosts; GPU providers own only assigned shards.
This controller owns one administrator's audit keys. It is not an independent
quorum and never substitutes a provider signature for complete numerical replay.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets
import shlex
import subprocess
import sys
import time
import fcntl
import traceback

from neuroshard.client import wire
from neuroshard.demo import client, protocol
from neuroshard.evolution import auditing, expert_lifecycle, service_admission, settlement
from neuroshard.evolution.app import code_hash
from neuroshard.evolution.reference_data import identity, save
from neuroshard.evolution.transactions import Outbox
from neuroshard.demo.network import edit_config
from ordinary_allocation import allocate as allocate_gpu, retire as retire_gpu
from ordinary_cloud import Cloud, REMOTE, REPO, PYTHON, PUBLIC
from operated_alpha_hosts import LedgerHosts, allocate as allocate_ledger, bootstrap as bootstrap_ledger
from portable_native_trial import Network
from prepare_provider_graph import prepare
import probe_provider_llm as probe

ROOT = Path(__file__).resolve().parents[1]
CPU = REPO+'/.neuroshard/native/bin/python'
ENGINE = REMOTE+'/cometbft'


def read(path):
    return json.loads(Path(path).read_bytes())


def publish(paths):
    """Only explicitly selected public files; never folders containing keys."""
    from neuroshard.evolution.sharded.retained_objects import transfer
    objects = {hashlib.sha256(path.read_bytes()).hexdigest(): path for path in paths}
    request = {'objects': {key: {'bytes': path.stat().st_size} for key, path in objects.items()}}
    signed = subprocess.run(['sudo', sys.executable, str(ROOT/'scripts/sign_ordinary_objects.py')],
        input=wire.canonical(request), capture_output=True, check=True)
    capabilities = json.loads(signed.stdout)
    receipts = {}
    for key, path in objects.items():
        receipts[key] = transfer(key, path.stat().st_size, PUBLIC+key,
                                source=path, put_url=capabilities[key]['put'])
    return receipts


def publish_metadata(home):
    metadata = home/'prepared/metadata'
    files = sorted(path for path in metadata.iterdir() if path.is_file())
    if not files or any(hashlib.sha256(path.read_bytes()).hexdigest() != path.name for path in files):
        raise ValueError('Require the content-addressed public policy metadata')
    save(home/'metadata-published.json', publish(files))


def starter_credits(home, ledger, network, freeze):
    folder = home/'starter-credits'
    owner = protocol.Identity.load_or_create(folder/'identity')
    ceiling = freeze['funding']['public_starter_credit_ceiling_atoms']
    network.send(3, 'fund-starter-credits', 'transfer', to=owner.public_key, amount=ceiling+10_000_000)
    config = {'home': REMOTE+'/starter-credits', 'node_rpc': 'http://127.0.0.1:26657',
        'chain_id': network.genesis['chain_id'],
        'manifest_root': identity(network.genesis['app_state']['manifest']),
        'grant_atoms': freeze['funding']['public_starter_grant_atoms'], 'ceiling_atoms': ceiling,
        'bind': '0.0.0.0', 'port': 18480}
    ledger.bundle(0, {'starter-credits/identity': (folder/'identity').read_bytes(),
                     'starter-credits/config.json': config})
    script = ('import json,sys; from neuroshard.demo import protocol; '
        'from neuroshard.evolution.provider_transport import certificate; from pathlib import Path; '
        'p=Path(sys.argv[1]); owner=protocol.Identity.load_or_create(p/"identity"); '
        'tls,pin=certificate(p,owner); print(json.dumps({"owner":owner.public_key,"certificate":pin}))')
    value = json.loads(ledger.command(0, ['env', 'PYTHONPATH='+REPO+'/src', CPU, '-c', script,
                                         config['home']]).stdout)
    if value['owner'] != owner.public_key:
        raise ValueError('The credit server loaded another funding key')
    unit(ledger, 0, 'neuroshard-alpha-credits', [CPU, REPO+'/scripts/operated_alpha_credits.py',
                                            '--config', config['home']+'/config.json'], native=True)
    return {**value, 'endpoint': 'https://'+ledger.hosts[0]['PublicIpAddress']+':18480',
            'grant_atoms': config['grant_atoms'], 'ceiling_atoms': ceiling}


def unit(hosts, index, name, argv, *, seconds=None, native=False):
    environment = ({'PYTHONPATH': REPO+'/src', 'ATEN_CPU_CAPABILITY': 'default',
        'MKL_ENABLE_INSTRUCTIONS': 'SSE4_2', 'OMP_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'} if native else dict(hosts.environment))
    if not native:
        # The GPU image selects its numerical libraries in the login shell.
        # Permanent units must preserve the same loader as Cloud.start and
        # the pinned reference; systemd otherwise produces a different build.
        facts = json.loads(hosts.command(index, ['python3', '-c',
            'import os,json; print(json.dumps({"loader":os.environ.get("LD_LIBRARY_PATH","")}))']).stdout)
        environment['LD_LIBRARY_PATH'] = facts['loader']
    content = ('[Unit]\nAfter=network-online.target\nWants=network-online.target\n'
        '[Service]\nUser=ubuntu\nWorkingDirectory='+REPO+'\n'
        + ''.join('Environment='+json.dumps(key+'='+value)+'\n' for key, value in environment.items())
        + 'ExecStart='+shlex.join(argv)+'\nRestart=on-failure\nRestartSec=5\n'
        + ('RuntimeMaxSec='+str(seconds)+'\n' if seconds else '')
        + '[Install]\nWantedBy=multi-user.target\n')
    script = ('import pathlib,sys; name=sys.argv[1]; '
              'assert name.startswith("neuroshard-alpha-") and all(c.isalnum() or c=="-" for c in name); '
              'pathlib.Path("/etc/systemd/system/"+name+".service").write_bytes(sys.stdin.buffer.read())')
    hosts.command(index, ['sudo', 'python3', '-c', script, name], input=content.encode())
    hosts.command(index, ['sudo', 'systemctl', 'daemon-reload'])
    hosts.command(index, ['sudo', 'systemctl', 'enable', '--now', name])


def connect(home, *, rpc_offset=0):
    """Open only controller RPC tunnels; remote native nodes keep running."""
    network = Network(home/'native')
    if rpc_offset:
        network.urls = [f'http://127.0.0.1:{node["rpc"]+rpc_offset}' for node in network.config['nodes']]
    hosts = LedgerHosts(home/'ledger-hosts')
    for index, node in enumerate(network.config['nodes'][:4]):
        with (home/f'ledger-tunnel-{index}.log').open('ab') as log:
            process = subprocess.Popen([*hosts.ssh(index), '-o', 'ExitOnForwardFailure=yes', '-N',
                '-L', f'127.0.0.1:{node["rpc"]+rpc_offset}:127.0.0.1:26657'], stdout=log, stderr=log)
        network.processes.append(process)
    network.until(lambda: all(client.query(url)['height'] > 1 and
        not client.rpc(url, 'status')['sync_info']['catching_up'] for url in network.urls), 150)
    return network


def native_nodes(home, manifest, engine):
    network = Network.create(home/'native', manifest, engine=str(engine), base_port=34400)
    economics = read(home/'freeze.json')['bootstrap_consensus']
    bond, outside = economics['operator_bond_atoms_per_validator'], economics['public_credit_limit_atoms']
    if bond % manifest['params']['bond_unit'] or 3*(3*bond) <= 2*(4*bond+outside):
        raise ValueError('Public starter credits must not buy a bootstrap blocking minority')
    for entry in network.genesis['app_state']['validators']:
        entry['bond'] = bond
        entry['liquid'] = economics['operator_liquid_atoms_per_validator']
    for entry in network.genesis['validators']:
        entry['power'] = str(bond // manifest['params']['bond_unit'])
    settlement.genesis(network.genesis['chain_id'], network.genesis['app_state']['validators'], manifest)
    save(home/'native/commitments.json', {'genesis': identity(network.genesis),
        'manifest': identity(manifest), 'chain_id': network.genesis['chain_id']})
    ledger, gpu = LedgerHosts(home/'ledger-hosts'), Cloud(home)
    ledger_ids = [row['id'] for row in network.config['nodes']]
    peer_addresses = [ledger_ids[i]+'@'+host['PrivateIpAddress']+':26656' for i, host in enumerate(ledger.hosts)]
    def install(hosts, index, folder, *, observer=False):
        if observer:
            subprocess.run([str(engine), 'init', '--home', str(folder)], check=True, capture_output=True)
        text = (folder/'config/config.toml').read_text()
        peers = ','.join(peer_addresses if observer else [p for i, p in enumerate(peer_addresses) if i != index])
        settings = [('', 'proxy_app', '"127.0.0.1:26658"'), ('', 'abci', '"grpc"'),
            ('', 'log_level', '"error"'), ('rpc', 'laddr', '"tcp://127.0.0.1:26657"'),
            ('rpc', 'max_body_bytes', '4194304'), ('rpc', 'timeout_broadcast_tx_commit', '"120s"'),
            ('p2p', 'laddr', '"tcp://0.0.0.0:26656"'), ('p2p', 'persistent_peers', json.dumps(peers)),
            ('p2p', 'addr_book_strict', 'false'), ('consensus', 'timeout_commit', '"500ms"'),
            ('consensus', 'timeout_propose', '"1s"'), ('consensus', 'timeout_prevote', '"500ms"'),
            ('consensus', 'timeout_precommit', '"500ms"')]
        for section, key, value in settings:
            text = edit_config(text, section, key, value)
        (folder/'config/config.toml').write_text(text)
        save(folder/'config/genesis.json', network.genesis)
        hosts.bundle(index, {'native/'+path.relative_to(folder).as_posix(): path.read_bytes()
                             for path in folder.rglob('*') if path.is_file()})
        unit(hosts, index, 'neuroshard-alpha-app', [CPU, '-m', 'neuroshard.evolution.app',
                                                 '--home', REMOTE+'/native', '--port', '26658'], native=True)
        unit(hosts, index, 'neuroshard-alpha-node', [ENGINE, 'start', '--home', REMOTE+'/native'])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: install(ledger, i, Path(network.config['nodes'][i]['home'])), range(4)))
    with ThreadPoolExecutor(max_workers=7) as pool:
        list(pool.map(lambda i: install(gpu, i, home/'observers'/str(i), observer=True), range(7)))
    network.close()
    return connect(home)


def provider(cloud, network, graph, rank, physical, index, offer_blocks):
    folder = REMOTE+'/providers/'+str(index)
    port = (18440 if index < 9 else 18460)+rank
    config = {'home': folder, 'node_rpc': 'http://127.0.0.1:26657',
        'chain_id': network.genesis['chain_id'], 'manifest_root': identity(network.genesis['app_state']['manifest']),
        'advertise': 'https://'+cloud.hosts[physical]['PublicIpAddress']+':'+str(port),
        'bind': '0.0.0.0', 'port': port, 'rank': rank, 'profile': REMOTE+'/profile.json',
        'inventory': REMOTE+'/object-lengths.json', 'source_home': REPO,
        'mirrors': [PUBLIC.rstrip('/')], 'max_bytes': 32*1024**3,
        'prepare_seconds': 600, 'frame_seconds': 120, 'run_seconds': cloud.remaining(),
        'collateral': 50_000_000, 'fee': 100+rank, 'offer_blocks': offer_blocks, 'maintain': True}
    cloud.put(physical, folder+'/config.json', config)
    cache = probe.MODEL_DISK+'/neuroshard-provider-models/'+str(index)
    cloud.command(physical, ['sudo', 'install', '-d', '-o', 'ubuntu', '-g', 'ubuntu', cache])
    cloud.command(physical, ['mkdir', '-p', folder+'/models'])
    cloud.command(physical, ['sudo', 'mount', '--bind', cache, folder+'/models'])
    public = json.loads(cloud.python(physical, ['-m', 'neuroshard.evolution.provider_runtime',
        '--config', folder+'/config.json', '--identity']).stdout)
    network.send(3, f'fund-provider-{index}', 'transfer', to=public['owner'], amount=250_000_000)
    offer = json.loads(cloud.python(physical, ['-m', 'neuroshard.evolution.provider_runtime',
        '--config', folder+'/config.json', '--publish-offer'], timeout=180).stdout)
    row = {'index': index, 'rank': rank, 'physical': physical, 'home': folder, 'config': config,
        'owner': public['owner'], 'offer': offer['offer_id'], 'unit': 'neuroshard-alpha-provider-'+str(index)}
    save(cloud.home/'providers'/f'{index}.json', row)
    return row


def launch_reference(cloud, graph, providers, generation):
    folder = REMOTE+'/full-audit-'+str(generation)
    service = {'key': 'full-audit-'+str(generation), 'graph': identity(graph),
        'placement': [r['physical'] for r in providers[:9]],
        'units': ['neuroshard-alpha-audit-'+str(generation)+'-'+str(i) for i in range(9)], 'folder': folder}
    def start(rank):
        row = providers[rank]
        model = row['home']+'/models/'+identity(graph)
        config = {'graph': REMOTE+'/graph.json', 'baseline': REMOTE+'/graph.json',
            'profile': REMOTE+'/profile.json', 'quality_policy': REMOTE+'/quality.json',
            'objects': model+'/objects', 'interpreter': model+'/interpreter', 'seed': model+'/seed',
            'source_home': REPO, 'inputs': REMOTE+'/unused-quality-inputs',
            'home': folder+'/owner-'+str(rank), 'max_seconds': min(21600, cloud.remaining())}
        path = folder+'/config-'+str(rank)+'.json'
        cloud.put(row['physical'], path, config)
        cloud.start(row['physical'], service['units'][rank], [REPO+'/scripts/run_native_expert_service.py',
            '--config', path], rank=rank, world=9, port=31000)
    probe.parallel(start, range(9))
    def ready():
        for rank, physical in enumerate(service['placement']):
            try:
                value = json.loads(cloud.read(physical, folder+'/owner-'+str(rank)+'/ready.json'))
            except subprocess.CalledProcessError:
                if not cloud.active(physical, service['units'][rank]):
                    raise RuntimeError('Reference owner stopped during initialization')
                return False
            if value['graph'] != identity(graph) or value['executor'] != graph['executor_root']:
                raise ValueError('The reference loaded another complete answering system')
        return True
    probe.until(ready, 600)
    service['started_at'] = time.time()
    save(cloud.home/'reference.json', service)
    return service


def setup(args):
    home = args.home.resolve()
    home.mkdir(parents=True, exist_ok=False)
    freeze = read(ROOT/'config/experiments/operated-alpha.json')
    preflight = read(args.preflight/'result.json')
    replay = read(args.preflight/'ledger-replay.json')
    if (not preflight['passed'] or not replay['passed'] or replay['issued_atoms'] != 0
            or read(args.preflight/'source.json')['code_hash'] != code_hash()):
        raise ValueError('The complete native admission preflight must pass first')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT):
        raise ValueError('Commit the alpha implementation and funding contract before allocation')
    save(home/'freeze.json', freeze)
    save(home/'preflight.json', {'execution': preflight, 'replay': replay})
    migrated = prepare(args.study, home/'prepared', freeze['accepted_graph'])
    manifest = read(home/'prepared/native-manifest.json')
    manifest['service_admission'] = service_admission.PROFILE
    save(home/'prepared/native-manifest.json', manifest)
    graph = read(home/'prepared/graph.json')
    owners = read(home/'prepared/owners.json')
    physical = [set() for _ in range(7)]
    for placement in freeze['placement'].values():
        for rank, host in enumerate(placement):
            physical[host].update(spec['sha256'] for spec in owners[rank]['files'].values())
    backbone = {spec['sha256'] for spec in graph['parent']['tensors'].values()}
    if any(backbone <= resident for resident in physical):
        raise ValueError('Replica placement would give a physical host the complete backbone')
    probe.validate_serving_policy(graph, probe.Objects(home/'prepared/policies'), ROOT)
    save(home/'deployment.json', {'source': revision, 'code_hash': code_hash(), 'manifest': identity(manifest),
        'graph': identity(graph), 'migration': migrated, 'phase': 'prepared'})
    # Publishing all new policy metadata is a separately recorded prerequisite;
    # existing model tensors are fetched from their immutable public catalog.
    if not (home/'metadata-published.json').exists():
        print(json.dumps({'phase': 'prepared', 'home': str(home), 'next': 'publish-metadata-then-launch'}))
        return


def launch(args):
    home = args.home.resolve()
    freeze, deployment = read(home/'freeze.json'), read(home/'deployment.json')
    if deployment['code_hash'] != code_hash() or not (home/'metadata-published.json').exists():
        raise ValueError('Publish and verify this exact source and complete policy metadata before launch')
    revision = deployment['source']
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != revision:
        raise ValueError('Deploy only the prepared committed revision')
    allocate_ledger(home/'ledger-hosts', freeze['ledger_resources'], revision)
    try:
        allocate_gpu(home, freeze['gpu_resources'], revision)
        ledger, cloud = LedgerHosts(home/'ledger-hosts'), Cloud(home)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(bootstrap_ledger, ledger, revision, args.engine),
                       pool.submit(probe.bootstrap, cloud, home/'prepared', revision, args.engine)]
            for future in futures:
                future.result()
        graph, manifest = read(home/'prepared/graph.json'), read(home/'prepared/native-manifest.json')
        network = native_nodes(home, manifest, args.engine)
        try:
            providers = []
            for replica in ('primary', 'secondary'):
                for rank, physical in enumerate(freeze['placement'][replica]):
                    providers.append(provider(cloud, network, graph, rank, physical, len(providers),
                                              freeze['service']['offer_blocks']))
            # Each replica is on different physical hosts. Restore its own
            # committed partitions instead of borrowing another host's cache.
            probe.restore(cloud, graph, providers[:9])
            probe.restore(cloud, graph, providers[9:])
            for row in providers:
                unit(cloud, row['physical'], row['unit'], [PYTHON, '-m',
                    'neuroshard.evolution.provider_runtime', '--config', row['home']+'/config.json'])
            reference = launch_reference(cloud, graph, providers, 0)
            for index in range(3):
                network.send(index, 'standing-inference', 'offer_audit_service', purpose='expert_inference',
                    scope=identity(graph), stage_limit=4096, capacity=2, expires_in=100000)
            recovery_home = home/'recovery'
            owner = protocol.Identity.load_or_create(recovery_home/'identity')
            network.send(3, 'fund-recovery', 'transfer', to=owner.public_key, amount=100_000_000)
            config = {'home': REMOTE+'/recovery', 'node_rpc': 'http://127.0.0.1:26657',
                'chain_id': network.genesis['chain_id'], 'manifest_root': identity(manifest),
                'run_seconds': ledger.remaining()}
            ledger.bundle(0, {'recovery/identity': (recovery_home/'identity').read_bytes(), 'recovery/config.json': config})
            unit(ledger, 0, 'neuroshard-alpha-recovery', [CPU, '-m', 'neuroshard.evolution.provider_control',
                                                      '--config', REMOTE+'/recovery/config.json'], native=True)
            faucet = starter_credits(home, ledger, network, freeze)
            for index, wallet in enumerate(probe.customer_wallets(home)):
                network.send(3, 'fund-gate-customer-'+str(index), 'transfer',
                             to=wallet.public_key, amount=3_000_000_000)
            genesis_file = Path(network.config['nodes'][0]['home'])/'config/genesis.json'
            bootstrap_receipts = publish([genesis_file, args.engine])
            save(home/'bootstrap-published.json', bootstrap_receipts)
            engine_sha = hashlib.sha256(args.engine.read_bytes()).hexdigest()
            genesis_sha = hashlib.sha256(genesis_file.read_bytes()).hexdigest()
            public = {'format': 'neuroshard-operated-alpha-access-v1', 'chain_id': network.genesis['chain_id'],
                'manifest_root': identity(manifest), 'source': revision, 'code_hash': code_hash(),
                'genesis_sha256': genesis_sha, 'genesis_url': PUBLIC+genesis_sha,
                'engine_sha256': engine_sha, 'engine_url': PUBLIC+engine_sha,
                'starter_credits': faucet,
                'peers': [node['id']+'@'+ledger.hosts[index]['PublicIpAddress']+':26656'
                          for index, node in enumerate(network.config['nodes'])],
                'graph': identity(graph), 'operator_count': 1, 'public_conversations': True,
                'gpu_funded_until': read(home/'allocation.json')['deadline'],
                'ledger_funded_until': read(home/'ledger-hosts/allocation.json')['deadline'],
                'availability': 'not released; deployment gate pending'}
            save(home/'access.json', public)
            save(home/'deployment.json', {**deployment, 'phase': 'running-release-gate-pending'})
        finally:
            network.close()
    except BaseException:
        if (home/'allocation.json').exists():
            retire_gpu(home)
        from operated_alpha_hosts import retire
        retire(home/'ledger-hosts')
        raise


def run(args):
    from operated_alpha_auditor import tick, maintain
    from operated_alpha_budget import observe
    home = args.home.resolve()
    freeze, deployment = read(home/'freeze.json'), read(home/'deployment.json')
    if deployment['code_hash'] != code_hash():
        raise ValueError('The controller must use its deployed immutable source')
    with (home/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cloud = Cloud(home)
        network = connect(home)
        providers = [read(home/'providers'/f'{index}.json') for index in range(18)]
        graph = read(home/'prepared/graph.json')
        service = read(home/'reference.json')
        generation = int(service['key'].rsplit('-', 1)[1])
        last_health = 0
        try:
            while True:
                try:
                    # Drain before the absolute GPU deadline. The independent
                    # ledger lifetime also permits refunds after a hard outage.
                    remaining = cloud.remaining(margin=0)
                    if remaining <= 180:
                        save(home/'controller-status.json', {'phase': 'funded_window_ended', 'time': time.time()})
                        return
                    closed = len(list((home/'audits').glob('*/intent.json')))
                    try:
                        costing = observe(home)
                        cost_stop = costing['close_future_capacity']
                    except Exception as error:
                        # Do not keep admitting new obligations if their declared
                        # spend watch is unavailable. Existing audits can finish.
                        cost_stop = True
                        save(home/'cost-watch-error.json', {'error': type(error).__name__, 'time': time.time()})
                    closing = (remaining < 5400 or cost_stop
                               or closed >= freeze['service']['max_settled_requests'])
                    if closing:
                        maintain(home, network, closing=True)
                    if network.query()['issued'] != 0:
                        raise RuntimeError('This serving-only alpha unexpectedly issued new tokens')
                    alive = all(cloud.active(physical, service['units'][rank])
                                for rank, physical in enumerate(service['placement']))
                    if not alive or time.time()-service['started_at'] > 18000:
                        cloud.stop(service)
                        generation += 1
                        service = launch_reference(cloud, graph, providers, generation)
                    result = tick(home, cloud, network, service)
                    if time.time()-last_health >= freeze['service']['health_probe_interval_seconds']:
                        # A capacity/ledger health check costs no customer tokens.
                        # Actual settled requests retain their separate full replay records.
                        status = network.query()
                        market = network.query('/hosting')
                        health = {'time': time.time(), 'height': status['height'],
                            'issued_atoms': status['issued'], 'reference_alive': alive,
                            'providers': len(market['providers']), 'offers': len(market['offers']),
                            'active_leases': len(market['leases']), 'closing': closing}
                        save(home/'health'/str(status['height']), health)
                        last_health = time.time()
                    if closing:
                        market = network.query('/hosting')
                        audits = network.query('/auditing')
                        if not market['leases'] and not audits['budgets']:
                            probe.collect_observations(cloud)
                            retire_gpu(home)
                            save(home/'controller-status.json', {'phase': 'drained', 'time': time.time(),
                                'height': network.query()['height']})
                            return
                    save(home/'controller-status.json', {**result, 'time': time.time(),
                        'closing': closing, 'reference_generation': generation})
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                    # No timeout or malformed result is an affirmative audit.
                    # Preserve the original signer journal and report the fault.
                    save(home/'controller-status.json', {'phase': 'retry', 'error': type(error).__name__,
                        'detail': str(error)[:1024], 'time': time.time()})
                    with (home/'controller-errors.log').open('a') as stream:
                        traceback.print_exc(file=stream)
                time.sleep(2)
        finally:
            network.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'publish-metadata', 'launch', 'run'])
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--study', type=Path)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--engine', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        if not args.study or not args.preflight:
            parser.error('prepare requires --study and --preflight')
        setup(args)
    elif args.action == 'publish-metadata':
        publish_metadata(args.home.resolve())
    elif args.action == 'launch':
        if not args.engine:
            parser.error('launch requires --engine')
        launch(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
