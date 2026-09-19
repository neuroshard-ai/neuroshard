#!/usr/bin/env python3
"""Operate the frozen, disposable accepted-LLM provider trial.

Seven machines, one administrator, fresh native chain. Never changes the live
network. Private keys and databases stay outside the public evidence allowlist.
The absolute EC2 shutdown timer also applies if this controller disappears.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import secrets
import subprocess
import time
import traceback

from neuroshard.client import provider_wire, wire
from neuroshard.client.hosted import Customer
from neuroshard.client.local_node import LocalNode
from neuroshard.demo import client
from neuroshard.demo.network import edit_config
from neuroshard.evolution import answering, auditing, expert_lifecycle, provider_assets
from neuroshard.evolution.app import code_hash
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.reference_data import identity, save
from neuroshard.evolution.transactions import Outbox
from ordinary_allocation import allocate, numerical_runtime, retire
from ordinary_cloud import Cloud, PUBLIC, REMOTE, REPO
from portable_native_trial import Network

ROOT = Path(__file__).resolve().parents[1]
CPU = REPO+'/.neuroshard/native/bin/python'
ENGINE = REMOTE+'/cometbft'
MODEL_DISK = '/opt/dlami/nvme'


def read(path):
    return json.loads(Path(path).read_bytes())


def customer_wallets(home):
    """Create local customer identities before allocating any paid resources."""
    return [wire.Wallet(home/f'customer-{index}/account.key', create=True) for index in range(2)]


def validate_serving_policy(graph, store, source_home):
    """Policy source commitments include operator scripts outside the executor."""
    from neuroshard.evolution.sharded.planned_graph import validate_configuration
    policy = answering.load(graph, store)
    validate_configuration(answering.core(graph), policy, source_home)
    return policy


def parallel(function, values):
    with ThreadPoolExecutor(max_workers=7) as pool:
        return list(pool.map(function, values))


def until(check, seconds=120):
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(.25)
    raise TimeoutError('The declared operated service condition timed out')


def bootstrap(cloud, prepared, revision, engine):
    archive = subprocess.check_output(['git', 'archive', '--format=tar.gz', revision], cwd=ROOT)
    metadata = {name: (prepared/name).read_bytes() for name in
                ('graph.json', 'profile.json', 'object-lengths.json', 'native-manifest.json')}
    metadata['quality.json'] = b'{}'
    metadata['cometbft'] = engine.read_bytes()
    setup = '''set -eu
sudo cloud-init status --wait
sudo systemctl is-active neuroshard-experiment-stop.timer
sudo apt-get -o Acquire::Retries=3 update -qq
sudo apt-get -o DPkg::Lock::Timeout=300 -o Acquire::Retries=3 install -y -qq python3-venv git
cd /home/ubuntu/neuroshard-study
python3 -m venv .neuroshard/venv
.neuroshard/venv/bin/python -m pip install -r docs/expert-execution-requirements.txt
python3 -m venv .neuroshard/native
.neuroshard/native/bin/python -m pip install -r docs/evolution-requirements.txt
chmod 700 /home/ubuntu/native-expert-live/cometbft
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
'''

    def one(physical):
        deadline = time.monotonic()+240
        while True:
            try:
                cloud.command(physical, ['true'], timeout=20)
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                if time.monotonic() > deadline:
                    raise
                time.sleep(3)
        cloud.command(physical, ['mkdir', '-p', REMOTE, REPO])
        cloud.command(physical, ['tar', '-xzf', '-', '-C', REPO], input=archive, timeout=180)
        cloud.bundle(physical, metadata)
        # Only the disposable allocation's AMI-managed instance store is used.
        # Keep native stores and provider keys on EBS; mount just model caches
        # below their normal paths. A missing mount must fail before pip/model
        # downloads, never silently fall back to the slow root volume.
        cloud.command(physical, ['sudo', 'cloud-init', 'status', '--wait'], timeout=240)
        cloud.command(physical, ['sudo', 'systemctl', 'start', 'dlami-nvme'], timeout=120)
        storage = json.loads(cloud.command(physical, ['python3', '-c',
            'import json,os,shutil,subprocess,sys; p=sys.argv[1]; '
            'm=json.loads(subprocess.check_output(["findmnt","--json","--target",p]))["filesystems"][0]; '
            'assert os.path.ismount(p) and m["target"]==p; '
            'assert shutil.disk_usage(p).free>=64*1024**3; '
            'print(json.dumps({"mount":m,"free_bytes":shutil.disk_usage(p).free}))', MODEL_DISK], timeout=30).stdout)
        save(cloud.home/f'storage-{physical}.json', storage)
        try:
            result = cloud.command(physical, ['timeout', '--kill-after=20', '1500', 'bash', '-s'],
                                   input=setup.encode(), timeout=1530)
        except subprocess.CalledProcessError as error:
            (cloud.home/f'bootstrap-{physical}.log').write_bytes((error.stdout or b'')+(error.stderr or b''))
            raise
        (cloud.home/f'bootstrap-{physical}.log').write_bytes(result.stdout+result.stderr)
        runtime = json.loads(cloud.python(physical, ['-c',
            'import os,json; from neuroshard.evolution import reference; r=reference.configure("cuda",2); '
            'r["allocator"]=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"); print(json.dumps(r))'],
            timeout=180).stdout)
        save(cloud.home/f'runtime-{physical}.json', runtime)
        return numerical_runtime(runtime)

    runtimes = parallel(one, range(7))
    if any(value != runtimes[0] for value in runtimes):
        raise ValueError('Actual GPU numerical runtimes disagree')
    if runtimes[0] != read(prepared/'profile.json')['runtime']:
        raise ValueError('Allocated hardware differs from the accepted numerical profile')


def native_nodes(cloud, manifest, engine):
    network = Network.create(cloud.home/'native', manifest, engine=str(engine), base_port=33400)
    nodes = network.config['nodes']
    for i in range(4, 7):
        home = cloud.home/'native/chain'/f'node{i}'
        subprocess.run([str(engine), 'init', '--home', str(home)], check=True, capture_output=True)
        node_id = subprocess.check_output([str(engine), 'show-node-id', '--home', str(home)], text=True).strip()
        nodes.append({'home': str(home), 'id': node_id, 'rpc': 33401+i*10})
    for index, node in enumerate(nodes):
        path = Path(node['home'])/'config/config.toml'
        text = path.read_text()
        peers = ','.join(other['id']+'@'+cloud.hosts[j]['PrivateIpAddress']+':26656'
                         for j, other in enumerate(nodes) if j != index)
        changes = [('', 'proxy_app', '"127.0.0.1:26658"'), ('', 'abci', '"grpc"'),
                   ('', 'log_level', '"error"'), ('rpc', 'laddr', '"tcp://127.0.0.1:26657"'),
                   ('rpc', 'max_body_bytes', '4194304'), ('rpc', 'timeout_broadcast_tx_commit', '"120s"'),
                   ('p2p', 'laddr', '"tcp://0.0.0.0:26656"'), ('p2p', 'persistent_peers', json.dumps(peers)),
                   ('p2p', 'addr_book_strict', 'false'), ('consensus', 'timeout_commit', '"500ms"'),
                   ('consensus', 'timeout_propose', '"1s"'), ('consensus', 'timeout_prevote', '"500ms"'),
                   ('consensus', 'timeout_precommit', '"500ms"')]
        for section, key, value in changes:
            text = edit_config(text, section, key, value)
        path.write_text(text)
        save(path.parent/'genesis.json', network.genesis)
    save(cloud.home/'native/network.json', network.config)

    def start(index):
        node = Path(nodes[index]['home'])
        cloud.bundle(index, {'native/'+path.relative_to(node).as_posix(): path.read_bytes()
                            for path in node.rglob('*') if path.is_file()})
        for role, command in (
            ('app', [CPU, '-m', 'neuroshard.evolution.app', '--home', REMOTE+'/native', '--port', '26658']),
            ('node', [ENGINE, 'start', '--home', REMOTE+'/native'])):
            cloud.command(index, ['sudo', 'systemd-run', '--unit=neuroshard-trial-'+role,
                '--property=User=ubuntu', '--property=WorkingDirectory='+REPO,
                '--property=RuntimeMaxSec='+str(cloud.remaining()),
                '--property=StandardOutput=append:'+REMOTE+'/native-'+role+'.log',
                '--property=StandardError=append:'+REMOTE+'/native-'+role+'.log',
                'env', 'PYTHONPATH='+REPO+'/src', 'ATEN_CPU_CAPABILITY=default',
                'MKL_ENABLE_INSTRUCTIONS=SSE4_2', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
                'MKL_NUM_THREADS=1', *command])
        with (cloud.home/f'tunnel-{index}.log').open('ab') as log:
            process = subprocess.Popen([*cloud.ssh(index), '-o', 'ExitOnForwardFailure=yes', '-N',
                '-L', f'127.0.0.1:{nodes[index]["rpc"]}:127.0.0.1:26657'], stdout=log, stderr=log)
        return process
    tunnels = parallel(start, range(7))
    network.processes.extend(tunnels)
    network.urls = ['http://127.0.0.1:'+str(node['rpc']) for node in nodes]
    network.until(lambda: all(client.query(url)['height'] > 1 and
        not client.rpc(url, 'status')['sync_info']['catching_up'] for url in network.urls), 150)
    return network


def provider(cloud, network, graph, rank, physical, index):
    home = REMOTE+'/providers/'+str(index)
    port = 18440+rank if index < 9 else 18460+rank if index < 18 else 18500+index-18
    config = {'home': home, 'node_rpc': 'http://127.0.0.1:26657',
        'chain_id': network.genesis['chain_id'], 'manifest_root': identity(network.genesis['app_state']['manifest']),
        'advertise': 'https://'+cloud.hosts[physical]['PublicIpAddress']+':'+str(port),
        'bind': '0.0.0.0', 'port': port, 'rank': rank,
        'profile': REMOTE+'/profile.json', 'inventory': REMOTE+'/object-lengths.json',
        'source_home': REPO, 'mirrors': [PUBLIC.rstrip('/')], 'max_bytes': 32*1024**3,
        'prepare_seconds': 600, 'frame_seconds': 120, 'run_seconds': cloud.remaining(),
        'collateral': 50_000_000, 'fee': 100+rank, 'offer_blocks': 20000}
    cloud.put(physical, home+'/config.json', config)
    cache = MODEL_DISK+'/neuroshard-provider-models/'+str(index)
    cloud.command(physical, ['sudo', 'install', '-d', '-o', 'ubuntu', '-g', 'ubuntu', cache])
    cloud.command(physical, ['mkdir', '-p', home+'/models'])
    cloud.command(physical, ['sudo', 'mount', '--bind', cache, home+'/models'])
    result = json.loads(cloud.python(physical, ['-m', 'neuroshard.evolution.provider_runtime',
        '--config', home+'/config.json', '--identity']).stdout)
    network.send(3, f'provider-{index}/fund', 'transfer', to=result['owner'], amount=100_000_000)
    offer = json.loads(cloud.python(physical, ['-m', 'neuroshard.evolution.provider_runtime',
        '--config', home+'/config.json', '--publish-offer'], timeout=180).stdout)
    value = {'index': index, 'rank': rank, 'physical': physical, 'home': home, 'config': config,
             'owner': result['owner'], 'offer': offer['offer_id'], 'unit': 'neuroshard-provider-'+str(index)}
    save(cloud.home/'providers'/f'{index}.json', value)
    return value


def restore(cloud, graph, providers):
    # First replica downloads only its owned immutable assets. The second uses
    # hard links to those same locally verified bytes, never another owner's
    # model. Fault replacements deliberately start with an empty model home.
    code = '''import json,sys
from pathlib import Path
from neuroshard.evolution import provider_assets
from neuroshard.evolution.reference_data import identity
c=json.load(open(sys.argv[1])); g=json.load(open(sys.argv[2]))
r=provider_assets.prepare(g,json.load(open(c['profile'])),c['rank'],
 Path(c['home'])/'models'/identity(g),c['source_home'],json.load(open(c['inventory'])),c['mirrors'],
 max_bytes=c['max_bytes'],max_seconds=c['prepare_seconds'])
print(json.dumps(r))
'''
    def one(row):
        try:
            raw = cloud.python(row['physical'], ['-c', code, row['home']+'/config.json', REMOTE+'/graph.json'], timeout=650).stdout
        except subprocess.CalledProcessError as error:
            (cloud.home/f'asset-failure-{row["index"]}.log').write_bytes((error.stderr or b'')[-65536:])
            raise
        save(cloud.home/'asset-receipts'/f'{row["index"]}.json', json.loads(raw))
    parallel(one, providers[:9])
    for row in providers[9:]:
        first = providers[row['rank']]
        if first['physical'] != row['physical']:
            raise ValueError('Replica hard links must remain on their physical shard owner')
        # Link through the common NVMe mount. Linux rejects hard links across
        # distinct bind mounts even when they expose the same underlying disk.
        disk = MODEL_DISK+'/neuroshard-provider-models/'
        cloud.command(row['physical'], ['cp', '-al', disk+str(first['index'])+'/'+identity(graph),
                                       disk+str(row['index'])+'/'])


def start_provider(cloud, row):
    cloud.start(row['physical'], row['unit'], ['-m', 'neuroshard.evolution.provider_runtime',
                '--config', row['home']+'/config.json'])


def cancel_offline_offer(cloud, row):
    # The trial kills a provider process, not its host or administrator. Explicit
    # operator departure prevents the known-dead process advertising more work.
    # This is not presented as third-party failure detection or automatic slashing.
    code = '''import json,sys
from pathlib import Path
from neuroshard.demo.protocol import Identity
from neuroshard.evolution.transactions import Outbox
c=json.load(open(sys.argv[1])); h=Path(c['home']); owner=Identity.load_or_create(h/'identity')
b=Outbox(h/'transactions.sqlite',c['node_rpc'],c['chain_id'],owner)
try:
 if b.pending(): b.confirm(b.pending(),timeout=30)
 b.send('leave-after-injected-loss','cancel_expert_offer',offer_id=sys.argv[2])
finally: b.close()
'''
    cloud.python(row['physical'], ['-c', code, row['home']+'/config.json', row['offer']], timeout=180)


def audit_service(cloud, graph, providers):
    service = {'key': 'provider-full-audits', 'graph': identity(graph),
        'placement': [row['physical'] for row in providers[:9]],
        'units': ['neuroshard-provider-auditor-'+str(i) for i in range(9)],
        'folder': REMOTE+'/full-audit-service'}
    def start(rank):
        row = providers[rank]
        model = row['home']+'/models/'+identity(graph)
        config = {'graph': REMOTE+'/graph.json', 'baseline': REMOTE+'/graph.json',
            'profile': REMOTE+'/profile.json', 'quality_policy': REMOTE+'/quality.json',
            'objects': model+'/objects', 'interpreter': model+'/interpreter', 'seed': model+'/seed',
            'source_home': REPO, 'inputs': REMOTE+'/unused-quality-inputs',
            'home': service['folder']+'/owner-'+str(rank), 'max_seconds': cloud.remaining()}
        path = service['folder']+'/config-'+str(rank)+'.json'
        cloud.put(row['physical'], path, config)
        cloud.start(row['physical'], service['units'][rank],
            [REPO+'/scripts/run_native_expert_service.py', '--config', path], rank=rank, world=9, port=31000)
    parallel(start, range(9))
    def ready():
        values = []
        for rank, physical in enumerate(service['placement']):
            try:
                values.append(json.loads(cloud.read(physical, service['folder']+'/owner-'+str(rank)+'/ready.json')))
            except subprocess.CalledProcessError:
                if not cloud.active(physical, service['units'][rank]):
                    raise RuntimeError('The complete numerical referee stopped during initialization')
        if len(values) != len(service['placement']):
            return None
        if any(row['graph'] != identity(graph) or row['executor'] != graph['executor_root'] for row in values):
            raise ValueError('Numerical referees loaded a different answering system')
        return values
    service['ready'] = until(ready, 600)
    save(cloud.home/'audit-service.json', service)
    return service


def settle(cloud, network, service, claim, folder):
    if claim['kind'] != 'expert_inference' or not claim.get('job_id'):
        raise ValueError('This serving trial never audits dormant learning or quality obligations')
    reports = []
    for index in range(3):
        started = time.monotonic()
        result = cloud.query(service, {'id': identity({'claim': claim['id'], 'replay': index}),
            'kind': 'inference_audit', 'claim': claim}, timeout=600)
        if result['status'] != 'completed' or not expert_lifecycle.replay_report(claim, result['report'])['valid']:
            raise ValueError('The native provider response failed complete numerical replay')
        save(folder/f'replay-{index}.json', result)
        reports.append({'report': result['report'], 'seconds': time.monotonic()-started})
    coverage = auditing.coverage(claim)
    intent_path = folder/'audit-intent.json'
    if not intent_path.exists():
        save(intent_path, {'claim': claim['id'], 'coverage': coverage,
                           'salts': [secrets.token_hex(32) for _ in range(3)]})
    intent = read(intent_path)
    if intent['claim'] != claim['id'] or intent['coverage'] != coverage:
        raise ValueError('Durable audit intent belongs to a different complete obligation')
    # SQLite connections belong to their creating thread. Keep each validator's
    # existing durable journal, with a connection opened in this audit thread.
    # The single audit worker exclusively writes these accounts during a batch;
    # the caller joins it before signing acceptance for another batch.
    boxes = [Outbox(network.home/f'outbox-{i}.sqlite', network.urls[0], network.genesis['chain_id'], owner,
                    rpc=network.outboxes[i].rpc, query=network.outboxes[i].query)
             for i, owner in enumerate(network.owners[:3])]
    try:
        for index in range(3):
            commitment = auditing.verdict_commitment(network.genesis['chain_id'], claim['id'],
                network.owners[index].public_key, coverage, intent['salts'][index], True)
            boxes[index].send(claim['id']+'/commit', 'audit_commit', claim_id=claim['id'], commitment=commitment)
        until(lambda: network.query()['height'] > network.query('/candidate')['audit_commit_end'])
        for index in range(3):
            boxes[index].send(claim['id']+'/reveal', 'audit_verdict', claim_id=claim['id'],
                         coverage_root=coverage, salt=intent['salts'][index], valid=True)
    finally:
        for box in boxes:
            box.close()
    result = network.settled(claim['id'], seconds=180)
    if not result['settlement']['accepted'] or result['status']['issued'] != 0:
        raise ValueError('Inference settlement changed supply or rejected the replayed claim')
    save(folder/'settlement.json', {'native': result, 'replays': reports})
    return reports


def batch(cloud, network, graph, service, providers, customers, cases, maximum, *, audit_pool):
    began = time.monotonic()
    rows, streams, measurements, assigned = [], {}, {}, {}
    audit_future = None
    for customer, case in zip(customers, cases):
        row = customer.prepare(case['messages'], maximum, 3_000_000_000)
        rows.append(row)
        measurements[row['id']] = {'case': case['id'], 'events': [], 'first_visible_seconds': None,
                                  'generated_seconds': None, 'settled_seconds': None}
    for customer, row in zip(customers, rows):
        customer.tick(row)
        for index in range(3):
            network.send(index, row['budget_id']+'/accept', 'accept_audit', budget_id=row['budget_id'])
    for customer, row in zip(customers, rows):
        assigned[row['id']] = time.monotonic()
        update = customer.tick(row)
        if update['status'] != 'serving':
            raise ValueError('A paid customer did not acquire an available native provider replica')
        measurements[row['id']]['job_id'] = row['job_id']
    # Generation latency begins at reservation submission, after quote/audit funding.
    # Customer-observed end-to-end time is reported separately, without hiding it.
    faults, pending_claims = {}, set()
    deadline = began + (1600 if any('fault_rank' in c for c in cases) else 1250)
    try:
        while time.monotonic() < deadline:
            for index, (customer, row, case) in enumerate(zip(customers, rows, cases)):
                update = customer.tick(row)
                result = measurements[row['id']]
                if update['status'] == 'finished':
                    if update['result']['status'] != 'completed':
                        raise ValueError('The frozen request did not complete: '+case['id'])
                    if result['settled_seconds'] is None:
                        result.update(settled_seconds=time.monotonic()-began, response=update['result'])
                    continue
                snapshot = update['snapshot']
                lease, epoch = snapshot['lease'], snapshot['lease']['assignment_root']
                if 'fault_rank' in case and row['id'] not in faults and lease['status'] == 'ready':
                    lost = next(p for p in providers if p['owner'] == lease['providers'][str(case['fault_rank'])]['owner'])
                    cloud.command(lost['physical'], ['sudo', 'systemctl', 'kill', '--signal=KILL', lost['unit']])
                    faults[row['id']] = {'lost': lost, 'epoch': epoch, 'at_seconds': time.monotonic()-began,
                                         'deadline': lease['work_deadline'], 'replacement': None}
                fault = faults.get(row['id'])
                if fault and fault['replacement'] is None and snapshot['height'] > fault['deadline']:
                    replacement = provider(cloud, network, graph, case['fault_rank'], case['replacement_physical'], len(providers))
                    providers.append(replacement)
                    offers = {rank: value['offer'] for rank, value in lease['providers'].items()}
                    offers[str(case['fault_rank'])] = replacement['offer']
                    network.send(3, row['id']+'/replace', 'replace_hosted_job', job_id=row['job_id'],
                        offers=offers, audit_budget=row['budget_id'])
                    replacement_view = network.query('/hosting/job', {'job_id': row['job_id']})
                    if (replacement_view['lease']['assignment_root'] == epoch
                            or identity(replacement_view['job']['request']) != row['quote']['request_root']):
                        raise ValueError('Replacement failed to fence the old execution and preserve the request')
                    start_provider(cloud, replacement)
                    fault['replacement'] = replacement
                    fault['new_assignment'] = replacement_view['lease']['assignment_root']
                    save(cloud.home/'cases'/case['id']/'fault.json', fault)
                    continue
                coordinator = lease['providers']['0']
                stream = streams.get(row['id'])
                if stream is None or stream['epoch'] != epoch:
                    if stream:
                        stream['connection'].close()
                    stream = {'epoch': epoch, 'after': 0, 'connection': provider_wire.PinnedConnection(
                        coordinator['endpoint'], coordinator['certificate'], timeout=1)}
                    streams[row['id']] = stream
                try:
                    event = provider_wire.poll(stream['connection'], customer.wallet, coordinator['owner'],
                        network.genesis['chain_id'], row['job_id'], epoch, stream['after'])
                    if event is not None:
                        if any(event[key] != row['quote'][key] for key in ('graph', 'tokenizer', 'request_root')):
                            raise RuntimeError('The streamed response changed its native commitment')
                        stream['after'] = event['sequence']
                        elapsed = time.monotonic()-assigned[row['id']]
                        result['events'].append({'seconds': elapsed, **event})
                        if event['text'] and result['first_visible_seconds'] is None:
                            result['first_visible_seconds'] = elapsed
                        if event['status'] in ('generated', 'submitted') and result['generated_seconds'] is None:
                            result['generated_seconds'] = elapsed
                except (OSError, ValueError):
                    stream['connection'].close()
            claim = network.query('/candidate')
            if claim and claim['id'] not in pending_claims:
                match = next((i for i, row in enumerate(rows) if row['job_id'] == claim['job_id']), None)
                if match is None:
                    raise ValueError('An unrelated obligation entered the inference-only trial')
                # Polling continues while all three complete replays execute.
                # Otherwise an audit of customer A would hide customer B's first token.
                pending_claims.add(claim['id'])
                folder = cloud.home/'cases'/cases[match]['id']
                save(folder/'claim.json', claim)
                if audit_future is not None:
                    audit_future.result()
                audit_future = audit_pool.submit(settle, cloud, network, service, claim, folder)
            if audit_future is not None and audit_future.done():
                audit_future.result()
            if all(row['phase'] == 'finished' for row in rows):
                if audit_future is not None:
                    audit_future.result()
                break
            time.sleep(.1)
        else:
            raise TimeoutError('Frozen provider request exceeded its settlement bound')
    finally:
        for stream in streams.values():
            stream['connection'].close()
        for row, case in zip(rows, cases):
            save(cloud.home/'cases'/case['id']/'measurement.json', measurements[row['id']])
    history = network.query('/hosting')['history']
    audits = network.query('/auditing')['history']
    for row, case in zip(rows, cases):
        matches = [value for value in history if value['job_id'] == row['job_id']]
        if len(matches) != 1:
            raise ValueError('A hosted request did not settle exactly once')
        response = measurements[row['id']]['response']
        if response['paid_atoms'] + response['refunded_atoms'] != row['quote']['execution_atoms']:
            raise ValueError('Execution settlement lost or duplicated prepaid funds')
        audited = [value for value in audits if value['id'] == row['budget_id']]
        if len(audited) != 1 or audited[0]['paid_atoms'] + audited[0]['refunded_atoms'] != row['quote']['verification_atoms']:
            raise ValueError('Complete verification failed to refund every unused atom')
        save(cloud.home/'cases'/case['id']/'hosting.json', matches[0])
        save(cloud.home/'cases'/case['id']/'audit-payment.json', audited[0])
        if row['id'] in faults:
            cancel_offline_offer(cloud, faults[row['id']]['lost'])
        if 'fault_rank' in case and (row['id'] not in faults or faults[row['id']]['replacement'] is None):
            raise ValueError('The declared owner-loss recovery was not actually exercised')
    return list(measurements.values())


def collect_observations(cloud):
    code = '''import json,pathlib
b=pathlib.Path('/home/ubuntu/native-expert-live'); out={}
names={'started.json','last-failure.json','result.json','submitted.json','ready.json'}
for directory in ('providers','full-audit-service'):
 for p in (b/directory).rglob('*.json'):
  if p.name in names and p.stat().st_size<8*1024**2:
   out[p.relative_to(b).as_posix()]=json.loads(p.read_bytes())
for p in b.glob('*.log'):
 if p.name.startswith(('neuroshard-provider-','native-')):
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-65536)); out[p.name]=f.read().decode(errors='replace')
out['network_counters']=pathlib.Path('/proc/net/dev').read_text()
out['memory']=pathlib.Path('/proc/meminfo').read_text()
print(json.dumps(out))
'''
    def one(physical):
        value = json.loads(cloud.command(physical, ['python3', '-c', code], timeout=45).stdout)
        save(cloud.home/'observations'/f'{physical}.json', value)
    parallel(one, range(7))


def evidence(cloud, network):
    # Snapshot stopped native stores. A private backup is kept for recovery;
    # publish only exported blocks, genesis and application states afterwards.
    def one(index):
        cloud.command(index, ['sudo', 'systemctl', 'stop', 'neuroshard-trial-node', 'neuroshard-trial-app'])
        raw = cloud.command(index, ['tar', '-czf', '-', '-C', REMOTE, 'native'], timeout=180).stdout
        destination = cloud.home/'private-native-backups'/f'{index}.tar.gz'
        destination.parent.mkdir(exist_ok=True, mode=0o700)
        destination.write_bytes(raw)
        destination.chmod(0o600)
        node = Path(network.config['nodes'][index]['home'])
        subprocess.run(['tar', '-xzf', str(destination), '--strip-components=1', '-C', str(node)], check=True)
    parallel(one, range(7))


def costs(home, prepared, freeze):
    """Conservative spend bound, not an invented NEURO/USD exchange rate.

    Charge all network bytes in both directions at the combined inter-zone and
    public-delivery ceiling, including free ingress and pip downloads. Allocation
    runtime covers idle, native nodes and every replay; do not add those twice.
    """
    allocation, finished = read(home/'allocation.json'), read(home/'resources-finished.json')
    seconds = finished['seconds_since_allocation']
    count = len(allocation['instances'])
    traffic = 0
    for path in (home/'observations').glob('*.json'):
        for line in read(path)['network_counters'].splitlines()[2:]:
            name, counters = line.split(':')
            if name.strip() != 'lo':
                values = counters.split()
                traffic += int(values[0])+int(values[8])
    # Current model plus ALL retained work objects in the final lineage catalog.
    # Count unique keys once. Evidence/source allowance is additional to those bytes.
    lengths = read(prepared/'object-lengths.json')
    for key, row in read(prepared/'retention-catalog.json').items():
        if key in lengths and lengths[key] != row['bytes']:
            raise ValueError('Retention inventories disagree on immutable bytes')
        lengths[key] = row['bytes']
    retained = sum(lengths.values()) + 1024**3
    # A shorter month gives the conservative denominator for per-second storage.
    month_fraction = seconds/(28*24*3600)
    parts = {'all_compute': finished['compute_cost_upper_estimate_usd'],
        'ebs_capacity': count*freeze['resources']['disk_gib']*.08*month_fraction,
        'ebs_extra_iops': count*(12000-3000)*.005*month_fraction,
        'ebs_extra_throughput': count*(500-125)*.04*month_fraction,
        'public_ipv4': count*.005*seconds/3600,
        'all_interface_traffic_upper': traffic/1e9*.15,
        'ninety_day_retention': retained/1e9*.023*3,
        'object_requests_and_evidence_allowance': 2.0,
        'prior_failed_allocation_allowance': freeze.get('retry', {}).get('prior_allocation_allowance_usd', 0)}
    report = {'format': 'neuroshard-finite-service-cost-v1', 'usd_upper_estimates': parts,
        'total_usd_upper': sum(parts.values()), 'measured_interface_bytes': traffic,
        'retained_model_work_and_evidence_bytes': retained, 'retention_days': 90,
        'sponsor_cap_usd': freeze['resources']['planning_cap_usd'],
        'funding': 'Existing operator AWS credit; finite trial sponsorship. Native NEURO has no assumed USD price.',
        'compute_scope': 'All seven instances from allocation through retirement, including native consensus, bootstrap, idle time and every complete numerical replay.',
        'transfer_scope': 'Conservative $0.15/decimal GB against sum of all non-loopback transmit and receive counters, including inbound bytes and all cross-zone directions; not billed usage.',
        'retention_scope': 'Current final-lineage public catalog and model metadata, deduplicated, plus 1 GiB source/evidence allowance for 90 days; original training compute is sunk, not charged as new serving work.',
        'price_sources': ['https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-east-1/index.csv',
            'https://aws.amazon.com/vpc/pricing/', 'https://aws.amazon.com/s3/pricing/',
            'https://aws.amazon.com/cloudfront/pricing/'],
        'limitations': 'Price-based pre-tax upper estimate, not an AWS invoice. Missing host counters make the cost result incomplete.',
        'all_host_counters_present': len(list((home/'observations').glob('*.json'))) == count}
    save(home/'costs.json', report)
    if report['total_usd_upper'] > report['sponsor_cap_usd'] or not report['all_host_counters_present']:
        raise ValueError('The complete service cost gate did not pass')


def run(args):
    home, prepared = args.home.resolve(), args.prepared.resolve()
    if home.exists() or subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT).strip():
        raise ValueError('Require a new trial directory and fully committed source')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    freeze = read(ROOT/'config/experiments/provider-llm-service.json')
    graph, manifest = read(prepared/'graph.json'), read(prepared/'native-manifest.json')
    provider_assets.validate_executor(graph, read(prepared/'profile.json'), ROOT)
    if validate_serving_policy(graph, Objects(prepared/'policies'), ROOT) != read(prepared/'policy.json'):
        raise ValueError('Prepared serving metadata differs from its committed policy')
    if manifest['code_hash'] != code_hash() or identity(graph) != read(prepared/'migration.json')['graph']:
        raise ValueError('Rebuild deployment metadata against the committed source before allocating')
    home.mkdir()
    save(home/'freeze.json', freeze)
    save(home/'migration.json', read(prepared/'migration.json'))
    save(home/'source.json', {'commit': revision, 'code_hash': code_hash()})
    engine = args.engine.resolve()
    network = cloud = None
    audit_pool = ThreadPoolExecutor(max_workers=1)
    boxes, providers = [], []
    started = time.monotonic()
    try:
        wallets = customer_wallets(home)
        allocate(home, freeze['resources'], revision)
        cloud = Cloud(home)
        bootstrap(cloud, prepared, revision, engine)
        network = native_nodes(cloud, manifest, engine)
        for replica in range(2):
            for rank, physical in enumerate(freeze['physical_placement']):
                providers.append(provider(cloud, network, graph, rank, physical, len(providers)))
        restoring = time.monotonic()
        restore(cloud, graph, providers)
        parallel(lambda row: start_provider(cloud, row), providers)
        service = audit_service(cloud, graph, providers)
        save(home/'cold-assets.json', {'seconds': time.monotonic()-restoring})
        customers = []
        for index, wallet in enumerate(wallets):
            network.send(3, f'customer-{index}/fund', 'transfer', to=wallet.public_key, amount=3_000_000_000)
            node = LocalNode(network.urls[index], network.genesis['chain_id'], identity(manifest))
            box = Outbox(home/f'customer-{index}/transactions.sqlite', node.url, node.chain_id, wallet)
            boxes.append(box)
            customers.append(Customer(home/f'customer-{index}/requests', node, wallet, box))
        warm = freeze['warmup']
        batch(cloud, network, graph, service, providers, customers,
              [{'id': 'warmup-'+str(i), 'messages': warm['messages']} for i in range(2)], warm['max_tokens'], audit_pool=audit_pool)
        # Both provider replicas must actually load and answer before declaring
        # cold readiness. This conservative bound also includes warmup settlement.
        cold_seconds = time.monotonic()-restoring
        save(home/'cold.json', {'seconds': cold_seconds, 'includes_both_paid_warmups_and_full_settlement': True})
        results = []
        for offset in range(0, 6, 2):
            results.extend(batch(cloud, network, graph, service, providers, customers,
                freeze['requests'][offset:offset+2], freeze['generation']['max_tokens'], audit_pool=audit_pool))
        for case in freeze['requests'][6:]:
            results.extend(batch(cloud, network, graph, service, providers, customers[:1], [case],
                                 freeze['generation']['max_tokens'], audit_pool=audit_pool))
        target = freeze['targets']
        ordinary = results[:6]
        passed = (cold_seconds <= target['cold_ready_seconds'] and
            all(row['generated_seconds'] is not None and row['generated_seconds'] <= target['warm_generation_p95_seconds']
                and row['first_visible_seconds'] is not None and row['first_visible_seconds'] <= target['warm_first_visible_p95_seconds']
                and row['settled_seconds'] <= target['settlement_seconds_per_request'] for row in ordinary) and
            all(row['settled_seconds'] <= target['recovered_settlement_seconds'] for row in results[6:]))
        save(home/'result.json', {'passed': passed, 'cold_seconds': cold_seconds, 'requests': results,
            'seconds': time.monotonic()-started, 'issued': network.query()['issued'],
            'note': 'One administrator, seven physical hosts. Three actual replays per response; not independent operators.'})
    except BaseException as error:
        save(home/'failure.json', {'error': type(error).__name__, 'detail': str(error)[:2000],
                                  'traceback': traceback.format_exc(), 'seconds': time.monotonic()-started})
        raise
    finally:
        audit_pool.shutdown(wait=True, cancel_futures=True)
        if cloud is not None:
            try:
                collect_observations(cloud)
            except Exception as error:
                save(home/'observation-failure.json', {'error': str(error)[:2000]})
            if network is not None:
                try:
                    evidence(cloud, network)
                except Exception as error:
                    save(home/'evidence-failure.json', {'error': str(error)[:2000]})
        for box in boxes:
            box.close()
        if network is not None:
            network.close()
        if (home/'allocation.json').exists():
            retire(home)
            costs(home, prepared, freeze)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--engine', type=Path, required=True)
    run(parser.parse_args())
