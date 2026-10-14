#!/usr/bin/env python3
"""General-route release qualification: one temporary client, fresh shadow services, evidence before shutdown.

    python scripts/release_check_general_supervise.py launch COMMIT --home <dir>
    ... prepare COMMIT --candidate <dir with network.json and assets/> ; start ; watch ; status

Adapted from A3's attempt-2 supervisor. The protected ledger and owner instances are never stopped or
terminated: only the shadow units on them (neuroshard-release-general-*) and the temporary client.
Evidence is copied before any service stops, and the client has an independent OS shutdown timer.
"""

import json
import pathlib
import subprocess
import sys
import time

HOME = pathlib.Path(sys.argv[sys.argv.index('--home') + 1]).resolve() if '--home' in sys.argv else pathlib.Path.cwd()
CANDIDATE = pathlib.Path(sys.argv[sys.argv.index('--candidate') + 1]).resolve() if '--candidate' in sys.argv else None
KEY = '/home/ubuntu/.ssh/id_ed25519'
SEED, OWNER = '172.31.73.85', '172.31.76.9'
PROTECTED = ('i-0d681a8ef83f72619', 'i-06bf7f1f01e6228bb', 'i-0515351dfce90dbd7', 'i-08b8035d33b53c0b7')
REGION, SUBNET, GROUP, IMAGE = 'us-east-1', 'subnet-06a30cbe7b35238a9', 'sg-075e8fd3ea722b364', 'ami-0fa5967347d08d2df'
STATE = '/home/ubuntu/.neuroshard/assistant-release-check-general'
CLIENT_MAX, SHADOW_MAX = '16h45min', '17h15min'
MAX_SECONDS = 17 * 3600 + 1800        # supervisor retirement after launch
OS_SHUTDOWN_SECONDS = 18 * 3600       # independent guard on the client itself
PYTHON = '/home/ubuntu/.venvs/neuroshard-release/bin/python -m neuroshard.assistant.release_check_general'
SSH = ['ssh', '-i', KEY, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=30',
       '-o', 'StrictHostKeyChecking=accept-new']


def read(name, default=None):
    path = HOME / name
    return json.loads(path.read_text()) if path.exists() else default


def save(name, value):
    path = HOME / name
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def ssh(host, command, timeout=120, **kwargs):
    known = ['-o', 'UserKnownHostsFile=' + str(HOME / 'known_hosts')]
    return subprocess.run(SSH + known + ['ubuntu@' + host, command], timeout=timeout, **kwargs)


def aws(*args, timeout=120):
    out = subprocess.run(['/home/ubuntu/.local/bin/aws', 'ec2', '--region', REGION, *args, '--output', 'json'], capture_output=True,
                         text=True, timeout=timeout, check=True).stdout
    return json.loads(out) if out.strip() else {}


def unit(description, command, maximum):
    return '\n'.join(['[Unit]', f'Description={description}', 'After=network-online.target', '[Service]',
                      'Type=simple', 'User=ubuntu', 'WorkingDirectory=/home/ubuntu/neuroshard-release',
                      'Environment=PYTHONPATH=/home/ubuntu/neuroshard-release/src',
                      'Environment=COMETBFT_BINARY=/home/ubuntu/neuroshard-tools/cometbft',
                      f'ExecStart={command}', f'RuntimeMaxSec={maximum}', 'TimeoutStopSec=30s',
                      'KillMode=control-group', 'Restart=no', ''])


def launch(commit):
    if read('allocation.json'):
        raise SystemExit('already launched')
    public = ' '.join(pathlib.Path(KEY + '.pub').read_text().split()[:2])
    deadline = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(time.time() + OS_SHUTDOWN_SECONDS))
    timer = f'[Timer]\nOnCalendar={deadline}\nPersistent=true\nAccuracySec=1s\n[Install]\nWantedBy=timers.target\n'
    cloud = {'ssh_authorized_keys': [public], 'write_files': [
        {'path': '/etc/systemd/system/neuroshard-expiry.service',
         'content': '[Service]\nType=oneshot\nExecStart=/usr/sbin/shutdown -h now\n'},
        {'path': '/etc/systemd/system/neuroshard-expiry.timer', 'content': timer}],
        'runcmd': [['systemctl', 'daemon-reload'], ['systemctl', 'enable', '--now', 'neuroshard-expiry.timer']]}
    tags = [{'Key': k, 'Value': v} for k, v in {'Name': 'neuroshard-release-check-general', 'Project': 'NeuroShard',
                                                'Purpose': 'assistant-release-check-general', 'Source': commit,
                                                'BudgetUSD': '40'}.items()]
    devices = [{'DeviceName': '/dev/sda1', 'Ebs': {'VolumeSize': 60, 'VolumeType': 'gp3', 'Encrypted': True,
                                                   'DeleteOnTermination': True}}]
    run = aws('run-instances', '--image-id', IMAGE, '--instance-type', 'm7i.2xlarge', '--count', '1',
              '--subnet-id', SUBNET, '--security-group-ids', GROUP, '--key-name', 'NextAI',
              '--instance-initiated-shutdown-behavior', 'terminate',
              '--metadata-options', 'HttpTokens=required,HttpPutResponseHopLimit=1',
              '--block-device-mappings', json.dumps(devices),
              '--tag-specifications', json.dumps([{'ResourceType': k, 'Tags': tags} for k in ('instance', 'volume')]),
              '--user-data', '#cloud-config\n' + json.dumps(cloud))
    instance = run['Instances'][0]
    allocation = {'region': REGION, 'id': instance['InstanceId'], 'private_ip': instance['PrivateIpAddress'],
                  'instance_type': 'm7i.2xlarge', 'commit': commit, 'launched_unix': time.time(),
                  'max_seconds': MAX_SECONDS, 'os_shutdown_utc': deadline, 'budget_usd': 40, 'attempt': 1}
    save('allocation.json', allocation)
    print(json.dumps(allocation, indent=1))


def wait_ssh(host, seconds=900):
    stop = time.time() + seconds
    while time.time() < stop:
        if ssh(host, 'true', timeout=20, capture_output=True).returncode == 0:
            return
        time.sleep(10)
    raise TimeoutError(f'{host} unreachable')


def install(host, commit, fresh):
    """The frozen commit in a detached checkout and the pinned runtime; protected hosts reuse their venv."""
    steps = ['set -euo pipefail']
    if fresh:
        uv = '/home/ubuntu/.venvs/bootstrap/bin/uv'
        steps += ['sudo cloud-init status --wait >/dev/null', 'systemctl is-active neuroshard-expiry.timer',
                  'sudo apt-get -o DPkg::Lock::Timeout=300 update -qq',
                  'sudo apt-get -o DPkg::Lock::Timeout=300 install -y -qq python3-venv git',
                  'python3 -m venv /home/ubuntu/.venvs/bootstrap',
                  '/home/ubuntu/.venvs/bootstrap/bin/python -m pip install -q uv==0.12.24',
                  f'{uv} python install 3.12.12',
                  f'{uv} venv --python 3.12.12 /home/ubuntu/.venvs/neuroshard-release',
                  'rm -rf /home/ubuntu/neuroshard-release',
                  'git clone -q https://github.com/neuroshard-ai/neuroshard.git /home/ubuntu/neuroshard-release',
                  f'git -C /home/ubuntu/neuroshard-release fetch -q origin {commit}',
                  f'git -C /home/ubuntu/neuroshard-release checkout -q --detach {commit}',
                  f'{uv} pip install -q --python /home/ubuntu/.venvs/neuroshard-release/bin/python '
                  '-r /home/ubuntu/neuroshard-release/docs/granite-shard-chain-requirements.txt']
    else:
        uv = '/home/ubuntu/.local/neuroshard-bootstrap/bin/uv'
        steps += [f'git -C /home/ubuntu/neuroshard fetch -q https://github.com/neuroshard-ai/neuroshard.git {commit}',
                  f'git -C /home/ubuntu/neuroshard-release checkout -q --detach {commit}',
                  f'{uv} pip install -q --python /home/ubuntu/.venvs/neuroshard-release/bin/python '
                  '-r /home/ubuntu/neuroshard-release/docs/granite-shard-chain-requirements.txt']
    steps += [f'test "$(git -C /home/ubuntu/neuroshard-release rev-parse HEAD)" = {commit}',
              'test -z "$(git -C /home/ubuntu/neuroshard-release status --porcelain)"']
    result = ssh(host, 'bash -s', timeout=2400, input='\n'.join(steps), capture_output=True, text=True)
    (HOME / f'install-{host}.log').write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        raise SystemExit(f'install failed on {host}; see install-{host}.log')


def units(client):
    seed = f'{PYTHON} seed --network {STATE}/network.json --home {STATE}/seed --host {SEED}'
    owner = lambda rank, host: (f'{PYTHON} owner --shard {rank} --network {STATE}/network.json '  # noqa: E731
                                f'--home {STATE}/owner --host {host}')
    worker = f'{PYTHON} client --network {STATE}/network.json --home {STATE}/client --host {client}'
    return {SEED: {'neuroshard-release-general-seed': unit('NeuroShard general-route release shadow chain', seed, SHADOW_MAX),
                   'neuroshard-release-general-owner': unit('NeuroShard general-route release shadow owner 1', owner(1, SEED),
                                                    SHADOW_MAX)},
            OWNER: {'neuroshard-release-general-owner': unit('NeuroShard general-route release shadow owner 2', owner(2, OWNER),
                                                     SHADOW_MAX)},
            client: {'neuroshard-release-general-client': unit('NeuroShard general-route release: 324 frozen conversations',
                                                       worker, CLIENT_MAX)}}


def prepare(commit):
    """Install the frozen commit; give each shadow owner its shard bytes (hard links of the live holdings) and the
    candidate's released files; give the client the candidate descriptor and assets."""
    client = read('allocation.json')['private_ip']
    wait_ssh(client)
    descriptor = (CANDIDATE / 'network.json').read_bytes()
    assets = subprocess.run(['tar', '-cf', '-', '-C', str(CANDIDATE / 'assets'), '.'], capture_output=True,
                            check=True).stdout
    version = json.loads(descriptor)['version']
    for host, rank in ((SEED, 1), (OWNER, 2)):
        holdings = f'{STATE}/owner/holdings/{version}/stage-{rank}'
        linked = (f'mkdir -p {STATE}/assets {holdings} && for component in config shard; do '
                  f'[ -e {holdings}/$component ] || cp -al /home/ubuntu/.neuroshard/assistant-owner/a2-u1/stage-{rank}/'
                  f'$component {holdings}/$component; done')
        ssh(host, linked, check=True, capture_output=True)
        ssh(host, f'cat > {STATE}/network.json', input=descriptor, check=True)
        ssh(host, f'tar -xf - -C {STATE}/assets', input=assets, check=True)
        install(host, commit, fresh=False)
    install(client, commit, fresh=True)
    ssh(client, f'mkdir -p {STATE}/assets && cat > {STATE}/network.json', input=descriptor, check=True)
    ssh(client, f'tar -xf - -C {STATE}/assets', input=assets, check=True)
    for host, files in units(client).items():
        for name, text in files.items():
            ssh(host, f'sudo tee /etc/systemd/system/{name}.service >/dev/null', input=text.encode(), check=True)
        ssh(host, 'sudo systemctl daemon-reload', check=True)
    save('prepared.json', {'commit': commit, 'client': client, 'unix': time.time(), 'version': version})
    print('prepared')


def start():
    client = read('allocation.json')['private_ip']
    ssh(SEED, 'sudo systemctl start neuroshard-release-general-seed', check=True)
    time.sleep(60)
    ssh(SEED, 'sudo systemctl start neuroshard-release-general-owner', check=True)
    ssh(OWNER, 'sudo systemctl start neuroshard-release-general-owner', check=True)
    time.sleep(240)
    ssh(client, 'sudo systemctl start neuroshard-release-general-client', check=True)
    save('started.json', {'unix': time.time()})
    print('started')


def remote_json(host, path, timeout=120):
    result = ssh(host, f'cat {path}', capture_output=True, text=True, timeout=timeout)
    return json.loads(result.stdout) if result.returncode == 0 and result.stdout.strip() else None


def collect(client):
    """Copy every comparison, the result and the logs before any service is stopped."""
    found = {}
    with (HOME / 'client-evidence.tar.gz').open('wb') as handle:
        found['client_tar'] = ssh(client, f'cd {STATE}/client && find . -maxdepth 1 -name "*.json" -print0 | '
                                          'tar -czf - --null -T -', stdout=handle, stderr=subprocess.PIPE,
                                  timeout=1800).returncode
    for host, names in ((client, ('neuroshard-release-general-client',)),
                        (SEED, ('neuroshard-release-general-owner', 'neuroshard-release-general-seed')),
                        (OWNER, ('neuroshard-release-general-owner',))):
        for name in names:
            log = ssh(host, f'journalctl --no-pager -n 400 -u {name}', capture_output=True, text=True, timeout=120)
            (HOME / f'{host}-{name}.log').write_text(log.stdout + log.stderr)
    for host in (SEED, OWNER):
        count = ssh(host, f'ls {STATE}/owner/logs 2>/dev/null | wc -l', capture_output=True, text=True, timeout=120)
        found[f'{host}_owner_logs'] = count.stdout.strip()
    return found


def retire(client, allocation):
    """Stop shadow units only, then terminate the temporary client; refuse anything protected."""
    stopped = {}
    for host, names in ((client, 'neuroshard-release-general-client'),
                        (SEED, 'neuroshard-release-general-owner neuroshard-release-general-seed'),
                        (OWNER, 'neuroshard-release-general-owner')):
        try:
            stopped[host] = ssh(host, f'sudo systemctl stop {names}', capture_output=True, timeout=180).returncode
        except subprocess.TimeoutExpired:
            stopped[host] = 'timeout'
    if allocation['id'] in PROTECTED:
        raise SystemExit('refusing to terminate a protected instance')
    described = aws('describe-instances', '--instance-ids', allocation['id'])
    tags = {t['Key']: t['Value'] for t in described['Reservations'][0]['Instances'][0].get('Tags', [])}
    if tags.get('Purpose') != 'assistant-release-check-general':
        raise SystemExit('instance purpose tag mismatch; refusing to terminate')
    aws('terminate-instances', '--instance-ids', allocation['id'])
    return stopped


def watch():
    allocation = read('allocation.json')
    client = allocation['private_ip']
    ended, reason = allocation['launched_unix'] + allocation['max_seconds'], 'time budget'
    while time.time() < ended:
        try:
            value = remote_json(client, f'{STATE}/client/result.json')
            if value:
                save('result.json', value)
                reason = 'passed' if value['passed'] else 'failed'
                break
            state = ssh(client, 'systemctl is-active neuroshard-release-general-client', capture_output=True, text=True,
                        timeout=60).stdout.strip()
            if read('started.json') and state in ('failed', 'inactive'):
                time.sleep(30)
                value = remote_json(client, f'{STATE}/client/result.json')
                if value:
                    save('result.json', value)
                    reason = 'passed' if value['passed'] else 'failed'
                else:
                    reason = f'client {state} without a result'
                break
            current = remote_json(client, f'{STATE}/client/progress.json')
            if current:
                save('progress.json', current)
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as error:
            save('watch-error.json', {'unix': time.time(), 'error': str(error)})
        time.sleep(60)
    finished = {'reason': reason, 'unix': time.time()}
    if reason == 'time budget':
        # Ask the worker to stop between conversations and record its result, then wait for it.
        ssh(client, 'sudo systemctl kill -s TERM neuroshard-release-general-client', capture_output=True, timeout=60)
        for _ in range(20):
            value = remote_json(client, f'{STATE}/client/result.json')
            if value:
                save('result.json', value)
                break
            time.sleep(30)
    try:
        finished['evidence'] = collect(client)
    except Exception as error:  # evidence failure must not block retirement
        finished['evidence_error'] = str(error)
    finished['stopped'] = retire(client, allocation)
    finished['instance_seconds'] = time.time() - allocation['launched_unix']
    finished['client_usd'] = finished['instance_seconds'] / 3600 * 0.4032
    save('resources-finished.json', finished)
    print(json.dumps(finished, indent=1))


def status():
    allocation = read('allocation.json')
    print(json.dumps({'allocation': allocation and allocation['id'], 'progress': read('progress.json'),
                      'result': (read('result.json') or {}).get('passed')}, indent=1))


if __name__ == '__main__':
    HOME.mkdir(parents=True, exist_ok=True)
    command = sys.argv[1]
    if command in ('launch', 'prepare'):
        globals()[command](sys.argv[2])
    else:
        globals()[command]()
