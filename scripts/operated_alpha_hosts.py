"""Four dedicated ledger hosts for a finite, single-administrator public alpha.

The GPU allocator remains separate. Native consensus can expire and refund jobs
when GPU funding ends. Every mutation targets only this recorded allocation.
"""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shlex
import subprocess
import time
import uuid

import boto3
from botocore.exceptions import ClientError
from ordinary_allocation import describe
from ordinary_cloud import Cloud, PROTECTED, REMOTE, REPO
from neuroshard.evolution.reference_data import save


class LedgerHosts(Cloud):
    def __init__(self, home):
        self.home = Path(home).resolve()
        allocation = json.loads((self.home/'allocation.json').read_bytes())
        self.hosts = sorted(allocation['instances'], key=lambda row: row['rank'])
        self.deadline = datetime.fromisoformat(allocation['deadline'])
        if (len(self.hosts) != 4 or [r['rank'] for r in self.hosts] != list(range(4))
                or len({r['InstanceId'] for r in self.hosts}) != 4
                or any(r['InstanceId'] in PROTECTED for r in self.hosts)):
            raise ValueError('Require the four recorded nonprotected ledger hosts')
        self.environment = {'PYTHONPATH': REPO+'/src', 'ATEN_CPU_CAPABILITY': 'default',
            'MKL_ENABLE_INSTRUCTIONS': 'SSE4_2', 'OMP_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}

    def ssh(self, physical):
        if type(physical) is not int or physical not in range(4):
            raise ValueError('Unknown ledger host')
        return ['ssh', '-i', '/home/ubuntu/.ssh/id_ed25519', '-o', 'BatchMode=yes',
            '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=accept-new',
            '-o', 'UserKnownHostsFile='+str(self.home/'known_hosts'),
            'ubuntu@'+self.hosts[physical]['PrivateIpAddress']]


def allocate(home, resources, revision):
    home = Path(home)
    home.mkdir(parents=True, exist_ok=False)
    ec2 = boto3.client('ec2', region_name=resources['region'])
    source = describe(ec2, ['i-0d681a8ef83f72619'])[0]
    images = ec2.describe_images(Owners=['099720109477'], Filters=[
        {'Name': 'name', 'Values': ['ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*']},
        {'Name': 'state', 'Values': ['available']}])['Images']
    image = max(images, key=lambda row: row['CreationDate'])
    now = datetime.now(timezone.utc)
    deadline = now + timedelta(hours=resources['hours'])
    name = 'neuroshard-alpha-ledger-'+uuid.uuid4().hex[:10]
    allocation = {'name': name, 'region': resources['region'], 'created': now.isoformat(),
        'deadline': deadline.isoformat(), 'source_commit': revision, 'instances': [],
        'image': image['ImageId'], 'resources': resources}
    save(home/'allocation.json', allocation)
    try:
        group = ec2.create_security_group(GroupName=name,
            Description='Operated alpha native peers; RPC remains loopback', VpcId=source['VpcId'])['GroupId']
        allocation['security_group'] = group
        save(home/'allocation.json', allocation)
        ec2.authorize_security_group_ingress(GroupId=group, IpPermissions=[
            {'IpProtocol': 'tcp', 'FromPort': 22, 'ToPort': 22,
             'IpRanges': [{'CidrIp': source['PrivateIpAddress']+'/32'}]},
            {'IpProtocol': 'tcp', 'FromPort': 26656, 'ToPort': 26656,
             'IpRanges': [{'CidrIp': '0.0.0.0/0'}]},
            {'IpProtocol': 'tcp', 'FromPort': 18480, 'ToPort': 18480,
             'IpRanges': [{'CidrIp': '0.0.0.0/0'}]}])
        user_data = {'ssh_authorized_keys': [' '.join(Path('/home/ubuntu/.ssh/id_ed25519.pub').read_text().split()[:2])],
            'write_files': [
                {'path': '/etc/systemd/system/neuroshard-ledger-expiry.service', 'content':
                 '[Unit]\nDescription=End funded alpha ledger window\n[Service]\nType=oneshot\nExecStart=/usr/sbin/shutdown -h now\n'},
                {'path': '/etc/systemd/system/neuroshard-ledger-expiry.timer', 'content':
                 '[Unit]\nDescription=Absolute funded ledger deadline\n[Timer]\nOnCalendar='+
                 deadline.strftime('%Y-%m-%d %H:%M:%S UTC')+
                 '\nPersistent=true\nAccuracySec=1s\n[Install]\nWantedBy=timers.target\n'}],
            'runcmd': [['systemctl', 'daemon-reload'], ['systemctl', 'enable', '--now', 'neuroshard-ledger-expiry.timer']]}
        subnets = sorted(ec2.describe_subnets(Filters=[{'Name': 'vpc-id', 'Values': [source['VpcId']]}])['Subnets'],
                         key=lambda row: (row['AvailabilityZone'], row['SubnetId']))
        zones = {}
        for subnet in subnets:
            zones.setdefault(subnet['AvailabilityZone'], subnet)
        if len(zones) < 3:
            raise ValueError('Declare at least three ledger availability zones')
        for rank in range(4):
            subnet = list(zones.values())[rank % len(zones)]
            tags = [{'Key': key, 'Value': value} for key, value in {'Name': name, 'Project': 'NeuroShard',
                'Purpose': 'operated-alpha-ledger', 'Rank': str(rank), 'ExpiresAt': deadline.isoformat(),
                'Source': revision, 'BudgetUSD': str(resources['planning_cap_usd'])}.items()]
            row = ec2.run_instances(ImageId=image['ImageId'], InstanceType='t3.medium', MinCount=1, MaxCount=1,
                KeyName=source['KeyName'], ClientToken=uuid.uuid5(uuid.NAMESPACE_URL, name+'/'+str(rank)).hex,
                UserData='#cloud-config\n'+json.dumps(user_data), InstanceInitiatedShutdownBehavior='terminate',
                MetadataOptions={'HttpTokens': 'required', 'HttpPutResponseHopLimit': 1},
                CreditSpecification={'CpuCredits': 'standard'},
                NetworkInterfaces=[{'DeviceIndex': 0, 'SubnetId': subnet['SubnetId'], 'Groups': [group],
                    'AssociatePublicIpAddress': True, 'DeleteOnTermination': True}],
                BlockDeviceMappings=[{'DeviceName': image['RootDeviceName'], 'Ebs': {'VolumeSize': 40,
                    'VolumeType': 'gp3', 'Encrypted': True, 'DeleteOnTermination': True}}],
                TagSpecifications=[{'ResourceType': value, 'Tags': tags} for value in ('instance', 'volume')])['Instances'][0]
            if row['InstanceId'] in PROTECTED:
                raise ValueError('A protected instance entered the ledger allocation')
            allocation['instances'].append({'rank': rank, 'InstanceId': row['InstanceId']})
            save(home/'allocation.json', allocation)
        deadline_start = time.monotonic() + 600
        while time.monotonic() < deadline_start:
            found = {r['InstanceId']: r for r in describe(ec2, [r['InstanceId'] for r in allocation['instances']])}
            if all(r['State']['Name'] == 'running' and r.get('PublicIpAddress') for r in found.values()):
                for row in allocation['instances']:
                    actual = found[row['InstanceId']]
                    row.update(PrivateIpAddress=actual['PrivateIpAddress'], PublicIpAddress=actual['PublicIpAddress'],
                        AvailabilityZone=actual['Placement']['AvailabilityZone'], LaunchTime=actual['LaunchTime'].isoformat(),
                        volumes=[v['Ebs']['VolumeId'] for v in actual['BlockDeviceMappings']])
                save(home/'allocation.json', allocation)
                return allocation
            time.sleep(3)
        raise TimeoutError('Ledger hosts did not become available')
    except BaseException:
        retire(home)
        raise


def retire(home):
    home = Path(home)
    value = json.loads((home/'allocation.json').read_bytes())
    ids = {r['InstanceId'] for r in value['instances']}
    if ids & PROTECTED:
        raise ValueError('Never retire a protected host')
    ec2 = boto3.client('ec2', region_name=value['region'])
    if ids:
        actual = describe(ec2, ids)
        if any(next((t['Value'] for t in r.get('Tags', []) if t['Key'] == 'Name'), None) != value['name'] for r in actual):
            raise ValueError('Ledger allocation ownership tag changed')
        ec2.terminate_instances(InstanceIds=sorted(ids))
    save(home/'retirement-requested.json', {'instances': sorted(ids), 'time': datetime.now(timezone.utc).isoformat()})
    until = time.monotonic()+600
    while ids and any(r['State']['Name'] != 'terminated' for r in describe(ec2, ids)):
        if time.monotonic() >= until:
            raise TimeoutError('Ledger termination remains incomplete')
        time.sleep(3)
    if value.get('security_group'):
        for _ in range(30):
            try:
                ec2.delete_security_group(GroupId=value['security_group'])
                break
            except ClientError as error:
                if error.response['Error']['Code'] == 'InvalidGroup.NotFound':
                    break
                if error.response['Error']['Code'] != 'DependencyViolation':
                    raise
                time.sleep(2)
        else:
            raise TimeoutError('Ledger security group still has dependencies')
    until = time.monotonic()+180
    while ec2.describe_volumes(Filters=[{'Name': 'tag:Name', 'Values': [value['name']]}])['Volumes']:
        if time.monotonic() >= until:
            raise ValueError('Retired ledger still has surviving tagged volumes')
        time.sleep(3)
    save(home/'resources-finished.json', {'terminated_instances': sorted(ids), 'remaining_volumes': [],
        'security_group_retired': True, 'finished': datetime.now(timezone.utc).isoformat()})


def bootstrap(hosts, revision, engine):
    from concurrent.futures import ThreadPoolExecutor
    root = Path(__file__).resolve().parents[1]
    archive = subprocess.check_output(['git', 'archive', '--format=tar.gz', revision], cwd=root)
    setup = '''set -eu
sudo cloud-init status --wait
sudo systemctl is-active neuroshard-ledger-expiry.timer
sudo apt-get -o Acquire::Retries=3 update -qq
sudo apt-get -o DPkg::Lock::Timeout=300 -o Acquire::Retries=3 install -y -qq python3-venv
cd /home/ubuntu/neuroshard-study
python3 -m venv .neuroshard/native
.neuroshard/native/bin/python -m pip install -r docs/evolution-requirements.txt
chmod 700 /home/ubuntu/native-expert-live/cometbft
'''
    def one(index):
        until = time.monotonic() + 300
        while True:
            try:
                hosts.command(index, ['true'], timeout=20)
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                if time.monotonic() > until:
                    raise
                time.sleep(3)
        hosts.command(index, ['mkdir', '-p', REMOTE, REPO])
        hosts.command(index, ['tar', '-xzf', '-', '-C', REPO], input=archive, timeout=180)
        hosts.bundle(index, {'cometbft': engine.read_bytes()})
        output = hosts.command(index, ['timeout', '--kill-after=20', '1200', 'bash', '-s'], input=setup.encode(), timeout=1230)
        (hosts.home/f'bootstrap-{index}.log').write_bytes(output.stdout+output.stderr)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, range(4)))
