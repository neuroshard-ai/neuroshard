"""Allocate and retire only the disposable owners of one frozen campaign."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time
import uuid

import boto3
from botocore.exceptions import ClientError

from neuroshard.evolution import expert_checkpoint
from neuroshard.evolution.reference_data import identity, save
from ordinary_cloud import Cloud, PROTECTED, REMOTE, REPO, PYTHON, PUBLIC

ROOT = Path(__file__).resolve().parents[1]
AMI = 'ami-0efd0f6e601922cad'


def numerical_runtime(observation):
    # Match the established sharded-training convention: a hostname identifies
    # an owner, not its arithmetic. Keep every actual numerical field.
    return {key: value for key, value in observation.items() if key != 'host'}


def describe(client, ids):
    requested = set(ids)
    if not requested:
        return []
    deadline, pause = time.monotonic()+120, .5
    while True:
        try:
            found = [instance for row in client.describe_instances(InstanceIds=sorted(requested))['Reservations']
                     for instance in row['Instances']]
            if {row['InstanceId'] for row in found} == requested:
                return found
        except ClientError as error:
            if error.response['Error']['Code'] not in ('InvalidInstanceID.NotFound', 'RequestLimitExceeded', 'Throttling'):
                raise
        if time.monotonic() >= deadline:
            raise TimeoutError('EC2 has not yet exposed the complete owned instance inventory')
        time.sleep(pause)
        pause = min(8, pause*2)


def inventories(graph, catalog):
    """No physical owner receives the whole frozen backbone or interpreter."""
    result = [dict() for _ in range(7)]

    def add(physical, spec, folder):
        key = spec['sha256']
        original = catalog[key]
        if original['sha256'] != key or original['bytes'] != spec['bytes']:
            raise ValueError('A public replica changed an immutable tensor identity')
        item = {'bytes': spec['bytes'], 'path': REMOTE+'/'+folder+'/'+key+'.safetensors',
                'urls': original['urls'] if 'urls' in original else [original['url']]}
        if key in result[physical] and result[physical][key] != item:
            raise ValueError('An owned tensor has conflicting storage prescriptions')
        result[physical][key] = item

    for name, spec in graph['parent']['tensors'].items():
        rank = expert_checkpoint.owner(name, graph['parent']['boundaries'])
        add(rank, spec, 'objects')
        if name in ('model.embed_tokens.weight', 'model.norm.weight') or (
                name.startswith('model.layers.') and int(name.split('.')[2]) >= graph['descriptor']['split']):
            for physical in range(3, 7):
                add(physical, spec, 'objects')
    for rank in range(3):
        for spec in graph['interpreter_assets']['partitions'][str(rank)]['tensors'].values():
            add(rank, spec, 'interpreter')
    for index, route in enumerate(graph['descriptor']['rules']):
        for spec in graph['experts'][route['id']]['tensors'].values():
            add(3+index, spec, 'objects')
    for physical in range(2, 7):
        for spec in graph['experts']['planner']['tensors'].values():
            add(physical, spec, 'objects')
    all_parent = {spec['sha256'] for spec in graph['parent']['tensors'].values()}
    if any(all_parent <= set(owner) for owner in result):
        raise ValueError('An owner must not acquire the complete backbone')
    return result


def allocate(home, resources, source_commit):
    home = Path(home)
    path = home/'allocation.json'
    if path.exists():
        raise ValueError('An allocation inventory already exists; inspect or retire it before another launch')
    if (resources['instance_types'] not in (['g6e.xlarge']*7, ['g5.xlarge']*7)
            or set(resources['protected_instances']) != PROTECTED):
        raise ValueError('Require the frozen seven-owner allocation')
    ec2 = boto3.client('ec2', region_name=resources['region'])
    protected = describe(ec2, PROTECTED)
    source = next(value for value in protected if value['InstanceId'] == 'i-0d681a8ef83f72619')
    save(home/'protected-before.json', {value['InstanceId']: value['State']['Name'] for value in protected})
    image = ec2.describe_images(ImageIds=[AMI])['Images'][0]
    if image['State'] != 'available' or image['OwnerId'] != '898082745236':
        raise ValueError('The previously used numerical image is unavailable')
    now = datetime.now(timezone.utc)
    deadline = now+timedelta(hours=resources['max_hours'])
    name = 'neuroshard-ordinary-'+uuid.uuid4().hex[:12]
    rate, sku = {'g5.xlarge': (1.006, '79BRGEEZC6TARVWJ'),
                 'g6e.xlarge': (1.861, 'HEU7PA78QPB8SDYY')}[resources['instance_types'][0]]
    allocation = {'name': name, 'region': resources['region'], 'created': now.isoformat(),
        'deadline': deadline.isoformat(), 'source_commit': source_commit, 'instances': [],
        'resources': resources, 'hourly_rate_usd': rate,
        'price_source': 'https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-east-1/index.csv',
        'price_sku': sku, 'price_effective_date': '2026-09-01'}
    save(path, allocation)
    try:
        group = ec2.create_security_group(GroupName=name, Description='Bounded ordinary native learning campaign',
                                           VpcId=source['VpcId'])['GroupId']
        allocation['security_group'] = group
        save(path, allocation)
        ec2.authorize_security_group_ingress(GroupId=group, IpPermissions=[
            {'IpProtocol': 'tcp', 'FromPort': 22, 'ToPort': 22,
             'IpRanges': [{'CidrIp': source['PrivateIpAddress']+'/32'}]},
            {'IpProtocol': 'tcp', 'FromPort': 1024, 'ToPort': 65535,
             'UserIdGroupPairs': [{'GroupId': group}]}])
        cloud = {'ssh_authorized_keys': [' '.join(Path('/home/ubuntu/.ssh/id_ed25519.pub').read_text().split()[:2])],
            'write_files': [
                {'path': '/etc/systemd/system/neuroshard-experiment-stop.service', 'content':
                 '[Unit]\nDescription=Retire bounded GPU experiment\n[Service]\nType=oneshot\nExecStart=/usr/sbin/shutdown -h now\n'},
                {'path': '/etc/systemd/system/neuroshard-experiment-stop.timer', 'content':
                 '[Unit]\nDescription=Absolute experiment deadline\n[Timer]\nOnCalendar='+deadline.strftime('%Y-%m-%d %H:%M:%S UTC')+
                 '\nPersistent=true\nAccuracySec=1s\n[Install]\nWantedBy=timers.target\n'}],
            'runcmd': [['systemctl', 'daemon-reload'], ['systemctl', 'enable', '--now', 'neuroshard-experiment-stop.timer']]}
        subnets = sorted(ec2.describe_subnets(Filters=[{'Name': 'vpc-id', 'Values': [source['VpcId']]}])['Subnets'],
                         key=lambda value: (value['AvailabilityZone'], value['SubnetId']))
        if ec2.describe_vpcs(VpcIds=[source['VpcId']])['Vpcs'][0]['IsDefault']:
            # EC2 can select available capacity across the default VPC's zones.
            # This changes placement only, never the frozen hardware or method.
            subnets.insert(0, {'SubnetId': None, 'AvailabilityZone': None})
        preferred = None
        for rank, kind in enumerate(resources['instance_types']):
            ordered = sorted(subnets, key=lambda value: value['AvailabilityZone'] != preferred) if preferred else subnets
            for subnet in ordered:
                tags = [{'Key': key, 'Value': value} for key, value in {'Name': name, 'Project': 'NeuroShard',
                    'Purpose': 'ordinary-native-campaign', 'Rank': str(rank), 'ExpiresAt': deadline.isoformat(),
                    'BudgetUSD': str(resources['planning_cap_usd']), 'Source': source_commit}.items()]
                request = {'ImageId': AMI, 'InstanceType': kind, 'MinCount': 1, 'MaxCount': 1, 'KeyName': source['KeyName'],
                    'ClientToken': uuid.uuid5(uuid.NAMESPACE_URL, name+'/'+str(rank)+'/'+str(subnet['SubnetId'])).hex,
                    'UserData': '#cloud-config\n'+json.dumps(cloud), 'InstanceInitiatedShutdownBehavior': 'terminate',
                    'MetadataOptions': {'HttpTokens': 'required', 'HttpPutResponseHopLimit': 1},
                    'NetworkInterfaces': [{'DeviceIndex': 0,
                                          **({'SubnetId': subnet['SubnetId']} if subnet['SubnetId'] else {}), 'Groups': [group],
                                          'AssociatePublicIpAddress': True, 'DeleteOnTermination': True}],
                    'BlockDeviceMappings': [{'DeviceName': image['RootDeviceName'], 'Ebs': {
                        'VolumeSize': resources['disk_gib'], 'VolumeType': 'gp3', 'Iops': 12000,
                        'Throughput': 500, 'Encrypted': True, 'DeleteOnTermination': True}}],
                    'TagSpecifications': [{'ResourceType': value, 'Tags': tags} for value in ('instance', 'volume')]}
                try:
                    instance = ec2.run_instances(**request)['Instances'][0]
                except ClientError as error:
                    if error.response['Error']['Code'] in ('InsufficientInstanceCapacity', 'Unsupported'):
                        allocation.setdefault('capacity_failures', []).append({'rank': rank,
                            'subnet': subnet['SubnetId'], 'error': error.response['Error']})
                        save(path, allocation)
                        continue
                    raise
                if instance['InstanceId'] in PROTECTED:
                    raise ValueError('A protected instance entered the disposable allocation')
                allocation['instances'].append({'rank': rank, 'InstanceId': instance['InstanceId']})
                preferred = instance['Placement']['AvailabilityZone']
                save(path, allocation)
                break
            else:
                raise RuntimeError('No capacity for the frozen owner type')
        until = time.monotonic()+600
        while time.monotonic() < until:
            found = {value['InstanceId']: value for value in describe(ec2, [row['InstanceId'] for row in allocation['instances']])}
            if all(value['State']['Name'] == 'running' and value.get('PrivateIpAddress') for value in found.values()):
                for row in allocation['instances']:
                    instance = found[row['InstanceId']]
                    row.update(PrivateIpAddress=instance['PrivateIpAddress'], LaunchTime=instance['LaunchTime'].isoformat(),
                        AvailabilityZone=instance['Placement']['AvailabilityZone'],
                        volumes=[value['Ebs']['VolumeId'] for value in instance['BlockDeviceMappings']])
                save(path, allocation)
                return allocation
            time.sleep(3)
        raise TimeoutError('The complete owner pool did not start')
    except BaseException:
        retire(home)
        raise


def bootstrap(home, catalog):
    home = Path(home)
    cloud = Cloud(home)
    allocation = json.loads((home/'allocation.json').read_bytes())
    graph = json.loads((home/'compiled/baseline-core.json').read_bytes())
    inventories_by_owner = inventories(graph, catalog)
    archive = subprocess.check_output(['git', 'archive', '--format=tar.gz', allocation['source_commit']], cwd=ROOT)
    if len(archive) > 128*1024**2:
        raise ValueError('Bound the source deployment archive')
    seed = {path.name: path.read_bytes() for path in (home/'compiled/seed').iterdir()}

    def one(rank):
        until = time.monotonic()+600
        while time.monotonic() < until:
            try:
                cloud.command(rank, ['true'], timeout=20)
                break
            except subprocess.SubprocessError:
                time.sleep(3)
        else:
            raise TimeoutError('A disposable owner is not reachable over SSH')
        cloud.command(rank, ['mkdir', '-p', REMOTE, REPO])
        cloud.command(rank, ['tar', '-xzf', '-', '-C', REPO], input=archive, timeout=180)
        cloud.bundle(rank, {'seed/'+name: raw for name, raw in seed.items()})
        setup = '''set -eu
sudo cloud-init status --wait
sudo systemctl is-active neuroshard-experiment-stop.timer
sudo apt-get -o Acquire::Retries=3 update -qq
sudo apt-get -o DPkg::Lock::Timeout=300 -o Acquire::Retries=3 install -y -qq python3-venv git
cd /home/ubuntu/neuroshard-study
python3 -m venv .neuroshard/venv
.neuroshard/venv/bin/python -m pip install -r docs/expert-execution-requirements.txt
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
'''
        result = cloud.command(rank, ['timeout', '--kill-after=20', '1500', 'bash', '-s'], input=setup.encode(), timeout=1530)
        (home/('setup-'+str(rank)+'.log')).write_bytes(result.stdout+result.stderr)
        assets = cloud.assets(rank, {'action': 'fetch', 'objects': inventories_by_owner[rank]})
        save(home/('owner-assets-'+str(rank)+'.json'), assets)
        runtime = json.loads(cloud.python(rank, ['-c', 'import os,json; from neuroshard.evolution import reference; '
            'r=reference.configure("cuda",2); r["allocator"]=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"); print(json.dumps(r))'], timeout=180).stdout)
        save(home/('runtime-'+str(rank)+'.json'), runtime)
        return numerical_runtime(runtime)

    with ThreadPoolExecutor(max_workers=7) as pool:
        runtimes = list(pool.map(one, range(7)))
    if any(runtime != runtimes[0] for runtime in runtimes):
        raise ValueError('The seven actual numerical runtimes disagree')
    save(home/'runtime.json', runtimes[0])
    return runtimes[0]


def retire(home):
    home = Path(home)
    allocation = json.loads((home/'allocation.json').read_bytes())
    client = boto3.client('ec2', region_name=allocation['region'])
    ids = {row['InstanceId'] for row in allocation['instances']}
    if ids & PROTECTED:
        raise ValueError('Refuse to terminate a protected host')
    if ids:
        client.terminate_instances(InstanceIds=sorted(ids))
        until = time.monotonic()+600
        while time.monotonic() < until:
            instances = describe(client, ids)
            if all(row['State']['Name'] == 'terminated' for row in instances):
                break
            time.sleep(3)
        else:
            raise TimeoutError('Disposable instance termination is not complete')
    if allocation.get('security_group'):
        for _ in range(30):
            try:
                client.delete_security_group(GroupId=allocation['security_group'])
                break
            except ClientError as error:
                if error.response['Error']['Code'] == 'InvalidGroup.NotFound':
                    break
                if error.response['Error']['Code'] != 'DependencyViolation':
                    raise
                time.sleep(2)
        else:
            raise TimeoutError('Disposable security group still has dependencies')
    until = time.monotonic()+180
    while True:
        volumes = client.describe_volumes(Filters=[{'Name': 'tag:Name', 'Values': [allocation['name']]}])['Volumes']
        if not volumes:
            break
        if time.monotonic() >= until:
            raise ValueError('Retired campaign still has surviving tagged volumes')
        time.sleep(3)
    protected = {row['InstanceId']: row['State']['Name'] for row in describe(client, PROTECTED)}
    if protected != json.loads((home/'protected-before.json').read_bytes()):
        raise ValueError('A protected host changed state during this campaign')
    now = datetime.now(timezone.utc)
    duration = max(0., (now-datetime.fromisoformat(allocation['created'])).total_seconds())
    result = {'terminated_instances': sorted(ids), 'remaining_volumes': [], 'security_group_retired': True,
        'protected_states': protected, 'finished': now.isoformat(), 'seconds_since_allocation': duration,
        'compute_cost_upper_estimate_usd': duration/3600*len(ids)*allocation['hourly_rate_usd'],
        'cost_scope': 'Compute upper estimate; record EBS, S3 and transfer separately.'}
    save(home/'resources-finished.json', result)
    return result
