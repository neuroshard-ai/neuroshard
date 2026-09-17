"""Bounded SSH execution for the frozen seven-host ordinary LLM campaign.

This module never provisions machines itself. Its only targets are the exact
unprotected instance inventory recorded by the campaign allocator. Model bytes
remain on their partition owners or stream between them without a local copy.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time
import uuid

from neuroshard.dataflow.store import canonical
from neuroshard.evolution.reference_data import identity, save

REMOTE = '/home/ubuntu/native-expert-live'
REPO = '/home/ubuntu/neuroshard-study'
PYTHON = REPO+'/.neuroshard/venv/bin/python'
PUBLIC = 'https://dwquwt9gkkeil.cloudfront.net/research/native-expert-live-20260916/objects/'
PROTECTED = {'i-0d681a8ef83f72619', 'i-06bf7f1f01e6228bb', 'i-0ebe86ca07e97cf29'}


class Cloud:
    def __init__(self, home):
        self.home = Path(home).resolve()
        allocation = json.loads((self.home/'allocation.json').read_bytes())
        self.hosts = sorted(allocation['instances'], key=lambda row: row['rank'])
        self.deadline = datetime.fromisoformat(allocation['deadline'])
        if (len(self.hosts) != 7 or [row['rank'] for row in self.hosts] != list(range(7))
                or any(row['InstanceId'] in PROTECTED for row in self.hosts)
                or len({row['InstanceId'] for row in self.hosts}) != 7):
            raise ValueError('Use exactly the seven declared disposable owners')
        self.environment = {'PYTHONPATH': REPO+'/src', 'ATEN_CPU_CAPABILITY': 'default',
            'MKL_ENABLE_INSTRUCTIONS': 'SSE4_2', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
            'MKL_NUM_THREADS': '2', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}

    def remaining(self, margin=120):
        seconds = (self.deadline-datetime.now(timezone.utc)).total_seconds()-margin
        if seconds <= 0:
            raise TimeoutError('The campaign allocation deadline has arrived')
        return int(seconds)

    def ssh(self, physical):
        if type(physical) is not int or physical not in range(7):
            raise ValueError('Unknown physical owner')
        return ['ssh', '-i', '/home/ubuntu/.ssh/id_ed25519', '-o', 'BatchMode=yes',
            '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=accept-new',
            '-o', 'UserKnownHostsFile='+str(self.home/'known_hosts'),
            'ubuntu@'+self.hosts[physical]['PrivateIpAddress']]

    def command(self, physical, argv, *, input=None, timeout=300):
        command = shlex.join(argv) if isinstance(argv, list) else argv
        return subprocess.run([*self.ssh(physical), command], input=input, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=True, timeout=min(timeout, self.remaining()))

    def python(self, physical, argv, *, input=None, timeout=300):
        return self.command(physical, ['env', *[key+'='+value for key, value in self.environment.items()],
            PYTHON, *argv], input=input, timeout=timeout)

    def read(self, physical, path, maximum=64*1024**2):
        script = ('import pathlib,sys; p=pathlib.Path(sys.argv[1]); '
            'n=int(sys.argv[2]); assert p.is_file() and not p.is_symlink() and p.stat().st_size<=n; '
            'sys.stdout.buffer.write(p.read_bytes())')
        return self.command(physical, ['python3', '-c', script, str(path), str(maximum)], timeout=60).stdout

    def put(self, physical, path, value):
        raw = value if isinstance(value, bytes) else canonical(value)
        script = ('import os,pathlib,sys,tempfile; p=pathlib.Path(sys.argv[1]); '
            'p.parent.mkdir(parents=True,exist_ok=True); '
            'f=tempfile.NamedTemporaryFile(dir=p.parent,delete=False); '
            'f.write(sys.stdin.buffer.read()); f.flush(); os.fsync(f.fileno()); f.close(); '
            'os.replace(f.name,p)')
        self.command(physical, ['python3', '-c', script, str(path)], input=raw, timeout=120)

    def bundle(self, physical, files):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w:gz') as archive:
            for name, raw in sorted(files.items()):
                if Path(name).is_absolute() or '..' in Path(name).parts:
                    raise ValueError('Bound metadata archive paths')
                raw = raw if isinstance(raw, bytes) else canonical(raw)
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(raw), 0o600
                archive.addfile(info, io.BytesIO(raw))
        self.command(physical, ['tar', '-xzf', '-', '-C', REMOTE], input=stream.getvalue(), timeout=180)

    def copy_directory(self, source, destination, old, new):
        """Stream activations between owners; the receiving reader checks hashes."""
        for path in (old, new):
            if not Path(path).is_relative_to(REMOTE):
                raise ValueError('Transfer only installed numerical directories')
        self.command(destination, ['mkdir', '-p', new])
        started = time.monotonic()
        producer = subprocess.Popen([*self.ssh(source), shlex.join(['tar', '-cf', '-', '-C', old, '.'])],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            consumer = subprocess.run([*self.ssh(destination), shlex.join(['tar', '-xf', '-', '-C', new])],
                stdin=producer.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=min(900, self.remaining()))
            producer.stdout.close()
            _, error = producer.communicate(timeout=60)
            if producer.returncode or consumer.returncode:
                raise OSError('Numerical owner transfer did not finish')
        finally:
            if producer.poll() is None:
                producer.kill()
                producer.wait(timeout=10)
        size = int(self.command(destination, ['du', '-sb', new]).stdout.split()[0])
        return {'from': source, 'to': destination, 'bytes': size, 'seconds': time.monotonic()-started}

    def start(self, physical, unit, argv, *, rank=None, world=None, port=None):
        environment = dict(self.environment)
        facts = json.loads(self.command(physical, ['python3', '-c',
            'import os,json; print(json.dumps({"loader":os.environ.get("LD_LIBRARY_PATH","")}))']).stdout)
        environment['LD_LIBRARY_PATH'] = facts['loader']
        if rank is not None:
            routes = json.loads(self.command(physical, ['ip', '-j', 'route', 'show', 'default']).stdout)
            environment.update(RANK=str(rank), WORLD_SIZE=str(world), MASTER_PORT=str(port),
                MASTER_ADDR=self.hosts[0]['PrivateIpAddress'], GLOO_SOCKET_IFNAME=routes[0]['dev'])
        command = ['sudo', 'systemd-run', '--unit='+unit, '--property=User=ubuntu',
            '--property=WorkingDirectory='+REPO, '--property=RuntimeMaxSec='+str(self.remaining()),
            '--property=StandardOutput=append:'+REMOTE+'/'+unit+'.log',
            '--property=StandardError=append:'+REMOTE+'/'+unit+'.log', 'env',
            *[key+'='+value for key, value in environment.items()], PYTHON, *argv]
        self.command(physical, command, timeout=45)

    def active(self, physical, unit):
        return self.command(physical, ['systemctl', 'show', '-p', 'ActiveState', '--value', unit]).stdout.strip() == b'active'

    def execute(self, physical, config):
        key = identity(config)
        invocation = uuid.uuid4().hex
        folder = REMOTE+'/operations/'+key+'/'+invocation
        config = {**config, 'result': folder+'/result.json'}
        self.put(physical, folder+'/config.json', config)
        unit = 'neuroshard-work-'+key[:20]+'-'+invocation[:8]
        self.start(physical, unit, [REPO+'/scripts/run_owned_expert_work.py', '--config', folder+'/config.json'])
        while self.remaining() > 0:
            try:
                value = json.loads(self.read(physical, config['result']))
                if value['operation'] != identity(config):
                    raise ValueError('An owner returned another numerical operation')
                return value
            except subprocess.CalledProcessError:
                if not self.active(physical, unit):
                    raise OSError('Numerical owner stopped before writing its result: '+unit) from None
                time.sleep(2)

    def assets(self, physical, request):
        result = self.python(physical, [REPO+'/scripts/run_owned_expert_assets.py'],
                              input=canonical(request), timeout=1800)
        return json.loads(result.stdout)

    @staticmethod
    def placement(graph):
        return [0, 1, 2, *[3+(index % 4) for index in range(len(graph['experts']))]]

    def service(self, key, graph, baseline, profile, quality, inputs, policy_store, *, slot):
        """Load a whole answering system across its physical partition owners."""
        placement = self.placement(graph)
        folder = 'services/'+key
        # Slots are assigned by installed roles, never derived from a claim.
        if type(slot) is not int or not 0 <= slot <= 15:
            raise ValueError('Choose a configured nonoverlapping process-group slot')
        units = ['neuroshard-service-'+key[:18]+'-'+str(rank) for rank in range(len(placement))]
        paths = {'graph': folder+'/graph.json', 'baseline': folder+'/baseline.json',
                 'profile': folder+'/profile.json', 'quality_policy': folder+'/quality-policy.json'}
        files = {paths['graph']: graph, paths['baseline']: baseline, paths['profile']: profile,
                 paths['quality_policy']: quality}
        files.update({folder+'/inputs/'+name: raw for name, raw in inputs.items()})
        for value in (graph, baseline):
            key_policy = value['answering']['policy_root']
            files['objects/policies/'+key_policy[:2]+'/'+key_policy] = policy_store.get(key_policy)
        with ThreadPoolExecutor(max_workers=7) as pool:
            list(pool.map(lambda physical: self.bundle(physical, files), sorted(set(placement))))
        configurations = []
        for rank, physical in enumerate(placement):
            config = {name: REMOTE+'/'+path for name, path in paths.items()}
            config.update(objects=REMOTE+'/objects', interpreter=REMOTE+'/interpreter', seed=REMOTE+'/seed',
                source_home=REPO, inputs=REMOTE+'/'+folder+'/inputs',
                home=REMOTE+'/'+folder+'/owner-'+str(rank), max_seconds=min(21600, self.remaining()))
            path = REMOTE+'/'+folder+'/config-'+str(rank)+'.json'
            self.put(physical, path, config)
            configurations.append(path)
        with ThreadPoolExecutor(max_workers=7) as pool:
            list(pool.map(lambda rank: self.start(placement[rank], units[rank],
                [REPO+'/scripts/run_native_expert_service.py', '--config', configurations[rank]],
                rank=rank, world=len(placement), port=31000+slot), range(len(placement))))
        service = {'key': key, 'graph': identity(graph), 'placement': placement, 'units': units,
                   'folder': REMOTE+'/'+folder, 'slot': slot}
        deadline = time.monotonic()+min(600, self.remaining())
        while time.monotonic() < deadline:
            ready = []
            for rank, physical in enumerate(placement):
                try:
                    ready.append(json.loads(self.read(physical, service['folder']+'/owner-'+str(rank)+'/ready.json')))
                except subprocess.CalledProcessError:
                    if not self.active(physical, units[rank]):
                        raise OSError('A serving owner stopped during initialization: '+units[rank]) from None
            if len(ready) == len(placement):
                if any(row['graph'] != identity(graph) or row['executor'] != identity(profile) for row in ready):
                    raise ValueError('Loaded serving owners disagree on the prescribed graph')
                service['ready'] = ready
                return service
            time.sleep(2)
        raise TimeoutError('Serving owners did not initialize within their startup bound')

    def query(self, service, request, *, timeout=3600):
        key = request['id']
        self.put(0, service['folder']+'/owner-0/requests/'+key+'.json', request)
        deadline = time.monotonic()+min(timeout, self.remaining())
        while time.monotonic() < deadline:
            values = []
            for rank, physical in enumerate(service['placement']):
                try:
                    values.append(json.loads(self.read(physical,
                        service['folder']+'/owner-'+str(rank)+'/results/'+key+'.json')))
                except subprocess.CalledProcessError:
                    if not self.active(physical, service['units'][rank]):
                        raise OSError('Serving owner stopped during the complete request') from None
            if len(values) == len(service['placement']):
                stable = lambda value: {key: item for key, item in value.items() if key != 'seconds'}
                if any(stable(value) != stable(values[0]) for value in values):
                    raise ValueError('Complete serving owner responses disagree')
                return values[0]
            time.sleep(2)
        raise TimeoutError('The complete serving request exceeded its bound')

    def stop(self, service):
        for rank, physical in enumerate(service['placement']):
            self.command(physical, ['sudo', 'systemctl', 'stop', service['units'][rank]], timeout=45)
