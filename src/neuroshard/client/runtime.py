"""Explicit managed CPU runtime installation; never modifies another environment."""
import hashlib,json,os,platform,shutil,subprocess,sys,tarfile,tempfile,venv
from pathlib import Path
from urllib.request import urlopen

from neuroshard import __version__

GO_VERSION='1.27.1'
GO_SHA256='63d339f0da5ab53635a56f2490a7984dfe12dfcff22ad749f63edaf590168445'
CPU_REQUIREMENTS=['torch==2.9.1+cpu']
RUNTIME_REQUIREMENTS=['numpy==2.2.6','grpcio==1.83.1','protobuf==6.33.6','cryptography==50.0.1','requests==2.34.2',
    'transformers==4.57.3','tokenizers==0.22.1','safetensors==0.7.0','huggingface-hub==0.36.0','urllib3==2.7.0','idna==3.19','filelock==3.32.6']


def root():return Path(os.environ.get('NEUROSHARD_STATE_DIR',str(Path.home()/'.neuroshard'))).expanduser()


def location():return root()/'runtimes'/__version__


def supported():return platform.system()=='Linux' and platform.machine() in ('x86_64','AMD64') and (3,10)<=sys.version_info[:2]<(3,13)


def ready():
    path=location();marker=path/'installed.json'
    if not marker.is_file() or not (path/'bin/python').is_file():return False
    try:
        expected=dict(item.split('==',1) for item in CPU_REQUIREMENTS+RUNTIME_REQUIREMENTS)
        expected['neuroshard-ai']=__version__
        saved=json.loads(marker.read_text())
        if saved.get('version')!=__version__ or saved.get('libraries')!=expected:return False
        actual=json.loads(subprocess.check_output([str(path/'bin/python'),'-c',
            'from importlib.metadata import version; import json; print(json.dumps({p:version(p) for p in '+repr(list(expected))+'}))'],
            text=True,stderr=subprocess.DEVNULL,timeout=15))
        return actual==expected
    except (ValueError,OSError,subprocess.SubprocessError):return False


def setup(package=None):
    if not supported():raise ValueError('Full nodes currently require Linux x86_64 and Python 3.10–3.12. Wallet and chat do not need the CPU runtime.')
    import fcntl
    path=location();path.parent.mkdir(parents=True,exist_ok=True)
    with (path.parent/(path.name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if ready():return path
        print(f'Installing the pinned CPU runtime in {path}. This downloads several hundred MB.',flush=True)
        venv.EnvBuilder(with_pip=True).create(path)
        python=str(path/'bin/python');log=path/'install.log'
        commands=[[python,'-m','pip','install','--disable-pip-version-check','--upgrade','pip>=26.2','setuptools>=83'],
                  [python,'-m','pip','install','--disable-pip-version-check',*CPU_REQUIREMENTS,'--index-url','https://download.pytorch.org/whl/cpu'],
                  [python,'-m','pip','install','--disable-pip-version-check',*RUNTIME_REQUIREMENTS],
                  [python,'-m','pip','install','--disable-pip-version-check','--no-deps',package or f'neuroshard-ai=={__version__}']]
        with log.open('a') as out:
            for command in commands:
                if subprocess.run(command,stdout=out,stderr=out).returncode:
                    raise ValueError(f'Runtime installation failed. Inspect {log}, then rerun neuroshard setup.')
        check=[python,'-c','from importlib.metadata import version; import json; print(json.dumps({p:version(p) for p in '+
               repr(['neuroshard-ai','torch','numpy','grpcio','protobuf','cryptography','requests','transformers','tokenizers','safetensors','huggingface-hub','urllib3','idna','filelock'])+'}))']
        actual=json.loads(subprocess.check_output(check,text=True))
        expected=dict(item.split('==',1) for item in CPU_REQUIREMENTS+RUNTIME_REQUIREMENTS)
        expected['neuroshard-ai']=__version__
        if actual!=expected:raise ValueError('Installed runtime does not match the pinned dependency versions')
        (path/'installed.json').write_text(json.dumps({'version':__version__,'python':sys.version.split()[0],'libraries':actual})+'\n')
        print('CPU runtime ready.',flush=True)
        return path


def engine():
    if not supported():raise ValueError('Consensus runtime requires Linux x86_64 and Python 3.10–3.12')
    import fcntl
    tools=root()/'tools';tools.mkdir(parents=True,exist_ok=True)
    binary=tools/'cometbft-0.38.26'
    manifest=Path(__file__).with_name('consensus')
    build_id=hashlib.sha256(GO_SHA256.encode()+b''.join((manifest/name).read_bytes()
        for name in ('go.mod','go.sum'))).hexdigest()
    receipt=tools/'consensus-build.json'
    with (tools/'engine.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            saved=json.loads(receipt.read_text())
            if (saved['build_id']==build_id and saved['sha256']==hashlib.sha256(binary.read_bytes()).hexdigest()
                    and subprocess.check_output([str(binary),'version'],text=True).strip()=='0.38.26'):
                return binary
        except (OSError,ValueError,KeyError,subprocess.SubprocessError):pass
        print('Preparing NeuroShard native consensus (CometBFT 0.38.26).',flush=True)
        toolchain=tools/f'go-{GO_VERSION}';go=toolchain/'go/bin/go'
        if not go.exists():
            with tempfile.TemporaryDirectory(dir=tools,prefix='.go-') as temporary:
                archive=Path(temporary)/'go.tar.gz';h=hashlib.sha256();size=0
                with urlopen(f'https://go.dev/dl/go{GO_VERSION}.linux-amd64.tar.gz',timeout=120) as response,archive.open('wb') as out:
                    if not response.geturl().startswith('https://'):raise ValueError('Insecure Go download redirect')
                    while chunk:=response.read(1024**2):
                        size+=len(chunk)
                        if size>150*1024**2:raise ValueError('Toolchain exceeds download limit')
                        h.update(chunk);out.write(chunk)
                if h.hexdigest()!=GO_SHA256:raise ValueError('Go toolchain checksum mismatch')
                toolchain.mkdir(exist_ok=True)
                with tarfile.open(archive,'r:gz') as tar:
                    # The authenticated official archive has no permission to escape this directory.
                    for item in tar.getmembers():
                        if item.name.startswith('/') or '..' in Path(item.name).parts or item.issym() or item.islnk():
                            raise ValueError('Unsafe toolchain archive entry')
                    tar.extractall(toolchain)
        if subprocess.check_output([str(go),'version'],text=True).strip()!=f'go version go{GO_VERSION} linux/amd64':
            raise ValueError('Wrong Go toolchain')
        with tempfile.TemporaryDirectory(dir=tools,prefix='.consensus-') as temporary:
            build=Path(temporary)
            for name in ('go.mod','go.sum'):shutil.copy2(manifest/name,build/name)
            env={**os.environ,'GOBIN':str(build),'GOTOOLCHAIN':'local','GOWORK':'off',
                 'GOFLAGS':'-mod=readonly','GOMAXPROCS':'2'}
            with (tools/'consensus-build.log').open('a') as out:
                result=subprocess.run([str(go),'install','github.com/cometbft/cometbft/cmd/cometbft'],
                    cwd=build,env=env,stdout=out,stderr=out)
            if result.returncode:raise ValueError(f'Consensus build failed. Inspect {tools/"consensus-build.log"}')
            if subprocess.check_output([str(build/'cometbft'),'version'],text=True).strip()!='0.38.26':
                raise ValueError('Wrong consensus executable version')
            digest=hashlib.sha256((build/'cometbft').read_bytes()).hexdigest()
            os.replace(build/'cometbft',binary)
            (build/'receipt.json').write_text(json.dumps({'build_id':build_id,'sha256':digest,
                'go_version':GO_VERSION,'cometbft_version':'0.38.26'})+'\n')
            os.replace(build/'receipt.json',receipt)
        return binary


def execute(module,args):
    path=setup(os.environ.get('NEUROSHARD_PACKAGE_SOURCE'))
    env={**os.environ,'ATEN_CPU_CAPABILITY':'default','MKL_ENABLE_INSTRUCTIONS':'SSE4_2',
         'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
    # A source checkout must never shadow the separately installed, versioned runtime.
    env.pop('PYTHONPATH',None)
    os.execve(str(path/'bin/python'),[str(path/'bin/python'),'-m',module,*map(str,args)],env)
