"""Bootstrap and run a native LLM testnet without changing reference-chain state."""
import argparse,base64,datetime,hashlib,json,os,re,signal,subprocess,sys,time
from pathlib import Path
from urllib.request import urlopen

from neuroshard.demo import network,protocol,work
from neuroshard.lab import state as ledger
from neuroshard.publicnet import bootstrap
from . import profile


def download(url,path,sha,limit=1024**3):
    if not url.startswith('https://'):raise ValueError('Artifacts require HTTPS')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
        if h.hexdigest()==sha:return
    import tempfile
    fd,name=tempfile.mkstemp(prefix='.download-',dir=path.parent)
    try:
        h=hashlib.sha256();size=0
        with os.fdopen(fd,'wb') as out,urlopen(url,timeout=120) as response:
            if not response.geturl().startswith('https://'):raise ValueError('Artifact redirected away from HTTPS')
            while chunk:=response.read(1024**2):
                size+=len(chunk)
                if size>limit:raise ValueError('Artifact exceeds download limit')
                h.update(chunk);out.write(chunk)
            out.flush();os.fsync(out.fileno())
        if h.hexdigest()!=sha:raise ValueError('Downloaded artifact failed SHA-256 check')
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def assets(spec,directory,mirrors=()):
    directory=Path(directory)
    for name,sha in spec['model']['files'].items():
        if Path(name).name!=name:raise ValueError('Invalid model filename')
        origins=[f'{mirror.rstrip("/")}/{sha}' for mirror in mirrors]
        model=spec['model']
        origins.append(f'https://huggingface.co/{model["repo"]}/resolve/{model["revision"]}/{name}')
        last=None
        for url in origins:
            try:download(url,directory/name,sha);break
            except (OSError,ValueError) as exc:last=exc
        else:raise ValueError(f'Cannot retrieve verified model asset {name}') from last
    return directory


def make_genesis(chain_id,declarations,output,model_dir,data_path):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    entries=[]
    for envelope in declarations:
        body,owner=protocol.verify(envelope)
        if set(body)!={'domain','chain_id','owner','consensus_key','bond','liquid','possession'} or (
            body['domain']!='neuroshard/genesis-declaration/v1' or body['chain_id']!=chain_id or body['owner']!=owner):
            raise ValueError('Invalid genesis declaration')
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(body['consensus_key'])).verify(bytes.fromhex(body['possession']),
            ledger.possession_message(chain_id,owner,body['consensus_key'],body['bond'],0))
        entries.append({k:body[k] for k in ('owner','consensus_key','bond','liquid')})
    if len(entries)<4 or len({v['owner'] for v in entries})!=len(entries):raise ValueError('Four distinct declarations required')
    data=json.loads(Path(data_path).read_text());spec=profile.build(model_dir,data)
    genesis={'genesis_time':datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00','Z'),
        'chain_id':chain_id,'initial_height':'1','consensus_params':{
            'block':{'max_bytes':'65536','max_gas':'-1'},
            'evidence':{'max_age_num_blocks':str(spec['params']['evidence_blocks']),
                        'max_age_duration':str(spec['params']['evidence_seconds']*1000000000),'max_bytes':'16384'},
            'validator':{'pub_key_types':['ed25519']},'version':{'app':'0'},'abci':{'vote_extensions_enable_height':'0'}},
        'validators':[{'address':ledger.consensus_address(v['consensus_key']),
            'pub_key':{'type':'tendermint/PubKeyEd25519','value':base64.b64encode(bytes.fromhex(v['consensus_key'])).decode()},
            'power':str(v['bond']//spec['params']['bond_unit']),'name':f'genesis-{i}'} for i,v in enumerate(entries)],
        'app_hash':'','app_state':{'manifest':spec,'validators':entries}}
    # Validate amounts, keys, and conservation before publishing a declaration bundle.
    ledger.genesis(chain_id,entries,spec)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    raw=work.canonical(genesis);(output/'genesis.json').write_bytes(raw)
    (output/'dataset.json').write_bytes(work.canonical(data))
    return {'chain_id':chain_id,'genesis_sha256':hashlib.sha256(raw).hexdigest(),'manifest_hash':work.digest(spec)}


def initialize(home,genesis_source,genesis_sha256,peers,engine,model_dir,data_path,base_port=26656,
               private=False,trusted_height=0,trusted_hash=None):
    home=Path(home).resolve();genesis,raw=bootstrap.load_genesis(str(genesis_source),genesis_sha256)
    spec=genesis['app_state']['manifest']
    if spec.get('version')!='neuroshard-llm-v1':raise ValueError('This is not an LLM testnet genesis')
    if not 1024<=base_port<=65531:raise ValueError('Invalid base port')
    if trusted_height or trusted_hash:
        if trusted_height<=0 or not re.fullmatch(r'[0-9a-fA-F]{64}',trusted_hash or ''):raise ValueError('Invalid trusted checkpoint')
    elif not private:
        age=time.time()-datetime.datetime.fromisoformat(genesis['genesis_time'].replace('Z','+00:00')).timestamp()
        if age>spec['params']['evidence_seconds']:raise ValueError('An older stake history requires a recent trusted checkpoint')
    if any((home/p).exists() for p in ('node.json','candidate.sqlite','data/blockstore.db')):
        raise ValueError('Node already initialized; start it without replacing its history')
    profile.check(spec,model_dir,json.loads(Path(data_path).read_text()))
    engine=bootstrap.engine_path(engine)
    if (home/'account.key').exists() and not (home/'config/priv_validator_key.json').exists():
        if {p.name for p in home.iterdir()}!={'account.key'}:raise ValueError('Initialize in a new node directory')
        subprocess.run([engine,'init','--home',str(home)],check=True,capture_output=True)
    identity=bootstrap.create_keys(home,engine)
    own_id=subprocess.check_output([engine,'show-node-id','--home',str(home)],text=True).strip()
    peers=[bootstrap.peer(p,private) for p in peers]
    config_path=home/'config/config.toml';text=config_path.read_text()
    values=[('','proxy_app',json.dumps(f'127.0.0.1:{base_port+2}')),('','abci','"grpc"'),('','log_level','"info"'),
        ('rpc','laddr',json.dumps(f'tcp://127.0.0.1:{base_port+1}')),('rpc','unsafe','false'),('rpc','timeout_broadcast_tx_commit','"90s"'),
        ('p2p','laddr',json.dumps(f'tcp://0.0.0.0:{base_port}')),('p2p','persistent_peers',json.dumps(','.join(peers))),
        ('p2p','pex','true'),('p2p','addr_book_strict','false' if private else 'true'),
        ('p2p','allow_duplicate_ip','true' if private else 'false'),('consensus','timeout_commit','"1s"'),
        ('consensus','timeout_propose','"60s"'),('consensus','timeout_prevote','"3s"'),('consensus','timeout_precommit','"3s"')]
    for section,key,value in values:text=network.edit_config(text,section,key,value)
    config_path.write_text(text);(home/'config/genesis.json').write_bytes(raw)
    (home/'dataset.json').write_bytes(Path(data_path).read_bytes())
    config={'home':str(home),'engine':engine,'base_port':base_port,'api_host':'127.0.0.1','peers':peers,'advertise':None,
        'chain_id':genesis['chain_id'],'genesis_sha256':genesis_sha256,'profile':'llm-testnet','public_key':identity.public_key,
        'node_id':own_id,'model_dir':str(Path(model_dir).resolve()),
        'trusted_checkpoint':{'height':trusted_height,'hash':trusted_hash.upper()} if trusted_hash else None}
    (home/'node.json').write_bytes(work.canonical(config));return config


def run(home):
    home=Path(home);config=json.loads((home/'node.json').read_text());logs=home/'logs';logs.mkdir(exist_ok=True)
    commands={'application':[sys.executable,'-m','neuroshard.inference.app','--home',str(home),'--port',str(config['base_port']+2)],
        'consensus':[bootstrap.engine_path(config['engine']),'start','--home',str(home)],
        'gateway':[sys.executable,'-m','neuroshard.inference.gateway','--home',str(home)]}
    children=[];stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for name,command in commands.items():
            with (logs/f'{name}.log').open('ab') as out:
                children.append(subprocess.Popen(command,stdout=out,stderr=out,start_new_session=True))
        print(json.dumps({'chain_id':config['chain_id'],'home':str(home),'status':'starting; verifying model and data'}),flush=True)
        while not stopping:
            if any(child.poll() is not None for child in children):raise RuntimeError(f'A node component stopped; see {logs}')
            checkpoint=config['trusted_checkpoint']
            if checkpoint:
                from neuroshard.demo import client
                try:
                    block=client.rpc(f'http://127.0.0.1:{config["base_port"]+1}','block',{'height':str(checkpoint['height'])})
                    if block['block_id']['hash']!=checkpoint['hash']:raise RuntimeError('Trusted checkpoint mismatch')
                    config['trusted_checkpoint']=None
                except (OSError,ValueError):pass
            time.sleep(0.5)
    finally:
        for child in reversed(children):
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        for child in children:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    gen=sub.add_parser('genesis');gen.add_argument('--chain-id',required=True);gen.add_argument('--declarations',nargs='+',required=True)
    gen.add_argument('--output',required=True);gen.add_argument('--model-dir',required=True);gen.add_argument('--data',required=True)
    init=sub.add_parser('init');init.add_argument('--home',required=True);init.add_argument('--genesis',required=True)
    init.add_argument('--genesis-sha256',required=True);init.add_argument('--peer',action='append',default=[])
    init.add_argument('--engine',required=True);init.add_argument('--model-dir',required=True);init.add_argument('--data',required=True)
    init.add_argument('--base-port',type=int,default=26656);init.add_argument('--private-network',action='store_true')
    start=sub.add_parser('run');start.add_argument('--home',required=True)
    args=parser.parse_args()
    if args.command=='run':run(args.home);return
    if args.command=='genesis':result=make_genesis(args.chain_id,[json.loads(Path(p).read_text()) for p in args.declarations],args.output,args.model_dir,args.data)
    else:result=initialize(args.home,args.genesis,args.genesis_sha256,args.peer,args.engine,args.model_dir,args.data,args.base_port,args.private_network)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
