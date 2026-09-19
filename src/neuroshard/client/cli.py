"""Join NeuroShard, keep your keys locally, and pay for model inference."""
import argparse,json,os,platform,shutil,sys,time
from decimal import Decimal,InvalidOperation
from pathlib import Path

from neuroshard import __version__
from . import runtime,wire


def network(path=None):
    path=Path(path) if path else Path(__file__).parent/'networks/llm-testnet.json'
    if not path.is_file():raise ValueError('No published network descriptor is installed. Use --network-file for a reviewed test network.')
    value=json.loads(path.read_text())
    required={'chain_id','genesis_sha256','genesis_url','rpc','api','peers','dataset_url','model_mirrors'}
    if not required<=set(value):raise ValueError('Incomplete network descriptor')
    wire.endpoint(value['rpc']);wire.endpoint(value['api'])
    return value


def atoms(value):
    try:
        amount=Decimal(value)*1000000
        if not amount.is_finite() or amount!=amount.to_integral_value() or not 0<amount<=2**53-1:raise ValueError()
        return int(amount)
    except (InvalidOperation,ValueError):raise ValueError('Use a positive NEURO amount with at most six decimal places') from None


def neuro(value):
    text=format(Decimal(value)/1000000,'f')
    return text.rstrip('0').rstrip('.') if '.' in text else text


def connected(args):
    config=network(args.network_file);rpc=wire.endpoint(args.rpc or config['rpc'])
    status=wire.query(rpc,'/summary')
    if status['chain_id']!=config['chain_id']:raise ValueError('Endpoint belongs to a different chain')
    return config,rpc,status


def wallet_action(args):
    path=args.home/'account.key'
    if args.action=='import':
        if path.exists():raise ValueError('A key already exists here; choose a separate --home to import another identity')
        value=json.loads(args.file.read_text())
        if value.get('format')!='neuroshard-key-v1':raise ValueError('Unsupported key-file format')
        seed=value.get('seed','')
        if len(seed)!=64 or any(c not in '0123456789abcdef' for c in seed):raise ValueError('Invalid native seed')
        from neuroshard.core.crypto.ecdsa import derive_keypair_from_token
        public=derive_keypair_from_token(seed).public_key_bytes.hex()
        if value.get('public_key')!=public:raise ValueError('Key file public identity mismatch')
        path.parent.mkdir(parents=True,exist_ok=True)
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:f.write(seed);f.flush();os.fsync(f.fileno())
    wallet=wire.Wallet(path,create=args.action=='create')
    if args.action=='export':
        args.file.parent.mkdir(parents=True,exist_ok=True)
        fd=os.open(args.file,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:
            json.dump({'format':'neuroshard-key-v1','seed':path.read_text().strip(),'public_key':wallet.public_key},f)
            f.write('\n');f.flush();os.fsync(f.fileno())
        print(f'Key backup saved to {args.file}. Anyone with this file can spend its NEURO.')
    elif args.action=='balance':
        _,rpc,_=connected(args);value=wire.query(rpc,'/account',{'public_key':wallet.public_key})
        if args.json:print(json.dumps(value))
        else:print(f'{neuro(value["balance"])} NEURO available · {neuro(value.get("locked",0))} locked\nPublic key: {wallet.public_key}')
    else:
        print(f'Public key: {wallet.public_key}\nKey file: {path}\nNo registration is required. Keep a private backup of this key.')


def chat(args):
    config,rpc,status=connected(args);wallet=wire.Wallet(args.home/'account.key')
    info=wire.http(config['api']+'/api/inference')
    if info['chain_id']!=config['chain_id'] or info['genesis_sha256']!=config['genesis_sha256']:
        raise ValueError('Inference service identity differs from the installed network')
    provider=args.provider or info.get('provider')
    if not provider:raise ValueError('No default provider is advertised. Select a provider with --provider.')
    if not args.provider and not info.get('provider_online',False):raise ValueError('The default provider is currently unavailable. Try again later or select --provider.')
    if not 1<=args.max_tokens<=info['max_tokens']:raise ValueError(f'Choose 1–{info["max_tokens"]} generated tokens')
    price=args.max_tokens*int(info['price_per_max_token_atoms']);fee=int(info['fee_atoms'])
    if price+fee>atoms(args.max_price):raise ValueError('The quoted total exceeds --max-price; nothing was submitted')
    account=wire.query(rpc,'/account',{'public_key':wallet.public_key})
    if account['balance']<price+fee:
        raise ValueError(f'Need {neuro(price+fee)} NEURO; available {neuro(account["balance"])}. Contribute with neuroshard join, or receive a native transfer.')
    signed=wallet.sign({'kind':'infer','chain_id':config['chain_id'],'nonce':account['nonce'],'provider':provider,
        'model_root':status['serving_root'],'request':{'prompt':args.prompt,'max_tokens':args.max_tokens},
        'price':price,'expires':status['height']+min(120,int(info['request_lifetime_blocks']))})
    request_id=wire.transaction_id(signed)
    if not args.json:print(f'{info["model_name"]}\nTotal: {neuro(price+fee)} NEURO ({neuro(fee)} fee). Prompts and responses are public.\nRequest: {request_id}',flush=True)
    # Save the signed request before submission so uncertain network outcomes are recoverable.
    directory=runtime.root()/'requests';directory.mkdir(parents=True,exist_ok=True)
    (directory/(request_id+'.json')).write_bytes(wire.canonical({'chain_id':config['chain_id'],'request':signed}))
    try:wire.broadcast(rpc,signed)
    except (OSError,ValueError) as exc:
        raise ValueError(f'Submission outcome requires checking: {exc}. Run neuroshard request {request_id} before sending another request.') from exc
    deadline=time.monotonic()+args.wait_seconds
    while time.monotonic()<deadline:
        try:value=wire.query(rpc,'/job',{'id':request_id})
        except (OSError,ValueError):time.sleep(1);continue
        if value['status']=='completed':
            if args.json:print(json.dumps(value))
            else:print(f'\n{value["output"]["text"]}\n\nSettled in block {value["height"]}. Provider paid {neuro(value["provider_paid"])} NEURO.')
            return
        if value['status']=='expired':raise ValueError(f'Provider timed out. {neuro(value["refunded"])} NEURO unlocked; the submission fee was spent.')
        time.sleep(1)
    print(json.dumps({'request_id':request_id,'status':'pending','next':f'neuroshard request {request_id}'}) if args.json
          else f'Still pending. Resume with: neuroshard request {request_id}')


def main(argv=None):
    parser=argparse.ArgumentParser(description='NeuroShard: contribute computation and use native NEURO for inference.',
        epilog='Experimental testnet. Keys stay on your machine. No website registration. Full nodes use Linux x86_64 CPUs.')
    parser.add_argument('--version',action='version',version=f'NeuroShard {__version__}')
    common=argparse.ArgumentParser(add_help=False)
    common.add_argument('--home',type=lambda p:Path(p).expanduser(),default=runtime.root()/'llm-testnet',help='Local key and node directory')
    common.add_argument('--network-file',type=Path,help='Reviewed network descriptor; defaults to the bundled public testnet')
    common.add_argument('--rpc',help='Use your own native RPC endpoint')
    common.add_argument('--json',action='store_true',help='Machine-readable results')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor',parents=[common],help='Check your machine, installation, and connection')
    sub.add_parser('setup',parents=[common],help='Install the pinned CPU runtime in a managed environment')
    for name,help_text in [('join','Set up a node, follow the ledger, and contribute'),('start','Start an initialized node'),
                           ('work','Contribute using an already running local node'),('serve','Serve paid requests using your local node')]:
        cmd=sub.add_parser(name,parents=[common],help=help_text)
        if name in ('join','start'):
            cmd.add_argument('--role',choices=('worker','observer','provider'),default='worker')
            cmd.add_argument('--base-port',type=int,default=26656)
            cmd.add_argument('--trusted-height',type=int,default=0);cmd.add_argument('--trusted-hash')
        cmd.add_argument('--stage',type=int,choices=(0,1),default=1,help='Training stage: 0 frozen features, 1 adapter update')
        cmd.add_argument('--coordinator',help='Sponsor URL; defaults to the project sponsor')
    sub.add_parser('status',parents=[common],help='Read chain progress and local account balance')
    wallet=sub.add_parser('wallet',help='Create or back up a local native key')
    actions=wallet.add_subparsers(dest='action',required=True)
    for name in ('create','show','balance','export','import'):
        cmd=actions.add_parser(name,parents=[common])
        if name in ('export','import'):cmd.add_argument('file',type=Path)
    ask=sub.add_parser('chat',parents=[common],help='Pay a bounded NEURO amount for a public model response')
    ask.add_argument('prompt',nargs='?');ask.add_argument('--max-tokens',type=int,default=32)
    ask.add_argument('--hosted-config',type=Path,help='Opt-in provider-network config with pinned local validating node')
    ask.add_argument('--resume',help='Resume a durable hosted request without signing another payment')
    ask.add_argument('--session',type=Path,help='Versioned multi-turn hosted conversation file')
    ask.add_argument('--quote-only',action='store_true',help='Show the complete hosted price without payment')
    ask.add_argument('--max-price',default='0.1',help='Maximum total NEURO, including provider, audit and transaction fees for hosted chat')
    ask.add_argument('--provider',help='Native public key of your chosen provider')
    ask.add_argument('--wait-seconds',type=int,default=120)
    request=sub.add_parser('request',parents=[common],help='Inspect a pending, completed, or expired inference request')
    request.add_argument('id')
    transfer=sub.add_parser('transfer',parents=[common],help='Sign a native NEURO transfer')
    transfer.add_argument('--to',required=True);transfer.add_argument('--amount',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='doctor':
            value={'client_version':__version__,'python':sys.version.split()[0],'platform':f'{platform.system()} {platform.machine()}',
                'full_node_supported':runtime.supported(),'cpu_runtime_installed':runtime.ready(),'home':str(args.home),
                'free_disk_gib':round(shutil.disk_usage(Path.home()).free/1024**3,1)}
            try:
                _,_,s=connected(args);value.update(chain_id=s['chain_id'],height=s['height'],training_round=s['round'])
            except (ValueError,OSError) as exc:value['network_status']=str(exc)
            if args.json:print(json.dumps(value))
            else:
                print(f'NeuroShard {__version__} · Python {value["python"]} · {value["platform"]}')
                print('Full-node CPU profile supported.' if value['full_node_supported'] else 'Wallet/chat supported; full nodes require Linux x86_64 and Python 3.10–3.12.')
                print(f'Free disk: {value["free_disk_gib"]} GiB. A full node downloads a CPU runtime and model; 8 GiB RAM and 5 GiB free disk are recommended.')
                print(f'Connected to {value["chain_id"]} at block {value["height"]}.' if 'chain_id' in value else f'Network: {value["network_status"]}')
                print('Next: neuroshard join — creates local keys, verifies the ledger, and contributes stage 1.' if value['full_node_supported']
                      else 'Next: neuroshard wallet create — creates a local account for transfers and remote chat.')
        elif args.command=='setup':runtime.setup(os.environ.get('NEUROSHARD_PACKAGE_SOURCE'))
        elif args.command in ('join','start','work','serve'):
            # Pass a single JSON argument, not a shell command; all child processes use argument arrays.
            value={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
            if not value['network_file']:value['network_file']=str(Path(__file__).parent/'networks/llm-testnet.json')
            runtime.execute('neuroshard.client.node_commands',[json.dumps(value)])
        elif args.command=='wallet':wallet_action(args)
        elif args.command=='chat':
            if not 1<=args.wait_seconds<=600:raise ValueError('wait-seconds must be 1–600')
            if args.hosted_config:
                from .hosted import run
                run(args, atoms(args.max_price))
            else:
                if not args.prompt or args.resume or args.session or args.quote_only:
                    raise ValueError('Public 0.4.0 chat requires a prompt; hosted sessions require --hosted-config')
                chat(args)
        elif args.command=='request':
            _,rpc,_=connected(args);print(json.dumps(wire.query(rpc,'/job',{'id':args.id}),indent=2))
        else:
            config,rpc,status=connected(args)
            if args.command=='status' and not (args.home/'account.key').exists():
                if args.json:print(json.dumps({'network':status,'account':None}))
                else:print(f'{config["chain_id"]} · block {status["height"]} · training round {status["round"]}\nNo local key yet. Run neuroshard join or neuroshard wallet create.')
                return
            wallet=wire.Wallet(args.home/'account.key')
            account=wire.query(rpc,'/account',{'public_key':wallet.public_key})
            if args.command=='status':
                if args.json:print(json.dumps({'network':status,'account':account}))
                else:print(f'{config["chain_id"]} · block {status["height"]} · training round {status["round"]}\n{neuro(account["balance"])} NEURO available · {neuro(account.get("locked",0))} locked\nPublic key: {wallet.public_key}')
            elif args.command=='transfer':
                from cryptography.hazmat.primitives.asymmetric import ec
                ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(),bytes.fromhex(args.to))
                amount=atoms(args.amount)
                if account['balance']<amount+status['params']['fee']:raise ValueError('Insufficient available NEURO')
                signed=wallet.sign({'kind':'transfer','chain_id':config['chain_id'],'nonce':account['nonce'],'to':args.to,'amount':amount})
                value=wire.broadcast(rpc,signed)
                print(json.dumps(value) if args.json else f'Submitted transfer: {value["hash"]}. Check the ledger for final acceptance.')
    except (OSError,ValueError,KeyError) as exc:
        parser.exit(1,f'NeuroShard: {exc}\n')
    except KeyboardInterrupt:parser.exit(130,'\nStopped. Your key and node history remain on disk.\n')


if __name__=='__main__':main()
