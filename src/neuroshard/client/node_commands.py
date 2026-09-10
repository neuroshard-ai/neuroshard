"""Full-node commands executed only inside the explicitly installed CPU runtime."""
import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path

from . import runtime,wire
from .cli import network,neuro


def main():
    args=json.loads(sys.argv[1]);home=Path(args['home']).expanduser().resolve();config=network(args['network_file'])
    command=args['command'];role=args.get('role','worker');coordinator=args.get('coordinator') or config['api']
    if command in ('join','start'):
        from neuroshard.inference import node
        if not (home/'node.json').exists():
            if command=='start':raise ValueError('No node initialized here. Run neuroshard join.')
            print('Checking the published genesis and downloading verified model/data files.',flush=True)
            cache=runtime.root()/'networks'/config['genesis_sha256'];cache.mkdir(parents=True,exist_ok=True)
            node.download(config['genesis_url'],cache/'genesis.json',config['genesis_sha256'],1024**2)
            genesis=json.loads((cache/'genesis.json').read_text());spec=genesis['app_state']['manifest']
            if genesis['chain_id']!=config['chain_id']:raise ValueError('Genesis chain identity mismatch')
            model=runtime.root()/'models'/spec['model']['revision']
            node.assets(spec,model,config['model_mirrors'])
            node.download(config['dataset_url'],cache/'dataset.json',spec['dataset_sha256'],8*1024**2)
            height,block_hash=args.get('trusted_height',0),args.get('trusted_hash')
            if not height and not block_hash:
                observed=wire.http(config['api']+'/api/network')
                if observed['chain_id']!=config['chain_id'] or observed['genesis_sha256']!=config['genesis_sha256']:
                    raise ValueError('Bootstrap checkpoint identity mismatch')
                height,block_hash=observed['latest_block_height'],observed['latest_block_hash']
                print(f'Using the project checkpoint at block {height}. Advanced users can supply an independently trusted height and hash.',flush=True)
            node.initialize(home,cache/'genesis.json',config['genesis_sha256'],config['peers'],runtime.engine(),model,
                cache/'dataset.json',args['base_port'],bool(config.get('private',False)),height,block_hash)
            print('Node initialized. Your key stays in '+str(home/'account.key'),flush=True)
        current=json.loads((home/'node.json').read_text())
        if current['genesis_sha256']!=config['genesis_sha256']:
            raise ValueError('This home belongs to a different genesis. Use a separate --home; no balance migration is defined.')
        lock=(home/'node.lock').open('a')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('This node is already running. Use neuroshard status or neuroshard work in another terminal.') from None
        children=[];stopping=False
        def stop(*_):
            nonlocal stopping
            stopping=True
        signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
        logs=home/'logs';logs.mkdir(exist_ok=True)
        def launch(label,argv):
            with (logs/(label+'.log')).open('ab') as out:
                child=subprocess.Popen([sys.executable,'-m',*argv],stdout=out,stderr=out,start_new_session=True)
            children.append(child)
        try:
            # Resume with the release's reviewed dependency graph even when this
            # home was initialized with an older build of the same Comet version.
            executable=str(runtime.engine())
            if current['engine']!=executable:
                current['engine']=executable
                temporary=home/'node.json.tmp'
                with temporary.open('w') as out:
                    json.dump(current,out);out.flush();os.fsync(out.fileno())
                os.replace(temporary,home/'node.json')
            launch('launcher',['neuroshard.inference.node','run','--home',str(home)])
            rpc=f'http://127.0.0.1:{current["base_port"]+1}';last_print=0;started=False
            print('Following the native ledger. Ctrl+C stops this node and its worker; keys and history remain.',flush=True)
            while not stopping:
                if any(child.poll() is not None for child in children):raise ValueError('A node component stopped. Inspect '+str(logs))
                try:
                    status=wire.query(rpc,'/summary');native=wire.rpc(rpc,'status')
                    ready=not native['sync_info']['catching_up'] and status['height']>0
                    checkpoint=current.get('trusted_checkpoint')
                    if ready and checkpoint:
                        observed=wire.rpc(rpc,'block',{'height':str(checkpoint['height'])})
                        if observed['block_id']['hash']!=checkpoint['hash']:
                            raise RuntimeError('Trusted checkpoint mismatch; refusing to contribute')
                    if ready and not started:
                        if role=='worker':
                            launch('worker',['neuroshard.publicnet.worker_cli','worker','--home',str(home),'--stage',str(args['stage']),'--coordinator',coordinator])
                        elif role=='provider':launch('provider',['neuroshard.inference.provider','--home',str(home)])
                        started=True
                    if time.monotonic()-last_print>=15:
                        account=wire.query(rpc,'/account',{'public_key':current['public_key']})
                        print(f'Block {status["height"]} · training round {status["round"]} · {neuro(account["balance"])} NEURO · '+
                              (f'{role} active' if ready else 'syncing'),flush=True);last_print=time.monotonic()
                except (OSError,ValueError,KeyError):
                    if time.monotonic()-last_print>=15:
                        print('Verifying the execution profile and waiting for peers…',flush=True);last_print=time.monotonic()
                time.sleep(1)
        finally:
            for child in reversed(children):
                if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
            for child in children:
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            lock.close()
    else:
        if not (home/'node.json').exists():raise ValueError('Run neuroshard join first, or select an initialized --home')
        current=json.loads((home/'node.json').read_text())
        if current['genesis_sha256']!=config['genesis_sha256']:raise ValueError('Node belongs to a different network')
        if command=='serve':
            from neuroshard.inference.provider import main as provider_main
            sys.argv=['neuroshard-serve','--home',str(home)];provider_main()
        else:
            from neuroshard.publicnet.worker_cli import main as work_main
            sys.argv=['neuroshard-work','worker','--home',str(home),'--stage',str(args['stage']),'--coordinator',coordinator]
            work_main()


if __name__=='__main__':
    try:main()
    except (OSError,ValueError,KeyError) as exc:sys.exit(f'NeuroShard: {exc}')
