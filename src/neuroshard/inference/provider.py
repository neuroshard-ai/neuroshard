"""Serve native paid requests, signing only outputs computed for a local finalized job."""
import argparse,json,threading,time,os
from pathlib import Path
from neuroshard.demo import client as wire,protocol,work
from .engine import Engine
from .settlement import submit


def run(home,stop=None,max_jobs=0):
    home=Path(home);config=json.loads((home/'node.json').read_text());identity=protocol.Identity.load_or_create(home/'account.key')
    spec=json.loads((home/'config/genesis.json').read_text())['app_state']['manifest']
    engine=Engine(config['model_dir'],spec['model']);rpc=f'http://127.0.0.1:{config["base_port"]+1}'
    stop=stop or threading.Event();completed=0;last_error=None;cache={}
    print(json.dumps({'provider':identity.public_key,'chain_id':config['chain_id'],'status':'waiting for funded requests'}),flush=True)
    while not stop.is_set():
        try:
            summary=wire.query(rpc,'/summary')
            if summary['chain_id']!=config['chain_id']:raise ValueError('Local chain identity changed')
            jobs=wire.query(rpc,'/jobs',{'provider':identity.public_key})['jobs']
            heartbeat=home/'provider-status.json'
            temporary=heartbeat.with_suffix('.tmp')
            temporary.write_text(json.dumps({'public_key':identity.public_key,'chain_id':config['chain_id'],
                'checked_at':time.time(),'height':summary['height']}))
            os.replace(temporary,heartbeat)
            cache={k:v for k,v in cache.items() if any(j['id']==k for j in jobs)}
            for item in jobs:
                if stop.is_set():break
                job=wire.query(rpc,'/job',{'id':item['id']})
                if job['status']!='pending' or job['provider']!=identity.public_key:continue
                if work.digest(job['weights'])!=job['model_root']:raise ValueError('Request checkpoint hash mismatch')
                if item['id'] not in cache:cache[item['id']]=engine.infer(job['weights'],job['request'])
                nonce=wire.query(rpc,'/account',{'public_key':identity.public_key})['nonce']
                signed=identity.sign({'kind':'respond','chain_id':config['chain_id'],'nonce':nonce,
                    'job_id':item['id'],'result_root':work.digest(cache[item['id']])})
                result=submit(rpc,signed)
                print(json.dumps({'request_id':item['id'],'settled_height':result['height']}),flush=True)
                completed+=1
                if max_jobs and completed>=max_jobs:return
            last_error=None
        except (OSError,ValueError,KeyError,TypeError) as exc:
            if str(exc)!=last_error:print(json.dumps({'provider_status':str(exc)}),flush=True)
            last_error=str(exc)
        stop.wait(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--home',type=Path,required=True)
    parser.add_argument('--max-jobs',type=int,default=0);args=parser.parse_args()
    import fcntl
    with (args.home/'provider.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(args.home,max_jobs=args.max_jobs)


if __name__=='__main__':main()
