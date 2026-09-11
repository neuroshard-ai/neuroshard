#!/usr/bin/env python3
"""Run an isolated four-validator fraud/settlement test; never reuse chain keys.

Default: a 10,384-parameter fixture on one physical host. --record and --objects
can instead supply a full-model conformance record. All processes stop on exit;
the fresh home retains logs, genesis and private test keys for investigation.
"""
import argparse
import base64
import copy
import json
import secrets
import subprocess
import sys
import time
from pathlib import Path

from neuroshard.evolution import schema
from neuroshard.evolution.model import torch,configure,grow,place
from neuroshard.evolution.objects import Objects,digest
from neuroshard.evolution.pipeline import Pipeline,LocalEndpoint,validate_record
from neuroshard.evolution.worker import Worker
from neuroshard.evolution.app import code_hash
from neuroshard.evolution.settlement import PARAMS,CHUNK_BYTES
from neuroshard.evolution.verification import bundle
from neuroshard.demo import protocol,client as wire
from neuroshard.demo.network import initialize,edit_config
from neuroshard.lab.app import native_parameters
from neuroshard.dataflow.store import canonical
from native_rpc import broadcast_finalized


def tiny_record(home,store):
    config=dict(hidden_size=16,intermediate_size=32,num_attention_heads=4,num_key_value_heads=2,
                vocab_size=64,rms_norm_eps=1e-5,rope_theta=100000.,max_position_embeddings=8192,num_hidden_layers=4)
    torch.manual_seed(42)
    components={}
    for name in ['embed','norm',*[f'block_{i:03}' for i in range(4)]]:
        values={k:torch.ones(shape) if len(shape)==1 else torch.randn(shape)*.05 for k,shape in schema.shapes(config,name).items()}
        components[name]={'root':store.put_tensors(values),'parameters':sum(t.numel() for t in values.values())}
    model=dict(format='neuroshard-model-v1',config=config,components=components,
               parameters=sum(c['parameters'] for c in components.values()),parent=None,origin={'synthetic_fixture':True})
    root=store.put_json(model)
    pipe=Pipeline(store,root,[LocalEndpoint(Worker(home/f'worker{i}',store)) for i in range(2)],[6000]*2,'native-check')
    try:return pipe.train({'input_ids':[[1,12,13,14,2]],'labels':[[-100,-100,13,14,2]]})
    finally:pipe.close()


def until(check,seconds=90):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        try:
            value=check()
            if value:return value
        except (OSError,ValueError,KeyError):pass
        time.sleep(.2)
    raise TimeoutError('Native condition timed out; inspect the retained logs')


def run(args):
    if bool(args.record)!=bool(args.objects):
        raise ValueError('--record and --objects must be supplied together')
    args.home=args.home.resolve()
    args.home.mkdir(parents=True,exist_ok=False)
    store=Objects(args.objects or args.home/'objects')
    record=json.loads(args.record.read_bytes())['training'] if args.record else tiny_record(args.home,store)
    config=initialize(args.home/'native',args.base_port,str(args.engine.resolve()))
    configure()
    params={**PARAMS,'availability_blocks':1024,'max_claim_blocks':4096}
    native={**native_parameters(params),'block_max_bytes':4*1024*1024}
    manifest={'params':params,'initial_model_root':record['parent'],'data_root':digest(canonical([record['batch']])),
              'training_batches':[record['batch']],'learning_rate':.003,'clip_norm':1.,'native_consensus':native,'code_hash':code_hash()}
    first=Path(config['nodes'][0]['home'])/'config/genesis.json'
    genesis=json.loads(first.read_bytes())
    owners=[protocol.Identity.load_or_create(args.home/f'founder{i}.key') for i in range(4)]
    entries=[]
    for i,node in enumerate(config['nodes']):
        key=json.loads((Path(node['home'])/'config/priv_validator_key.json').read_bytes())['pub_key']['value']
        entries.append({'owner':owners[i].public_key,'consensus_key':base64.b64decode(key).hex(),'bond':2500000,'liquid':1000000000})
        genesis['validators'][i]['power']='10'
    genesis.update(chain_id='neuroshard-evolution-check-'+secrets.token_hex(6),app_state={'manifest':manifest,'validators':entries})
    genesis['consensus_params']['block']['max_bytes']=str(native['block_max_bytes'])
    genesis['consensus_params']['evidence'].update(max_age_num_blocks=str(params['evidence_blocks']),max_age_duration=str(params['evidence_seconds']*1000000000))
    processes=[]
    try:
        for i,node in enumerate(config['nodes']):
            path=Path(node['home'])/'config/config.toml'
            text=edit_config(path.read_text(),'rpc','max_body_bytes','4194304')
            text=edit_config(text,'rpc','timeout_broadcast_tx_commit','"120s"')
            path.write_text(edit_config(text,'mempool','max_tx_bytes','2097152'))
            (path.parent/'genesis.json').write_bytes(canonical(genesis))
            commands=[('app',[sys.executable,'-m','neuroshard.evolution.app','--home',node['home'],'--port',str(node['abci'])]),
                      ('node',[config['engine'],'start','--home',node['home']])]
            for name,command in commands:
                with (args.home/f'{name}{i}.log').open('wb') as log:
                    processes.append(subprocess.Popen(command,stdout=log,stderr=log))
        urls=[f'http://127.0.0.1:{n["rpc"]}' for n in config['nodes']]
        url=urls[0]
        until(lambda:all(wire.query(u)['height']>0 for u in urls))
        def send(owner,kind,**fields):
            account=wire.query(url,'/account',{'public_key':owner.public_key})
            return broadcast_finalized(url,owner.sign({'kind':kind,'chain_id':genesis['chain_id'],'nonce':account['nonce'],**fields}))
        def claim(value):
            status=wire.query(url)
            workers=[owners[i%3] for i in range(len(value['traces']))]
            send(owners[0],'reserve',parent=status['model_root'],round=status['training_round'],workers=[o.public_key for o in workers])
            assignment=wire.query(url)['assignment']['id']
            receipts=[owner.sign({'domain':'neuroshard/evolution/work/v1','chain_id':genesis['chain_id'],
                      'assignment':assignment,'record_root':value['record_root'],'stage':i,'trace_root':value['traces'][i]}) for i,owner in enumerate(workers)]
            send(owners[0],'claim',record_root=value['record_root'],metadata=bundle(store,value['record_root']),
                 workers=receipts,data_root=manifest['data_root'],sequence_index=0)
            return wire.query(url)['candidate']['id']
        stage=len(record['traces'])-1
        forged=copy.deepcopy(store.json(record['record_root']))
        trace=store.json(record['traces'][stage])
        name=trace['partition']['components'][0]
        trace['components'][name]=store.json(record['parent'])['components'][name]
        forged['traces'][stage]=store.put_json(trace)
        model=store.json(record['model_root'])
        model['components'][name]=trace['components'][name]
        forged['model_root']=store.put_json(model)
        forged['record_root']=store.put_json(forged)
        assert validate_record(store,forged['record_root'])['valid']
        claim_id=claim(forged)
        send(owners[3],'challenge',claim_id=claim_id,stage=stage,challenge_kind='fraud',object_root=None)
        needed=wire.query(url)['candidate']['challenge']['needed']
        started=time.monotonic()
        uploaded=transactions=0
        for key in needed:
            raw=store.get(key)
            for i,start in enumerate(range(0,len(raw),CHUNK_BYTES)):
                chunk=raw[start:start+CHUNK_BYTES]
                send(owners[3],'upload',claim_id=claim_id,object_root=key,index=i,data=base64.b64encode(chunk).decode())
                uploaded+=len(chunk);transactions+=1
            send(owners[3],'seal',claim_id=claim_id,object_root=key);transactions+=1
        upload_seconds=time.monotonic()-started
        print(json.dumps({'phase':'training_replay_uploaded','bytes':uploaded,'transactions':transactions,'seconds':upload_seconds}),flush=True)
        started=time.monotonic()
        send(owners[3],'resolve',claim_id=claim_id)
        resolve_seconds=time.monotonic()-started
        fraud=wire.query(url)
        assert fraud['issued']==0 and 'optimizer update' in fraud['settled'][-1]['reason']
        print(json.dumps({'phase':'forged_training_rejected','seconds':resolve_seconds}),flush=True)
        claim(record)
        accepted=until(lambda:s if (s:=wire.query(url))['training_round']==1 else None)
        assert accepted['issued']==1000000 and accepted['model_root']==record['model_root']
        print(json.dumps({'phase':'valid_training_settled','issued_atoms':accepted['issued']}),flush=True)
        # Native growth has its own bonded dispute path and mints no reward.
        parent=accepted['model_root']
        grown,grown_model=grow(parent,store,4)
        bad=copy.deepcopy(grown_model)
        old_depth=store.json(parent)['config']['num_hidden_layers']
        component=copy.deepcopy(bad['components'][f'block_{old_depth:03}'])
        tensors=store.tensors(component['root'])
        tensors['mlp.down_proj.weight'][0,0]=1.
        component['root']=store.put_tensors(tensors)
        for depth in range(old_depth,old_depth+4):bad['components'][f'block_{depth:03}']=component
        bad_root=store.put_json(bad)
        def propose_growth(root,value):
            send(owners[0],'grow',parent=parent,model_root=root,
                 metadata={parent:store.json(parent),root:value},capacities=[48000000]*4)
            return wire.query(url)['candidate']['id']
        bad_id=propose_growth(bad_root,bad)
        send(owners[3],'challenge',claim_id=bad_id,stage=0,challenge_kind='fraud',object_root=None)
        growth_inputs=wire.query(url)['candidate']['challenge']['needed']
        assert len(growth_inputs)==1
        growth_bytes=0
        for key in growth_inputs:
            raw=store.get(key)
            for i,start in enumerate(range(0,len(raw),CHUNK_BYTES)):
                chunk=raw[start:start+CHUNK_BYTES]
                send(owners[3],'upload',claim_id=bad_id,object_root=key,index=i,data=base64.b64encode(chunk).decode())
                growth_bytes+=len(chunk)
            send(owners[3],'seal',claim_id=bad_id,object_root=key)
        send(owners[3],'resolve',claim_id=bad_id)
        growth_fraud=wire.query(url)
        assert growth_fraud['issued']==1000000 and growth_fraud['model_root']==parent
        assert growth_fraud['settled'][-1]['reason']=='objective replay mismatch: identity growth'
        propose_growth(grown,grown_model)
        growth_accepted=until(lambda:s if (s:=wire.query(url))['period_growths']==1 else None)
        assert growth_accepted['model_root']==grown and growth_accepted['issued']==1000000
        assert growth_accepted['serving_root']==record['parent']
        capacities=[48000000]*4
        count=len(place(grown_model,capacities))
        workers=[LocalEndpoint(Worker(args.home/f'grown-worker{i}',store)) for i in range(count)]
        pipe=Pipeline(store,grown,workers,capacities,'after-native-growth',start_step=1)
        try:next_record=pipe.train(store.json(record['batch']))
        finally:pipe.close()
        claim(next_record)
        final=until(lambda:s if (s:=wire.query(url))['training_round']==2 else None)
        assert final['issued']==2000000 and final['model_root']==next_record['model_root']
        # A CometBFT block header commits the previous height's app hash.
        # Wait for the next header so agreement includes the final paid step.
        height=final['height']+1
        until(lambda:all(wire.query(u)['height']>=height for u in urls))
        states=[wire.query(u) for u in urls]
        assert all(s['issued']==2000000 and s['training_round']==2 and s['model_root']==next_record['model_root'] for s in states)
        headers=[wire.rpc(u,'block',{'height':str(height)})['block']['header'] for u in urls]
        assert len({h['app_hash'] for h in headers})==1
        result={'chain_id':genesis['chain_id'],'source_hash':manifest['code_hash'],'genesis_hash':digest(canonical(genesis)),
                'validators':4,'physical_hosts':1,'operators':1,'fixture':'external-record' if args.record else 'synthetic-10384-parameters',
                'uploaded_replay_bytes':uploaded,'upload_transactions':transactions,'upload_seconds':upload_seconds,
                'dispute_resolution_seconds':resolve_seconds,'fraud_rejection':fraud['settled'][-1],
                'accepted_training':accepted['settled'][-1],'issued_atoms':accepted['issued'],
                'growth_fraud':growth_fraud['settled'][-1],'growth_replay_bytes':growth_bytes,
                'accepted_growth':growth_accepted['settled'][-1],'grown_parameters':grown_model['parameters'],
                'training_after_growth':final['settled'][-1],'final_issued_atoms':final['issued'],
                'agreement_includes_height':final['height'],
                'matching_app_hash_height':height,'matching_app_hash':headers[0]['app_hash']}
        (args.home/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    finally:
        for process in processes:
            if process.poll() is None:process.terminate()
        for process in processes:
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home',type=Path,required=True)
    parser.add_argument('--engine',type=Path,required=True)
    parser.add_argument('--base-port',type=int,default=49650)
    parser.add_argument('--record',type=Path)
    parser.add_argument('--objects',type=Path)
    run(parser.parse_args())
