#!/usr/bin/env python3
"""Settle compact SGD refutations on an isolated native four-validator chain.

Optionally upload the same forged stage for the old full referee first. Every
node uses a fresh genesis and keys; all processes stop on exit. Test issuance is
separate from the public network. The numerical benchmark supplies the record.
"""
import argparse
import base64
import json
import secrets
import subprocess
import sys
import time
from pathlib import Path

from neuroshard.dataflow.store import canonical
from neuroshard.demo import protocol, client as wire
from neuroshard.demo.network import initialize, edit_config
from neuroshard.evolution.app import code_hash
from neuroshard.evolution.objects import Objects, digest
from neuroshard.evolution.settlement import PARAMS, CHUNK_BYTES
from neuroshard.evolution.verification import bundle
from neuroshard.lab.app import native_parameters
from experiment_evolution_native import until
from native_rpc import broadcast_finalized


def run(args):
    report = json.loads(args.record.read_bytes())
    if report['source_hash'] != code_hash():
        raise ValueError('Use the numerical benchmark\'s exact package source')
    args.home = args.home.resolve()
    args.home.mkdir(parents=True,exist_ok=False)
    store = Objects(args.objects)
    training, forged = report['training'], report['forged_training']
    cases = report['portable_cases']
    config = initialize(args.home/'native',args.base_port,str(args.engine.resolve()))
    params = {**PARAMS,'availability_blocks':1024,'max_claim_blocks':4096,'challenge_blocks':64}
    native = {**native_parameters(params),'block_max_bytes':4*1024*1024}
    manifest = {'params':params,'initial_model_root':training['parent'],
        'data_root':digest(canonical([training['batch']])),'training_batches':[training['batch']],
        'learning_rate':float.fromhex(training['learning_rate_hex']),'clip_norm':float.fromhex(training['clip_norm_hex']),
        'native_consensus':native,'code_hash':code_hash()}
    genesis = json.loads((Path(config['nodes'][0]['home'])/'config/genesis.json').read_bytes())
    owners = [protocol.Identity.load_or_create(args.home/f'founder{i}.key') for i in range(4)]
    validators = []
    for i,node in enumerate(config['nodes']):
        key = json.loads((Path(node['home'])/'config/priv_validator_key.json').read_bytes())['pub_key']['value']
        validators.append({'owner':owners[i].public_key,'consensus_key':base64.b64decode(key).hex(),
                           'bond':2_500_000,'liquid':1_000_000_000})
        genesis['validators'][i]['power'] = '10'
    genesis.update(chain_id='neuroshard-update-check-'+secrets.token_hex(6),
                   app_state={'manifest':manifest,'validators':validators})
    genesis['consensus_params']['block']['max_bytes'] = str(native['block_max_bytes'])
    genesis['consensus_params']['evidence'].update(max_age_num_blocks=str(params['evidence_blocks']),
        max_age_duration=str(params['evidence_seconds']*1_000_000_000))
    processes = []
    try:
        for i,node in enumerate(config['nodes']):
            path = Path(node['home'])/'config/config.toml'
            text = edit_config(path.read_text(),'rpc','max_body_bytes','4194304')
            text = edit_config(text,'mempool','max_tx_bytes','2097152')
            path.write_text(text)
            (path.parent/'genesis.json').write_bytes(canonical(genesis))
            commands = [('app',[sys.executable,'-m','neuroshard.evolution.app','--home',node['home'],'--port',str(node['abci'])]),
                        ('node',[config['engine'],'start','--home',node['home']])]
            for name,command in commands:
                with (args.home/f'{name}{i}.log').open('wb') as log:
                    processes.append(subprocess.Popen(command,stdout=log,stderr=log))
        urls = [f'http://127.0.0.1:{node["rpc"]}' for node in config['nodes']]
        url = urls[0]
        until(lambda:all(wire.query(endpoint)['height']>0 for endpoint in urls))

        def send(owner,kind,**fields):
            nonce = wire.query(url,'/account',{'public_key':owner.public_key})['nonce']
            envelope = owner.sign({'kind':kind,'chain_id':genesis['chain_id'],'nonce':nonce,**fields})
            started = time.perf_counter()
            receipt = broadcast_finalized(url,envelope)
            return {'seconds':time.perf_counter()-started,'signed_bytes':len(canonical(envelope)),
                    'height':int(receipt['height']),'hash':receipt['hash']}

        def claim(record):
            status = wire.query(url)
            workers = [owners[i%3] for i in range(len(record['traces']))]
            send(owners[0],'reserve',parent=status['model_root'],round=status['training_round'],workers=[o.public_key for o in workers])
            assignment = wire.query(url)['assignment']['id']
            receipts = [owner.sign({'domain':'neuroshard/evolution/work/v1','chain_id':genesis['chain_id'],
                'assignment':assignment,'record_root':record['record_root'],'stage':stage,'trace_root':record['traces'][stage]})
                for stage,owner in enumerate(workers)]
            send(owners[0],'claim',record_root=record['record_root'],metadata=bundle(store,record['record_root']),
                 workers=receipts,data_root=manifest['data_root'],sequence_index=0)
            return wire.query(url)['candidate']['id']

        full = None
        if args.compare_full:
            identity = claim(forged)
            started = time.perf_counter()
            receipts = [send(owners[3],'challenge',claim_id=identity,stage=0,challenge_kind='fraud',object_root=None)]
            payload_bytes = 0
            for key in wire.query(url)['candidate']['challenge']['needed']:
                raw = store.get(key)
                for index,start in enumerate(range(0,len(raw),CHUNK_BYTES)):
                    chunk = raw[start:start+CHUNK_BYTES]
                    receipts.append(send(owners[3],'upload',claim_id=identity,object_root=key,index=index,data=base64.b64encode(chunk).decode()))
                    payload_bytes += len(chunk)
                receipts.append(send(owners[3],'seal',claim_id=identity,object_root=key))
            resolution = send(owners[3],'resolve',claim_id=identity)
            receipts.append(resolution)
            state = wire.query(url)
            if state['candidate'] or state['issued'] or state['settled'][-1]['accepted']:
                raise ValueError('Full referee failed to reject the same forged update')
            full = {'seconds':time.perf_counter()-started,'input_bytes':payload_bytes,
                    'signed_transaction_bytes':sum(row['signed_bytes'] for row in receipts),
                    'transactions':len(receipts),'resolve':resolution,'rejection':state['settled'][-1]}
            print(json.dumps({'phase':'full_refutation',**full}),flush=True)
        identity = claim(forged)
        case = cases['forged']
        compact = send(owners[3],'refute_update',claim_id=identity,stage=case['stage'],
                       tensor_index=case['tensor_index'],witness=case['witness'])
        state = wire.query(url)
        if state['candidate'] or state['issued'] or state['settled'][-1]['accepted']:
            raise ValueError('Compact witness failed to refute the optimizer error')
        compact['rejection'] = state['settled'][-1]
        print(json.dumps({'phase':'compact_refutation',**compact}),flush=True)

        # A valid opening of an honest update burns the accuser's bond and
        # cannot change the candidate's deadline or authorize early payment.
        identity = claim(training)
        before = wire.query(url)
        case = cases['honest']
        false_accusation = send(owners[3],'refute_update',claim_id=identity,stage=case['stage'],
                                tensor_index=case['tensor_index'],witness=case['witness'])
        after = wire.query(url)
        if (after['issued'] or after['candidate']['deadline']!=before['candidate']['deadline']
                or after['burned']-before['burned']!=params['fee']+params['challenge_bond']):
            raise ValueError('False accusation changed settlement timing or accounting')
        final = until(lambda:s if (s:=wire.query(url))['training_round']==1 else None,180)
        if final['issued']!=1_000_000 or final['model_root']!=training['model_root']:
            raise ValueError('Valid work failed to settle exactly once')
        height = final['height']+1
        until(lambda:all(wire.query(endpoint)['height']>=height for endpoint in urls))
        headers = [wire.rpc(endpoint,'block',{'height':str(height)})['block']['header'] for endpoint in urls]
        if len({header['app_hash'] for header in headers})!=1:
            raise ValueError('Native validators disagree')
        result = {'chain_id':genesis['chain_id'],'source_hash':code_hash(),
            'genesis_sha256':digest(canonical(genesis)),'validators':4,'validator_hosts':1,'operators':1,
            'full_refutation':full,'compact_refutation':compact,'false_accusation':false_accusation,
            'issued_atoms':final['issued'],'full_referee_calls':final['audit_count'],
            'compact_checks':final['update_check_count'],'matching_header_height':height,
            'matching_header_app_hash':headers[0]['app_hash'],'serving_root':final['serving_root'],
            'scope':'isolated native SGD refutations; no public cutover or model-quality claim'}
        (args.home/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
        return result
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home',type=Path,required=True)
    parser.add_argument('--record',type=Path,required=True)
    parser.add_argument('--objects',type=Path,required=True)
    parser.add_argument('--engine',type=Path,required=True)
    parser.add_argument('--base-port',type=int,default=53550)
    parser.add_argument('--compare-full',action='store_true')
    run(parser.parse_args())
