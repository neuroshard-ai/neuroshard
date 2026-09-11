#!/usr/bin/env python3
"""Resume a completed isolated lifecycle trial and train its second cohort.

Reopens the original node homes, keys and databases. This is a bounded test
driver for experiment_lifecycle_native.py, not a public operator service.
"""
import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

from neuroshard.demo import protocol, client as wire
from neuroshard.evolution.app import code_hash
from neuroshard.evolution.model import place
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.pipeline import Pipeline, LocalEndpoint
from neuroshard.evolution.verification import Metadata, bundle, audit
from neuroshard.evolution.worker import Worker
from experiment_evolution_native import until
from native_rpc import broadcast_finalized


def run(args):
    home = args.home.resolve()
    original = json.loads((home/'result.json').read_bytes())
    config = json.loads((home/'native/network.json').read_bytes())
    output = home/'continuation-result.json'
    if output.exists():
        raise ValueError('This bounded continuation already has a result; do not repeat it')
    if (original['source_hash'] != code_hash() or original['fresh_cohorts_activated'] != 2
            or not original['chain_id'].startswith('neuroshard-lifecycle-check-')):
        raise ValueError('Use the exact source of a completed isolated lifecycle trial')
    if len(config['nodes']) != 4:
        raise ValueError('Expected the four retained isolated node homes')
    for node in config['nodes']:
        path = Path(node['home']).resolve()
        if not path.is_relative_to(home) or not (path/'evolution.sqlite').is_file():
            raise ValueError('Refusing to create, relocate or replace existing node state')
        for port in (node['rpc'],node['abci'],node['p2p']):
            with socket.socket() as probe:
                probe.settimeout(1)
                if probe.connect_ex(('127.0.0.1',port)) == 0:
                    raise ValueError('Stop the original experiment before reopening its nodes')
        genesis = json.loads((path/'config/genesis.json').read_bytes())
        if genesis['chain_id'] != original['chain_id'] or genesis['app_state']['manifest']['code_hash'] != code_hash():
            raise ValueError('Retained genesis differs from the original trial')
    keys = [home/f'founder{i}.key' for i in range(4)]
    if any(not path.is_file() for path in keys):
        raise ValueError('Missing original test identity; never generate replacement keys')
    owners = [protocol.Identity.load_or_create(path) for path in keys]
    store = Objects(home/'objects')
    capacities = original.get('declared_capacities',[6000]*2)
    endpoints = None
    if original['worker_transport'] == 'http':
        if not args.workers_config:
            raise ValueError('Supply the original worker configuration')
        from neuroshard.evolution.transport import Endpoint
        workers = json.loads(args.workers_config.read_bytes())['workers'][:len(capacities)]
        if len(workers) != len(capacities):
            raise ValueError('Not enough worker endpoints')
        endpoints = [Endpoint(w['url'],(args.workers_config.resolve().parent/Path(w['token_file']).expanduser()).read_text().strip(),store) for w in workers]
    urls = [f'http://127.0.0.1:{node["rpc"]}' for node in config['nodes']]
    def query(path='/status', options=None):return wire.query(urls[0],path,options)
    def send(owner,kind,**fields):
        nonce = query('/account',{'public_key':owner.public_key})['nonce']
        signed = owner.sign({'kind':kind,'chain_id':original['chain_id'],'nonce':nonce,**fields})
        broadcast_finalized(urls[0],signed)
    processes, replays, steps = [], 0, []
    started = time.monotonic()
    try:
        for i,node in enumerate(config['nodes']):
            for name,command in [('app',[sys.executable,'-m','neuroshard.evolution.app','--home',node['home'],'--port',str(node['abci'])]),
                                 ('node',[config['engine'],'start','--home',node['home']])]:
                with (home/f'continuation-{name}{i}.log').open('ab') as log:
                    processes.append(subprocess.Popen(command,stdout=log,stderr=log))
        until(lambda:all(wire.query(url)['height']>original['matching_app_hash_height'] for url in urls),120)
        before, data, life = query(), query('/data'), query('/lifecycle')
        if before['assignment'] or before['candidate'] or life['evaluation']:
            raise ValueError('Pending work needs explicit recovery before this continuation')
        if (not data or data['closed'] or data['step'] != 0
                or before['training_round'] != original['training_steps'] or before['issued'] != original['issued_atoms']):
            raise ValueError('Expected the untouched second admitted cohort and preserved prior payments')
        # The native schedule mixes current fresh windows and retained replay
        # windows. Derive the distinction from state, not mutable input files.
        fresh_training = {batch for doc in data['train'] for batch in doc['batches']}
        for index,batch_root in enumerate(data['schedule']):
            status = query()
            count = len(place(store.json(status['model_root']),capacities))
            send(owners[0],'reserve',parent=status['model_root'],round=status['training_round'],workers=[o.public_key for o in owners[:count]])
            reservation = query()['assignment']
            if reservation['data_root'] != data['root'] or reservation['sequence_index'] != index or reservation['batch'] != batch_root:
                raise ValueError('Continuation reservation differs from the native schedule')
            workers = endpoints[:count] if endpoints else [LocalEndpoint(Worker(home/f'continuation-worker{i}',store)) for i in range(count)]
            pipeline = Pipeline(store,status['model_root'],workers,capacities,
                'continue-'+original['chain_id']+'-'+str(status['training_round']),start_step=status['training_round'])
            try:record = pipeline.train(data['batches'][batch_root])
            finally:pipeline.close()
            metadata = Metadata(bundle(store,record['record_root']))
            for stage in range(len(record['traces'])):
                if not audit(store,metadata,record['record_root'],stage)['valid']:
                    raise ValueError('Observer rejected a continuation training stage')
                replays += 1
            receipts = [owner.sign({'domain':'neuroshard/evolution/work/v1','chain_id':original['chain_id'],
                'assignment':reservation['id'],'record_root':record['record_root'],'stage':stage,'trace_root':record['traces'][stage]}) for stage,owner in enumerate(owners[:count])]
            send(owners[0],'claim',record_root=record['record_root'],metadata=metadata.values,workers=receipts,
                 data_root=data['root'],sequence_index=index)
            settled = until(lambda:s if (s:=query())['candidate'] is None else None,120)
            if settled['training_round'] != status['training_round']+1:
                raise ValueError('Continuation did not settle the prescribed training step')
            steps.append({'round':settled['training_round'],'batch':batch_root,'replay':batch_root not in fresh_training,
                          'model_root':settled['model_root'],'record_root':record['record_root']})
            print(json.dumps({'phase':'next_cohort_training_settled',**steps[-1]}),flush=True)
        after = query()
        if (after['serving_root'] != before['serving_root'] or query('/data')['step'] != len(data['schedule'])
                or after['issued'] != before['issued']+len(steps)*1_000_000
                or sum(step['replay'] for step in steps) != len(steps)//4):
            raise ValueError('Continuation violated serving protection or issuance accounting')
        height = after['height']+2
        until(lambda:all(wire.query(url)['height']>=height for url in urls),120)
        headers = [wire.rpc(url,'block',{'height':str(height)})['block']['header'] for url in urls]
        if len({header['app_hash'] for header in headers}) != 1:
            raise ValueError('Continuation validators disagree')
        result = {'chain_id':original['chain_id'],'source_hash':code_hash(),'data_root':data['root'],
            'retained_original_node_state':True,'serving_root_unchanged':after['serving_root'],
            'before_round':before['training_round'],'after_round':after['training_round'],
            'before_issued_atoms':before['issued'],'after_issued_atoms':after['issued'],
            'steps':steps,'replay_steps':sum(step['replay'] for step in steps),'independent_stage_replays':replays,
            'matching_app_hash_height':height,'matching_app_hash':headers[0]['app_hash'],
            'seconds':time.monotonic()-started,'next_required_action':'evaluate the second candidate before another data admission',
            'scope':'bounded second-cohort continuation; not an autonomous public operator service or quality improvement claim'}
        output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home',type=Path,required=True)
    parser.add_argument('--workers-config',type=Path)
    run(parser.parse_args())
