#!/usr/bin/env python3
"""Isolated native data -> training -> evaluation -> serving -> paid inference.

Uses a tiny synthetic model and synthetic token documents, not a real tokenizer
or evidence of LLM quality. Four fresh validators share one physical host.
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

from neuroshard.dataflow.store import canonical
from neuroshard.demo import protocol, client as wire
from neuroshard.demo.network import initialize, edit_config
from neuroshard.evolution import cohorts, forward, lifecycle, auditing
from neuroshard.evolution.app import code_hash
from neuroshard.evolution.objects import Objects, digest
from neuroshard.evolution.pipeline import Pipeline, LocalEndpoint
from neuroshard.evolution.model import place, grow
from neuroshard.evolution.text import TextCodec
from neuroshard.evolution.worker import Worker
from neuroshard.evolution.settlement import PARAMS, CHUNK_BYTES
from neuroshard.evolution.verification import bundle, Metadata, audit, audit_growth
from neuroshard.lab.app import native_parameters
from experiment_evolution_native import tiny_record, until
from native_rpc import broadcast_finalized


def fixture(store, previous, index=0, cursors=None):
    values, documents, windows = {}, [], []
    def add(value):
        key = store.put_json(value)
        values[key] = value
        return key
    for group, role in enumerate(('train','retention','fresh')):
        source = add({'repo':'fixture/'+role,'revision':'a'*40,'split':'train','license':'CC0-1.0','role':role})
        start = (cursors or {}).get(source,0)
        windows.append({'source':source,'start':start,'end':start+32})
        for row in range(32):
            batch = {'input_ids':[[1,3+row,40+group+index*3,13,14,2]],'labels':[[-100,-100,-100,13,14,2]]}
            documents.append({'id':digest(canonical(['fixture-doc',index,role,row])), 'source':source,
                'row':start+row,'object':digest(canonical(['fixture-raw',index,role,row])),
                'batches':[add(batch)],'role':role,'omitted_targets':0})
    key = add({'format':cohorts.FORMAT,'previous':previous,'tokenizer_root':'b'*64,
               'windows':windows,'documents':documents})
    return key,values


def run(args):
    home = args.home.resolve()
    home.mkdir(parents=True,exist_ok=False)
    store = Objects(home/'objects')
    real = args.model_root is not None
    if real:
        if not all((args.objects,args.cohort,args.next_cohort)):
            raise ValueError('Real-model runs require objects and two prepared cohorts')
        source = Objects(args.objects)
        def fetch(key):
            try:return source.get(key)
            except FileNotFoundError:return None
        store.fetchers.append(fetch)
        initial = args.model_root
        model = store.json(initial)
        codec = TextCodec.load(store,model['tokenizer_root'])
        codec.check_model(model)
        prepared = json.loads(args.cohort.read_bytes())
        second_prepared = json.loads(args.next_cohort.read_bytes())
        if any(value['status']!='prepared_for_review' for value in (prepared,second_prepared)):
            raise ValueError('Both real cohorts must be complete proposals')
        cohort = prepared['metadata'][prepared['data_root']]
        previous = cohort['previous']
        cursors = {window['source']:window['start'] for window in cohort['windows']}
        eos = [codec.tokenizer.eos_token_id]
        prompt = codec.prompt([{'role':'user','content':'What is the capital of France?'}])
        capacities = [48000000]*(4 if args.growth_layers else 3)
    else:
        if any((args.objects,args.cohort,args.next_cohort,args.workers_config,args.growth_layers)):
            raise ValueError('External data or workers require --model-root')
        initial = tiny_record(home,store)['parent']
        model = store.json(initial)
        model['tokenizer_root'] = 'b'*64  # an explicit synthetic text identity
        initial = store.put_json(model)
        previous,cursors,eos,prompt,capacities = 'c'*64,{},[2],[1,3],[6000]*2
    count = len(place(model,capacities))
    endpoints = None
    if args.workers_config:
        from neuroshard.evolution.transport import Endpoint
        workers = json.loads(args.workers_config.read_bytes())['workers'][:len(capacities)]
        if len(workers)!=len(capacities):raise ValueError('Not enough configured workers')
        endpoints = [Endpoint(w['url'],(args.workers_config.resolve().parent/Path(w['token_file']).expanduser()).read_text().strip(),store) for w in workers]
    config = initialize(home/'native',args.base_port,str(args.engine.resolve()))
    profile = {'format':lifecycle.FORMAT,'tokenizer_root':model['tokenizer_root'],'vocabulary':model['config']['vocab_size'],'eos_ids':eos,
        'initial_model':model,'source_cursors':cursors,'steps_per_cohort':4,'data_vote_blocks':256,
        'evaluation_blocks':10000,'price_per_token':1000}
    params = {**PARAMS,'challenge_blocks':8,'availability_blocks':1024,'max_claim_blocks':4096}
    native = {**native_parameters(params),'block_max_bytes':4*1024*1024}
    manifest = {'params':params,'initial_model_root':initial,'data_root':previous,'training_batches':[],
        'learning_rate':.003,'clip_norm':1.,'native_consensus':native,'code_hash':code_hash(),'lifecycle':profile}
    if args.funded_audits:
        manifest['auditing'] = {**auditing.PROFILE, 'commit_blocks':256, 'reveal_blocks':32}
    genesis = json.loads((Path(config['nodes'][0]['home'])/'config/genesis.json').read_bytes())
    owners = [protocol.Identity.load_or_create(home/f'founder{i}.key') for i in range(4)]
    auditor_key = args.auditor_key or home/'auditor.key'
    auditor = protocol.Identity.load_or_create(auditor_key) if args.funded_audits else None
    validators = []
    for index,node in enumerate(config['nodes']):
        key = json.loads((Path(node['home'])/'config/priv_validator_key.json').read_bytes())['pub_key']['value']
        validators.append({'owner':owners[index].public_key,'consensus_key':base64.b64decode(key).hex(),
                           'bond':2500000,'liquid':1_000_000_000})
        genesis['validators'][index]['power'] = '10'
    genesis.update(chain_id='neuroshard-lifecycle-check-'+secrets.token_hex(6), initial_height='1',
                   app_state={'manifest':manifest,'validators':validators})
    config['chain_id'] = genesis['chain_id']
    (home/'native/network.json').write_bytes(canonical(config))
    genesis['consensus_params']['block']['max_bytes'] = str(native['block_max_bytes'])
    genesis['consensus_params']['evidence'].update(max_age_num_blocks=str(params['evidence_blocks']),
        max_age_duration=str(params['evidence_seconds']*1_000_000_000))
    for node in config['nodes']:
        path = Path(node['home'])/'config/config.toml'
        text = edit_config(path.read_text(),'rpc','max_body_bytes','4194304')
        text = edit_config(text,'rpc','timeout_broadcast_tx_commit','"120s"')
        path.write_text(edit_config(text,'mempool','max_tx_bytes','2097152'))
        (path.parent/'genesis.json').write_bytes(canonical(genesis))
    processes = {}
    def start(index):
        node = config['nodes'][index]
        for name, command in [('app',[sys.executable,'-m','neuroshard.evolution.app','--home',node['home'],'--port',str(node['abci'])]),
                              ('node',[config['engine'],'start','--home',node['home']])]:
            with (home/f'{name}{index}.log').open('ab') as log:
                processes[name+str(index)] = subprocess.Popen(command,stdout=log,stderr=log)
    def stop(index):
        for name in ('node','app'):
            process = processes.get(name+str(index))
            if process and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
    urls = [f'http://127.0.0.1:{node["rpc"]}' for node in config['nodes']]
    url = urls[0]
    def query(path='/status',options=None):return wire.query(url,path,options)
    def send(owner,kind,**fields):
        if args.funded_audits and kind in ('claim', *auditing.CLAIM_KINDS):
            from neuroshard.evolution.audit_worker import required_objects
            values = fields['metadata']
            claim_kind = {'claim':'training', 'grow':'growth', 'score':'score', 'respond':'inference'}[kind]
            if kind == 'grow':
                value = {'kind':'growth', 'parent':fields['parent'], 'model_root':fields['model_root'],
                         'assignments':place(values[fields['model_root']], fields['capacities'])}
                record_root = store.put_json(value)
                values = {**values, record_root:value}
            else:
                record_root = fields['record_root']
            preview = {'kind':claim_kind, 'record_root':record_root, 'metadata':values,
                       'model_root':values[record_root]['model_root']}
            for key in required_objects(preview):
                store.get(key)  # Publish every object before the audit window opens.
        if args.funded_audits and kind in ('reserve', *auditing.CLAIM_KINDS):
            if kind == 'reserve':
                stages = len(fields['workers'])
            elif kind == 'grow':
                stages = 1
            else:
                stages = len(forward.trace_roots(Metadata(fields['metadata']), fields['record_root']))
            budget = send(owners[0], 'fund_audit', publisher=owner.public_key,
                          auditors=[auditor.public_key], stage_limit=stages, expires_in=1024)
            until(lambda: query('/auditing')['budgets'][budget]['auditors'][auditor.public_key]['bond'], 60)
            fields['audit_budget'] = budget
        nonce = query('/account',{'public_key':owner.public_key})['nonce']
        signed = owner.sign({'kind':kind,'chain_id':genesis['chain_id'],'nonce':nonce,**fields})
        broadcast_finalized(url,signed)
        return protocol.transaction_id(signed)
    def settled():return until(lambda:s if (s:=query())['candidate'] is None else None,360)
    def pipe(root,name,step=0):
        count = len(place(store.json(root),capacities))
        workers = endpoints[:count] if endpoints else [LocalEndpoint(Worker(home/(name+str(i)),store)) for i in range(count)]
        return Pipeline(store,root,workers,capacities,'native-'+genesis['chain_id']+'-'+name,start_step=step)
    def emit(phase,**values):print(json.dumps({'phase':phase,**values}),flush=True)
    audits = 0
    def check(record_root, is_forward=True):
        nonlocal audits
        metadata = Metadata(forward.bundle(store,record_root) if is_forward else bundle(store,record_root))
        count = len(forward.trace_roots(metadata,record_root)) if is_forward else len(metadata.json(record_root)['traces'])
        for stage in range(count):
            if not audit(store,metadata,record_root,stage)['valid']:
                raise ValueError('Independent observer rejected an honest record')
            audits += 1
        return metadata.values
    def dispute(stage):
        claim_id = query()['candidate']['id']
        send(owners[3],'challenge',claim_id=claim_id,stage=stage,challenge_kind='fraud',object_root=None)
        needed = query('/candidate')['challenge']['needed']
        size = 0
        for key in needed:
            raw = store.get(key)
            for index,start_at in enumerate(range(0,len(raw),CHUNK_BYTES)):
                chunk = raw[start_at:start_at+CHUNK_BYTES]
                send(owners[3],'upload',claim_id=claim_id,object_root=key,index=index,data=base64.b64encode(chunk).decode())
                size += len(chunk)
            send(owners[3],'seal',claim_id=claim_id,object_root=key)
        send(owners[3],'resolve',claim_id=claim_id)
        return size
    try:
        for index in range(4):start(index)
        until(lambda:all(wire.query(u)['height']>0 for u in urls))
        if args.funded_audits:
            send(owners[0], 'transfer', to=auditor.public_key, amount=100_000_000)
            command = [sys.executable, '-m', 'neuroshard.evolution.audit_worker',
                       '--home', str(home/'auditor'), '--rpc', url,
                       '--genesis-sha256', digest(canonical(genesis)), '--key', str(auditor_key),
                       '--sponsor', owners[0].public_key, '--objects', str(home/'objects')]
            if args.auditor_command:
                template = json.loads(args.auditor_command.read_bytes())
                if not isinstance(template, list) or not all(isinstance(v, str) for v in template):
                    raise ValueError('Auditor command must be a JSON argument list')
                command = [part.format(genesis_sha256=digest(canonical(genesis)),
                           sponsor=owners[0].public_key, rpc=url) for part in template]
            with (home/'auditor.log').open('ab') as log:
                processes['auditor'] = subprocess.Popen(command, stdout=log, stderr=log)
        key, values = (prepared['data_root'],prepared['metadata']) if real else fixture(store,manifest['data_root'])
        proposal = send(owners[0],'propose_data',data_root=key,metadata=values)
        for owner in owners[:2]:send(owner,'vote_data',proposal_id=proposal,approve=True)
        activate_after = query('/lifecycle')['proposal']['activate_after']
        until(lambda:query()['height']>activate_after)
        assert query('/data') is None
        send(owners[2],'vote_data',proposal_id=proposal,approve=True)
        until(lambda:query('/data'))
        emit('data_activated',data_root=key)
        if args.growth_layers:
            grown, grown_model = grow(initial,store,args.growth_layers)
            metadata = Metadata({initial:store.json(initial),grown:grown_model})
            assert audit_growth(store,metadata,initial,grown)['valid']
            send(owners[0],'grow',parent=initial,model_root=grown,metadata=metadata.values,capacities=capacities)
            settled()
            assert query()['model_root']==grown and query()['issued']==0
            emit('growth_settled_without_issuance',parameters=grown_model['parameters'])
        for step in range(4):
            status = query()
            count = len(place(store.json(status['model_root']),capacities))
            send(owners[0],'reserve',parent=status['model_root'],round=step,workers=[o.public_key for o in owners[:count]])
            reservation = query()['assignment']
            worker_pipe = pipe(status['model_root'],'train'+str(step),step)
            try:value = worker_pipe.train(query('/data')['batches'][reservation['batch']])
            finally:worker_pipe.close()
            metadata = check(value['record_root'],False)
            receipts = [owner.sign({'domain':'neuroshard/evolution/work/v1','chain_id':genesis['chain_id'],
                'assignment':reservation['id'],'record_root':value['record_root'],'stage':i,'trace_root':value['traces'][i]}) for i,owner in enumerate(owners[:count])]
            send(owners[0],'claim',record_root=value['record_root'],metadata=metadata,workers=receipts,
                 data_root=key,sequence_index=step)
            settled()
        assert query()['issued'] == 4_000_000
        emit('cohort_training_settled',issued_atoms=query()['issued'])
        evaluation_id = send(owners[0],'open_evaluation')
        evaluation = query('/evaluation')
        for side in ('baseline','candidate'):
            worker_pipe = pipe(evaluation[side],side)
            try:
                for role in ('retention','fresh'):
                    for offset in range(0,len(cohorts.evaluation_rows(query('/data'),role)),4):
                        _, batch = cohorts.evaluation_batch(query('/data'),role,offset)
                        value = worker_pipe.evaluate_record(batch)
                        send(owners[0],'score',evaluation_id=evaluation_id,side=side,role=role,offset=offset,
                             record_root=value['record_root'],metadata=check(value['record_root']))
                        settled()
                    emit('evaluation_group_settled',side=side,group=role)
            finally:worker_pipe.close()
        expected = cohorts.decision(query('/evaluation')['measurements'],query('/data'))
        send(owners[0],'finish_evaluation',evaluation_id=evaluation_id)
        decision = query('/lifecycle')['evaluations'][-1]['decision']
        assert decision == expected
        assert query()['serving_root'] == (evaluation['candidate'] if decision['promote'] else initial)
        emit('serving_decision',promoted=decision['promote'])
        serving = query()['serving_root']
        job_id = send(owners[0],'infer',model_root=serving,provider=owners[1].public_key,
                      prompt_ids=prompt,max_tokens=1,max_price=7000,expires_in=4096)
        worker_pipe = pipe(serving,'generate')
        try:value = worker_pipe.generate_record(prompt,1,eos)
        finally:worker_pipe.close()
        good = value['record_root']
        record = store.json(store.json(good)['records'][0])
        head = store.json(record['traces'][-1])
        wrong = (value['token_ids'][0]+1)%profile['vocabulary']
        head['result']['next_ids'] = [wrong]
        record['result'] = head['result']
        record['traces'][-1] = store.put_json(head)
        forged = store.json(good)
        forged['records'] = [store.put_json(record)]
        forged['token_ids'] = [wrong]
        forged_root = store.put_json(forged)
        send(owners[1],'respond',job_id=job_id,record_root=forged_root,metadata=forward.bundle(store,forged_root))
        if args.funded_audits:
            md = Metadata(forward.bundle(store, forged_root))
            replay_bytes = sum(len(store.get(key)) for key in forward.dependencies(md, record['traces'][-1]))
            settled()  # The separate auditor process must detect and refute it.
        else:
            replay_bytes = dispute(len(record['traces'])-1)
        assert query()['settled'][-1]['reason'] == 'objective replay mismatch: forward evaluation result'
        send(owners[1],'respond',job_id=job_id,record_root=good,metadata=check(good))
        settled()
        paid = query('/inference',{'id':job_id})
        assert paid['status'] == 'completed' and paid['paid_atoms'] == 1000 and paid['refunded_atoms'] == 6000
        emit('paid_inference_and_fraud_checked',replay_bytes=replay_bytes)
        life = query('/lifecycle')
        new_key, new_values = (second_prepared['data_root'],second_prepared['metadata']) if real else fixture(store,key,1,life['cursors'])
        proposal = send(owners[0],'propose_data',data_root=new_key,metadata=new_values)
        for owner in owners[:3]:send(owner,'vote_data',proposal_id=proposal,approve=True)
        until(lambda:d if (d:=query('/data'))['root']==new_key else None)
        assert query()['training_round'] == 4 and query('/data')['step'] == 0
        emit('second_fresh_cohort_activated',data_root=new_key)
        expected_issued = 4_000_000
        if args.continue_cohort:
            if not args.funded_audits:
                raise ValueError('The continuous operator requires funded auditing')
            worker_config = (workers if endpoints else [{'capacity':v} for v in capacities])
            if endpoints:
                worker_config = [{**w, 'capacity':capacities[i],
                    'token_file':str((args.workers_config.resolve().parent/Path(w['token_file']).expanduser()).absolute())}
                    for i,w in enumerate(worker_config)]
            operator_config = {'home':str(home/'operator'), 'rpc':url,
                'genesis_sha256':digest(canonical(genesis)), 'key':str(home/'founder0.key'),
                'worker_keys':[str(home/f'founder{i}.key') for i in range(len(capacities))],
                'auditors':[auditor.public_key], 'objects':str(home/'objects'),
                'workers':worker_config, 'budget':{'training_round_limit':8,'minimum_balance':100_000_000}}
            operator_path = home/'operator.json'
            operator_path.write_bytes(canonical(operator_config))
            command = [sys.executable, str(Path(__file__).with_name('run_lifecycle_operator.py')),
                       '--config',str(operator_path)]
            def start_operator():
                with (home/'operator.log').open('ab') as log:
                    processes['operator'] = subprocess.Popen(command, stdout=log, stderr=log)
            start_operator()
            until(lambda:query()['assignment'],60)
            # Crash the coordinator after a durable reservation. The restarted
            # process must recover it, then finish training and quality scoring.
            processes['operator'].kill()
            processes['operator'].wait(timeout=10)
            start_operator()
            until(lambda:query()['training_round']==8 and query('/data')['closed'],3600)
            expected_issued = 8_000_000
            emit('continuous_operator_completed_second_cohort', issued_atoms=expected_issued,
                 serving_decision=query('/lifecycle')['evaluations'][-1]['decision'])
        stop(3)
        before = query()['height']
        until(lambda:query()['height']>before+2)
        stop(2)
        time.sleep(2)
        halted = query()['height']
        time.sleep(3)
        assert query()['height'] == halted
        start(2)
        until(lambda:query()['height']>halted+2)
        start(3)
        until(lambda:all(wire.query(u)['height']>halted for u in urls))
        height = query()['height']+2
        until(lambda:all(wire.query(u)['height']>=height for u in urls))
        headers = [wire.rpc(u,'block',{'height':str(height)})['block']['header'] for u in urls]
        assert len({header['app_hash'] for header in headers}) == 1
        assert all(wire.query(u)['issued'] == expected_issued for u in urls)
        result = {'chain_id':genesis['chain_id'],'source_hash':manifest['code_hash'],
            'genesis_hash':digest(canonical(genesis)),'fixture':'real-model' if real else 'synthetic-10384-parameters',
            'validator_physical_hosts':1,'worker_transport':'http' if endpoints else 'local-single-process',
            'operators':1,'validators':4,'independent_stage_replays':audits,'parameters':model['parameters'],
            'candidate_parameters':store.json(evaluation['candidate'])['parameters'],
            'growth_layers':args.growth_layers,'declared_capacities':capacities,
            'fresh_cohorts_activated':2,'training_steps':expected_issued//1_000_000,'issued_atoms':expected_issued,
            'continuous_operator_recovered_reservation':args.continue_cohort,
            'serving_decision':decision,'llm_quality_improvement_claimed':False,
            'inference':paid,'forged_inference_replay_bytes':replay_bytes,
            'one_validator_down_progress':True,'half_voting_power_down_halts':True,
            'restart_catchup_agreement':True,'matching_app_hash_height':height,
            'matching_app_hash':headers[0]['app_hash']}
        if args.funded_audits:
            result['funded_auditing'] = query('/auditing')
            assert result['funded_auditing']['paid_atoms'] > 0
            assert not result['funded_auditing']['budgets']
            result['serving_evaluations'] = query('/lifecycle')['evaluations']
        (home/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
    finally:
        if 'operator' in processes:
            processes['operator'].terminate()
            try: processes['operator'].wait(timeout=5)
            except subprocess.TimeoutExpired: processes['operator'].kill(); processes['operator'].wait()
        if 'auditor' in processes:
            processes['auditor'].terminate()
            try: processes['auditor'].wait(timeout=5)
            except subprocess.TimeoutExpired: processes['auditor'].kill(); processes['auditor'].wait()
        for index in range(4):stop(index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home',type=Path,required=True)
    parser.add_argument('--engine',type=Path,required=True)
    parser.add_argument('--base-port',type=int,default=50650)
    parser.add_argument('--model-root')
    parser.add_argument('--objects',type=Path)
    parser.add_argument('--cohort',type=Path)
    parser.add_argument('--next-cohort',type=Path)
    parser.add_argument('--workers-config',type=Path)
    parser.add_argument('--growth-layers',type=int,default=0,choices=range(0,5),help='Real-model trial: add up to four identity blocks before training')
    parser.add_argument('--funded-audits', action='store_true', help='Require prepaid complete audit reports for every execution claim')
    parser.add_argument('--auditor-key', type=Path, help='Optional precreated experiment auditor identity')
    parser.add_argument('--auditor-command', type=Path, help='JSON argv template for a separately operated audit process')
    parser.add_argument('--continue-cohort', action='store_true', help='Crash and recover the continuous operator through the second cohort and evaluation')
    run(parser.parse_args())
