#!/usr/bin/env python3
"""Resume a lifecycle trial stopped after paid inference at next-data admission.

Retains the original genesis, keys, blocks, balances and numerical profile.
This bounded recovery driver records its interruption; it is not a migration.
"""
import argparse
import base64
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

from neuroshard.dataflow.store import canonical
from neuroshard.demo import protocol, client as wire
from neuroshard.evolution import cohorts, forward
from neuroshard.evolution.app import code_hash
from neuroshard.evolution.objects import digest
from neuroshard.evolution.verification import Metadata
from experiment_evolution_native import until
from native_rpc import broadcast_finalized


def run(args):
    home = args.home.resolve()
    if (home/'result.json').exists():raise ValueError('Original trial already has a completed result')
    config = json.loads((home/'native/network.json').read_bytes())
    if len(config['nodes']) != 4:raise ValueError('Expected four original isolated node homes')
    genesis = None
    for node in config['nodes']:
        path = Path(node['home']).resolve()
        if not path.is_relative_to(home) or not (path/'evolution.sqlite').is_file():
            raise ValueError('Missing retained node state; recovery never initializes nodes')
        value = json.loads((path/'config/genesis.json').read_bytes())
        if genesis is not None and value != genesis:raise ValueError('Retained nodes disagree on genesis')
        genesis = value
        for port in (node['rpc'],node['p2p'],node['abci']):
            with socket.socket() as probe:
                probe.settimeout(1)
                if probe.connect_ex(('127.0.0.1',port)) == 0:raise ValueError('Original node port is still running')
    manifest = genesis['app_state']['manifest']
    if not genesis['chain_id'].startswith('neuroshard-lifecycle-check-') or manifest['code_hash'] != code_hash():
        raise ValueError('Use the exact package profile of the original isolated trial')
    keys = [home/f'founder{i}.key' for i in range(4)]
    if any(not path.is_file() for path in keys):raise ValueError('Missing original test key')
    owners = [protocol.Identity.load_or_create(path) for path in keys]
    prepared = json.loads(args.next_cohort.read_bytes())
    if prepared['status'] != 'prepared_for_review':raise ValueError('Next cohort is incomplete')
    cohorts.metadata(prepared['metadata'])
    phases = [json.loads(line) for line in args.run_log.read_text().splitlines() if line.startswith('{"phase":')]
    inference_phase = next(value for value in phases if value['phase']=='paid_inference_and_fraud_checked')
    processes, urls = {}, [f'http://127.0.0.1:{node["rpc"]}' for node in config['nodes']]
    def query(path='/status',options=None):return wire.query(urls[0],path,options)
    def send(owner,kind,**fields):
        nonce = query('/account',{'public_key':owner.public_key})['nonce']
        value = owner.sign({'kind':kind,'chain_id':genesis['chain_id'],'nonce':nonce,**fields})
        broadcast_finalized(urls[0],value)
        return protocol.transaction_id(value)
    def start(index):
        node = config['nodes'][index]
        for name,command in [('app',[sys.executable,'-m','neuroshard.evolution.app','--home',node['home'],'--port',str(node['abci'])]),
                             ('node',[config['engine'],'start','--home',node['home']])]:
            with (home/f'recovery-{name}{index}.log').open('ab') as log:
                processes[name+str(index)] = subprocess.Popen(command,stdout=log,stderr=log)
    def stop(index):
        for name in ('node','app'):
            process = processes.get(name+str(index))
            if process and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
    try:
        for index in range(4):start(index)
        until(lambda:all(wire.query(url)['height']>0 for url in urls),120)
        before, life, data = query(),query('/lifecycle'),query('/data')
        if (before['assignment'] or before['candidate'] or before['training_round'] != 4 or before['issued'] != 4_000_000
                or not data['closed'] or life['evaluation'] or len(life['evaluations']) != 1
                or len(life['data_history']) != 1 or not life['data_history'][0]['accepted']):
            raise ValueError('Ledger is not at the expected interrupted admission boundary')
        evaluation = life['evaluations'][0]
        if (manifest['lifecycle']['initial_model']['parameters'] != 134515008
                or evaluation['candidate_model']['parameters'] != 148675392):
            raise ValueError('This recovery driver is bounded to the recorded real growth trial')
        if evaluation['decision'] != cohorts.decision(evaluation['measurements'],data):
            raise ValueError('Retained evaluation differs from complete native measurements')
        paid = list(life['results'].values())
        if (len(paid) != 1 or paid[0]['status'] != 'completed' or paid[0]['paid_atoms'] != 1000
                or paid[0]['refunded_atoms'] != 6000 or before['audit_count'] != 1):
            raise ValueError('Prior inference payment and fraud dispute did not complete')
        if not any(not entry['accepted'] and entry['kind']=='inference' and entry['reason']=='objective replay mismatch: forward evaluation result' for entry in before['settled']):
            raise ValueError('Missing the original refuted output-head claim')
        # Recover the completed driver's replay count from its finalized claims.
        # Every honest claim in that driver calls check() before submission.
        audits, counts = 0, {}
        for entry in before['settled']:
            counts[entry['kind']] = counts.get(entry['kind'],0)+1
            if not entry['accepted'] or entry['kind']=='growth':continue
            height = entry['height']-manifest['params']['challenge_blocks']-1
            txs = wire.rpc(urls[0],'block',{'height':str(height)})['block']['data']['txs']
            envelopes = [protocol.parse_json(base64.b64decode(tx)) for tx in txs or []]
            envelope = next((tx for tx in envelopes if protocol.transaction_id(tx)==entry['id']),None)
            if envelope is None:raise ValueError('Finalized claim cannot be reconstructed from its original block')
            body,_ = protocol.verify(envelope)
            metadata = Metadata(body['metadata'])
            audits += (len(metadata.json(body['record_root'])['traces']) if entry['kind']=='training'
                       else len(forward.trace_roots(metadata,body['record_root'])))
        if counts != {'growth':1,'training':4,'score':74,'inference':2}:
            raise ValueError('Unexpected original settlement history')
        proposal = send(owners[0],'propose_data',data_root=prepared['data_root'],metadata=prepared['metadata'])
        for owner in owners[:3]:send(owner,'vote_data',proposal_id=proposal,approve=True)
        until(lambda:query('/data')['root']==prepared['data_root'],120)
        assert query()['issued']==before['issued'] and query()['training_round']==before['training_round']
        print(json.dumps({'phase':'second_fresh_cohort_activated_after_recovery','data_root':prepared['data_root']}),flush=True)
        stop(3); height=query()['height']; until(lambda:query()['height']>height+2,120)
        stop(2); time.sleep(2); halted=query()['height']; time.sleep(3); assert query()['height']==halted
        start(2); until(lambda:query()['height']>halted+2,120)
        start(3); until(lambda:all(wire.query(url)['height']>halted for url in urls),120)
        height=query()['height']+2
        until(lambda:all(wire.query(url)['height']>=height for url in urls),120)
        headers=[wire.rpc(url,'block',{'height':str(height)})['block']['header'] for url in urls]
        assert len({header['app_hash'] for header in headers})==1
        model=manifest['lifecycle']['initial_model']; candidate=evaluation['candidate_model']
        result={'chain_id':genesis['chain_id'],'source_hash':code_hash(),'genesis_hash':digest(canonical(genesis)),
            'fixture':'real-model','validator_physical_hosts':1,'worker_transport':'http','operators':1,'validators':4,
            'parameters':model['parameters'],'candidate_parameters':candidate['parameters'],
            'growth_layers':candidate['config']['num_hidden_layers']-model['config']['num_hidden_layers'],
            'declared_capacities':[48000000]*4,'fresh_cohorts_activated':2,'training_steps':4,'issued_atoms':4_000_000,
            'serving_decision':evaluation['decision'],'llm_quality_improvement_claimed':False,'inference':paid[0],
            'forged_inference_replay_bytes':inference_phase['replay_bytes'],'independent_stage_replays':audits,
            'replay_count_basis':'reconstructed from finalized honest claims and completed original driver checkpoints',
            'prior_settlement_counts':counts,'one_validator_down_progress':True,'half_voting_power_down_halts':True,
            'restart_catchup_agreement':True,'matching_app_hash_height':height,'matching_app_hash':headers[0]['app_hash'],
            'recovery':{'reason':'second proposal repeated two previously admitted token windows',
                'original_genesis_keys_blocks_and_balances_retained':True,'before_height':before['height'],
                'corrected_data_root':prepared['data_root']}}
        (home/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
    finally:
        for index in range(4):stop(index)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home',type=Path,required=True)
    parser.add_argument('--next-cohort',type=Path,required=True)
    parser.add_argument('--run-log',type=Path,required=True)
    run(parser.parse_args())
