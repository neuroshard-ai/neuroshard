#!/usr/bin/env python3
"""Operate the frozen ordinary-learning campaign on disposable shard owners.

Allocation/setup and the numerical campaign are separate commands so actual
runtime observations can be committed before training. Every command uses the
same absolute cloud retirement deadline. The existing networks are untouched.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from neuroshard.dataflow.store import canonical
from neuroshard.demo import client, protocol
from neuroshard.evolution import auditing, expert_admission, expert_lifecycle as life
from neuroshard.evolution import expert_preparation, ordinary_operation, settlement
from neuroshard.evolution.app import code_hash
from neuroshard.evolution.reference_data import identity, save, sha256
from neuroshard.evolution.transactions import Outbox
from neuroshard.lab.app import native_parameters
from ordinary_allocation import allocate, bootstrap, retire
from ordinary_campaign_backend import Backend
from ordinary_cloud import Cloud
from portable_native_trial import Network

ROOT = Path(__file__).resolve().parents[1]


def source_check(home):
    freeze = json.loads((home/'source-freeze.json').read_bytes())
    if any(sha256(ROOT/path) != digest for path, digest in freeze['sources'].items()):
        raise ValueError('Campaign source differs from its pre-training commitment')
    if any(sha256(home/'compiled'/path) != digest for path, digest in freeze['compiled_files'].items()):
        raise ValueError('Compiled campaign inputs differ from their pre-training commitment')
    if subprocess.check_output(['git', 'rev-parse', freeze['revision']], cwd=ROOT).decode().strip() != freeze['revision']:
        raise ValueError('The declared source commit is unavailable')
    operation = json.loads((home/'operation.json').read_bytes())
    if identity(operation) != freeze['operation']:
        raise ValueError('The complete campaign prescription changed after freezing')
    return freeze, operation


def initial_job(backend):
    """Prepare and review actual initialization, then encode a new genesis.

    The preparation view below is explicitly pre-genesis. It cannot submit
    transactions and is never reported as committed ledger state.
    """
    freeze, store = backend.freeze, backend.store
    graph = store.json(freeze['baseline_graph'])
    quality = store.json(freeze['quality_rule'])
    rows = store.json(freeze['sources']['bootstrap/test']['records'])
    from neuroshard.evolution.data import document_identity
    quality['roles'] = {'test': expert_preparation.record_set(store, 'bootstrap-test',
        [{**row, 'id': document_identity(row['messages'])} for row in rows]), **quality['retention_anchors']}
    quality_root = store.put_json(quality)
    view = {'data_root': identity({'pre_genesis': graph}),
        'manifest': {'auditing': auditing.QUORUM_PROFILE,
            'expert_admission': {'data_policy': freeze['data_policy']},
            'expert_lifecycle': {'format': life.PROSPECTIVE, 'serving_graph': graph,
                'price_per_token': 1, 'max_tokens': 64, 'quality': {'policy_root': quality_root}}},
        'expert_lifecycle': {'serving_graph': graph, 'history': [], 'quality_closed': True,
            'admission': {'active': None, 'proposal': None, 'seen_jobs': {}, 'cursors': {},
                          'seen_documents': {}, 'trained_documents': {}}}}
    job = backend.preparer().prepare(view, freeze['entries'][0])
    job = copy.deepcopy(job)
    job['data']['previous'] = None
    save(backend.home/'initial-job.json', job)
    return job


def create_network(backend, engine):
    backend.publish_metadata()
    backend.advance_feed(0)
    job = initial_job(backend)
    backend.publish_metadata()
    params = {**settlement.PARAMS, 'steps_per_period': 4096,
              'lease_blocks': 7200, 'max_claim_blocks': 20000}
    manifest = {'params': params, 'initial_model_root': job['work']['parent']['state_root'],
        'data_root': identity(job['data']), 'code_hash': code_hash(),
        'native_consensus': {**native_parameters(params), 'block_max_bytes': 4*1024**2},
        'auditing': {**auditing.QUORUM_PROFILE, 'commit_blocks': 10000, 'reveal_blocks': 64},
        'expert_work': job['work'], 'expert_lifecycle': job['lifecycle'],
        'expert_admission': {'format': expert_admission.FORMAT, 'proposal_blocks': 1024,
            'job_blocks': 200000, 'data_policy': backend.freeze['data_policy'], 'initial_data': job['data']}}
    network = Network.create(backend.home/'native', manifest, engine=engine, base_port=39950)
    genesis_path = Path(network.config['nodes'][0]['home'])/'config/genesis.json'
    save(backend.home/'native.json', {'home': network.config['nodes'][0]['home'],
        'genesis_sha256': sha256(genesis_path), 'genesis_root': identity(network.genesis),
        'rpc': network.urls[0], 'chain_id': network.genesis['chain_id']})
    save(backend.home/'native-manifest.json', manifest)
    try:
        network.start()
    except BaseException:
        network.close()
        raise
    return network


def commands(backend, network):
    native = json.loads((backend.home/'native.json').read_bytes())
    base = [sys.executable, str(ROOT/'scripts/ordinary_campaign_backend.py'), '--home', str(backend.home)]
    publisher_backend = {'argv': [*base, '--actor', '0'], 'timeout_seconds': 14400}
    config = {'workers': {'prefix': [key.public_key for key in backend.keys],
                         'training': backend.training_key.public_key},
        'execution': publisher_backend, 'preparation': publisher_backend,
        'limits': {'max_jobs': 5, 'max_steps': 128, 'max_attempts': 2, 'window_updates': 4}}
    save(backend.home/'publisher-config.json', config)
    publisher = [sys.executable, '-m', 'neuroshard.evolution.expert_operator', '--home', str(backend.home/'publisher'),
        '--native-home', native['home'], '--genesis-sha256', native['genesis_sha256'], '--rpc', native['rpc'],
        '--key', str(network.home/'owner-0.key'), '--config', str(backend.home/'publisher-config.json'), '--once']
    auditors = []
    for actor in range(1, 4):
        execution = {'argv': [*base, '--actor', str(actor), '--audit'], 'timeout_seconds': 14400}
        review = {'argv': [*base, '--actor', str(actor)], 'timeout_seconds': 14400}
        save(backend.home/('auditor-'+str(actor)+'-execution.json'), execution)
        save(backend.home/('auditor-'+str(actor)+'-curation.json'), review)
        auditors.append([sys.executable, '-m', 'neuroshard.evolution.audit_worker', '--home', str(backend.home/('auditor-'+str(actor))),
            '--rpc', native['rpc'], '--genesis-sha256', native['genesis_root'],
            '--native-home', native['home'], '--native-genesis-sha256', native['genesis_sha256'],
            '--key', str(network.home/('owner-'+str(actor)+'.key')), '--max-stages', '4096',
            '--objects', str(backend.home/'compiled/objects'),
            '--sponsor', network.owners[0].public_key,
            '--execution-backend', str(backend.home/('auditor-'+str(actor)+'-execution.json')),
            '--admission-backend', str(backend.home/('auditor-'+str(actor)+'-curation.json'))])
    return publisher, auditors


def paid_inference(backend, network):
    """After publisher shutdown, reuse its sole durable outbox for one reply."""
    from neuroshard.evolution import answering
    from neuroshard.evolution.sharded.graph_service import inference_transcript
    state = backend.state()
    graph = state['expert_lifecycle']['serving_graph']
    owners = [protocol.Identity.load_or_create(backend.home/'keys'/('serving-'+str(rank)+'.key'))
              for rank in range(3+len(graph['experts']))]
    box = Outbox(backend.home/'publisher/outbox.sqlite', network.urls[0], state['chain_id'], network.owners[0])
    try:
        quote = answering.quote(graph, 64, 1)
        messages = [{'role': 'user', 'content': 'What is the name of the NeuroShard client package?'}]
        box.send('ordinary-paid/request', 'infer_expert', graph=identity(graph), question=messages, max_tokens=64,
            workers=[owner.public_key for owner in owners], max_price=quote['maximum_atoms'], expires_in=10000)
        job_id = box.logical_id('ordinary-paid/request')
        job = backend.state()['expert_lifecycle']['jobs'][job_id]
        response = backend.cloud.query(backend.serving(backend.state()), {'id': identity({'paid': job_id}),
            'kind': 'generate', 'graph': identity(graph), 'question': messages, 'max_tokens': 64})
        if response['status'] != 'completed':
            raise ValueError('Paid ordinary inference was unavailable')
        value = response['result']
        claim = {'kind': 'expert_inference', 'job_id': job_id, 'graph': graph, 'model_root': identity(graph),
            'executor_root': graph['executor_root'], 'request': job['request'], 'outputs': value['outputs'],
            'text': value['text'], 'response': value['answering'], 'stages': value['answering']['generated_tokens'],
            'record_root': None}
        transcript = identity(inference_transcript(claim, value))
        box.send('ordinary-paid/fund', 'fund_audit', publisher=network.owners[0].public_key,
                 auditors=[], stage_limit=claim['stages'], expires_in=100000)
        budget = box.logical_id('ordinary-paid/fund')
        network.until(lambda: auditing.enough(backend.state()['auditing']['budgets'][budget], lambda row: bool(row['bond'])), seconds=180)
        box.send('ordinary-paid/respond', 'respond_answering', job_id=job_id, response=value['answering'],
            transcript_root=transcript, audit_budget=budget, workers=[owner.sign(
                life.answering_receipt(state['chain_id'], job, value['answering'], transcript, rank))
                for rank, owner in enumerate(owners)])
        network.until(lambda: job_id in backend.state()['expert_lifecycle']['results'], seconds=1800)
        outcome = backend.state()['expert_lifecycle']['results'][job_id]
        if outcome.get('status') != 'completed' or backend.state()['issued'] != state['issued']:
            raise ValueError('Paid inference did not settle with zero extra issuance')
        save(backend.home/'paid-inference.json', {'request': messages, 'response': response, 'settlement': outcome,
            'issued_before': state['issued'], 'issued_after': backend.state()['issued']})
        return outcome
    finally:
        box.close()


def operate(home, engine):
    source_check(home)
    backend = Backend(home)
    runtime = json.loads((home/'runtime.json').read_bytes())
    if backend.profile['runtime'] != runtime:
        raise ValueError('Commit the observed numerical runtime before training')
    if (home/'native').exists():
        network = Network(home/'native')
        # A supervisor may be restarted only after its earlier owned processes
        # have exited. CometBFT retains its own signing and application journals.
        old = json.loads((home/'native/processes.json').read_bytes()) if (home/'native/processes.json').exists() else {'pids': []}
        if any((Path('/proc')/str(pid)).exists() for pid in old['pids']):
            raise ValueError('A previous native supervisor still owns live processes')
        try:
            network.start()
        except BaseException:
            network.close()
            raise
    else:
        network = create_network(backend, engine)
    auditors, publisher = [], None
    launches, probes, last_probe = 0, 0, 0.
    try:
        publisher_command, auditor_commands = commands(backend, network)
        for actor, command in enumerate(auditor_commands, 1):
            with (home/('auditor-'+str(actor)+'.log')).open('ab') as log:
                auditors.append(subprocess.Popen(command, stdout=log, stderr=log, cwd=ROOT))
        backend.probe(backend.state(), 'initial-native-serving')
        while backend.cloud.remaining(900) > 0:
            state = backend.state()
            sequence = ordinary_operation.outcome_sequence(state, backend.freeze['entries'])
            save(home/'progress.json', {'phase': 'operating', 'time': datetime.now(timezone.utc).isoformat(),
                'height': state['height'], 'sequence': sequence, 'issued_atoms': state['issued'],
                'serving_root': state['serving_root'], 'current_step': state['expert_work']['checkpoint']['step'],
                'publisher_process_starts': launches, 'serving_probes': probes})
            if sequence['status'] in ('complete', 'quality_stop') or (home/'comparison-stop.json').exists():
                if publisher is not None and publisher.poll() is None:
                    publisher.wait(timeout=90)
                # Record the last native quality outcome in the publisher's
                # durable journal before handing that sole signer to inference.
                with (home/'publisher.log').open('ab') as log:
                    subprocess.run(publisher_command, stdout=log, stderr=log, cwd=ROOT, check=True, timeout=180)
                launches += 1
                break
            if any(process.poll() is not None for process in auditors):
                raise RuntimeError('A configured native auditor stopped')
            if publisher is None or publisher.poll() is not None:
                if publisher is not None and publisher.returncode:
                    raise RuntimeError('The native publisher process failed')
                with (home/'publisher.log').open('ab') as log:
                    publisher = subprocess.Popen(publisher_command, stdout=log, stderr=log, cwd=ROOT)
                launches += 1
            if time.monotonic()-last_probe > 120:
                backend.probe(state, 'during-native-work-'+str(probes))
                probes += 1
                last_probe = time.monotonic()
            time.sleep(1)
        else:
            raise TimeoutError('The campaign stopped at its bounded retirement margin')
        state = backend.state()
        sequence = ordinary_operation.outcome_sequence(state, backend.freeze['entries'])
        if sequence['status'] == 'complete' and not (home/'comparison-stop.json').exists():
            paid_inference(backend, network)
            state = backend.state()
        settlement.invariant(state)
        save(home/'final-state.json', state)
        height = state['height']+1
        network.until(lambda: all(int(client.rpc(url, 'status')['sync_info']['latest_block_height']) >= height for url in network.urls), seconds=60)
        headers = [client.rpc(url, 'block', {'height': str(height)})['block']['header'] for url in network.urls]
        if any(header['app_hash'].lower() != identity(state) for header in headers):
            raise ValueError('The four native validators did not commit the same final state')
        save(home/'matching-headers.json', headers)
        result = {'sequence': sequence, 'publisher_process_starts': launches, 'probes_during_work': probes,
            'issued_atoms': state['issued'], 'trained_documents': len(expert_admission.bookkeeping(state)['trained_documents']),
            'serving_root': state['serving_root'], 'state_root': identity(state), 'height': state['height'],
            'manifest_unchanged': state['manifest'] == json.loads((home/'native-manifest.json').read_bytes()),
            'scope': 'Four native validators under one administrator; actual GPU shards on seven disposable hosts.'}
        save(home/'result.json', result)
        return result
    finally:
        if publisher is not None and publisher.poll() is None:
            publisher.terminate()
            try:
                publisher.wait(timeout=30)
            except subprocess.TimeoutExpired:
                publisher.kill()
                publisher.wait(timeout=10)
        for process in auditors:
            if process.poll() is None:
                process.terminate()
        for process in auditors:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        network.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--phase', choices=('allocate', 'bootstrap', 'run', 'retire'), required=True)
    parser.add_argument('--engine', default='/home/ubuntu/neuroshard-native-candidate/.neuroshard/tools/cometbft')
    args = parser.parse_args()
    os.environ['PYTHONPATH'] = str(ROOT/'src')
    args.home = args.home.resolve()
    if args.phase == 'retire':
        print(json.dumps(retire(args.home)))
    elif args.phase == 'allocate':
        frozen, operation = source_check(args.home)
        print(json.dumps({'instances': len(allocate(args.home, operation['resources'], frozen['revision'])['instances'])}))
    elif args.phase == 'bootstrap':
        source_check(args.home)
        try:
            print(json.dumps(bootstrap(args.home, json.loads((args.home/'catalog.json').read_bytes()))))
        except BaseException:
            retire(args.home)
            raise
    else:
        try:
            print(json.dumps(operate(args.home, args.engine)))
        finally:
            retire(args.home)
