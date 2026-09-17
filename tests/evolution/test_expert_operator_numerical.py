"""Two automatic native cohorts with real neural production and replay.

Small synthetic models and local sources isolate the operator integration.
The ledger uses real signed transitions with a controlled block clock, not an
operated CometBFT network. Quality calls run on six actual shard processes.
"""
import copy
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
import torch.distributed as dist
import torch.multiprocessing as mp

from neuroshard.demo import protocol
from neuroshard.evolution import auditing, expert_admission, expert_data, expert_preparation
from neuroshard.evolution import expert_lifecycle as life, expert_work, settlement
from neuroshard.evolution.reference_data import identity, save
from neuroshard.evolution.sharded import expert_execution, graph_quality, prefix_execution
from neuroshard.evolution.sharded.graph_execution import GraphNetwork
from test_expert_data import prepared
from test_expert_admission import renamed
from test_expert_operator import Node, build
from test_expert_lifecycle import send
from test_native_audit_quorum import verdicts
from test_settlement import blocks

SOURCE = Path(__file__).resolve().parents[2]


def serving_owner(rank, home):
    """Keep the accepted shards loaded while another cohort executes."""
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'test'
    home = Path(home)
    graph = json.loads((home/'continuous-serving/graph.json').read_bytes())
    dist.init_process_group('gloo', init_method='file://'+str(home/'continuous-serving/rendezvous'),
                            rank=rank, world_size=3+len(graph['experts']), timeout=timedelta(seconds=120))
    try:
        profile = json.loads((home/'profile.json').read_bytes())
        net = GraphNetwork(graph, profile, objects=home/'objects', interpreter=home/'interpreter',
                           seed=home/'seed', source_home=SOURCE, rank=rank)
        previous = None
        while True:
            request = [None]
            if rank == 0:
                path = home/'continuous-serving/request.json'
                if path.exists():
                    request[0] = json.loads(path.read_bytes())
            dist.broadcast_object_list(request, src=0)
            request = request[0]
            if request and request.get('stop'):
                break
            if request and request['id'] != previous:
                assert request['serving_root'] == identity(graph)
                answer = net.answer('word3 word7', 3)
                destination = home/'continuous-serving'/(request['id']+'.owner-'+str(rank)+'.json')
                save(destination.with_suffix('.partial'), {'context': request, 'answer': answer})
                destination.with_suffix('.partial').replace(destination)
                previous = request['id']
            time.sleep(.1)
    finally:
        dist.destroy_process_group()


def quality_owner(rank, home, request):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'test'
    home, request = Path(home), Path(request)
    value = json.loads(request.read_bytes())
    dist.init_process_group('gloo', init_method='file://'+str(request.with_suffix('.rendezvous')),
                            rank=rank, world_size=6, timeout=timedelta(seconds=120))
    try:
        profile = json.loads((home/'profile.json').read_bytes())
        net = GraphNetwork(value['candidate'], profile, objects=home/'objects', interpreter=home/'interpreter',
                           seed=home/'seed', source_home=SOURCE, rank=rank)
        if value['claim'] is None:
            result = graph_quality.evaluate(value['policy'], request.parent, value['baseline'], value['candidate'], net)
        else:
            report, result = graph_quality.quality_report(value['claim'], value['policy'], request.parent, net)
            assert life.replay_report(value['claim'], report)['valid']
        save(request.with_suffix('.owner-'+str(rank)+'.json'), result)
    finally:
        dist.destroy_process_group()


def test_two_automatic_cohorts_execute_replay_and_reject_quality_without_overwriting_serving(prepared):
    home, _, initial_job, policy, store, tokenizer, upstream, original_plan, _ = prepared
    owners = [protocol.Identity('numerical-operator-'+str(i)) for i in range(4)]
    validators = [{'owner': owner.public_key, 'consensus_key': Ed25519PrivateKey.from_private_bytes(
        hashlib.sha256(('numerical-validator-'+str(i)).encode()).digest()).public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw).hex(), 'bond': settlement.PARAMS['bond_unit'], 'liquid': 10**10}
        for i, owner in enumerate(owners)]
    initial_job = copy.deepcopy(initial_job)
    initial_job['data']['previous'] = None
    manifest = {'params': settlement.PARAMS, 'initial_model_root': initial_job['work']['parent']['state_root'],
        'data_root': identity(initial_job['data']),
        'auditing': {**auditing.QUORUM_PROFILE, 'commit_blocks': 16, 'reveal_blocks': 16},
        'expert_work': initial_job['work'], 'expert_lifecycle': initial_job['lifecycle'],
        'expert_admission': {'format': expert_admission.FORMAT, 'proposal_blocks': 64,
                            'job_blocks': 2048, 'data_policy': identity(policy), 'initial_data': initial_job['data']}}
    node = Node(settlement.genesis('numerical-cohort-operator', validators, manifest))
    serving = node.state['serving_root']
    worker_keys = [protocol.Identity('operator-prefix-'+str(i)) for i in range(3)]
    training_key = protocol.Identity('operator-training')
    contexts, calls, serving_checks = {}, [], []
    serving_home = home/'continuous-serving'
    serving_home.mkdir()
    save(serving_home/'graph.json', initial_job['lifecycle']['serving_graph'])
    serving_processes = mp.spawn(serving_owner, args=(str(home),),
        nprocs=3+len(initial_job['lifecycle']['serving_graph']['experts']), join=False)

    def request_serving(label):
        request = {'id': str(len(serving_checks)), 'phase': label, 'height': node.state['height'],
                   'serving_root': node.state['serving_root']}
        save(serving_home/'next.json', request)
        (serving_home/'next.json').replace(serving_home/'request.json')
        return request

    def check_serving(request):
        paths = [serving_home/(request['id']+'.owner-'+str(i)+'.json')
                 for i in range(len(serving_processes.processes))]
        deadline = time.monotonic()+120
        while not all(path.exists() for path in paths):
            assert all(process.is_alive() for process in serving_processes.processes)
            if time.monotonic() > deadline:
                raise TimeoutError('Accepted serving shards did not answer during the cohort')
            time.sleep(.1)
        responses = [json.loads(path.read_bytes()) for path in paths]
        assert all(row == responses[0] for row in responses)
        if serving_checks:
            assert responses[0]['answer'] == serving_checks[0]['answer']
        serving_checks.append(responses[0])

    def records(source, start, count):
        original = upstream(source, 0, 2)
        pairs = ([('word20 word21', 'word22'), ('word23 word24', 'word25')]
                 if source['role'] == 'train' else [('word26 word27', 'word28'), ('word27 word28', 'word29; word30')])
        new = [{'messages': [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}],
                'stratum': 'single' if index == 0 else 'composed',
                'topics': ['one'] if index == 0 else ['one', 'two'], 'answers': answer.split('; ')}
               for index, (question, answer) in enumerate(pairs)]
        return (original+new)[start:start+count]

    def next_job(state):
        if len(expert_admission.bookkeeping(state)['seen_jobs']) == 2:
            return None
        plan = copy.deepcopy(original_plan)
        plan['previous_graph'] = identity(state['expert_lifecycle']['serving_graph']['descriptor'])
        quality = store.json(initial_job['lifecycle']['quality']['policy_root'])
        windows = [{'source': source, 'count': 2} for source in initial_job['data']['sources'].values()]
        rejected_entry = identity({'windows': windows, 'fixture': 'repeated-original-rows'})
        if not operator.db.execute('SELECT id FROM cohorts WHERE id=?', ('data/'+rejected_entry,)).fetchone():
            try:
                expert_preparation.prepare(state, plan, policy, store, tokenizer,
                    lambda source, start, count: upstream(source, 0, count), windows=windows, batch_size=1)
            except ValueError as error:
                assert 'Fresh selection repeats admitted history' in str(error)
                return {'rejected_data': {'entry': rejected_entry,
                    'review': {'mechanical_checks_passed': False, 'reason': str(error)}}}
            raise AssertionError('Repeated original documents were admitted as fresh data')
        bundle = expert_preparation.prepare(state, plan, policy, store, tokenizer, records, windows=windows, batch_size=1)
        inputs = store.json(bundle['prepared'])
        initial = renamed(initial_job['work']['parent'], initial_job['work']['checkpoint'], expert_data.job_identity(plan, inputs))
        job, review = expert_preparation.seal(state, bundle, initial, initial_job['lifecycle']['candidate_template'],
                                             quality, policy, store, tokenizer, records)
        assert review['mechanical_checks_passed']
        return job

    def context(job):
        key = identity(job)
        if key not in contexts:
            inputs = store.json(job['work']['prepared'])
            plan = store.json(inputs['plan'])
            directory = home/key
            directory.mkdir()
            for role, spec in inputs['roles'].items():
                (directory/(role+'.jsonl')).write_bytes(store.get(spec['sha256']))
            contexts[key] = {'job': job, 'plan': plan, 'prepared': inputs, 'inputs': directory}
        return contexts[key]

    def quality(job, work, label, claim=None):
        ctx = context(job)
        candidate = life.materialize_graph(job['lifecycle']['candidate_template'], work['checkpoint'])
        boundary = ctx['inputs']/'producer'/work['checkpoint']['checkpoint']/('shard-%06d'%work['checkpoint']['step'])
        for path in boundary.glob('*.safetensors'):
            destination = home/'objects'/path.name
            if not destination.exists():
                shutil.copyfile(path, destination)
        policy_value = store.json(job['lifecycle']['quality']['policy_root'])
        for spec in policy_value['roles'].values():
            destination = ctx['inputs']/spec['file']
            if store.path(spec['sha256']).exists():
                destination.write_bytes(store.get(spec['sha256']))
            else:
                shutil.copyfile(home/'quality-inputs'/spec['file'], destination)
        request = ctx['inputs']/(label+'.json')
        save(request, {'policy': policy_value, 'baseline': job['lifecycle']['serving_graph'],
                       'candidate': candidate, 'claim': claim})
        processes = mp.spawn(quality_owner, args=(str(home), str(request)), nprocs=6, join=False)
        try:
            while not processes.join(timeout=60):
                pass
        finally:
            for process in processes.processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=10)
        results = [json.loads(request.with_suffix('.owner-'+str(i)+'.json').read_bytes()) for i in range(6)]
        assert all(result == results[0] for result in results)
        return results[0], candidate

    def execute_backend(request):
        job, phase, work = request['job'], request['phase'], request['work']
        ctx = context(job)
        calls.append((identity(job), phase))
        paths = {'inputs': ctx['inputs'], 'objects': home/'objects', 'bank_home': ctx['inputs']/'unused',
                 'checkpoint_store': ctx['inputs']/'producer', 'max_seconds': 60}
        if phase == 'prefix':
            produced = prefix_execution.produce_features(job['work'], ctx['plan'], ctx['prepared'], **paths)
            ctx['prefix'] = produced
            output = {key: produced[key] for key in ('feature_root', 'batch_roots')}
            return {**output, 'transcript_root': produced['transcript_root'], 'workers': [key.sign(
                expert_work.receipt(node.state['chain_id'], request['assignment'], output, produced['transcript_root'], rank))
                for rank, key in enumerate(worker_keys)]}
        if phase == 'training':
            profile = expert_work.resolve_prefix(job['work'], work['feature_root'], work['batch_roots'])
            paths['bank_home'] = paths['checkpoint_store']/'prefix'/ctx['prefix']['transcript_root']/'rank-2/features'
            produced = expert_execution.produce_training(work['checkpoint'], request['stages'], profile,
                                                         ctx['plan'], ctx['prepared'], **paths)
            return {**produced, 'worker': training_key.sign(expert_work.receipt(node.state['chain_id'],
                request['assignment'], produced['window']['output']['checkpoint'], identity(produced['window']), 0))}
        measured, candidate = quality(job, work, 'publisher-quality')
        report = {'format': life.FORMAT+'/quality', 'policy_root': job['lifecycle']['quality']['policy_root'],
            'baseline_graph': identity(job['lifecycle']['serving_graph']), 'candidate_graph': identity(candidate),
            'prepared': job['lifecycle']['quality']['prepared'], 'passed': measured['decision']['passed'],
            'results_root': identity(measured)}
        claim = {'kind': 'expert_quality', 'graph': candidate, 'model_root': identity(candidate),
                 'executor_root': candidate['executor_root'], 'record_root': None, 'report': report,
                 'baseline_graph': job['lifecycle']['serving_graph'], 'stages': job['lifecycle']['quality']['stages']}
        return {'report': report, 'transcript_root': identity(graph_quality.quality_transcript(claim, measured))}

    def backend(request):
        probe = request_serving(request['phase'])
        result = execute_backend(request)
        check_serving(probe)
        return result

    operator, outbox = build(home/'operator', node, owners, backend, next_job)
    audited, restarted = [], False
    try:
        check_serving(request_serving('initial'))
        for _ in range(80):
            result = operator.tick()
            if result['phase'] in ('no_new_cohort', 'cohort_budget_complete'):
                break
            if result['phase'] == 'quality_rejected' and not restarted:
                operator.close()
                outbox.close()
                operator, outbox = build(home/'operator', node, owners, backend, next_job)
                restarted = True
                check_serving(request_serving('publisher-restarted-after-rejection'))
            if result['phase'] == 'data_rejected':
                check_serving(request_serving('repeated-source-rejected'))
            admission = expert_admission.bookkeeping(node.state)
            if admission['proposal']:
                proposal = admission['proposal']
                review = expert_data.review(node.state, proposal['job'], policy, store, tokenizer, records)
                for owner in owners[1:]:
                    node.state = send(node.state, owner, 'vote_expert_job', proposal_id=proposal['id'],
                                      approve=review['mechanical_checks_passed'], review_root=identity(review))
                node.state = blocks(node.state, node.state['manifest']['params']['activation_blocks'])
            if result['phase'] == 'waiting_for_auditors':
                for owner in owners[1:]:
                    node.state = send(node.state, owner, 'accept_audit', budget_id=result['budget'])
            claim = node.state['candidate']
            if claim:
                active = admission['active']
                job = active['job'] if active else initial_job
                ctx = context(job)
                votes = []
                for index, owner in enumerate(owners[1:]):
                    paths = {'inputs': ctx['inputs'], 'objects': home/'objects',
                             'bank_home': ctx['inputs']/'auditor-bank',
                             'checkpoint_store': ctx['inputs']/('auditor-'+str(index)), 'max_seconds': 60}
                    if claim['kind'] == 'expert_features':
                        report = prefix_execution.execute_features(claim, job['work'], ctx['plan'], ctx['prepared'], **paths)
                    elif claim['kind'] == 'expert_training':
                        profile = expert_work.execution_profile(node.state)
                        paths['bank_home'] = paths['checkpoint_store']/'prefix'/ctx['prefix']['transcript_root']/'rank-2/features'
                        report = expert_execution.execute_training(claim, profile, ctx['plan'], ctx['prepared'], **paths)
                    else:
                        measured, _ = quality(job, node.state['expert_work'], 'audit-quality-'+str(index), claim)
                        votes.append((owner, identity(measured) == claim['report']['results_root']))
                        continue
                    votes.append((owner, expert_work.replay_report(claim, report)['valid']))
                assert all(valid for _, valid in votes)
                node.state = verdicts(node.state, votes)
                node.state = blocks(node.state, node.state['candidate']['deadline']-node.state['height']+1)
                audited.append(claim['kind'])
            assert node.state['serving_root'] == serving
        else:
            raise AssertionError('Automatic native loop failed to reach the end of its feed')
        outcomes = [json.loads(row[0]) for row in operator.db.execute('SELECT outcome FROM cohorts')]
        assert [row['phase'] for row in outcomes] == ['quality_rejected', 'data_rejected', 'quality_rejected']
        assert restarted and len(serving_checks) == 9
        save(home/'continuous-serving/results.json', serving_checks)
        assert audited == ['expert_features', 'expert_training', 'expert_quality']*2
        assert node.state['issued'] == 4*settlement.PARAMS['reward_atoms']
        assert len(expert_admission.bookkeeping(node.state)['trained_documents']) == 4
        assert node.state['manifest'] == manifest
        assert len(calls) == 6
        settlement.invariant(node.state)
    finally:
        operator.close()
        outbox.close()
        save(serving_home/'next.json', {'stop': True})
        (serving_home/'next.json').replace(serving_home/'request.json')
        try:
            serving_processes.join(timeout=10)
        finally:
            for process in serving_processes.processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=10)
