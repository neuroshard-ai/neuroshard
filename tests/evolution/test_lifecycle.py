"""Native lifecycle safety: immutable inputs, adjudication, escrow and recovery."""
import base64
import copy
import hashlib
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from neuroshard.dataflow.store import canonical
from neuroshard.demo import protocol
from neuroshard.evolution import cohorts, forward, lifecycle, settlement as state
from neuroshard.evolution.objects import Objects, digest
from neuroshard.evolution.pipeline import Pipeline, LocalEndpoint
from neuroshard.evolution.worker import Worker
from neuroshard.evolution.verification import Metadata, audit, bundle


def send(s, owner, kind, store=None, **fields):
    return state.transition(s, owner.sign({'chain_id':s['chain_id'], 'nonce':s['accounts'][owner.public_key]['nonce'],
        'kind':kind, **fields}), store, True, audit)


def blocks(s, count):
    for _ in range(count):
        s, _ = state.advance(s, s['height']+1, s['time_ns']+1_000_000_000)
    return s


def fixture_cohort(store, s, index=0):
    documents, windows = [], []
    values = {}
    def add(value):
        key = store.put_json(value)
        values[key] = value
        return key
    for group, role in enumerate(('train','retention','fresh')):
        spec = {'repo':'fixture/'+role, 'revision':'a'*40, 'split':'train', 'license':'CC0-1.0', 'role':role}
        source = add(spec)
        start = s['lifecycle']['cursors'].get(source,0)
        windows.append({'source':source, 'start':start, 'end':start+32})
        for row in range(32):
            batch = {'input_ids':[[1, 3+row, 40+group+index*3, 13, 14, 2]],
                     'labels':[[-100,-100,-100,13,14,2]]}
            documents.append({'id':digest(canonical(['document',index,role,row])), 'source':source,
                'row':start+row, 'object':digest(canonical(['raw-document',index,role,row])),
                'batches':[add(batch)], 'role':role, 'omitted_targets':0})
    key = add({'format':cohorts.FORMAT, 'previous':s['data_root'],
               'tokenizer_root':s['manifest']['lifecycle']['tokenizer_root'], 'windows':windows, 'documents':documents})
    return key, values


@pytest.fixture
def case(seed, tmp_path):
    store, _, model = seed
    model['tokenizer_root'] = 'b'*64
    root = store.put_json(model)
    owners = [protocol.Identity('native-lifecycle-test-'+str(i)) for i in range(4)]
    validators = []
    for i, owner in enumerate(owners):
        key = Ed25519PrivateKey.from_private_bytes(hashlib.sha256(str(i).encode()).digest())
        validators.append({'owner':owner.public_key, 'consensus_key':key.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw).hex(),
                           'bond':2500000, 'liquid':1_000_000_000})
    profile = {'format':lifecycle.FORMAT, 'tokenizer_root':model['tokenizer_root'], 'vocabulary':64,
        'eos_ids':[2], 'initial_model':model, 'source_cursors':{}, 'steps_per_cohort':4,
        'data_vote_blocks':256, 'evaluation_blocks':10000, 'price_per_token':1000}
    manifest = {'params':{**state.PARAMS,'challenge_blocks':2}, 'initial_model_root':root,
        'data_root':'c'*64, 'training_batches':[], 'learning_rate':.003, 'clip_norm':1., 'lifecycle':profile}
    s = state.genesis('lifecycle-tests',validators,manifest)
    return s, owners, store, tmp_path


def activate(case):
    s, owners, store, home = case
    key, values = fixture_cohort(store,s)
    s = send(s,owners[0],'propose_data',data_root=key,metadata=values)
    proposal = s['lifecycle']['proposal']['id']
    for owner in owners[:3]:
        s = send(s,owner,'vote_data',proposal_id=proposal,approve=True)
    s = blocks(s,s['manifest']['params']['activation_blocks'])
    assert s['data_root'] == key
    return s, owners, store, home


def pipeline(store, root, home, session, step=0):
    return Pipeline(store,root,[LocalEndpoint(Worker(home/(session+str(i)),store)) for i in range(2)],
                    [6000]*2,session,start_step=step)


def train_cohort(case):
    s, owners, store, home = activate(case)
    for _ in s['lifecycle']['active']['schedule']:
        s = send(s,owners[0],'reserve',parent=s['model_root'],round=s['training_round'],workers=[o.public_key for o in owners[:2]])
        reserved = s['assignment']
        pipe = pipeline(store,s['model_root'],home,'training'+str(s['training_round']),s['training_round'])
        try:
            value = pipe.train(s['lifecycle']['active']['batches'][reserved['batch']])
        finally:
            pipe.close()
        receipts = [owner.sign({'domain':'neuroshard/evolution/work/v1','chain_id':s['chain_id'],
            'assignment':reserved['id'],'record_root':value['record_root'],'stage':i,'trace_root':value['traces'][i]})
                    for i, owner in enumerate(owners[:2])]
        s = send(s,owners[0],'claim',record_root=value['record_root'],metadata=bundle(store,value['record_root']),
                 workers=receipts,data_root=s['data_root'],sequence_index=reserved['sequence_index'])
        s = blocks(s,3)
    return s, owners, store, home


def test_data_requires_supermajority_delay_and_no_implicit_mining(case):
    s, owners, store, _ = case
    with pytest.raises(ValueError,match='Activate fresh data'):
        send(s,owners[0],'reserve',parent=s['model_root'],round=0,workers=[owners[0].public_key])
    key, values = fixture_cohort(store,s)
    old = s['data_root']
    s = send(s,owners[0],'propose_data',data_root=key,metadata=values)
    proposal = s['lifecycle']['proposal']['id']
    for owner in owners[:2]:
        s = send(s,owner,'vote_data',proposal_id=proposal,approve=True)
    s = blocks(s,17)
    assert s['data_root'] == old and s['issued'] == 0
    with pytest.raises(ValueError,match='one explicit vote'):
        send(s,owners[0],'vote_data',proposal_id=proposal,approve=True)
    s = send(s,owners[2],'vote_data',proposal_id=proposal,approve=True)
    s = blocks(s,1)
    assert s['data_root'] == key and s['issued'] == 0 and not s['lifecycle']['proposal']
    assert len(s['lifecycle']['active']['schedule']) == 4
    state.invariant(s)


@pytest.mark.parametrize('fault',['cursor','source_role','document','batch','incomplete','codec'])
def test_data_rejects_invalid_freshness_or_protected_relabeling(case,fault):
    s, owners, store, _ = case
    key, values = fixture_cohort(store,s)
    value = values[key]
    if fault == 'cursor':value['windows'][0]['start'] = 1
    elif fault == 'source_role':value['documents'][-1]['role'] = 'train'
    elif fault == 'document':value['documents'][-1]['id'] = value['documents'][0]['id']
    elif fault == 'batch':value['documents'][-1]['batches'] = value['documents'][0]['batches']
    elif fault == 'incomplete':value['documents'][-1]['omitted_targets'] = 1
    elif fault == 'codec':value['tokenizer_root'] = 'f'*64
    del values[key]
    key = store.put_json(value)
    values[key] = value
    with pytest.raises(ValueError):
        send(s,owners[0],'propose_data',data_root=key,metadata=values)


def test_unapproved_proposal_expires_and_accounts_for_collateral(case):
    s, owners, store, _ = case
    key, values = fixture_cohort(store,s)
    before = s['accounts'][owners[0].public_key]['balance']
    s = send(s,owners[0],'propose_data',data_root=key,metadata=values)
    s = blocks(s,257)
    assert s['lifecycle']['proposal'] is None and s['lifecycle']['active'] is None
    assert s['accounts'][owners[0].public_key]['balance'] == before-state.PARAMS['fee']-state.PARAMS['claim_bond']//10
    state.invariant(s)


def test_forward_graph_replays_each_partition_and_head(case):
    s, _, store, home = case
    pipe = pipeline(store,s['model_root'],home,'forward')
    try:
        value = pipe.evaluate_record({'input_ids':[[1,3,13,14,2]],'labels':[[-100,-100,13,14,2]]})
        generated = pipe.generate_record([1,3],3,[2])
    finally:
        pipe.close()
    for item in (value,generated):
        metadata = Metadata(forward.bundle(store,item['record_root']))
        for index in range(len(forward.trace_roots(metadata,item['record_root']))):
            assert audit(store,metadata,item['record_root'],index)['valid']
    forward.validate_generation(store,generated['record_root'])


def test_cohort_training_budget_is_finite_and_survives_json_restart(case):
    s, owners, store, _ = train_cohort(case)
    assert s['issued'] == 4_000_000 and s['lifecycle']['active']['step'] == 4
    restored = json.loads(canonical(s))
    state.invariant(restored)
    for saved in (s,restored):
        with pytest.raises(ValueError,match='finish the current cohort evaluation'):
            send(saved,owners[0],'reserve',parent=saved['model_root'],round=4,workers=[owners[0].public_key])
    assert s['serving_root'] != s['model_root']


def test_paired_score_replays_and_incomplete_evaluation_cannot_promote(case):
    s, owners, store, home = train_cohort(case)
    s = send(s,owners[0],'open_evaluation')
    evaluation = s['lifecycle']['evaluation']
    _, batch = cohorts.evaluation_batch(s['lifecycle']['active'],'fresh',0)
    pipe = pipeline(store,evaluation['candidate'],home,'score')
    try:value = pipe.evaluate_record(batch)
    finally:pipe.close()
    metadata = Metadata(forward.bundle(store,value['record_root']))
    assert all(audit(store,metadata,value['record_root'],i)['valid'] for i in range(3))
    s = send(s,owners[0],'score',evaluation_id=evaluation['id'],side='candidate',role='fresh',offset=0,
             record_root=value['record_root'],metadata=metadata.values)
    s = blocks(s,3)
    assert len(s['lifecycle']['evaluation']['measurements']['candidate']['fresh']) == 4
    with pytest.raises(ValueError,match='Every paired document score'):
        send(s,owners[0],'finish_evaluation',evaluation_id=evaluation['id'])
    with pytest.raises(ValueError,match='scored once'):
        send(s,owners[0],'score',evaluation_id=evaluation['id'],side='candidate',role='fresh',offset=0,
             record_root=value['record_root'],metadata=metadata.values)
    state.invariant(s)


def test_evaluation_timeout_returns_to_serving_without_erasing_mining(case):
    s, owners, _, _ = train_cohort(case)
    serving = s['serving_root']
    s = send(s,owners[0],'open_evaluation')
    s['height'] = s['lifecycle']['evaluation']['expires']
    s = blocks(s,1)
    assert s['model_root'] == s['serving_root'] == serving
    assert s['issued'] == 4_000_000 and len(s['paid_work']) == 4
    assert s['lifecycle']['active']['closed']
    assert not s['lifecycle']['evaluations'][-1]['decision']['promote']
    state.invariant(s)


def test_new_cohort_uses_fresh_rows_and_bounded_replay(case):
    s, owners, store, _ = train_cohort(case)
    s = send(s,owners[0],'open_evaluation')
    s['height'] = s['lifecycle']['evaluation']['expires']
    s = blocks(s,1)
    previous = s['lifecycle']['active']
    key, values = fixture_cohort(store,s,1)
    s = send(s,owners[0],'propose_data',data_root=key,metadata=values)
    proposal = s['lifecycle']['proposal']['id']
    for owner in owners[:3]:s = send(s,owner,'vote_data',proposal_id=proposal,approve=True)
    s = blocks(s,16)
    active = s['lifecycle']['active']
    assert active['step'] == 0 and s['training_round'] == 4
    assert sum(key in previous['batches'] for key in active['schedule']) == 1
    assert all(cursor == 64 for cursor in s['lifecycle']['cursors'].values())


def test_paid_generation_pins_serving_root_and_settles_exact_escrow(case):
    s, owners, store, home = case
    before = s['accounts'][owners[0].public_key]['balance']
    s = send(s,owners[0],'infer',model_root=s['serving_root'],provider=owners[1].public_key,
             prompt_ids=[1,3],max_tokens=2,max_price=7000,expires_in=256)
    job = next(iter(s['lifecycle']['jobs'].values()))
    assert s['accounts'][owners[0].public_key]['balance'] == before-8000
    pipe = pipeline(store,job['model_root'],home,'inference')
    try:value = pipe.generate_record(job['prompt_ids'],job['max_tokens'],job['eos_ids'])
    finally:pipe.close()
    s = send(s,owners[1],'respond',job_id=job['id'],record_root=value['record_root'],metadata=forward.bundle(store,value['record_root']))
    s = blocks(s,3)
    result = s['lifecycle']['results'][job['id']]
    assert result['status'] == 'completed' and result['paid_atoms'] == len(value['token_ids'])*1000
    assert s['accounts'][owners[0].public_key]['balance'] == before-1000-result['paid_atoms']
    assert s['issued'] == 0 and not s['lifecycle']['jobs']
    state.invariant(s)


def test_missing_inference_response_refunds_request_not_fee(case):
    s, owners, _, _ = case
    before = s['accounts'][owners[0].public_key]['balance']
    s = send(s,owners[0],'infer',model_root=s['serving_root'],provider=owners[1].public_key,
             prompt_ids=[1,3],max_tokens=2,max_price=7000,expires_in=10)
    s = blocks(s,11)
    assert s['accounts'][owners[0].public_key]['balance'] == before-1000
    assert next(iter(s['lifecycle']['results'].values()))['status'] == 'expired'
    state.invariant(s)


def test_forged_paid_output_is_refuted_from_native_inputs(case):
    s, owners, store, home = case
    s = send(s,owners[0],'infer',model_root=s['serving_root'],provider=owners[1].public_key,
             prompt_ids=[1,3],max_tokens=1,max_price=1000,expires_in=512)
    job = next(iter(s['lifecycle']['jobs'].values()))
    pipe = pipeline(store,job['model_root'],home,'forgery')
    try:value = pipe.generate_record(job['prompt_ids'],1,job['eos_ids'])
    finally:pipe.close()
    generation = store.json(value['record_root'])
    record = store.json(generation['records'][0])
    head = store.json(record['traces'][-1])
    wrong = (generation['token_ids'][0]+1)%64
    head['result']['next_ids'] = [wrong]
    record['result'] = head['result']
    record['traces'][-1] = store.put_json(head)
    generation['records'][0] = store.put_json(record)
    generation['token_ids'] = [wrong]
    key = store.put_json(generation)
    metadata = forward.bundle(store,key)
    forward.validate_generation(Metadata(metadata),key)  # plausible graph, false arithmetic
    s = send(s,owners[1],'respond',job_id=job['id'],record_root=key,metadata=metadata)
    claim_id = s['candidate']['id']
    s = send(s,owners[2],'challenge',claim_id=claim_id,stage=2,challenge_kind='fraud',object_root=None)
    artifacts = Objects(home/'chain-evidence')
    for key in list(s['candidate']['challenge']['needed']):
        s = send(s,owners[2],'upload',store=artifacts,claim_id=claim_id,object_root=key,index=0,data=base64.b64encode(store.get(key)).decode())
        s = send(s,owners[2],'seal',store=artifacts,claim_id=claim_id,object_root=key)
    s = send(s,owners[2],'resolve',store=artifacts,claim_id=claim_id)
    assert s['candidate'] is None and not s['lifecycle']['results']
    assert s['settled'][-1]['reason'] == 'objective replay mismatch: forward evaluation result'
    assert s['lifecycle']['jobs'][job['id']]['claim_id'] is None
    state.invariant(s)


def test_integer_quality_gate_rejects_regression_and_threshold_equality():
    before = [2.0.hex()]*32
    assert cohorts.comparison(before,[1.9.hex()]*32,-1000)['passes']
    assert not cohorts.comparison(before,before,-1000)['passes']
    assert not cohorts.comparison(before,[2.1.hex()]*32,20000)['passes']
    assert not cohorts.comparison(before,[1.999.hex()]*32,-1000)['passes']
    with pytest.raises(ValueError,match='Invalid finite loss'):
        cohorts.comparison(before,['nan']*32,-1000)


def test_absolute_fraud_deadline_does_not_confiscate_publisher_bond(case):
    s, owners, store, home = case
    s = send(s,owners[0],'infer',model_root=s['serving_root'],provider=owners[1].public_key,
             prompt_ids=[1,3],max_tokens=1,max_price=1000,expires_in=100)
    job = next(iter(s['lifecycle']['jobs'].values()))
    pipe = pipeline(store,job['model_root'],home,'expired-accusation')
    try:value = pipe.generate_record([1,3],1,[2])
    finally:pipe.close()
    before = s['accounts'][owners[1].public_key]['balance']
    s = send(s,owners[1],'respond',job_id=job['id'],record_root=value['record_root'],metadata=forward.bundle(store,value['record_root']))
    s = send(s,owners[2],'challenge',claim_id=s['candidate']['id'],stage=2,challenge_kind='fraud',object_root=None)
    s['height'] = s['candidate']['expires']
    s = blocks(s,1)
    assert s['accounts'][owners[1].public_key]['balance'] == before-state.PARAMS['fee']
    assert s['lifecycle']['results'][job['id']]['status'] == 'expired'
    assert s['settled'][-1]['accepted'] is False and s['issued'] == 0
    state.invariant(s)


def test_training_deadline_recovers_without_discarding_data_history(case):
    s, _, _, _ = activate(case)
    s['height'] = s['lifecycle']['active']['expires']
    s = blocks(s,1)
    assert s['lifecycle']['active']['closed']
    assert s['lifecycle']['active']['termination_reason'] == 'cohort training deadline missed'
    assert len(s['lifecycle']['seen_documents']) == 96 and s['issued'] == 0
    state.invariant(s)


@pytest.mark.parametrize('fault',['batch','dependency','stop','early_stop'])
def test_generation_graph_rejects_substitutions(case,fault):
    s, _, store, home = case
    pipe = pipeline(store,s['serving_root'],home,'mutation-'+fault)
    try:value = pipe.generate_record([1,3],2,[])
    finally:pipe.close()
    generation = store.json(value['record_root'])
    if fault == 'batch':generation['prompt_ids'] = [1,4]
    elif fault == 'dependency':generation['records'][1] = generation['records'][0]
    elif fault == 'stop':generation['eos_ids'] = [generation['token_ids'][0]]
    elif fault == 'early_stop':generation['token_ids'].pop();generation['records'].pop()
    key = store.put_json(generation)
    with pytest.raises(ValueError):forward.validate_generation(store,key)


def test_complete_real_arithmetic_scores_drive_native_decision(case):
    s, owners, store, home = train_cohort(case)
    s = send(s,owners[0],'open_evaluation')
    evaluation = s['lifecycle']['evaluation']
    initial = s['serving_root']
    for side in ('baseline','candidate'):
        pipe = pipeline(store,evaluation[side],home,'complete-'+side)
        try:
            for role in ('retention','fresh'):
                for offset in range(0,32,4):
                    _, batch = cohorts.evaluation_batch(s['lifecycle']['active'],role,offset)
                    value = pipe.evaluate_record(batch)
                    s = send(s,owners[0],'score',evaluation_id=evaluation['id'],side=side,role=role,offset=offset,
                             record_root=value['record_root'],metadata=forward.bundle(store,value['record_root']))
                    s = blocks(s,3)
        finally:pipe.close()
    expected = cohorts.decision(s['lifecycle']['evaluation']['measurements'],s['lifecycle']['active'])
    s = send(s,owners[0],'finish_evaluation',evaluation_id=evaluation['id'])
    assert s['lifecycle']['evaluations'][-1]['decision'] == expected
    assert s['serving_root'] == (evaluation['candidate'] if expected['promote'] else initial)
    assert s['model_root'] == s['serving_root'] and s['lifecycle']['active']['closed']
    assert s['issued'] == 4_000_000
    state.invariant(s)


def test_multiwindow_evaluation_weights_targets_and_counts_documents_once(case):
    s, _, store, _ = case
    key, values = fixture_cohort(store,s)
    value = values.pop(key)
    doc = next(d for d in value['documents'] if d['role']=='retention')
    extra = {'input_ids':[[1,3,55,13,14,2]], 'labels':[[-100,-100,-100,-100,-100,2]]}
    extra_root = store.put_json(extra)
    values[extra_root] = extra
    doc['batches'].append(extra_root)
    key = store.put_json(value)
    values[key] = value
    _, roles = cohorts.validate(cohorts.metadata(values),key,s)
    active = {**roles,'batches':{key:values[key] for doc in value['documents'] for key in doc['batches']}}
    assert len(cohorts.evaluation_rows(active,'retention')) == 33
    documents, tail = cohorts.evaluation_batch(active,'retention',32)
    assert len(documents) == len(tail['input_ids']) == 1
    losses = [2.0.hex(),6.0.hex()]+[4.0.hex()]*31
    grouped = cohorts.document_losses(active,'retention',losses)
    assert len(grouped) == 32 and float.fromhex(grouped[0]) == 3.0
    with pytest.raises(ValueError,match='Missing evaluation windows'):
        cohorts.document_losses(active,'retention',losses[:-1])
