import base64
import copy
import hashlib

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding,PublicFormat

from neuroshard.demo import protocol
from neuroshard.evolution import settlement as state
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.pipeline import Pipeline,LocalEndpoint
from neuroshard.evolution.worker import Worker
from neuroshard.evolution.verification import bundle,Metadata,audit,validate_record


@pytest.fixture
def scenario(seed,tmp_path):
    store,root,model=seed
    owners=[protocol.Identity(f'evolution-test-{i}') for i in range(4)]
    validators=[]
    for i,owner in enumerate(owners):
        key=Ed25519PrivateKey.from_private_bytes(hashlib.sha256(str(i).encode()).digest())
        validators.append({'owner':owner.public_key,'consensus_key':key.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw).hex(),'bond':250000,'liquid':1000000000})
    pipe=Pipeline(store,root,[LocalEndpoint(Worker(tmp_path/f'w{i}',store)) for i in range(2)],[6000]*2,'settlement')
    record=pipe.train([[1,12,13,14,2]])
    pipe.close()
    manifest={'params':state.PARAMS,'initial_model_root':root,'data_root':'a'*64,'training_batches':[record['batch']], 'learning_rate':.003,'clip_norm':1.}
    s=state.genesis('evolution-tests',validators,manifest)
    s=state.transition(s,tx(s,owners[0],'reserve',parent=s['model_root'],round=0,workers=[o.public_key for o in owners[:2]]))
    return s,owners,store,record,Objects(tmp_path/'consensus-artifacts')


def tx(s,owner,kind,**fields):
    return owner.sign({'kind':kind,'chain_id':s['chain_id'],'nonce':s['accounts'].get(owner.public_key,{'nonce':0})['nonce'],**fields})


def claim(s,owners,store,record):
    receipts=[owner.sign({'domain':'neuroshard/evolution/work/v1','chain_id':s['chain_id'],'assignment':s['assignment']['id'],'record_root':record['record_root'], 'stage':i,'trace_root':record['traces'][i]}) for i,owner in enumerate(owners[:2])]
    return tx(s,owners[0],'claim',record_root=record['record_root'],metadata=bundle(store,record['record_root']),workers=receipts,data_root=s['data_root'],sequence_index=0)


def blocks(s,count):
    for _ in range(count):
        s,_=state.advance(s,s['height']+1,s['time_ns']+1000000000)
    return s


def test_normal_training_settles_without_calling_neural_referee(scenario):
    s,owners,store,record,artifacts=scenario
    def forbidden(*_):raise AssertionError('Ordinary claims must not execute a neural model')
    signed=claim(s,owners,store,record)
    s=state.transition(s,signed,artifacts,referee=forbidden)
    assert s['model_root']==record['parent'] and s['issued']==0
    s=blocks(s,state.PARAMS['challenge_blocks']+1)
    assert s['model_root']==record['model_root'] and s['issued']==1000000
    assert s['serving_root']==record['parent'] # quality approval remains separate
    with pytest.raises(ValueError,match='nonce'):
        state.transition(s,signed,artifacts)
    state.invariant(s)


def test_missing_data_rejects_claim_and_never_pays_training(scenario):
    s,owners,store,record,artifacts=scenario
    s=state.transition(s,claim(s,owners,store,record))
    target=store.json(record['model_root'])['components']['embed']['root']
    s=state.transition(s,tx(s,owners[2],'challenge',claim_id=s['candidate']['id'],stage=0,challenge_kind='availability',object_root=target))
    s=blocks(s,state.PARAMS['availability_blocks']+1)
    assert s['issued']==0 and s['candidate'] is None
    assert s['settled'][-1]['reason']=='data availability deadline missed'
    state.invariant(s)


def test_corrupt_update_is_refuted_from_native_uploaded_inputs(scenario):
    s,owners,store,record,artifacts=scenario
    trace=copy.deepcopy(store.json(record['traces'][0]))
    parent=store.json(record['parent'])
    trace['components']['embed']=parent['components']['embed']
    forged=copy.deepcopy(store.json(record['record_root']))
    forged['traces'][0]=store.put_json(trace)
    changed=store.json(record['model_root'])
    changed['components']['embed']=parent['components']['embed']
    forged['model_root']=store.put_json(changed)
    forged['record_root']=store.put_json(forged)
    # Graph structure alone cannot establish whether a claimed gradient update
    # happened. Only the bounded objective replay rejects this counterfeit.
    assert validate_record(store,forged['record_root'])['valid']
    s=state.transition(s,claim(s,owners,store,forged))
    claim_id=s['candidate']['id']
    s=state.transition(s,tx(s,owners[2],'challenge',claim_id=claim_id,stage=0,challenge_kind='fraud',object_root=None))
    with pytest.raises(ValueError,match='first be published'):
        state.transition(s,tx(s,owners[2],'resolve',claim_id=claim_id),artifacts,True,audit)
    for key in list(s['candidate']['challenge']['needed']):
        raw=store.get(key)
        # All tiny fixture objects fit in one native chunk.
        assert len(raw)<state.CHUNK_BYTES
        upload=tx(s,owners[2],'upload',claim_id=claim_id,object_root=key,index=0,data=base64.b64encode(raw).decode())
        s=state.transition(s,upload,artifacts,True,audit)
        s=state.transition(s,tx(s,owners[2],'seal',claim_id=claim_id,object_root=key),artifacts,True,audit)
    s=state.transition(s,tx(s,owners[2],'resolve',claim_id=claim_id),artifacts,True,audit)
    assert s['candidate'] is None and s['issued']==0 and s['audit_count']==1
    assert 'optimizer update' in s['settled'][-1]['reason']
    state.invariant(s)


def test_third_party_cannot_poison_availability_upload(scenario):
    s,owners,store,record,artifacts=scenario
    s=state.transition(s,claim(s,owners,store,record))
    target=store.json(record['model_root'])['components']['embed']['root']
    claim_id=s['candidate']['id']
    s=state.transition(s,tx(s,owners[2],'challenge',claim_id=claim_id,stage=0,challenge_kind='availability',object_root=target))
    with pytest.raises(ValueError,match='evidence publisher'):
        state.transition(s,tx(s,owners[3],'upload',claim_id=claim_id,object_root=target,index=0,data=base64.b64encode(b'poison').decode()),artifacts,True,audit)


def test_budget_renews_by_period_without_resetting_training_history(scenario):
    s,owners,store,record,artifacts=scenario
    s['period_steps']=s['manifest']['params']['steps_per_period']
    with pytest.raises(ValueError,match='budget exhausted'):
        state.transition(s,claim(s,owners,store,record))
    # Advance to the next budget boundary, preserving the native state.
    s['height']=state.PARAMS['budget_period_blocks']-1
    s['assignment']['expires']=s['height']+10
    s,_=state.advance(s,s['height']+1,s['time_ns']+1000000000)
    assert s['period_steps']==0 and s['period']==1
    assert state.transition(s,claim(s,owners,store,record))['candidate']


def test_native_growth_settles_without_minting_and_next_session_continues_round(scenario,tmp_path):
    from neuroshard.evolution.model import grow
    s,owners,store,record,artifacts=scenario
    s=state.transition(s,claim(s,owners,store,record))
    s=blocks(s,state.PARAMS['challenge_blocks']+1)
    parent=s['model_root']
    grown,value=grow(parent,store,2)
    metadata={parent:store.json(parent),grown:value}
    s=state.transition(s,tx(s,owners[0],'grow',parent=parent,model_root=grown,metadata=metadata,capacities=[6000]*3))
    s=blocks(s,state.PARAMS['challenge_blocks']+1)
    assert s['model_root']==grown and s['serving_root']==record['parent']
    assert s['issued']==1000000 and s['training_round']==1 and s['period_growths']==1
    assert s['settled'][-1]['kind']=='growth'
    with pytest.raises(ValueError,match='growth budget'):
        state.transition(s,tx(s,owners[0],'grow',parent=grown,model_root=grown,metadata=metadata,capacities=[6000]*3))
    # A fresh worker session after architecture change uses the ledger's round,
    # rather than silently starting again at round zero.
    workers=[LocalEndpoint(Worker(tmp_path/f'grown-worker{i}',store)) for i in range(2)]
    pipe=Pipeline(store,grown,workers,[9000]*2,'after-growth',start_step=1,journal=tmp_path/'grown.json')
    second=pipe.train(store.json(record['batch']))
    pipe.close()
    assert second['step']==1
    assert all(audit(store,Metadata(bundle(store,second['record_root'])),second['record_root'],i)['valid'] for i in range(2))
    s=state.transition(s,tx(s,owners[0],'reserve',parent=grown,round=1,workers=[o.public_key for o in owners[:2]]))
    s=state.transition(s,claim(s,owners,store,second))
    s=blocks(s,state.PARAMS['challenge_blocks']+1)
    assert s['model_root']==second['model_root'] and s['issued']==2000000
    state.invariant(s)


def test_native_growth_fraud_uses_only_last_parent_block(scenario):
    from neuroshard.evolution.model import grow,torch
    s,owners,store,record,artifacts=scenario
    s=state.transition(s,claim(s,owners,store,record))
    s=blocks(s,state.PARAMS['challenge_blocks']+1)
    parent=s['model_root']
    _,value=grow(parent,store,2)
    component=copy.deepcopy(value['components']['block_004'])
    tensors=store.tensors(component['root'])
    tensors['mlp.down_proj.weight']=torch.ones_like(tensors['mlp.down_proj.weight'])
    component['root']=store.put_tensors(tensors)
    for name in ('block_004','block_005'):value['components'][name]=component
    forged=store.put_json(value)
    s=state.transition(s,tx(s,owners[0],'grow',parent=parent,model_root=forged,
        metadata={parent:store.json(parent),forged:value},capacities=[6000]*3))
    claim_id=s['candidate']['id']
    s=state.transition(s,tx(s,owners[3],'challenge',claim_id=claim_id,stage=0,challenge_kind='fraud',object_root=None))
    needed=s['candidate']['challenge']['needed']
    assert needed==[store.json(parent)['components']['block_003']['root']]
    key=needed[0]
    s=state.transition(s,tx(s,owners[3],'upload',claim_id=claim_id,object_root=key,index=0,data=base64.b64encode(store.get(key)).decode()),artifacts,True,audit)
    s=state.transition(s,tx(s,owners[3],'seal',claim_id=claim_id,object_root=key),artifacts,True,audit)
    s=state.transition(s,tx(s,owners[3],'resolve',claim_id=claim_id),artifacts,True,audit)
    assert s['model_root']==parent and s['period_growths']==0 and s['issued']==1000000
    assert s['settled'][-1]['reason']=='objective replay mismatch: identity growth'
    state.invariant(s)


def test_converged_weights_cannot_mint_again_under_new_ancestry(scenario,tmp_path):
    from neuroshard.evolution.model import torch
    from neuroshard.evolution.verification import work_identity
    s,owners,store,record,_=scenario
    model=copy.deepcopy(store.json(record['parent']))
    for c in model['components'].values():
        c['root']=store.put_tensors({name:torch.zeros_like(value) for name,value in store.tensors(c['root']).items()})
    zero_root=store.put_json(model)
    manifest={**s['manifest'],'initial_model_root':zero_root}
    entries=[{'owner':v['owner'],'consensus_key':key,'bond':v['amount'],'liquid':1000000000}
             for key,v in s['validators'].items()]
    s=state.genesis('converged-model-test',entries,manifest)
    pipe=Pipeline(store,zero_root,[LocalEndpoint(Worker(tmp_path/f'zero{i}',store)) for i in range(2)],[6000]*2,'zero')
    first=pipe.train(store.json(record['batch']))
    second=pipe.train(store.json(record['batch']))
    pipe.close()
    assert first['parent']!=second['parent']
    assert store.json(first['model_root'])['components']==store.json(second['model_root'])['components']==model['components']
    assert work_identity(store,first['record_root'])==work_identity(store,second['record_root'])
    s=state.transition(s,tx(s,owners[0],'reserve',parent=zero_root,round=0,workers=[o.public_key for o in owners[:2]]))
    s=state.transition(s,claim(s,owners,store,first))
    s=blocks(s,state.PARAMS['challenge_blocks']+1)
    s=state.transition(s,tx(s,owners[0],'reserve',parent=s['model_root'],round=1,workers=[o.public_key for o in owners[:2]]))
    with pytest.raises(ValueError,match='already been paid'):
        state.transition(s,claim(s,owners,store,second))
    assert s['issued']==1000000 and len(s['paid_work'])==1
