import copy,hashlib

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding,PublicFormat

from neuroshard.demo import protocol,work
from neuroshard.lab import state as ledger
from neuroshard.inference import state


@pytest.fixture
def setup():
    owners=[protocol.Identity('llm-validator-'+str(i)) for i in range(4)]
    validators=[]
    for i,owner in enumerate(owners):
        key=Ed25519PrivateKey.from_private_bytes(hashlib.sha256(str(i).encode()).digest())
        validators.append({'owner':owner.public_key,'consensus_key':key.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw).hex(),
                           'bond':250000,'liquid':20000000})
    params={**ledger.PARAMS,'inference_token_price':1000,'inference_blocks':24}
    s=state.genesis('llm-tests',validators,{'params':params,'model':{'max_new_tokens':64}}, {'test':0},5.0)
    return s,owners,protocol.Identity('new-zero-balance-provider')


def tx(s,owner,kind,**fields):
    return owner.sign({'kind':kind,'chain_id':s['chain_id'],'nonce':s['accounts'].get(owner.public_key,{'nonce':0})['nonce'],**fields})


def request(s,owner,provider,price=100000):
    return tx(s,owner,'infer',provider=provider.public_key,model_root=s['serving_root'],
        request={'prompt':'Explain training','max_tokens':16},price=price,expires=s['height']+12)


def oracle(s,task):
    if 'validate_request' in task:return None
    job=task['job']
    return {'text':'A checked response','token_ids':[42],'model_root':job['model_root'],
            'request_root':work.digest(job['request'])}


def apply(s,envelope):
    s,_=state.advance(s,s['height']+1,s['time_ns']+1000000000,
        committers={ledger.consensus_address(k) for k in s['validators']})
    return state.transition(s,envelope,oracle)


def test_request_locks_budget_and_zero_balance_provider_can_settle_without_issuance(setup):
    s,owners,provider=setup
    before=s['accounts'][owners[0].public_key]['balance']
    signed=request(s,owners[0],provider);job_id=protocol.transaction_id(signed)
    s=apply(s,signed)
    assert state.available(s,owners[0].public_key)==before-101000
    output=oracle(s,{'job':s['jobs'][job_id]})
    s=apply(s,tx(s,provider,'respond',job_id=job_id,result_root=work.digest(output)))
    assert s['accounts'][provider.public_key]['balance']==80000
    assert s['issued']==0 and s['round']==0
    assert state.locked(s,owners[0].public_key)==0
    assert s['results'][job_id]['output']==output
    state.invariant(s)


def test_locked_budget_cannot_be_transferred_or_used_for_another_job(setup):
    s,owners,provider=setup
    signed=request(s,owners[0],provider,price=19900000);s=apply(s,signed)
    with pytest.raises(ValueError,match='available balance'):
        apply(s,tx(s,owners[0],'transfer',to=owners[1].public_key,amount=100000))
    with pytest.raises(ValueError,match='available balance'):
        apply(s,request(s,owners[0],provider))


def test_expiry_unlocks_customer_budget_and_provider_cannot_collect_late(setup):
    s,owners,provider=setup
    before=s['accounts'][owners[0].public_key]['balance'];signed=request(s,owners[0],provider)
    job_id=protocol.transaction_id(signed);s=apply(s,signed)
    while s['jobs']:
        s,_=state.advance(s,s['height']+1,s['time_ns']+1000000000)
    assert state.available(s,owners[0].public_key)==before-ledger.PARAMS['fee']
    assert s['results'][job_id]['refunded']==100000
    with pytest.raises(ValueError,match='No live'):
        apply(s,tx(s,provider,'respond',job_id=job_id,result_root='a'*64))


def test_wrong_provider_wrong_result_and_replay_never_pay(setup):
    s,owners,provider=setup;signed=request(s,owners[0],provider);job_id=protocol.transaction_id(signed)
    s=apply(s,signed);original=copy.deepcopy(s)
    with pytest.raises(ValueError,match='No live'):
        apply(s,tx(s,owners[1],'respond',job_id=job_id,result_root='a'*64))
    with pytest.raises(ValueError,match='full replay'):
        apply(s,tx(s,provider,'respond',job_id=job_id,result_root='a'*64))
    assert s==original
    response=tx(s,provider,'respond',job_id=job_id,result_root=work.digest(oracle(s,{'job':s['jobs'][job_id]})))
    s=apply(s,response)
    with pytest.raises(ValueError,match='nonce'):
        apply(s,response)


def test_request_retains_checkpoint_after_training_state_advances(setup):
    s,owners,provider=setup;signed=request(s,owners[0],provider);job_id=protocol.transaction_id(signed)
    s=apply(s,signed);root=s['serving_root']
    s['weights']={'test':1};s['model_root']=work.digest(s['weights'])
    s['serving_weights']={'test':1};s['serving_root']=s['model_root']
    assert s['jobs'][job_id]['model_root']==root
    output=oracle(s,{'job':s['jobs'][job_id]})
    s=apply(s,tx(s,provider,'respond',job_id=job_id,result_root=work.digest(output)))
    assert s['results'][job_id]['output']['model_root']==root


def test_queue_bound_and_wrong_chain_are_rejected(setup):
    s,owners,provider=setup
    for _ in range(state.MAX_JOBS):
        signed=request(s,owners[0],provider)
        s=state.transition(s,signed,oracle)
    with pytest.raises(ValueError,match='queue is full'):
        state.transition(s,request(s,owners[0],provider),oracle)
    bad=request(s,owners[1],provider)['body'];bad['chain_id']='different-chain'
    with pytest.raises(ValueError,match='Wrong chain'):
        state.transition(s,owners[1].sign(bad),oracle)


@pytest.mark.parametrize('envelope',[None,[],42,'tx',{}, {'body':None},{'body':[]}])
def test_malformed_envelopes_are_rejected_without_crashing_application(setup,envelope):
    s,_,_=setup
    with pytest.raises(ValueError):state.transition(s,envelope,oracle)
