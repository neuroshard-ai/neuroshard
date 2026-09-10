"""Native model jobs: spend locks, bounded independent inference, and adapter training.

The v2 bonded ledger remains the accounting/consensus primitive. Inference budgets
stay in account totals under a native spend lock until settlement or expiry. Every
spending transition enforces those locks; no outside payment or admission service.
"""
import copy

from neuroshard.demo import protocol, work
from neuroshard.lab import state as ledger


MAX_JOBS = 32
MAX_RESULTS = 128


def locked(state, owner):
    return sum(job['price'] for job in state['jobs'].values() if job['owner'] == owner)


def available(state, owner):
    return state['accounts'].get(owner, {'balance':0})['balance'] - locked(state, owner)


def invariant(state):
    ledger.invariant(state)
    assert len(state['jobs']) <= MAX_JOBS
    assert all(available(state, owner) >= 0 for owner in state['accounts'])
    assert len(state['results']) <= MAX_RESULTS
    assert work.digest(state['weights']) == state['model_root']
    assert work.digest(state['serving_weights']) == state['serving_root']
    for job in state['jobs'].values():
        assert work.digest(job['weights']) == job['model_root']


def genesis(chain_id, validators, spec, initial, validation_loss):
    state = ledger.genesis(chain_id, validators, spec)
    root = work.digest(initial)
    state.update(weights=initial, model_root=root, serving_weights=copy.deepcopy(initial),serving_root=root,
        validation_loss_hex=float(validation_loss).hex(), serving_loss_hex=float(validation_loss).hex(),
        jobs={}, results={}, last_training_evaluation=None)
    invariant(state)
    return state


def advance(previous, height, time_ns, evidence=(), committers=None):
    state, updates = ledger.advance(previous,height,time_ns,evidence,committers)
    for job_id, job in list(state['jobs'].items()):
        if height > job['expires']:
            remember(state,job_id,{'status':'expired','owner':job['owner'],'provider':job['provider'],
                'model_root':job['model_root'],'refunded':job['price'],'height':height})
            del state['jobs'][job_id]
    invariant(state)
    return state, updates


def remember(state, key, value):
    state['results'][key] = value
    while len(state['results']) > MAX_RESULTS:
        oldest = min(state['results'], key=lambda k:(state['results'][k]['height'],k))
        del state['results'][oldest]


def checked(previous, envelope, fields):
    body,owner = protocol.verify(envelope)
    ledger.public_key(owner)
    if set(body) != {'kind','chain_id','nonce'} | fields:
        raise ValueError('Invalid transaction schema')
    if body['chain_id'] != previous['chain_id']:
        raise ValueError('Wrong chain')
    nonce=ledger.integer(body['nonce'])
    if nonce != previous['accounts'].get(owner,{'nonce':0})['nonce']:
        raise ValueError('Wrong account nonce')
    return body,owner


def transition(previous,envelope,execute):
    if not isinstance(envelope,dict) or not isinstance(envelope.get('body'),dict):
        raise ValueError('Invalid signed transaction envelope')
    kind=envelope['body'].get('kind')
    params=ledger.parameters(previous)
    if kind in ('transfer','bond','unbond','withdraw','reserve','submit'):
        body,owner=protocol.verify(envelope)
        spend=params['fee']
        if kind in ('transfer','bond'):
            spend += ledger.integer(body.get('amount'),1)
        if kind == 'reserve':
            if body.get('task_kind') != 'train':
                raise ValueError('Use an independent inference request')
            spend += params['reservation_bond']
        if available(previous,owner) < spend:
            raise ValueError('Insufficient available balance; inference funds are locked')
        execution=[]
        def replay(candidate):
            value=execute(candidate,None);execution.append(value);return value
        state=ledger.transition(previous,envelope,replay)
        if kind == 'submit':
            value=execution[-1]
            evaluation=value['validation_loss_hex']
            state['validation_loss_hex']=evaluation
            promoted=float.fromhex(evaluation) < float.fromhex(state['serving_loss_hex'])
            if promoted:
                state['serving_root']=state['model_root']
                state['serving_weights']=copy.deepcopy(state['weights'])
                state['serving_loss_hex']=evaluation
            state['last_training_evaluation']={'height':state['height'],'round':state['round'],
                'model_root':state['model_root'],'validation_loss_hex':evaluation,'promoted':promoted}
        invariant(state)
        return state
    if kind == 'infer':
        body,owner=checked(previous,envelope,{'provider','model_root','request','price','expires'})
        ledger.public_key(body['provider'])
        request=body['request']
        if (not isinstance(request,dict) or set(request) != {'prompt','max_tokens'}
                or not isinstance(request['prompt'],str) or not 1 <= len(request['prompt'].encode()) <= 2048):
            raise ValueError('Invalid inference prompt')
        tokens=ledger.integer(request['max_tokens'],1,previous['manifest']['model']['max_new_tokens'])
        price=ledger.integer(body['price'],tokens*params['inference_token_price'])
        expires=ledger.integer(body['expires'],previous['height']+2,previous['height']+params['inference_blocks'])
        if body['model_root'] != previous['serving_root']:
            raise ValueError('Request must pin the current serving checkpoint')
        if len(previous['jobs']) >= MAX_JOBS:
            raise ValueError('Inference queue is full')
        if available(previous,owner) < price+params['fee']:
            raise ValueError('Insufficient available balance for inference budget and fee')
        # Tokenizer validation is cheap and part of the frozen execution profile.
        execute(previous,{'validate_request':request})
        state=copy.deepcopy(previous);sender=ledger.account(state,owner)
        sender['nonce']+=1;sender['balance']-=params['fee'];state['burned']+=params['fee']
        job_id=protocol.transaction_id(envelope)
        state['jobs'][job_id]={'owner':owner,'provider':body['provider'],'model_root':body['model_root'],
            'weights':copy.deepcopy(state['serving_weights']),'request':request,'price':price,
            'expires':expires,'created_height':state['height']}
    elif kind == 'respond':
        body,owner=checked(previous,envelope,{'job_id','result_root'})
        job=previous['jobs'].get(body['job_id'])
        if not job or job['provider'] != owner or previous['height'] > job['expires']:
            raise ValueError('No live inference request for this provider')
        if not isinstance(body['result_root'],str) or len(body['result_root']) != 64:
            raise ValueError('Invalid inference result commitment')
        result=execute(previous,{'job_id':body['job_id'],'job':job})
        if body['result_root'] != work.digest(result):
            raise ValueError('Inference result differs from full replay')
        state=copy.deepcopy(previous)
        ledger.account(state,owner)['nonce']+=1
        ledger.account(state,job['owner'])['balance']-=job['price']
        paid=job['price']*4//5
        ledger.account(state,owner)['balance']+=paid
        if state['pending_verifier_reward'] is not None:
            raise ValueError('Previous verifier reward has not settled')
        state['pending_verifier_reward']={'height':state['height'],'budget':job['price']-paid,
            'powers':ledger.voting_power(state,state['height'])}
        remember(state,body['job_id'],{'status':'completed','owner':job['owner'],'provider':owner,
            'model_root':job['model_root'],'output':result,'price':job['price'],'provider_paid':paid,
            'height':state['height']})
        state['last_inference']=result
        del state['jobs'][body['job_id']]
    else:
        raise ValueError('Unknown native transaction kind')
    invariant(state)
    return state
