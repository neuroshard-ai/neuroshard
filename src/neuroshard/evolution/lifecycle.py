"""Opt-in native lifecycle for curated data, serving decisions and paid inference.

This is a new-genesis experimental profile, not a migration of the 0.4 ledger.
The numerical paths remain optimistic and require complete observer coverage.
"""
from neuroshard.dataflow.store import canonical
from neuroshard.demo import protocol
from neuroshard.lab import state as ledger
from . import cohorts, forward, schema
from .objects import digest
from .verification import Metadata

FORMAT = 'neuroshard-lifecycle-v1'
FIELDS = {
    'propose_data':{'data_root','metadata'},
    'vote_data':{'proposal_id','approve'},
    'open_evaluation':set(),
    'score':{'evaluation_id','side','role','offset','record_root','metadata'},
    'finish_evaluation':{'evaluation_id'},
    'infer':{'model_root','provider','prompt_ids','max_tokens','max_price','expires_in'},
    'respond':{'job_id','record_root','metadata'},
}


def initialize(state):
    profile = state['manifest']['lifecycle']
    if set(profile) != {'format','tokenizer_root','vocabulary','eos_ids','initial_model',
                        'source_cursors','steps_per_cohort','data_vote_blocks','evaluation_blocks','price_per_token'} or profile['format'] != FORMAT:
        raise ValueError('Invalid native lifecycle profile')
    schema.root(profile['tokenizer_root'])
    schema.integer(profile['vocabulary'], 16, 131072)
    schema.integer(profile['steps_per_cohort'], 1, 64)
    schema.integer(profile['data_vote_blocks'], 16, 100000)
    schema.integer(profile['evaluation_blocks'], 1024, 100000)
    schema.integer(profile['price_per_token'], 1, 10**9)
    if not isinstance(profile['eos_ids'], list) or len(profile['eos_ids']) > 8 or len(set(profile['eos_ids'])) != len(profile['eos_ids']):
        raise ValueError('Invalid profile stop tokens')
    for token in profile['eos_ids']:
        schema.integer(token, 0, profile['vocabulary']-1)
    model = schema.model(profile['initial_model'])
    if (digest(canonical(model)) != state['model_root'] or model.get('tokenizer_root') != profile['tokenizer_root']
            or model['config']['vocab_size'] != profile['vocabulary']):
        raise ValueError('Initial model does not bind this text contract')
    if not isinstance(profile['source_cursors'], dict) or len(profile['source_cursors']) > 128:
        raise ValueError('Invalid initial source cursors')
    for key, cursor in profile['source_cursors'].items():
        schema.root(key)
        schema.integer(cursor, 0, 2**53-1)
    state['lifecycle'] = {'active':None, 'proposal':None, 'cursors':dict(profile['source_cursors']),
        'seen_documents':{}, 'seen_batches':{}, 'evaluation':None, 'evaluations':[],
        'serving_model':model, 'jobs':{}, 'results':{}, 'data_history':[]}


def escrow(state):
    life = state.get('lifecycle')
    if not life:
        return 0
    return ((life['proposal']['bond'] if life['proposal'] else 0)
            + sum(job['escrow'] for job in life['jobs'].values()))


def weights(state):
    result = {}
    for key, power in ledger.voting_power(state, max(1,state['height'])).items():
        owner = state['validators'][key]['owner']
        result[owner] = result.get(owner, 0)+power
    return result


def quorum(electorate, votes):
    return 3*sum(power for key,power in electorate.items() if votes.get(key) is True) > 2*sum(electorate.values())


def end_proposal(state, accepted, reason):
    life = state['lifecycle']
    proposal = life['proposal']
    if accepted:
        metadata = cohorts.metadata(proposal['metadata'])
        value, roles = cohorts.validate(metadata, proposal['data_root'], state)
        replay = life['active']['train'] if life['active'] else []
        plan = cohorts.schedule(proposal['data_root'], roles['train'], replay,
                                state['manifest']['lifecycle']['steps_per_cohort'])
        batches = {key:metadata.json(key) for d in value['documents'] for key in d['batches']}
        if replay:
            batches.update({key:life['active']['batches'][key] for d in replay for key in d['batches'] if key in plan})
        life['active'] = {'root':proposal['data_root'], **roles, 'schedule':plan,
                         'batches':batches, 'step':0, 'closed':False, 'activated_height':state['height'],
                         'expires':state['height']+2*state['manifest']['lifecycle']['evaluation_blocks']}
        for window in value['windows']:
            life['cursors'][window['source']] = window['end']
        for document in value['documents']:
            life['seen_documents'][document['id']] = document['role']
            for key in document['batches']:
                life['seen_batches'][key] = document['role']
        state['data_root'] = proposal['data_root']
    refund = proposal['bond'] if accepted else proposal['bond']*9//10
    ledger.account(state, proposal['owner'])['balance'] += refund
    state['burned'] += proposal['bond']-refund
    life['data_history'].append({'id':proposal['id'], 'data_root':proposal['data_root'],
        'accepted':accepted, 'reason':reason, 'height':state['height']})
    life['data_history'] = life['data_history'][-128:]
    life['proposal'] = None


def finish_evaluation(state, reason=None):
    life = state['lifecycle']
    evaluation = life['evaluation']
    result = {'promote':False, 'reason':reason} if reason else cohorts.decision(evaluation['measurements'],life['active'])
    if result['promote']:
        state['serving_root'] = evaluation['candidate']
        life['serving_model'] = evaluation['candidate_model']
    else:
        # A rejected learning branch is retained in history; the next cohort
        # starts from the serving model. Paid-task identities survive rollback.
        state['model_root'] = evaluation['baseline']
    life['evaluations'].append({**evaluation, 'decision':result, 'finished_height':state['height']})
    life['evaluations'] = life['evaluations'][-32:]
    life['evaluation'] = None
    life['active']['closed'] = True


def advance(state):
    if 'lifecycle' not in state:
        return
    life = state['lifecycle']
    height = state['height']
    proposal = life['proposal']
    if proposal:
        if height > proposal['expires']:
            end_proposal(state, False, 'data admission expired')
        elif (height >= proposal['activate_after'] and quorum(proposal['electorate'],proposal['votes'])
                and quorum(weights(state), proposal['votes']) and not state['assignment']
                and not state['candidate'] and not life['evaluation']
                and (life['active'] is None or life['active']['closed'])):
            end_proposal(state, True, 'native supermajority data admission')
    evaluation = life['evaluation']
    if evaluation and height > evaluation['expires'] and not state['candidate']:
        finish_evaluation(state, 'evaluation deadline missed')
    active = life['active']
    if (active and not active['closed'] and height > active['expires']
            and not life['evaluation'] and not state['candidate'] and not state['assignment']):
        state['model_root'] = state['serving_root']
        active.update(closed=True, termination_reason='cohort training deadline missed')
    for key, job in list(life['jobs'].items()):
        if height > job['expires'] and not job['claim_id']:
            ledger.account(state,job['payer'])['balance'] += job['escrow']
            life['results'][key] = {'id':key, 'status':'expired', 'model_root':job['model_root'],
                                    'refunded_atoms':job['escrow'], 'height':height}
            del life['jobs'][key]
    trim_results(life)


def trim_results(life):
    # JSON serialization sorts keys; never rely on insertion order for pruning.
    while len(life['results']) > 128:
        key = min(life['results'], key=lambda k:(life['results'][k]['height'],k))
        del life['results'][key]


def assignment(state):
    life = state['lifecycle']
    active = life['active']
    if (not active or active['closed'] or state['height'] > active['expires']
            or life['evaluation'] or active['step'] >= len(active['schedule'])):
        raise ValueError('Activate fresh data or finish the current cohort evaluation before training')
    return {'data_root':active['root'], 'sequence_index':active['step'], 'batch':active['schedule'][active['step']]}


def new_claim(state, owner, envelope, record_root, metadata, **fields):
    if state['candidate'] or state['assignment']:
        raise ValueError('Finish pending native work before another execution claim')
    amount = state['manifest']['params']['claim_bond']
    account = ledger.account(state,owner)
    if account['balance'] < amount:
        raise ValueError('Insufficient execution claim bond')
    account['balance'] -= amount
    p = state['manifest']['params']
    state['candidate'] = {'id':protocol.transaction_id(envelope), 'owner':owner, 'bond':amount,
        'record_root':record_root, 'metadata':metadata, 'workers':[], 'challenge':None,
        'deadline':state['height']+p['challenge_blocks'], 'expires':state['height']+p['max_claim_blocks'], **fields}


def apply(state, owner, body, envelope):
    if 'lifecycle' not in state:
        raise ValueError('This genesis does not enable the native model lifecycle')
    life, profile = state['lifecycle'], state['manifest']['lifecycle']
    kind, height = body['kind'], state['height']
    account = ledger.account(state,owner)
    if kind == 'propose_data':
        if life['proposal']:
            raise ValueError('A bounded data admission is already pending')
        if life['active'] and not life['active']['closed']:
            raise ValueError('Finish the active cohort before proposing its replacement')
        metadata = cohorts.metadata(body['metadata'])
        cohorts.validate(metadata, body['data_root'], state)
        bond = state['manifest']['params']['claim_bond']
        if account['balance'] < bond:
            raise ValueError('Insufficient data admission bond')
        account['balance'] -= bond
        life['proposal'] = {'id':protocol.transaction_id(envelope), 'owner':owner, 'bond':bond,
            'data_root':body['data_root'], 'metadata':body['metadata'], 'votes':{}, 'electorate':weights(state),
            'activate_after':height+state['manifest']['params']['activation_blocks'],
            'expires':height+profile['data_vote_blocks']}
    elif kind == 'vote_data':
        proposal = life['proposal']
        if not proposal or body['proposal_id'] != proposal['id'] or height > proposal['expires']:
            raise ValueError('No matching live data proposal')
        if type(body['approve']) is not bool or owner not in proposal['electorate'] or owner in proposal['votes']:
            raise ValueError('Each snapshotted validator owner may cast one explicit vote')
        proposal['votes'][owner] = body['approve']
    elif kind == 'open_evaluation':
        active = life['active']
        if (state['candidate'] or state['assignment'] or life['evaluation'] or not active
                or active['closed'] or active['step'] != len(active['schedule'])):
            raise ValueError('Complete the prescribed cohort training before evaluation')
        life['evaluation'] = {'id':protocol.transaction_id(envelope), 'baseline':state['serving_root'],
            'candidate':state['model_root'], 'data_root':state['data_root'], 'candidate_model':None,
            'expires':height+profile['evaluation_blocks'], 'opened_height':height,
            'measurements':{side:{role:[] for role in ('retention','fresh')} for side in ('baseline','candidate')}}
    elif kind == 'score':
        evaluation = life['evaluation']
        if (not evaluation or body['evaluation_id'] != evaluation['id']
                or height+state['manifest']['params']['challenge_blocks'] >= evaluation['expires']):
            raise ValueError('No matching live evaluation')
        side, role = body['side'], body['role']
        if side not in ('baseline','candidate') or role not in ('retention','fresh'):
            raise ValueError('Unknown evaluation side or group')
        offset = schema.integer(body['offset'], 0, len(cohorts.evaluation_rows(life['active'],role))-1)
        if offset != len(evaluation['measurements'][side][role]):
            raise ValueError('Evaluation documents must be scored once in assigned order')
        _, batch = cohorts.evaluation_batch(life['active'], role, offset)
        metadata = Metadata(body['metadata'])
        value = forward.validate(metadata,body['record_root'])
        if value['model_root'] != evaluation[side] or metadata.json(value['batch']) != batch:
            raise ValueError('Evaluation substituted its model or assigned documents')
        new_claim(state, owner, envelope, body['record_root'], body['metadata'], kind='score',
            model_root=value['model_root'], evaluation_id=evaluation['id'], side=side, role=role,
            offset=offset, losses=value['result']['losses_hex'])
        state['candidate']['expires'] = min(state['candidate']['expires'], evaluation['expires'])
    elif kind == 'finish_evaluation':
        evaluation = life['evaluation']
        if (not evaluation or body['evaluation_id'] != evaluation['id'] or state['candidate']
                or any(len(values) != len(cohorts.evaluation_rows(life['active'],role))
                       for roles in evaluation['measurements'].values() for role,values in roles.items())):
            raise ValueError('Every paired document score must settle before the serving decision')
        finish_evaluation(state)
    elif kind == 'infer':
        if len(life['jobs']) >= 16 or body['model_root'] != state['serving_root']:
            raise ValueError('Inference queue is full or serving model changed')
        provider = ledger.public_key(body['provider'])
        prompt = body['prompt_ids']
        if not isinstance(prompt,list) or not 2 <= len(prompt) <= 192:
            raise ValueError('Inference prompt requires 2–192 tokens')
        from .batches import unpack
        unpack([prompt],profile['vocabulary'])
        maximum = schema.integer(body['max_tokens'],1,8)
        escrow = schema.integer(body['max_price'],maximum*profile['price_per_token'],2**60)
        duration = schema.integer(body['expires_in'],2*state['manifest']['params']['challenge_blocks']+2,4096)
        if account['balance'] < escrow:
            raise ValueError('Insufficient inference escrow')
        account['balance'] -= escrow
        key = protocol.transaction_id(envelope)
        life['jobs'][key] = {'id':key, 'payer':owner, 'provider':provider, 'model_root':state['serving_root'],
            'prompt_ids':prompt, 'max_tokens':maximum, 'eos_ids':profile['eos_ids'], 'escrow':escrow,
            'unit_price':profile['price_per_token'], 'expires':height+duration, 'claim_id':None}
    elif kind == 'respond':
        job = life['jobs'].get(schema.root(body['job_id']))
        if not job or job['provider'] != owner or job['claim_id'] or height+state['manifest']['params']['challenge_blocks'] >= job['expires']:
            raise ValueError('No assigned inference job with sufficient dispute time')
        metadata = Metadata(body['metadata'])
        value = forward.validate_generation(metadata,body['record_root'])
        if any(value[key] != job[key] for key in ('model_root','prompt_ids','max_tokens','eos_ids')):
            raise ValueError('Generation differs from the paid request')
        new_claim(state, owner, envelope, body['record_root'], body['metadata'], kind='inference',
                  model_root=job['model_root'], job_id=job['id'], token_ids=value['token_ids'])
        state['candidate']['expires'] = min(state['candidate']['expires'],job['expires'])
        job['claim_id'] = state['candidate']['id']
    else:
        raise ValueError('Unknown lifecycle transaction')


def settled(state, claim, accepted):
    """Called after the existing claim-bond refund or slash has been accounted."""
    if 'lifecycle' not in state:
        return
    life = state['lifecycle']
    kind = claim.get('kind','training')
    if kind == 'training' and accepted:
        life['active']['step'] += 1
    elif kind == 'score' and accepted:
        evaluation = life['evaluation']
        if not evaluation or evaluation['id'] != claim['evaluation_id']:
            raise ValueError('Settling score lost its evaluation reservation')
        scores = evaluation['measurements'][claim['side']][claim['role']]
        if len(scores) != claim['offset']:
            raise ValueError('Score already settled')
        scores.extend(claim['losses'])
        if claim['side'] == 'candidate':
            evaluation['candidate_model'] = claim['metadata'][claim['model_root']]
    elif kind == 'inference':
        job = life['jobs'][claim['job_id']]
        job['claim_id'] = None
        if accepted:
            paid = len(claim['token_ids'])*job['unit_price']
            ledger.account(state,job['provider'])['balance'] += paid
            ledger.account(state,job['payer'])['balance'] += job['escrow']-paid
            life['results'][job['id']] = {'id':job['id'], 'status':'completed', 'model_root':job['model_root'],
                'token_ids':claim['token_ids'], 'provider':job['provider'], 'paid_atoms':paid,
                'refunded_atoms':job['escrow']-paid, 'height':state['height']}
            del life['jobs'][job['id']]
            trim_results(life)


def dispute(metadata, record_root, stage):
    traces = forward.trace_roots(metadata,record_root)
    schema.integer(stage,0,len(traces)-1)
    needed = forward.dependencies(metadata,traces[stage])
    outputs = [metadata.json(key)['result']['output'] for key in traces
               if metadata.json(key)['request']['phase'] == 'forward']
    outputs += [component['root'] for component in metadata.json(metadata.json(record_root)['model_root'])['components'].values()]
    return needed, outputs
