"""Optimistic training settlement: compact claims, bonded disputes, native uploads.

This module is an experimental state machine. Its security assumes an honest
observer checks each accepted execution graph during its challenge window.
It never treats a signature or an unchallenged claim as a cryptographic proof.
"""
import base64
import copy
import json

from neuroshard.dataflow.store import canonical
from neuroshard.demo import protocol
from neuroshard.lab import state as ledger
from .objects import digest, MAX_OBJECT_BYTES
from .schema import root, integer
from .verification import Metadata, bundle, validate_record, dependencies,validate_growth,work_identity
from . import lifecycle, auditing, portable_work, portable_lifecycle

CHUNK_BYTES = 1024*1024
MAX_TX_BYTES = 2*1024*1024
PARAMS = {**ledger.PARAMS, 'fee':1000, 'epoch_blocks':64, 'activation_blocks':16,
          'lease_blocks':1024,
          'evidence_blocks':1024, 'evidence_seconds':4096,
          'challenge_blocks':32, 'availability_blocks':256,
          'claim_bond':20_000_000, 'challenge_bond':2_000_000,
          'reward_atoms':1_000_000, 'budget_period_blocks':86400,
          'steps_per_period':256, 'max_tx_bytes':MAX_TX_BYTES}
PARAMS['max_claim_blocks'] = 1024
PARAMS['growths_per_period'] = 1


def genesis(chain_id, validators, manifest):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    base = {'chain_id':chain_id,'height':0,'time_ns':0,'manifest':copy.deepcopy(manifest),
            'accounts':{},'validators':{},'evidence_seen':[],'issued':0,'burned':0,'initial_supply':0}
    p = manifest['params']
    for entry in validators:
        owner,key = ledger.public_key(entry['owner']),root(entry['consensus_key'])
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(key))
        amount,liquid = integer(entry['bond'],p['bond_unit'],2**60),integer(entry['liquid'],0,2**60)
        if key in base['validators'] or amount%p['bond_unit']:
            raise ValueError('Invalid or reused genesis consensus bond')
        account(base,owner)['balance'] += liquid
        base['validators'][key] = {'owner':owner,'amount':amount,'status':'active',
            'history':[[1,amount//p['bond_unit']]],'emit_at':None,'removed_at':None,
            'release_height':None,'release_time_ns':None}
        base['initial_supply'] += amount+liquid
    if not base['validators']:
        raise ValueError('Genesis requires bonded native consensus weight')
    base.update(model_root=root(manifest['initial_model_root']), serving_root=manifest['initial_model_root'],
                candidate=None, settled=[], period=0, period_steps=0, period_growths=0,audit_count=0,
                training_round=0, paid_work={},data_root=root(manifest['data_root']), assignment=None,
                update_check_count=0)
    if 'lifecycle' in manifest:
        lifecycle.initialize(base)
    if 'auditing' in manifest:
        auditing.initialize(base)
    if 'portable_work' in manifest:
        portable_work.initialize(base)
    if 'portable_lifecycle' in manifest:
        portable_lifecycle.initialize(base)
    invariant(base)
    return base


def invariant(s):
    escrow = lifecycle.escrow(s) + auditing.escrow(s) + portable_lifecycle.escrow(s)
    if s['assignment']:
        escrow += s['assignment']['bond']
    if s['candidate']:
        escrow += s['candidate']['bond']
        escrow += s['candidate']['challenge']['bond'] if s['candidate']['challenge'] else 0
    total = sum(a['balance'] for a in s['accounts'].values())
    total += sum(v['amount'] for v in s['validators'].values())
    if s['initial_supply']+s['issued'] != total+escrow+s['burned']:
        raise ValueError('Supply conservation failed')
    if any(type(a['balance']) is not int or a['balance']<0 or a['nonce']<0 for a in s['accounts'].values()):
        raise ValueError('Invalid account balance or nonce')
    if s['issued'] != s['training_round']*s['manifest']['params']['reward_atoms']:
        raise ValueError('Training issuance accounting failed')
    if len(s['paid_work']) != s['training_round']:
        raise ValueError('A numerical task must be paid at most once')
    if 'portable_lifecycle' in s:
        if (s['model_root'] != s['portable_work']['checkpoint']['state_root']
                or s['serving_root'] != s['portable_lifecycle']['serving_checkpoint']['state_root']):
            raise ValueError('Portable learned and serving commitments lost their separate bindings')


def account(s, owner):
    return ledger.account(s,owner)


def close(s, accepted, reason, refund_bond=False, proven_fault=False):
    claim = s['candidate']
    p = s['manifest']['params']
    challenge = claim['challenge']
    if accepted and not auditing.complete(s, claim):
        raise ValueError('Complete funded audit reports are required for settlement')
    if accepted:
        account(s,claim['owner'])['balance'] += claim['bond']
        if challenge:
            # A successful data-availability response returns the challenger
            # bond: requesting the public data is not itself misconduct.
            account(s,challenge['owner'])['balance'] += challenge['bond']
        if claim.get('kind','training')=='growth':
            s['period_growths'] += 1
        elif claim.get('kind') == 'portable_training':
            portable_work.settle(s, claim)
        elif claim.get('kind','training')=='training':
            if claim['work_identity'] in s['paid_work']:
                raise ValueError('Task was already paid')
            s['paid_work'][claim['work_identity']]=claim['id']
            reward = p['reward_atoms']
            s['issued'] += reward
            s['training_round'] += 1
            s['period_steps'] += 1
            count = len(claim['workers'])
            for index,owner in enumerate(claim['workers']):
                account(s,owner)['balance'] += reward//count + (1 if index<reward%count else 0)
        if claim.get('kind','training') in ('training','growth'):
            s['model_root'] = claim['model_root']
    elif refund_bond:
        account(s,claim['owner'])['balance'] += claim['bond']
        if challenge:
            s['burned'] += challenge['bond']
    else:
        if challenge:
            account(s,challenge['owner'])['balance'] += challenge['bond'] + claim['bond']//2
            s['burned'] += claim['bond']-claim['bond']//2
        else:
            s['burned'] += claim['bond']
    if 'auditing' in s:
        auditing.finish(s, claim['audit_budget'], accepted=accepted, claim=claim,
                        proven_fault=proven_fault, reason=reason)
    lifecycle.settled(s,claim,accepted)
    portable_lifecycle.settled(s,claim,accepted)
    s['settled'].append({'id':claim['id'],'accepted':accepted,'reason':reason,
                         'kind':claim.get('kind','training'),
                         'model_root':claim['model_root'],'height':s['height']})
    s['settled'] = s['settled'][-128:]
    s['candidate'] = None


def advance(previous,height,time_ns,evidence=(),committers=None):
    s = copy.deepcopy(previous)
    p = s['manifest']['params']
    if height != s['height']+1 or time_ns < s['time_ns']:
        raise ValueError('Nonmonotonic native block metadata')
    s['height'],s['time_ns'] = height,time_ns
    updates = {}
    for item in evidence:
        event = f'{item["kind"]}:{item["address"]}:{item["height"]}'
        if event in s['evidence_seen']:
            continue
        if item['kind'] not in (1,2) or not 1<=item['height']<height:
            raise ValueError('Invalid consensus evidence')
        for key,v in s['validators'].items():
            if ledger.consensus_address(key) != item['address']:
                continue
            if not ledger.voting_power(s,item['height']).get(key) or v['status']=='withdrawn':
                raise ValueError('Evidence does not refer to a liable validator')
            penalty = (v['amount']+3)//4
            v['amount'] -= penalty
            s['burned'] += penalty
            if v['status'] not in ('cooldown','jailed'):
                v.update(status='jailed',emit_at=None,removed_at=height+2,release_height=None,release_time_ns=None)
                v['history'].append([height+2,0])
                updates[key] = 0
            s['evidence_seen'].append(event)
            break
        else:
            raise ValueError('Unknown evidence validator')
    for key,v in s['validators'].items():
        if v['emit_at'] == height:
            if v['status']=='pending':
                power = v['amount']//p['bond_unit']
                v.update(status='active',emit_at=None)
                v['history'].append([height+2,power])
                updates[key] = power
            elif v['status']=='leaving':
                v.update(status='cooldown',emit_at=None,removed_at=height+2)
                v['history'].append([height+2,0])
                updates[key] = 0
        if v['removed_at']==height and v['release_height'] is None:
            v['release_height'] = height+p['evidence_blocks']
            v['release_time_ns'] = time_ns+p['evidence_seconds']*1_000_000_000
    if updates and not ledger.voting_power(s,height+2):
        raise ValueError('Cannot remove all native consensus weight')
    period = height//p['budget_period_blocks']
    if period != s['period']:
        s['period'],s['period_steps'] = period,0
        s['period_growths'] = 0
    assignment = s['assignment']
    if assignment and height>assignment['expires']:
        s['burned'] += assignment['bond']//10
        account(s,assignment['owner'])['balance'] += assignment['bond']-assignment['bond']//10
        if 'auditing' in s:
            auditing.finish(s, assignment['audit_budget'], reason='training reservation expired')
        s['assignment'] = None
    claim = s['candidate']
    if claim:
        challenge = claim['challenge']
        if height>claim['expires']:
            # An unfinished accusation must not burn an honest publisher's
            # collateral. No computation is accepted or paid at this deadline.
            # Failure to answer a fully elapsed availability deadline remains
            # attributable to the publisher and is handled as unavailability.
            unavailable = challenge and challenge['kind']=='availability' and height>challenge['deadline']
            close(s,False,'data availability deadline missed' if unavailable else 'absolute claim deadline expired',
                  refund_bond=not unavailable)
        elif challenge and height>challenge['deadline']:
            if challenge['kind']=='availability':
                close(s,False,'data availability deadline missed')
            else:
                # An accuser who never presents the required evidence loses its
                # bond. Keep the claim open for other observers afterward.
                s['burned'] += challenge['bond']
                claim['challenge'] = None
                claim['deadline'] = max(height+p['challenge_blocks'], auditing.minimum_deadline(s, claim))
        elif not challenge and auditing.rejected(s, claim) and height > claim['deadline']:
            close(s, False, 'native audit quorum rejected execution')
        elif (not challenge and 'auditing' in s and height > claim['audit_reveal_end']
              and not auditing.complete(s, claim)):
            if not auditing.rejected(s, claim):
                close(s, False, 'funded audit coverage deadline missed', refund_bond=True)
        elif not challenge and height>claim['deadline']:
            close(s,True,'challenge window elapsed')
    lifecycle.advance(s)
    auditing.advance(s)
    portable_lifecycle.advance(s)
    invariant(s)
    return s,updates


def transition(previous,envelope,artifacts=None,commit_artifacts=False,referee=None):
    if len(canonical(envelope))>MAX_TX_BYTES:
        raise ValueError('Transaction exceeds native byte limit')
    body,owner = protocol.verify(envelope)
    ledger.public_key(owner)
    if body.get('chain_id') != previous['chain_id']:
        raise ValueError('Wrong chain')
    integer(body.get('nonce'),0,2**53-1)
    if body['nonce'] != previous['accounts'].get(owner,{'nonce':0})['nonce']:
        raise ValueError('Wrong nonce')
    extras = {
        'transfer':{'to','amount'},'bond':{'consensus_key','amount','possession'},
        'unbond':{'consensus_key'},'withdraw':{'consensus_key'},
        'reserve':{'parent','round','workers'},
        'claim':{'record_root','metadata','workers','data_root','sequence_index'},
        'grow':{'parent','model_root','metadata','capacities'},
        'challenge':{'claim_id','stage','challenge_kind','object_root'},
        'upload':{'claim_id','object_root','index','data'},
        'seal':{'claim_id','object_root'},'resolve':{'claim_id'},
        'refute_update':{'claim_id','stage','tensor_index','witness'},
        **lifecycle.FIELDS,
        **auditing.FIELDS,
        **portable_work.FIELDS,
        **portable_lifecycle.FIELDS,
    }
    if 'auditing' in previous:
        for name in ('reserve', *auditing.CLAIM_KINDS):
            extras[name] = extras[name] | {'audit_budget'}
    kind = body.get('kind')
    if kind not in extras or set(body) != {'kind','chain_id','nonce'}|extras[kind]:
        raise ValueError('Invalid transaction schema')
    if 'portable_work' in previous and kind in ('reserve', 'claim', 'grow', *lifecycle.FIELDS):
        raise ValueError('This genesis accepts only its prepared portable execution profile')
    s = copy.deepcopy(previous)
    p = s['manifest']['params']
    sender = account(s,owner)
    def debit(amount):
        if sender['balance']<amount:
            raise ValueError('Insufficient available balance')
        sender['balance'] -= amount
    debit(p['fee'])
    s['burned'] += p['fee']
    sender['nonce'] += 1
    if kind in portable_lifecycle.FIELDS:
        portable_lifecycle.apply(s, owner, body, envelope)
    elif kind in portable_work.FIELDS:
        portable_work.apply(s, owner, body, envelope)
    elif kind in auditing.FIELDS:
        auditing.apply(s, owner, body, envelope)
    elif kind in lifecycle.FIELDS:
        lifecycle.apply(s,owner,body,envelope)
    elif kind=='transfer':
        amount = integer(body['amount'],1,2**60)
        recipient = ledger.public_key(body['to'])
        debit(amount)
        account(s,recipient)['balance'] += amount
    elif kind=='bond':
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        key = root(body['consensus_key'])
        amount = integer(body['amount'],p['bond_unit'],2**60)
        if key in s['validators'] or amount%p['bond_unit']:
            raise ValueError('Invalid or reused consensus bond')
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(key)).verify(bytes.fromhex(body['possession']),
                ledger.possession_message(s['chain_id'],owner,key,amount,body['nonce']))
        except Exception as exc:
            raise ValueError('Invalid consensus-key possession proof') from exc
        debit(amount)
        s['validators'][key] = {'owner':owner,'amount':amount,'status':'pending','history':[],
            'emit_at':ledger.next_epoch(s['height']+p['activation_blocks'],p),
            'removed_at':None,'release_height':None,'release_time_ns':None}
    elif kind in ('unbond','withdraw'):
        key = root(body['consensus_key'])
        v = s['validators'][key]
        if v['owner']!=owner:
            raise ValueError('Bond belongs to another account')
        if kind=='unbond':
            if v['status']!='active' or not any(k!=key and x['status']=='active' for k,x in s['validators'].items()):
                raise ValueError('A replacement must be active before the last validator leaves')
            v.update(status='leaving',emit_at=ledger.next_epoch(s['height']+1,p))
        else:
            if auditing.held(s, key):
                raise ValueError('Bond remains committed to a native audit obligation')
            if v['status'] not in ('cooldown','jailed') or v['release_height'] is None or s['height']<v['release_height'] or s['time_ns']<=v['release_time_ns']:
                raise ValueError('Bond remains exposed to consensus evidence')
            sender['balance'] += v['amount']
            v.update(amount=0,status='withdrawn')
    elif kind=='grow':
        if 'lifecycle' in s:
            lifecycle.assignment(s)
            if s['lifecycle']['active']['step']:
                raise ValueError('Growth is allowed only before a fresh cohort starts training')
        if s['assignment'] or s['candidate'] or s['period_growths']>=p['growths_per_period']:
            raise ValueError('Pending work or growth budget exhausted')
        if body['parent']!=s['model_root']:
            raise ValueError('Growth must extend the current learning model')
        metadata = Metadata(body['metadata'])
        candidate_root = root(body['model_root'])
        validate_growth(metadata,body['parent'],candidate_root)
        from .model import place
        # Placement inspects metadata only. Declared capacities are admission
        # inputs, not a proof that independent hardware is online.
        assignments = place(metadata.json(candidate_root),body['capacities'])
        debit(p['claim_bond'])
        record = {'kind':'growth','parent':body['parent'],'model_root':candidate_root,
                  'assignments':assignments}
        record_root = digest(canonical(record))
        values = {**metadata.values,record_root:record}
        Metadata(values)
        s['candidate'] = {'kind':'growth','id':protocol.transaction_id(envelope),'owner':owner,
            'bond':p['claim_bond'],'model_root':candidate_root,'record_root':record_root,'metadata':values,
            'workers':[],'deadline':s['height']+p['challenge_blocks'],
            'expires':s['height']+p['max_claim_blocks'],'challenge':None}
    elif kind=='reserve':
        if s['candidate'] or s['assignment'] or s['period_steps']>=p['steps_per_period']:
            raise ValueError('Pending work or current period budget exhausted')
        if body['parent']!=s['model_root'] or type(body['round']) is not int or body['round']!=s['training_round']:
            raise ValueError('Wrong reservation parent or round')
        if not isinstance(body['workers'],list) or not 1<=len(body['workers'])<=64:
            raise ValueError('A reservation requires bounded worker identities')
        workers = [ledger.public_key(key) for key in body['workers']]
        debit(p['claim_bond'])
        s['assignment'] = {'id':protocol.transaction_id(envelope),'owner':owner,'workers':workers,
                           'bond':p['claim_bond'],'expires':s['height']+p['lease_blocks']}
        if 'lifecycle' in s:
            s['assignment'].update(lifecycle.assignment(s))
        if 'auditing' in s:
            auditing.lock(s, body['audit_budget'], owner, workers, s['assignment']['id'])
            s['assignment']['audit_budget'] = body['audit_budget']
    elif kind=='claim':
        if s['candidate'] or s['period_steps']>=p['steps_per_period']:
            raise ValueError('Pending candidate or current period budget exhausted')
        assignment = s['assignment']
        if not assignment or assignment['owner']!=owner or s['height']>assignment['expires']:
            raise ValueError('Reserve this training round before computing its worker receipts')
        metadata = Metadata(body['metadata'])
        record = metadata.json(root(body['record_root']))
        validate_record(metadata,body['record_root'])
        identity = work_identity(metadata,body['record_root'])
        if identity in s['paid_work']:
            raise ValueError('This prescribed computation has already been paid')
        if record['parent'] != s['model_root'] or record['step'] != s['training_round']:
            raise ValueError('Wrong training parent or round')
        if body['data_root'] != s['data_root']:
            raise ValueError('Wrong active dataset')
        if 'lifecycle' in s:
            expected = lifecycle.assignment(s)
            integer(body['sequence_index'],0,len(s['lifecycle']['active']['schedule'])-1)
            if any(assignment.get(k)!=v for k,v in expected.items()) or body['sequence_index']!=expected['sequence_index'] or record['batch']!=expected['batch']:
                raise ValueError('Work does not use its reserved cohort batch')
        else:
            data = s['manifest']['training_batches']
            index = integer(body['sequence_index'],0,len(data)-1)
            if index != s['training_round']%len(data) or record['batch'] != data[index]:
                raise ValueError('Work does not use the assigned training batch')
        if record['learning_rate_hex'] != float(s['manifest']['learning_rate']).hex() or record['clip_norm_hex'] != float(s['manifest']['clip_norm']).hex():
            raise ValueError('Unapproved optimizer settings')
        workers = body['workers']
        if not isinstance(workers,list) or len(workers)!=len(record['traces']):
            raise ValueError('Missing worker receipts')
        identities = []
        for stage,signed in enumerate(workers):
            receipt,key = protocol.verify(signed)
            expected = {'domain':'neuroshard/evolution/work/v1','chain_id':s['chain_id'],
                        'assignment':assignment['id'],'record_root':body['record_root'],
                        'stage':stage,'trace_root':record['traces'][stage]}
            if receipt != expected:
                raise ValueError('Receipt does not bind this graph and stage')
            identities.append(ledger.public_key(key))
        if identities != assignment['workers']:
            raise ValueError('Work rewards belong to the workers reserved before execution')
        s['candidate'] = {'id':protocol.transaction_id(envelope),'owner':owner,'bond':assignment['bond'],
            'work_identity':identity,
            'model_root':record['model_root'],'record_root':body['record_root'],'metadata':body['metadata'],
            'workers':identities,'deadline':s['height']+p['challenge_blocks'],
            'expires':s['height']+p['max_claim_blocks'],'challenge':None}
        if 'auditing' in s:
            auditing.attach(s, assignment['audit_budget'])
        s['assignment'] = None
    else:
        claim = s['candidate']
        if not claim or body['claim_id'] != claim['id']:
            raise ValueError('No matching pending claim')
        if claim.get('kind') in ('portable_training', *portable_lifecycle.SERVICE_KINDS):
            raise ValueError('Portable work requires the native weighted replay verdict path')
        metadata = Metadata(claim['metadata'])
        record = metadata.json(claim['record_root'])
        if kind=='refute_update':
            if claim.get('kind','training') != 'training' or claim['challenge'] or s['height']>claim['deadline']:
                raise ValueError('Compact update refutation requires an unchallenged live training claim')
            from .update_witness import check
            verdict = check(metadata,claim['record_root'],body['stage'],body['tensor_index'],body['witness'])
            debit(p['challenge_bond'])
            s['update_check_count'] += 1
            if verdict['valid']:
                # A disproved accusation cannot extend or reset the work slot.
                s['burned'] += p['challenge_bond']
            else:
                claim['challenge'] = {'owner':owner,'bond':p['challenge_bond'],'kind':'compact_update'}
                close(s,False,'objective update witness: '+verdict['mismatch'], proven_fault=True)
        elif kind=='challenge':
            if claim['challenge'] or s['height']>claim['deadline']:
                raise ValueError('Challenge is already active or too late')
            growth = record.get('kind')=='growth'
            forward_claim = claim.get('kind') in ('score','inference')
            if forward_claim:
                needed,outputs = lifecycle.dispute(metadata,claim['record_root'],body['stage'])
                stage = body['stage']
            else:
                stage = integer(body['stage'],0,0 if growth else len(record['traces'])-1)
            if body['challenge_kind'] not in ('fraud','availability'):
                raise ValueError('Unknown challenge kind')
            if forward_claim:
                pass
            elif growth:
                parent = metadata.json(record['parent'])
                last_block = f'block_{parent["config"]["num_hidden_layers"]-1:03}'
                needed = [parent['components'][last_block]['root']]
            else:
                needed = dependencies(metadata,record['traces'][stage])
            if not forward_claim:
                outputs = [c['root'] for c in metadata.json(record['model_root'])['components'].values()]
            requested = body['object_root']
            if body['challenge_kind']=='availability':
                if requested not in needed+outputs:
                    raise ValueError('Availability request is unrelated to the claim')
                needed = [requested]
            elif requested is not None:
                raise ValueError('Fraud challenge specifies a stage, not one object')
            debit(p['challenge_bond'])
            if 'auditing' in s:
                claim['audit_interrupted'] = True
            claim['challenge'] = {'owner':owner,'bond':p['challenge_bond'],'kind':body['challenge_kind'],
                'stage':stage,'needed':needed,'uploads':{},'sealed':[],
                'deadline':s['height']+p['availability_blocks']}
        else:
            challenge = claim['challenge']
            if not challenge or s['height']>challenge['deadline']:
                raise ValueError('No live challenge')
            if kind=='upload':
                expected_owner = claim['owner'] if challenge['kind']=='availability' else challenge['owner']
                if owner != expected_owner:
                    raise ValueError('Only the evidence publisher can append upload chunks')
                key = root(body['object_root'])
                if key not in challenge['needed']:
                    raise ValueError('Upload is unrelated to active evidence')
                index = integer(body['index'],0,MAX_OBJECT_BYTES//CHUNK_BYTES-1)
                raw = base64.b64decode(body['data'],validate=True)
                if not 0<len(raw)<=CHUNK_BYTES:
                    raise ValueError('Invalid chunk size')
                chunks = challenge['uploads'].setdefault(key,[])
                if index!=len(chunks) or key in challenge['sealed'] or (chunks and chunks[-1]['size']!=CHUNK_BYTES):
                    raise ValueError('Nonconsecutive or already sealed upload')
                chunks.append({'root':digest(raw),'size':len(raw)})
                if commit_artifacts:
                    artifacts.put(raw)
            elif kind=='seal':
                key = root(body['object_root'])
                chunks = challenge['uploads'].get(key)
                if not chunks or key in challenge['sealed']:
                    raise ValueError('No unsealed upload')
                raw = b''.join(artifacts.get(c['root']) for c in chunks)
                if digest(raw)!=key or len(raw)>MAX_OBJECT_BYTES:
                    raise ValueError('Finalized chunks differ from requested artifact')
                if commit_artifacts:
                    artifacts.put(raw)
                challenge['sealed'].append(key)
                if challenge['kind']=='availability':
                    account(s,challenge['owner'])['balance'] += challenge['bond']
                    claim['challenge'] = None
                    claim['deadline'] = max(s['height']+p['challenge_blocks'], auditing.minimum_deadline(s, claim))
            elif kind=='resolve':
                if challenge['kind']!='fraud' or set(challenge['sealed'])!=set(challenge['needed']):
                    raise ValueError('Every replay input must first be published in native blocks')
                if referee is None:
                    raise ValueError('No objective referee configured')
                verdict = referee(artifacts,metadata,claim['record_root'],challenge['stage'])
                s['audit_count'] += 1
                if verdict['valid']:
                    s['burned'] += challenge['bond']
                    claim['challenge'] = None
                    claim['deadline'] = max(s['height']+p['challenge_blocks'], auditing.minimum_deadline(s, claim))
                else:
                    close(s,False,'objective replay mismatch: '+verdict['mismatch'], proven_fault=True)
    if 'auditing' in s and kind in auditing.CLAIM_KINDS:
        auditing.lock(s, body['audit_budget'], owner, [], s['candidate']['id'])
        auditing.attach(s, body['audit_budget'])
    invariant(s)
    return s
