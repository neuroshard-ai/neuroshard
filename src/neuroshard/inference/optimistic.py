"""Bonded optimistic serving: owners bond per shard, jobs escrow payment, fraud proofs slash.

Serving work is accepted unless an auditor proves fraud within the challenge window.
Validators replay a fraud proof only when a challenge arrives, so honest serving costs
them no recomputation. Owner 0 runs on the user's own device and is not bonded; every
other shard of a job has one bonded owner whose signed log the job commits to.

A committed log answers its job's request. The job registers the user's session key;
the user's device signs the transcript of every message it sends under that key, and
each owner signs the transcript of every message it passes on. An owner commits a log
whose header names the job and carries its upstream sender's signature over the log's
inputs, so a log from any other session is refused, and a log whose entries depart
from what was signed is provable fraud.
"""

import copy
import hashlib
import json

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from neuroshard.core.crypto.ecdsa import ecdsa_verify

PARAMS = {'fee': 1000, 'owner_bond_minimum': 5_000_000, 'minimum_price': 100_000, 'challenge_blocks': 20,
          'job_blocks': 60, 'auditor_share_ppm': 500_000, 'max_jobs': 32, 'max_results': 128}
FIELDS = {'owner_bond': {'model_root', 'shard', 'log_key', 'amount', 'possession'},
          'owner_unbond': {'log_key'}, 'owner_withdraw': {'log_key'},
          'serve_open': {'model_root', 'owners', 'request_root', 'session_key', 'price'},
          'log_commit': {'job_id', 'statement_root', 'entries_root', 'header', 'log_signature'},
          'challenge': {'job_id', 'log_key', 'proof_root'}}
OWNER_LOG_FORMAT = 'neuroshard-granite-owner-log/2'
HEADER_FIELDS = {'format', 'rank', 'public_key', 'session', 'upstream'}
LINK_DOMAIN = 'neuroshard/serving-link/v1'


class NoVerdict(Exception):
    """This validator cannot judge a challenged proof now. That is no evidence about the proof, and never a rejection."""


class ProofUnavailable(NoVerdict):
    """This validator does not hold the bytes at the challenged content address: missing, incomplete or different."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def integer(value, minimum=0, maximum=2 ** 60):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError('Integer outside protocol bounds')
    return value


def hex_digest(value):
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise ValueError('Noncanonical digest')
    bytes.fromhex(value)
    return value


def account_key(value):
    if not isinstance(value, str) or len(value) != 66 or value != value.lower():
        raise ValueError('Noncanonical account key')
    ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(), bytes.fromhex(value))
    return value


def ed25519_key(value):
    hex_digest(value)
    Ed25519PublicKey.from_public_bytes(bytes.fromhex(value))
    return value


log_key = ed25519_key


def ed25519_valid(public, message, signature):
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public)).verify(bytes.fromhex(signature), message)
        return True
    except Exception:
        return False


def verify(envelope):
    """The body and signing account of a secp256k1-signed envelope."""
    if not isinstance(envelope, dict) or set(envelope) != {'body', 'public_key', 'signature'}:
        raise ValueError('Invalid signed envelope')
    if not isinstance(envelope['body'], dict) or not isinstance(envelope['signature'], str):
        raise ValueError('Invalid signed envelope')
    public = account_key(envelope['public_key'])
    if not ecdsa_verify(canonical(envelope['body']).decode(), envelope['signature'], bytes.fromhex(public)):
        raise ValueError('Signature verification failed')
    return envelope['body'], public


def transaction_id(envelope):
    return digest({'body': envelope['body'], 'public_key': envelope['public_key']})


def possession_message(chain_id, account, key, shard, amount, nonce):
    return canonical({'domain': 'neuroshard/owner-bond/v1', 'chain_id': chain_id, 'account': account, 'log_key': key,
                      'shard': shard, 'amount': amount, 'nonce': nonce})


def commitment_message(chain_id, job_id, statement_root):
    return canonical({'domain': 'neuroshard/owner-log/v1', 'chain_id': chain_id, 'job_id': job_id,
                      'statement_root': statement_root})


def link_seed(chain_id, job_id, hop):
    """The empty transcript of one serving link of a job: messages from owner ``hop`` to the next owner."""
    return digest({'domain': LINK_DOMAIN, 'chain_id': chain_id, 'job_id': job_id, 'hop': hop})


def link_step(head, op, value, tensor=None):
    """The transcript after one more command, or one more message whose tensor has digest ``tensor``."""
    return digest({'head': head, 'op': op, 'value': value, 'tensor': tensor})


def link_message(chain_id, job_id, hop, entries, head):
    """What a sender signs with every message: the link's transcript head after ``entries`` steps."""
    return canonical({'domain': LINK_DOMAIN, 'chain_id': chain_id, 'job_id': job_id, 'hop': hop,
                      'entries': entries, 'head': head})


def log_statement(header, entries_root):
    """The digest an owner commits for a log: its binding header and the digest of its entries."""
    return digest({'domain': 'neuroshard/owner-log/v2', 'header': header, 'entries_root': entries_root})


def genesis(chain_id, model_root, shards, allocations, params=None):
    """``shards`` is the number of owner partitions; owners 1 to shards - 1 are bonded."""
    state = {'chain_id': chain_id, 'height': 0, 'params': dict(params or PARAMS), 'model_root': hex_digest(model_root),
             'shards': integer(shards, 2, 64), 'accounts': {}, 'owners': {}, 'jobs': {}, 'results': {},
             'initial_supply': 0, 'burned': 0}
    for owner, amount in sorted(allocations.items()):
        state['accounts'][account_key(owner)] = {'balance': integer(amount), 'nonce': 0}
        state['initial_supply'] += amount
    invariant(state)
    return state


def root(state):
    return digest(state)


def invariant(state):
    liquid = sum(a['balance'] for a in state['accounts'].values())
    bonded = sum(o['amount'] for o in state['owners'].values())
    escrow = sum(j['price'] for j in state['jobs'].values())
    assert all(a['balance'] >= 0 and a['nonce'] >= 0 for a in state['accounts'].values())
    assert all(o['amount'] >= 0 for o in state['owners'].values())
    assert state['initial_supply'] == liquid + bonded + escrow + state['burned']
    assert len(state['jobs']) <= state['params']['max_jobs']
    assert len(state['results']) <= state['params']['max_results']


def remember(state, key, value):
    state['results'][key] = {**value, 'height': state['height']}
    while len(state['results']) > state['params']['max_results']:
        oldest = min(state['results'], key=lambda k: (state['results'][k]['height'], k))
        del state['results'][oldest]


def account(state, owner):
    return state['accounts'].setdefault(owner, {'balance': 0, 'nonce': 0})


def advance(previous, height):
    """Block boundary: settle every job whose challenge window has closed; refund jobs never fully committed."""
    if height != previous['height'] + 1:
        raise ValueError('Nonmonotonic block height')
    state = copy.deepcopy(previous)
    state['height'] = height
    for job_id in sorted(state['jobs']):
        job = state['jobs'][job_id]
        if job['deadline'] is not None and height > job['deadline']:
            share = job['price'] // len(job['owners'])
            for key in job['owners']:
                account(state, state['owners'][key]['account'])['balance'] += share
            state['burned'] += job['price'] - share * len(job['owners'])
            remember(state, job_id, {'status': 'settled', 'user': job['user'], 'owners': job['owners'],
                                     'paid_each': share})
            del state['jobs'][job_id]
        elif job['deadline'] is None and height > job['expires']:
            account(state, job['user'])['balance'] += job['price']
            remember(state, job_id, {'status': 'expired', 'user': job['user'], 'owners': job['owners'],
                                     'committed': sorted(job['commits'])})
            del state['jobs'][job_id]
    invariant(state)
    return state


def open_jobs(state, key):
    return [job_id for job_id, job in state['jobs'].items() if key in job['owners']]


def challenge_request(state, body):
    """What a challenge's proof must establish; raises when the challenge cannot apply, without any replay."""
    job, key = state['jobs'].get(body['job_id']), body['log_key']
    if job is None or key not in job['commits']:
        raise ValueError('No committed log to challenge')
    if job['deadline'] is not None and state['height'] > job['deadline']:
        raise ValueError('Challenge window closed')
    return {'chain_id': state['chain_id'], 'job_id': body['job_id'], 'shard': state['owners'][key]['shard'],
            'log_key': key, 'statement_root': job['commits'][key], 'proof_root': hex_digest(body['proof_root'])}


def admit(previous, envelope):
    """The checks every transaction passes before it changes state or any proof is replayed.

    Signature, schema, chain, nonce and fee apply to every kind; a challenge must also name
    a committed log in an open window. Returns the body, the sender and, for a challenge,
    the request its proof must establish.
    """
    body, sender = verify(envelope)
    kind = body.get('kind')
    if kind not in FIELDS or set(body) != {'kind', 'chain_id', 'nonce'} | FIELDS[kind]:
        raise ValueError('Invalid transaction schema')
    if body['chain_id'] != previous['chain_id']:
        raise ValueError('Wrong chain')
    payer = previous['accounts'].get(sender, {'balance': 0, 'nonce': 0})
    if integer(body['nonce']) != payer['nonce']:
        raise ValueError('Wrong account nonce')
    if payer['balance'] < previous['params']['fee']:
        raise ValueError('Insufficient transaction fee')
    return body, sender, challenge_request(previous, body) if kind == 'challenge' else None


def upstream_key(job, shard):
    """Who sends shard ``shard`` its inputs: the user's device for shard 1, otherwise the previous shard's owner."""
    return job['session_key'] if shard == 1 else job['owners'][shard - 2]


def bound(state, job, job_id, key, header):
    """Refuse a log that does not answer this job.

    Its header must describe the committing owner, name this chain, job and request, and
    carry the upstream sender's signature over the transcript of the log's inputs.
    """
    if not isinstance(header, dict) or set(header) != HEADER_FIELDS:
        raise ValueError('Invalid log header')
    shard = state['owners'][key]['shard']
    if header['format'] != OWNER_LOG_FORMAT or type(header['rank']) is not int or header['rank'] != shard:
        raise ValueError('Log header does not describe this owner')
    if header['session'] != {'chain_id': state['chain_id'], 'job_id': job_id, 'request_root': job['request_root']}:
        raise ValueError('Log is not bound to this job request')
    upstream, sender = header['upstream'], upstream_key(job, shard)
    if not isinstance(upstream, dict) or set(upstream) != {'key', 'entries', 'head', 'signature'}:
        raise ValueError('Log inputs are not signed by their upstream sender')
    message = link_message(state['chain_id'], job_id, shard - 1, integer(upstream['entries'], 1),
                           hex_digest(upstream['head']))
    if upstream['key'] != sender or not isinstance(upstream['signature'], str) or not ed25519_valid(
            sender, message, upstream['signature']):
        raise ValueError('Log inputs are not signed by their upstream sender')


def transition(previous, envelope, execute):
    """One signed transaction. ``execute(state, request)`` judges a challenge's proof.

    It is called only for challenges, and only after every check in ``admit`` has passed.
    """
    params = previous['params']
    body, sender, challenge = admit(previous, envelope)
    kind, nonce = body['kind'], body['nonce']
    state = copy.deepcopy(previous)
    payer = account(state, sender)
    payer['balance'] -= params['fee']
    payer['nonce'] += 1
    state['burned'] += params['fee']

    def debit(amount):
        if payer['balance'] < amount:
            raise ValueError('Insufficient spendable balance')
        payer['balance'] -= amount

    if kind == 'owner_bond':
        key, shard = log_key(body['log_key']), integer(body['shard'], 1, state['shards'] - 1)
        amount = integer(body['amount'], params['owner_bond_minimum'])
        if body['model_root'] != state['model_root'] or key in state['owners']:
            raise ValueError('Wrong model or previously used log key')
        if not isinstance(body['possession'], str) or not ed25519_valid(
                key, possession_message(state['chain_id'], sender, key, shard, amount, nonce), body['possession']):
            raise ValueError('Log-key possession proof failed')
        debit(amount)
        state['owners'][key] = {'account': sender, 'shard': shard, 'amount': amount, 'status': 'active',
                                'release_height': None}
    elif kind in ('owner_unbond', 'owner_withdraw'):
        owner = state['owners'].get(body['log_key'])
        if owner is None or owner['account'] != sender:
            raise ValueError('Owner bond belongs to a different account')
        if open_jobs(state, body['log_key']):
            raise ValueError('Owner is named in an unsettled job')
        if kind == 'owner_unbond':
            if owner['status'] != 'active':
                raise ValueError('Only an active owner can start withdrawal')
            owner.update(status='leaving', release_height=state['height'] + params['challenge_blocks'])
        else:
            if owner['status'] != 'leaving' or state['height'] < owner['release_height']:
                raise ValueError('Owner bond is still exposed to challenges')
            payer['balance'] += owner['amount']
            owner.update(amount=0, status='withdrawn')
    elif kind == 'serve_open':
        owners = body['owners']
        if body['model_root'] != state['model_root'] or not isinstance(owners, list) or len(owners) != state['shards'] - 1:
            raise ValueError('A job names one bonded owner per audited shard of the current model')
        for shard, key in enumerate(owners, start=1):
            owner = state['owners'].get(key)
            if owner is None or owner['status'] != 'active' or owner['shard'] != shard:
                raise ValueError('Job owners must be active and bonded for their shards in order')
        hex_digest(body['request_root'])
        session_key = ed25519_key(body['session_key'])
        price = integer(body['price'], params['minimum_price'])
        if len(state['jobs']) >= params['max_jobs']:
            raise ValueError('Job queue is full')
        debit(price)
        state['jobs'][transaction_id(envelope)] = {
            'user': sender, 'owners': list(owners), 'request_root': body['request_root'], 'session_key': session_key,
            'price': price, 'opened': state['height'], 'expires': state['height'] + params['job_blocks'],
            'commits': {}, 'deadline': None}
    elif kind == 'log_commit':
        job = state['jobs'].get(body['job_id'])
        if job is None or job['deadline'] is not None:
            raise ValueError('No job awaiting log commitments')
        named = [key for key in job['owners'] if state['owners'][key]['account'] == sender and key not in job['commits']]
        if not named:
            raise ValueError('Sender has no uncommitted owner in this job')
        header = body['header']
        key = header.get('public_key') if isinstance(header, dict) else None
        if key not in named:
            raise ValueError('Log header names no uncommitted owner of the sender')
        statement_root = hex_digest(body['statement_root'])
        if log_statement(header, hex_digest(body['entries_root'])) != statement_root:
            raise ValueError('Log statement does not match its header')
        bound(state, job, body['job_id'], key, header)
        if not isinstance(body['log_signature'], str) or not ed25519_valid(
                key, commitment_message(state['chain_id'], body['job_id'], statement_root), body['log_signature']):
            raise ValueError('Log commitment is not signed by the owner log key')
        job['commits'][key] = statement_root
        if len(job['commits']) == len(job['owners']):
            job['deadline'] = state['height'] + params['challenge_blocks']
    else:
        job, key = state['jobs'][body['job_id']], body['log_key']
        owner = state['owners'][key]
        if execute(previous, challenge) is not True:
            raise ValueError('Fraud proof does not verify')
        slashed = owner['amount']
        reward = slashed * params['auditor_share_ppm'] // 1_000_000
        payer['balance'] += reward
        state['burned'] += slashed - reward
        owner.update(amount=0, status='slashed')
        account(state, job['user'])['balance'] += job['price']
        remember(state, body['job_id'], {'status': 'fraud', 'user': job['user'], 'owners': job['owners'],
                                         'guilty': key, 'challenger': sender, 'slashed': slashed, 'reward': reward,
                                         'proof_root': body['proof_root']})
        del state['jobs'][body['job_id']]
        # A proven cheater is paid for nothing else still open.
        for other in sorted(open_jobs(state, key)):
            voided = state['jobs'].pop(other)
            account(state, voided['user'])['balance'] += voided['price']
            remember(state, other, {'status': 'voided', 'user': voided['user'], 'owners': voided['owners'],
                                    'guilty': key})
    invariant(state)
    return state
