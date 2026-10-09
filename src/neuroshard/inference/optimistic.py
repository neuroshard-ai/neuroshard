"""Bonded optimistic serving: owners bond per shard, jobs escrow payment, fraud proofs slash.

Serving work is accepted unless an auditor proves fraud within the challenge window.
Validators replay a fraud proof only when a challenge arrives, so honest serving costs
them no recomputation. Owner 0 runs on the user's own device and is not bonded; every
other shard of a job has one bonded owner whose signed log the job commits to. An active
owner publishes the endpoint where users reach its shard, and accounts transfer NEURO.

A committed log answers its job's request. The job registers the user's session key;
the user's device signs the transcript of every message it sends under that key, and
each owner signs the transcript of every message it passes on. An owner commits a log
whose header names the job and carries its upstream sender's signature over the log's
inputs, so a log from any other session is refused, and a log whose entries depart
from what was signed is provable fraud.

A job buys a budget of token positions at its price. Every link signature also covers
the positions the link has carried, so each commitment states, under the upstream
sender's signature, how much work the log covers. Settlement pays each owner for those
positions, never more than the user's device sent or the job bought, and returns the
rest of the escrow to the user.

A challenge takes two transactions. Opening one names a committed log and a proof's
content address and locks a deposit; nothing is replayed. Validators then judge the
open challenge's proof, and a ``prove`` transaction lands it only if it verifies: the
owner is slashed and the deposit returned. A challenge not proven within its proof
window lapses and its deposit is burned, so every replay a challenge causes is paid
for. A job does not settle while a challenge of it is open.
"""

import copy
import hashlib
import json
import re

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from neuroshard.core.crypto.ecdsa import ecdsa_verify

PARAMS = {'fee': 1000, 'owner_bond_minimum': 5_000_000, 'minimum_price': 100_000, 'challenge_blocks': 20,
          'job_blocks': 60, 'auditor_share_ppm': 500_000, 'max_jobs': 32, 'max_results': 128,
          'challenge_deposit': 1_000_000, 'proof_blocks': 20}
FIELDS = {'owner_bond': {'model_root', 'shard', 'log_key', 'amount', 'possession'},
          'owner_unbond': {'log_key'}, 'owner_withdraw': {'log_key'},
          'serve_open': {'model_root', 'owners', 'request_root', 'session_key', 'price', 'positions'},
          'log_commit': {'job_id', 'statement_root', 'entries_root', 'header', 'log_signature'},
          'challenge': {'job_id', 'log_key', 'proof_root'}, 'prove': {'job_id', 'challenge_id'},
          'transfer': {'to', 'amount'}, 'owner_endpoint': {'log_key', 'endpoint'}}
ENDPOINT = re.compile(r'[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?:[1-9][0-9]{0,4}')
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


def endpoint(value):
    """Where users reach an owner's shard: a lowercase host name or IPv4 address and a port."""
    if not isinstance(value, str) or not ENDPOINT.fullmatch(value) or int(value.rsplit(':', 1)[1]) > 65535:
        raise ValueError('Endpoint must be a lowercase host:port')
    return value


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


def link_message(chain_id, job_id, hop, entries, head, positions):
    """What a sender signs with every message: the link's transcript head after ``entries`` steps,
    whose messages carried ``positions`` token positions in all."""
    return canonical({'domain': LINK_DOMAIN, 'chain_id': chain_id, 'job_id': job_id, 'hop': hop,
                      'entries': entries, 'head': head, 'positions': positions})


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
    escrow = sum(j['price'] + sum(c['deposit'] for c in j['challenges'].values()) for j in state['jobs'].values())
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


def billable(job):
    """The token positions each owner of a fully committed job is paid for.

    An owner's are those its committed log covers, as its upstream sender signed them, but
    never more than the user's device sent (owner 1's count, signed by the session key) or
    the positions the job bought.
    """
    sent = min(job['served'][job['owners'][0]], job['positions'])
    return [min(job['served'][key], sent) for key in job['owners']]


def advance(previous, height):
    """Block boundary: burn the deposits of challenges not proven in time, then settle every job
    whose challenge window has closed and refund jobs never fully committed.

    A settled job pays each owner its equal share of the price for every position it served;
    whatever is not paid out returns to the user. A job with an open challenge waits.
    """
    if height != previous['height'] + 1:
        raise ValueError('Nonmonotonic block height')
    state = copy.deepcopy(previous)
    state['height'] = height
    for job_id in sorted(state['jobs']):
        job = state['jobs'][job_id]
        for challenge_id in sorted(job['challenges']):
            lapsed = job['challenges'][challenge_id]
            if height > lapsed['deadline']:
                state['burned'] += lapsed['deposit']
                remember(state, challenge_id, {'status': 'forfeited', 'job_id': job_id, **lapsed})
                del job['challenges'][challenge_id]
        if job['challenges']:
            continue
        if job['deadline'] is not None and height > job['deadline']:
            served = billable(job)
            paid = [job['price'] * positions // (job['positions'] * len(job['owners'])) for positions in served]
            for key, amount in zip(job['owners'], paid):
                account(state, state['owners'][key]['account'])['balance'] += amount
            refunded = job['price'] - sum(paid)
            account(state, job['user'])['balance'] += refunded
            remember(state, job_id, {'status': 'settled', 'user': job['user'], 'owners': job['owners'],
                                     'positions': served, 'paid': paid, 'refunded': refunded})
            del state['jobs'][job_id]
        elif job['deadline'] is None and height > job['expires']:
            refund(state, job)
            remember(state, job_id, {'status': 'expired', 'user': job['user'], 'owners': job['owners'],
                                     'committed': sorted(job['commits'])})
            del state['jobs'][job_id]
    invariant(state)
    return state


def open_jobs(state, key):
    return [job_id for job_id, job in state['jobs'].items() if key in job['owners']]


def challenge_request(state, job_id, challenge):
    """What the proof of an open challenge of ``job_id`` must establish against the committed log."""
    key = challenge['log_key']
    return {'chain_id': state['chain_id'], 'job_id': job_id, 'shard': state['owners'][key]['shard'],
            'log_key': key, 'statement_root': state['jobs'][job_id]['commits'][key], 'proof_root': challenge['proof_root']}


def opening(state, body):
    """Refuse a challenge that cannot open: it must name a committed log in an open window."""
    job, key = state['jobs'].get(body['job_id']), body['log_key']
    if job is None or key not in job['commits']:
        raise ValueError('No committed log to challenge')
    if job['deadline'] is not None and state['height'] > job['deadline']:
        raise ValueError('Challenge window closed')
    hex_digest(body['proof_root'])


def proving(state, body):
    """The request a ``prove`` transaction's proof must establish: that of a challenge opened in an earlier block.

    A challenge opened in the same block has no committed deposit yet, so its proof is refused
    before any replay.
    """
    job = state['jobs'].get(body['job_id'])
    challenge = job['challenges'].get(hex_digest(body['challenge_id'])) if job else None
    if challenge is None:
        raise ValueError('No open challenge to prove')
    if challenge['deadline'] - state['params']['proof_blocks'] >= state['height']:
        raise ValueError('A proof must come in a later block than its challenge')
    return challenge_request(state, body['job_id'], challenge)


def admit(previous, envelope):
    """The checks every transaction passes before it changes state or any proof is replayed.

    Signature, schema, chain, nonce and fee apply to every kind. A challenge must also name
    a committed log in an open window, and a proof an open challenge. Returns the body, the
    sender and, for a proof, the request it must establish. Nothing else is ever replayed.
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
    if kind == 'challenge':
        opening(previous, body)
    return body, sender, proving(previous, body) if kind == 'prove' else None


def upstream_key(job, shard):
    """Who sends shard ``shard`` its inputs: the user's device for shard 1, otherwise the previous shard's owner."""
    return job['session_key'] if shard == 1 else job['owners'][shard - 2]


def bound(state, job, job_id, key, header):
    """Refuse a log that does not answer this job.

    Its header must describe the committing owner, name this chain, job and request, and
    carry the upstream sender's signature over the transcript of the log's inputs. Returns
    the token positions that signature says the inputs carried.
    """
    if not isinstance(header, dict) or set(header) != HEADER_FIELDS:
        raise ValueError('Invalid log header')
    shard = state['owners'][key]['shard']
    if header['format'] != OWNER_LOG_FORMAT or type(header['rank']) is not int or header['rank'] != shard:
        raise ValueError('Log header does not describe this owner')
    if header['session'] != {'chain_id': state['chain_id'], 'job_id': job_id, 'request_root': job['request_root']}:
        raise ValueError('Log is not bound to this job request')
    upstream, sender = header['upstream'], upstream_key(job, shard)
    if not isinstance(upstream, dict) or set(upstream) != {'key', 'entries', 'head', 'positions', 'signature'}:
        raise ValueError('Log inputs are not signed by their upstream sender')
    positions = integer(upstream['positions'], 1)
    message = link_message(state['chain_id'], job_id, shard - 1, integer(upstream['entries'], 1),
                           hex_digest(upstream['head']), positions)
    if upstream['key'] != sender or not isinstance(upstream['signature'], str) or not ed25519_valid(
            sender, message, upstream['signature']):
        raise ValueError('Log inputs are not signed by their upstream sender')
    return positions


def refund(state, job):
    """Return a job's escrow to its user and the deposits of its open challenges to their challengers."""
    account(state, job['user'])['balance'] += job['price']
    for challenge in job['challenges'].values():
        account(state, challenge['challenger'])['balance'] += challenge['deposit']


def transition(previous, envelope, execute):
    """One signed transaction. ``execute(state, request)`` judges a proof of an open challenge.

    It is called only for ``prove``, and only after every check in ``admit`` has passed.
    """
    params = previous['params']
    body, sender, request = admit(previous, envelope)
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
        price, positions = integer(body['price'], params['minimum_price']), integer(body['positions'], 1)
        if len(state['jobs']) >= params['max_jobs']:
            raise ValueError('Job queue is full')
        debit(price)
        state['jobs'][transaction_id(envelope)] = {
            'user': sender, 'owners': list(owners), 'request_root': body['request_root'], 'session_key': session_key,
            'price': price, 'positions': positions, 'opened': state['height'],
            'expires': state['height'] + params['job_blocks'], 'commits': {}, 'served': {}, 'challenges': {},
            'deadline': None}
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
        served = bound(state, job, body['job_id'], key, header)
        if not isinstance(body['log_signature'], str) or not ed25519_valid(
                key, commitment_message(state['chain_id'], body['job_id'], statement_root), body['log_signature']):
            raise ValueError('Log commitment is not signed by the owner log key')
        job['commits'][key] = statement_root
        job['served'][key] = served
        if len(job['commits']) == len(job['owners']):
            job['deadline'] = state['height'] + params['challenge_blocks']
    elif kind == 'transfer':
        recipient, amount = account_key(body['to']), integer(body['amount'], 1)
        debit(amount)
        account(state, recipient)['balance'] += amount
    elif kind == 'owner_endpoint':
        owner = state['owners'].get(body['log_key'])
        if owner is None or owner['account'] != sender or owner['status'] != 'active':
            raise ValueError('Only an active owner can publish its endpoint')
        owner['endpoint'] = endpoint(body['endpoint'])
    elif kind == 'challenge':
        debit(params['challenge_deposit'])
        state['jobs'][body['job_id']]['challenges'][transaction_id(envelope)] = {
            'challenger': sender, 'log_key': body['log_key'], 'proof_root': body['proof_root'],
            'deposit': params['challenge_deposit'], 'deadline': state['height'] + params['proof_blocks']}
    else:
        if execute(previous, request) is not True:
            raise ValueError('Fraud proof does not verify')
        job = state['jobs'].pop(body['job_id'])
        proven = job['challenges'].pop(body['challenge_id'])
        key = proven['log_key']
        owner = state['owners'][key]
        slashed = owner['amount']
        reward = slashed * params['auditor_share_ppm'] // 1_000_000
        account(state, proven['challenger'])['balance'] += proven['deposit'] + reward
        state['burned'] += slashed - reward
        owner.update(amount=0, status='slashed')
        # The job's other challenges are moot: their deposits return with the user's escrow.
        refund(state, job)
        remember(state, body['job_id'], {'status': 'fraud', 'user': job['user'], 'owners': job['owners'],
                                         'guilty': key, 'challenger': proven['challenger'],
                                         'challenge_id': body['challenge_id'], 'slashed': slashed, 'reward': reward,
                                         'proof_root': proven['proof_root']})
        # A proven cheater is paid for nothing else still open.
        for other in sorted(open_jobs(state, key)):
            voided = state['jobs'].pop(other)
            refund(state, voided)
            remember(state, other, {'status': 'voided', 'user': voided['user'], 'owners': voided['owners'],
                                    'guilty': key})
    invariant(state)
    return state
