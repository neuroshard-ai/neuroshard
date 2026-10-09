import hashlib
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from neuroshard.core.crypto.ecdsa import derive_keypair_from_token, ecdsa_sign
from neuroshard.inference import optimistic as ledger

CHAIN = 'neuroshard-optimistic-test'
MODEL = 'ab' * 32
PARAMS = {**ledger.PARAMS, 'challenge_blocks': 3, 'job_blocks': 6, 'proof_blocks': 4}
DEPOSIT = PARAMS['challenge_deposit']
# Who sends each shard its inputs: the user's device under its session key, then owner 1.
UPSTREAM = {1: 'session', 2: 'log-1'}
# The token positions a test job buys; the default logs cover all of them.
POSITIONS = 8


class Account:
    def __init__(self, name):
        self.key = derive_keypair_from_token(hashlib.sha256(name.encode()).hexdigest())
        self.public = self.key.public_key_bytes.hex()
        self.nonce = 0

    def sign(self, kind, **fields):
        body = {'kind': kind, 'chain_id': CHAIN, 'nonce': self.nonce, **fields}
        self.nonce += 1
        return {'body': body, 'public_key': self.public,
                'signature': ecdsa_sign(ledger.canonical(body).decode(), self.key.private_key_bytes)}


def log_identity(name):
    key = Ed25519PrivateKey.from_private_bytes(hashlib.sha256(name.encode()).digest())
    public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
    return key, public


def bond(account, shard, name, amount=PARAMS['owner_bond_minimum']):
    key, public = log_identity(name)
    possession = key.sign(ledger.possession_message(CHAIN, account.public, public, shard, amount, account.nonce)).hex()
    return public, account.sign('owner_bond', model_root=MODEL, shard=shard, log_key=public, amount=amount,
                                possession=possession)


def open_job(account, keys, request='cd' * 32, price=1_000_000, positions=POSITIONS):
    return account.sign('serve_open', model_root=MODEL, owners=[keys[1], keys[2]], request_root=request,
                        session_key=log_identity('session')[1], price=price, positions=positions)


def header(job_id, shard, request='cd' * 32, entries=4, head='77' * 32, signer=None, hop=None, chain=CHAIN,
           positions=POSITIONS):
    """A bound log's header: its owner, its job and request, and the upstream signature over its inputs."""
    key, public = log_identity(signer or UPSTREAM[shard])
    message = ledger.link_message(chain, job_id, shard - 1 if hop is None else hop, entries, head, positions)
    return {'format': ledger.OWNER_LOG_FORMAT, 'rank': shard, 'public_key': log_identity(f'log-{shard}')[1],
            'session': {'chain_id': chain, 'job_id': job_id, 'request_root': request},
            'upstream': {'key': public, 'entries': entries, 'head': head, 'positions': positions,
                         'signature': key.sign(message).hex()}}


def statement(job_id, shard, entries_root, **binding):
    return ledger.log_statement(header(job_id, shard, **binding), entries_root)


def commit(account, shard, job_id, head=None, entries_root=None, signer=None):
    entries_root = entries_root or f'{shard}{shard}' * 32
    head = head or header(job_id, shard)
    root = ledger.log_statement(head, entries_root)
    key, _ = log_identity(signer or f'log-{shard}')
    return account.sign('log_commit', job_id=job_id, statement_root=root, entries_root=entries_root, header=head,
                        log_signature=key.sign(ledger.commitment_message(CHAIN, job_id, root)).hex())


@pytest.fixture
def market():
    people = {name: Account(name) for name in ('user', 'owner-1', 'owner-2', 'auditor')}
    state = ledger.genesis(CHAIN, MODEL, 3, {p.public: 20_000_000 for p in people.values()}, PARAMS)
    keys = {}
    for shard, name in ((1, 'owner-1'), (2, 'owner-2')):
        keys[shard], envelope = bond(people[name], shard, f'log-{shard}')
        state = ledger.transition(state, envelope, None)
    return state, people, keys


def refuse(state, envelope, match, execute=None):
    with pytest.raises(ValueError, match=match):
        ledger.transition(state, envelope, execute)


def opened(state, people, keys, price=1_000_000, request='cd' * 32, positions=POSITIONS):
    envelope = open_job(people['user'], keys, request, price, positions)
    return ledger.transition(state, envelope, None), ledger.transaction_id(envelope)


def settle(state, job_id):
    while job_id in state['jobs']:
        state = ledger.advance(state, state['height'] + 1)
    return state


def committed(state, people, job_id, request='cd' * 32):
    for shard in (1, 2):
        state = ledger.transition(state, commit(people[f'owner-{shard}'], shard, job_id, header(job_id, shard, request)),
                                  None)
    return state


def challenge(account, job_id, log_key, proof_root='ee' * 32):
    """A challenge of the log ``log_key`` committed for ``job_id``, and the ID a proof of it names."""
    envelope = account.sign('challenge', job_id=job_id, log_key=log_key, proof_root=proof_root)
    return envelope, ledger.transaction_id(envelope)


def prove(account, job_id, challenge_id):
    return account.sign('prove', job_id=job_id, challenge_id=challenge_id)


def test_an_honest_job_settles_after_the_challenge_window_and_conserves_money(market):
    state, people, keys = market
    state, job_id = opened(state, people, keys)
    state = committed(state, people, job_id)
    deadline = state['jobs'][job_id]['deadline']
    assert deadline == state['height'] + PARAMS['challenge_blocks']
    before = {name: state['accounts'][p.public]['balance'] for name, p in people.items()}
    while state['height'] <= deadline:
        state = ledger.advance(state, state['height'] + 1)
        assert (job_id in state['jobs']) == (state['height'] <= deadline)
    assert state['results'][job_id] == {**state['results'][job_id], 'status': 'settled', 'positions': [8, 8],
                                        'paid': [500_000, 500_000], 'refunded': 0}
    for shard in (1, 2):
        assert state['accounts'][people[f'owner-{shard}'].public]['balance'] == before[f'owner-{shard}'] + 500_000
    assert state['accounts'][people['user'].public]['balance'] == before['user']


def test_a_settled_job_pays_for_the_positions_its_logs_cover_and_refunds_the_rest(market):
    state, people, keys = market
    for budget, served, billed, paid in (
            # Both owners commit a one-message prefix of the session.
            (8, (1, 1), [1, 1], [62_500, 62_500]),
            # Owner 2's log covers less than owner 1's.
            (8, (8, 3), [8, 3], [500_000, 187_500]),
            # Owner 2 is paid for no more than the user's device sent to owner 1,
            (8, (2, 6), [2, 2], [125_000, 125_000]),
            # and no owner for more than the job bought.
            (8, (20, 20), [8, 8], [500_000, 500_000]),
            # What rounding leaves unpaid goes back to the user.
            (3, (1, 1), [1, 1], [166_666, 166_666])):
        state, job_id = opened(state, people, keys, positions=budget)
        for shard in (1, 2):
            state = ledger.transition(state, commit(people[f'owner-{shard}'], shard, job_id,
                                                    header(job_id, shard, positions=served[shard - 1])), None)
        names = ('user', 'owner-1', 'owner-2')
        before = [state['accounts'][people[name].public]['balance'] for name in names]
        state = settle(state, job_id)
        refunded = 1_000_000 - sum(paid)
        assert state['results'][job_id] == {**state['results'][job_id], 'status': 'settled', 'positions': billed,
                                            'paid': paid, 'refunded': refunded}
        assert [state['accounts'][people[name].public]['balance'] for name in names] == [
            before[0] + refunded, before[1] + paid[0], before[2] + paid[1]]


def test_a_job_buys_a_positive_number_of_positions(market):
    state, people, keys = market
    for bad in (0, -1, 1.5, True, None):
        refuse(state, open_job(people['user'], keys, positions=bad), 'outside protocol bounds')
        people['user'].nonce -= 1
    unpriced = people['user'].sign('serve_open', model_root=MODEL, owners=[keys[1], keys[2]], request_root='cd' * 32,
                                   session_key=log_identity('session')[1], price=1_000_000)
    refuse(state, unpriced, 'Invalid transaction schema')
    people['user'].nonce -= 1
    state, job_id = opened(state, people, keys, positions=5)
    assert state['jobs'][job_id]['positions'] == 5 and state['jobs'][job_id]['served'] == {}


def test_owner_bonds_need_their_log_key_and_stay_exposed_while_named_or_within_the_window(market):
    state, people, keys = market
    stranger = Account('stranger')
    key, public = log_identity('log-x')
    forged = stranger.sign('owner_bond', model_root=MODEL, shard=1, log_key=public, amount=PARAMS['owner_bond_minimum'],
                           possession=key.sign(b'other message').hex())
    refuse(ledger.genesis(CHAIN, MODEL, 3, {stranger.public: 10_000_000}, PARAMS), forged, 'possession')
    reused = bond(people['owner-2'], 2, 'log-1')[1]
    refuse(state, reused, 'previously used')
    people['owner-2'].nonce -= 1
    state, job_id = opened(state, people, keys)
    refuse(state, people['owner-1'].sign('owner_unbond', log_key=keys[1]), 'unsettled job')
    people['owner-1'].nonce -= 1
    state = committed(state, people, job_id)
    for _ in range(PARAMS['challenge_blocks'] + 1):
        state = ledger.advance(state, state['height'] + 1)
    state = ledger.transition(state, people['owner-1'].sign('owner_unbond', log_key=keys[1]), None)
    refuse(state, open_job(people['user'], keys), 'active and bonded')
    people['user'].nonce -= 1
    refuse(state, people['owner-1'].sign('owner_withdraw', log_key=keys[1]), 'still exposed')
    people['owner-1'].nonce -= 1
    for _ in range(PARAMS['challenge_blocks']):
        state = ledger.advance(state, state['height'] + 1)
    balance = state['accounts'][people['owner-1'].public]['balance']
    state = ledger.transition(state, people['owner-1'].sign('owner_withdraw', log_key=keys[1]), None)
    assert state['owners'][keys[1]]['status'] == 'withdrawn'
    assert state['accounts'][people['owner-1'].public]['balance'] == balance - PARAMS['fee'] + PARAMS['owner_bond_minimum']


def test_an_active_owner_publishes_where_users_reach_its_shard(market):
    state, people, keys = market
    for address in ('shard1.example.org:28700', '203.0.113.7:28701'):
        state = ledger.transition(state, people['owner-1'].sign('owner_endpoint', log_key=keys[1], endpoint=address), None)
        assert state['owners'][keys[1]]['endpoint'] == address
    refuse(state, people['user'].sign('owner_endpoint', log_key=keys[1], endpoint='other.example:1'), 'active owner')
    for bad in ('shard2.example.org', 'Shard2.example.org:1', 'host:0', 'host:65536', 'host:80/path', 'a' * 300 + ':1', 7):
        refuse(state, people['owner-2'].sign('owner_endpoint', log_key=keys[2], endpoint=bad), 'host:port')
        people['owner-2'].nonce -= 1
    state = ledger.transition(state, people['owner-2'].sign('owner_unbond', log_key=keys[2]), None)
    refuse(state, people['owner-2'].sign('owner_endpoint', log_key=keys[2], endpoint='shard2.example.org:1'), 'active owner')


def test_accounts_transfer_neuro_and_conserve_money(market):
    state, people, keys = market
    newcomer = Account('newcomer')
    state = ledger.transition(state, people['user'].sign('transfer', to=newcomer.public, amount=3_000_000), None)
    assert state['accounts'][newcomer.public] == {'balance': 3_000_000, 'nonce': 0}
    assert state['accounts'][people['user'].public]['balance'] == 20_000_000 - 3_000_000 - PARAMS['fee']
    for bad in (0, -1, '5', None):
        refuse(state, people['user'].sign('transfer', to=newcomer.public, amount=bad), 'Integer')
        people['user'].nonce -= 1
    with pytest.raises(ValueError):
        ledger.transition(state, people['user'].sign('transfer', to='00' * 33, amount=1), None)
    refuse(state, newcomer.sign('transfer', to=people['user'].public, amount=3_000_000), 'spendable')


def test_a_job_registers_a_valid_session_key(market):
    state, people, keys = market
    for bad in ('cd' * 31, 'AB' * 32, 7, None):
        envelope = people['user'].sign('serve_open', model_root=MODEL, owners=[keys[1], keys[2]], request_root='cd' * 32,
                                       session_key=bad, price=1_000_000, positions=POSITIONS)
        people['user'].nonce -= 1
        with pytest.raises(ValueError):
            ledger.transition(state, envelope, None)
    state, job_id = opened(state, people, keys)
    assert state['jobs'][job_id]['session_key'] == log_identity('session')[1]


def test_commitments_must_come_from_a_named_owner_and_be_signed_by_its_log_key(market):
    state, people, keys = market
    state, job_id = opened(state, people, keys)
    refuse(state, commit(people['auditor'], 1, job_id), 'no uncommitted owner')
    refuse(state, commit(people['owner-1'], 2, job_id), 'names no uncommitted owner of the sender')
    people['owner-1'].nonce -= 1
    refuse(state, commit(people['owner-1'], 1, job_id, signer='log-2'), 'not signed by the owner log key')
    people['owner-1'].nonce -= 1
    state = ledger.transition(state, commit(people['owner-1'], 1, job_id), None)
    refuse(state, commit(people['owner-1'], 1, job_id, entries_root='33' * 32), 'no uncommitted owner')
    assert state['jobs'][job_id]['deadline'] is None
    assert state['jobs'][job_id]['commits'] == {keys[1]: statement(job_id, 1, '11' * 32)}


def test_a_log_that_does_not_answer_this_jobs_request_is_refused_at_commitment(market):
    state, people, keys = market
    state, old = opened(state, people, keys)
    state = committed(state, people, old)
    state, job_id = opened(state, people, keys, request='ef' * 32)
    stale = header(old, 1)
    for head, match in (
            # The internally correct log of an earlier job, reused verbatim.
            (stale, 'not bound to this job request'),
            # Relabelled for this job, but its inputs were signed for the earlier session only.
            ({**header(job_id, 1, 'ef' * 32), 'upstream': stale['upstream']}, 'not signed by their upstream sender'),
            # This job with another request, or another chain.
            (header(job_id, 1, 'cd' * 32), 'not bound to this job request'),
            (header(job_id, 1, 'ef' * 32, chain='another-chain'), 'not bound to this job request'),
            # Inputs signed by anyone but the user's session key, or for another link of the job.
            (header(job_id, 1, 'ef' * 32, signer='log-2'), 'not signed by their upstream sender'),
            (header(job_id, 1, 'ef' * 32, hop=1), 'not signed by their upstream sender'),
            ({**header(job_id, 1, 'ef' * 32), 'rank': 2}, 'does not describe this owner'),
            ({**header(job_id, 1, 'ef' * 32), 'format': 'neuroshard-granite-owner-log/1'}, 'does not describe this owner'),
            ({**header(job_id, 1, 'ef' * 32), 'upstream': None}, 'not signed by their upstream sender'),
            # A count of positions the upstream sender did not sign, or none at all.
            ({**header(job_id, 1, 'ef' * 32), 'upstream': {**header(job_id, 1, 'ef' * 32)['upstream'], 'positions': 9}},
             'not signed by their upstream sender'),
            (header(job_id, 1, 'ef' * 32, positions=0), 'outside protocol bounds'),
            ({**header(job_id, 1, 'ef' * 32), 'extra': 1}, 'Invalid log header')):
        refuse(state, commit(people['owner-1'], 1, job_id, head), match)
        people['owner-1'].nonce -= 1
    # Owner 2's inputs come from owner 1, so the user's session key cannot vouch for them.
    refuse(state, commit(people['owner-2'], 2, job_id, header(job_id, 2, 'ef' * 32, signer='session')),
           'not signed by their upstream sender')
    people['owner-2'].nonce -= 1
    # The committed statement must be the one the header and entries produce.
    body = commit(people['owner-1'], 1, job_id, header(job_id, 1, 'ef' * 32))['body']
    people['owner-1'].nonce -= 1
    key, _ = log_identity('log-1')
    forged = people['owner-1'].sign('log_commit', job_id=job_id, statement_root='00' * 32, entries_root=body['entries_root'],
                                    header=body['header'],
                                    log_signature=key.sign(ledger.commitment_message(CHAIN, job_id, '00' * 32)).hex())
    refuse(state, forged, 'does not match its header')
    people['owner-1'].nonce -= 1
    state = committed(state, people, job_id, 'ef' * 32)
    assert set(state['jobs'][job_id]['commits']) == {keys[1], keys[2]}
    assert state['jobs'][job_id]['served'] == {keys[1]: POSITIONS, keys[2]: POSITIONS}


def test_a_verified_fraud_proof_slashes_the_owner_pays_the_auditor_and_refunds_every_affected_user(market):
    state, people, keys = market
    state, first = opened(state, people, keys)
    state, second = opened(state, people, keys, price=2_000_000)
    state = committed(committed(state, people, first), people, second)
    seen = []

    def execute(previous, request):
        seen.append(request)
        return True

    balance = lambda name: state['accounts'][people[name].public]['balance']
    user, auditor, other = balance('user'), balance('auditor'), balance('owner-2')
    opening, challenge_id = challenge(people['auditor'], first, keys[1])
    state = ledger.transition(state, opening, execute)
    # Opening a challenge locks its deposit and replays nothing.
    assert seen == [] and balance('auditor') == auditor - PARAMS['fee'] - DEPOSIT
    # Other challenges of the proven job and of the voided one are moot.
    state = ledger.transition(state, challenge(people['owner-2'], first, keys[1], 'dd' * 32)[0], None)
    state = ledger.transition(state, challenge(people['user'], second, keys[2], 'dd' * 32)[0], None)
    # A proof in the challenge's own block is refused before any replay: its deposit is not committed yet.
    refuse(state, prove(people['owner-2'], first, challenge_id), 'later block', execute)
    people['owner-2'].nonce -= 1
    assert seen == []
    state = ledger.advance(state, state['height'] + 1)
    state = ledger.transition(state, prove(people['owner-2'], first, challenge_id), execute)
    assert seen == [{'chain_id': CHAIN, 'job_id': first, 'shard': 1, 'log_key': keys[1],
                     'statement_root': statement(first, 1, '11' * 32), 'proof_root': 'ee' * 32}]
    reward = PARAMS['owner_bond_minimum'] * PARAMS['auditor_share_ppm'] // 1_000_000
    assert state['owners'][keys[1]] == {**state['owners'][keys[1]], 'amount': 0, 'status': 'slashed'}
    # The reward goes to whoever opened the challenge, whoever submits its proof.
    assert balance('auditor') == auditor - PARAMS['fee'] + reward
    assert balance('owner-2') == other - 2 * PARAMS['fee']
    assert balance('user') == user - PARAMS['fee'] + 1_000_000 + 2_000_000
    assert state['results'][first] == {**state['results'][first], 'status': 'fraud', 'guilty': keys[1],
                                       'challenger': people['auditor'].public, 'challenge_id': challenge_id}
    assert state['results'][second]['status'] == 'voided' and not state['jobs']
    refuse(state, open_job(people['user'], keys), 'active and bonded')


def test_an_unproven_challenge_forfeits_its_deposit_and_holds_settlement_only_until_it_lapses(market):
    state, people, keys = market
    state, job_id = opened(state, people, keys)
    called = []
    replay = lambda *a: called.append(a) or True
    refuse(state, challenge(people['auditor'], job_id, keys[1])[0], 'No committed log', replay)
    people['auditor'].nonce -= 1
    state = committed(state, people, job_id)
    deadline = state['jobs'][job_id]['deadline']
    opening, challenge_id = challenge(people['auditor'], job_id, keys[1])
    state = ledger.transition(state, opening, replay)
    lapse = state['height'] + PARAMS['proof_blocks']
    assert lapse > deadline and not called
    assert state['jobs'][job_id]['challenges'] == {challenge_id: {
        'challenger': people['auditor'].public, 'log_key': keys[1], 'proof_root': 'ee' * 32, 'deposit': DEPOSIT,
        'deadline': lapse}}
    refuse(state, prove(people['auditor'], job_id, challenge_id), 'later block', replay)
    people['auditor'].nonce -= 1
    state = ledger.advance(state, state['height'] + 1)
    refuse(state, prove(people['auditor'], job_id, challenge_id), 'does not verify', lambda previous, request: False)
    people['auditor'].nonce -= 1
    burned, owner = state['burned'], state['accounts'][people['owner-1'].public]['balance']
    while state['height'] < lapse:
        state = ledger.advance(state, state['height'] + 1)
        assert job_id in state['jobs']
    # The window has closed, but the open challenge holds the job; no new challenge may open.
    refuse(state, challenge(people['auditor'], job_id, keys[2])[0], 'Challenge window closed')
    people['auditor'].nonce -= 1
    state = ledger.advance(state, state['height'] + 1)
    assert state['results'][challenge_id] == {**state['results'][challenge_id], 'status': 'forfeited',
                                              'job_id': job_id, 'challenger': people['auditor'].public,
                                              'deposit': DEPOSIT}
    assert state['burned'] == burned + DEPOSIT and state['results'][job_id]['status'] == 'settled'
    assert state['accounts'][people['owner-1'].public]['balance'] == owner + 500_000
    refuse(state, prove(people['auditor'], job_id, challenge_id), 'No open challenge', replay)
    people['auditor'].nonce -= 1
    refuse(state, challenge(people['auditor'], job_id, keys[1])[0], 'No committed log', replay)
    assert not called


def test_a_challenge_needs_its_deposit(market):
    state, people, keys = market
    state, job_id = opened(state, people, keys)
    state = committed(state, people, job_id)
    poor = Account('poor')
    state['accounts'][poor.public] = {'balance': PARAMS['fee'] + DEPOSIT - 1, 'nonce': 0}
    state['initial_supply'] += PARAMS['fee'] + DEPOSIT - 1
    refuse(state, challenge(poor, job_id, keys[1])[0], 'Insufficient spendable balance')
    state['accounts'][poor.public]['balance'] += 1
    state['initial_supply'] += 1
    poor.nonce = 0
    state = ledger.transition(state, challenge(poor, job_id, keys[1])[0], None)
    assert state['accounts'][poor.public]['balance'] == 0


def challenges_failing_admission(state, people, keys, job_id, pending):
    """A challenge failing each cheap admission check in turn, with the error each must give."""
    auditor = people['auditor']

    def signed(account, nonce=0, chain=CHAIN, **fields):
        body = {'kind': 'challenge', 'chain_id': chain, 'nonce': nonce, 'job_id': job_id, 'log_key': keys[1],
                'proof_root': 'ee' * 32, **fields}
        return {'body': body, 'public_key': account.public,
                'signature': ecdsa_sign(ledger.canonical(body).decode(), account.key.private_key_bytes)}

    good = signed(auditor)
    return [({**good, 'signature': signed(people['user'])['signature']}, 'Signature verification failed'),
            ({**good, 'body': {**good['body'], 'kind': 'challenge', 'extra': 1}}, 'Signature verification failed'),
            (signed(auditor, extra=1), 'Invalid transaction schema'),
            (signed(auditor, chain='another-chain'), 'Wrong chain'),
            (signed(auditor, nonce=5), 'Wrong account nonce'),
            (signed(Account('stranger')), 'Insufficient transaction fee'),
            (signed(auditor, job_id='00' * 32), 'No committed log'),
            (signed(auditor, job_id=pending), 'No committed log'),
            (signed(auditor, log_key=log_identity('log-x')[1]), 'No committed log'),
            (signed(auditor, proof_root='EE' * 32), 'Noncanonical digest')], good


def test_every_cheap_check_precedes_a_proof_replay(market):
    state, people, keys = market
    state, job_id = opened(state, people, keys)
    state, pending = opened(state, people, keys)
    state = committed(state, people, job_id)
    replays = []
    replay = lambda previous, request: replays.append(request) or True
    failing, good = challenges_failing_admission(state, people, keys, job_id, pending)
    for envelope, match in failing:
        with pytest.raises(ValueError, match=match):
            ledger.admit(state, envelope)
        refuse(state, envelope, match, replay)
    # Opening a challenge replays nothing, even when every check passes.
    assert ledger.admit(state, good)[2] is None
    state = ledger.transition(state, good, replay)
    challenge_id, people['auditor'].nonce = ledger.transaction_id(good), 1
    for target, named, match in ((pending, challenge_id, 'No open challenge'), (job_id, 'ff' * 32, 'No open challenge'),
                                 (job_id, 'FF' * 32, 'Noncanonical digest'), (job_id, ['ff' * 32], 'Noncanonical digest')):
        refuse(state, prove(people['auditor'], target, named), match, replay)
        people['auditor'].nonce -= 1
    with pytest.raises(ValueError, match='later block'):
        ledger.admit(state, prove(people['auditor'], job_id, challenge_id))
    people['auditor'].nonce -= 1
    assert replays == []
    state = ledger.advance(state, state['height'] + 1)
    request = ledger.admit(state, prove(people['auditor'], job_id, challenge_id))[2]
    assert request == {'chain_id': CHAIN, 'job_id': job_id, 'shard': 1, 'log_key': keys[1],
                       'statement_root': state['jobs'][job_id]['commits'][keys[1]], 'proof_root': 'ee' * 32}
    assert ledger.admit(state, open_job(people['user'], keys))[2] is None


def test_a_job_its_owners_never_finish_committing_expires_with_a_full_refund(market):
    state, people, keys = market
    user = state['accounts'][people['user'].public]['balance']
    state, job_id = opened(state, people, keys)
    state = ledger.transition(state, commit(people['owner-1'], 1, job_id), None)
    for _ in range(PARAMS['job_blocks'] + 1):
        state = ledger.advance(state, state['height'] + 1)
    assert state['results'][job_id] == {**state['results'][job_id], 'status': 'expired', 'committed': [keys[1]]}
    assert state['accounts'][people['user'].public]['balance'] == user - PARAMS['fee']


def test_independent_replays_of_the_same_blocks_reach_the_same_root(market):
    state, people, keys = market
    envelope = open_job(people['user'], keys)
    job_id = ledger.transaction_id(envelope)
    blocks = [[envelope], [commit(people['owner-1'], 1, job_id)], [commit(people['owner-2'], 2, job_id)], [], [], [], []]
    roots = []
    for _ in range(2):
        replica = state
        for transactions in blocks:
            replica = ledger.advance(replica, replica['height'] + 1)
            for transaction in transactions:
                replica = ledger.transition(replica, transaction, None)
        roots.append(ledger.root(replica))
    assert roots[0] == roots[1] and replica['results'][job_id]['status'] == 'settled'


def test_the_ledger_never_loads_torch():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    probe = ('import sys, neuroshard.inference.optimistic, neuroshard.inference.optimistic_app; '
             'assert "torch" not in sys.modules')
    subprocess.run([sys.executable, '-c', probe], check=True, cwd=root, env={'PYTHONPATH': str(root / 'src')})


def application(tmp_path, check, name='settlement.sqlite', people=None):
    """A fresh ABCI application at genesis with the test parties' allocations."""
    from neuroshard.demo import abci_pb2 as pb
    from neuroshard.inference import optimistic_app as app

    people = people or {name: Account(name) for name in ('user', 'owner-1', 'owner-2', 'auditor')}
    terms = {'model_root': MODEL, 'shards': 3, 'allocations': {p.public: 20_000_000 for p in people.values()},
             'params': PARAMS}
    value = app.Application(tmp_path / name, check)
    value.InitChain(pb.RequestInitChain(chain_id=CHAIN, app_state_bytes=json.dumps(terms).encode(), initial_height=1),
                    None)
    return value, people


def finalize(value, txs):
    """Execute and commit ``txs`` as the next block, without a proposal round."""
    from neuroshard.demo import abci_pb2 as pb

    results = value.FinalizeBlock(pb.RequestFinalizeBlock(txs=txs, height=value.state['height'] + 1), None).tx_results
    value.Commit(pb.RequestCommit(), None)
    return results


def drained(value):
    """Wait for every proof the application queued for background judging."""
    value.check.worker.submit(lambda: None).result()


def opened_challenge(value, people, keys, job_id, proof_root='aa' * 32):
    """The auditor's challenge, committed in its own block and judged in the background; the raw proof of it."""
    opening, challenge_id = challenge(people['auditor'], job_id, keys[1], proof_root)
    assert [r.code for r in finalize(value, [ledger.canonical(opening)])] == [0]
    drained(value)
    return ledger.canonical(prove(people['auditor'], job_id, challenge_id))


def request_of(value, proof):
    """What a proof must establish, admitted in the next block as validators admit it."""
    return ledger.admit(ledger.advance(value.state, value.state['height'] + 1), json.loads(proof))[2]


def committed_job(value, people):
    """Both bonds, one opened job and both of its log commitments, each in its own committed block."""
    keys = {}
    for shard, name in ((1, 'owner-1'), (2, 'owner-2')):
        keys[shard], envelope = bond(people[name], shard, f'log-{shard}')
        finalize(value, [ledger.canonical(envelope)])
    opened = open_job(people['user'], keys)
    job_id = ledger.transaction_id(opened)
    finalize(value, [ledger.canonical(opened)])
    finalize(value, [ledger.canonical(commit(people[f'owner-{s}'], s, job_id)) for s in (1, 2)])
    return keys, job_id


def test_the_abci_application_admits_proposes_and_finalizes_only_transactions_that_apply(tmp_path):
    from neuroshard.demo import abci_pb2 as pb
    from neuroshard.inference import optimistic_app as app

    replays = []

    def check(state, request):
        replays.append(request['proof_root'])
        return request['proof_root'] == 'aa' * 32

    application_, people = application(tmp_path, check)
    raw = lambda envelope: ledger.canonical(envelope)

    def block(txs):
        height = application_.state['height'] + 1
        assert application_.ProcessProposal(pb.RequestProcessProposal(txs=txs, height=height), None).status == 1
        results = application_.FinalizeBlock(pb.RequestFinalizeBlock(txs=txs, height=height), None).tx_results
        application_.Commit(pb.RequestCommit(), None)
        return results

    keys = {}
    for shard, name in ((1, 'owner-1'), (2, 'owner-2')):
        keys[shard], envelope = bond(people[name], shard, f'log-{shard}')
        assert application_.CheckTx(pb.RequestCheckTx(tx=raw(envelope)), None).code == 0
        block([raw(envelope)])
    opened = open_job(people['user'], keys)
    job_id = ledger.transaction_id(opened)
    commits = [raw(commit(people[f'owner-{s}'], s, job_id)) for s in (1, 2)]
    # A proposal keeps only transactions that apply in order; one that does not apply is dropped.
    proposed = application_.PrepareProposal(pb.RequestPrepareProposal(txs=commits + [raw(opened)] + commits,
                                                                      height=application_.state['height'] + 1,
                                                                      max_tx_bytes=app.MAX_TX_BYTES), None).txs
    assert list(proposed) == [raw(opened)] + commits
    assert application_.ProcessProposal(pb.RequestProcessProposal(txs=commits, height=application_.state['height'] + 1),
                                        None).status == 2
    block(list(proposed))
    # A challenge opens without a replay; its proof is judged once, in the background, after its block commits.
    bad, challenge_id = challenge(people['auditor'], job_id, keys[1])
    assert application_.CheckTx(pb.RequestCheckTx(tx=raw(bad)), None).code == 0 and replays == []
    block([raw(bad)])
    drained(application_)
    assert replays == ['ee' * 32]
    failed = raw(prove(people['auditor'], job_id, challenge_id))
    for _ in range(2):
        rejected = application_.CheckTx(pb.RequestCheckTx(tx=failed), None)
        assert rejected.code == 1 and rejected.log == 'Fraud proof does not verify'
    while job_id in application_.state['jobs']:
        block([])
        drained(application_)
    assert replays == ['ee' * 32]
    assert application_.state['results'][job_id]['status'] == 'settled'
    assert application_.state['results'][challenge_id]['status'] == 'forfeited'
    reloaded = app.Application(tmp_path / 'settlement.sqlite', check)
    assert ledger.root(reloaded.state) == ledger.root(application_.state)
    queried = json.loads(application_.Query(pb.RequestQuery(path='/state'), None).value)
    assert queried['root'] == ledger.root(application_.state)
    assert application_.Info(pb.RequestInfo(), None).last_block_app_hash == bytes.fromhex(queried['root'])


def test_only_challenges_whose_deposits_are_locked_on_chain_are_ever_replayed(tmp_path):
    from neuroshard.demo import abci_pb2 as pb

    replays = []
    value, people = application(tmp_path, lambda state, request: replays.append(request) or True)
    keys, job_id = committed_job(value, people)
    pending = open_job(people['user'], keys)
    finalize(value, [ledger.canonical(pending)])
    failing, good = challenges_failing_admission(value.state, people, keys, job_id, ledger.transaction_id(pending))
    for envelope, match in failing:
        response = value.CheckTx(pb.RequestCheckTx(tx=ledger.canonical(envelope)), None)
        assert response.code == 1 and match in response.log, (match, response.log)
    assert value.CheckTx(pb.RequestCheckTx(tx=ledger.canonical(good)), None).code == 0
    drained(value)
    assert replays == []
    finalize(value, [ledger.canonical(good)])
    for _ in range(3):
        drained(value)
        finalize(value, [])
    assert len(replays) == 1


def test_a_bundle_this_validator_lacks_gives_no_verdict_until_it_arrives(tmp_path):
    from neuroshard.demo import abci_pb2 as pb
    from neuroshard.inference import optimistic_app as app

    held, replays = set(), []

    def check(state, request):
        if request['proof_root'] not in held:
            raise ledger.ProofUnavailable('bundle not delivered')
        replays.append(request['proof_root'])
        return True

    value, people = application(tmp_path, check)
    keys, job_id = committed_job(value, people)
    proof = opened_challenge(value, people, keys, job_id)
    height = value.state['height'] + 1
    refused = value.CheckTx(pb.RequestCheckTx(tx=proof), None)
    assert refused.code == app.NO_VERDICT and 'bundle not delivered' in refused.log
    # It is neither proposed nor voted for here, and nothing about the proof is remembered.
    assert list(value.PrepareProposal(pb.RequestPrepareProposal(txs=[proof], height=height,
                                                                max_tx_bytes=app.MAX_TX_BYTES), None).txs) == []
    assert value.ProcessProposal(pb.RequestProcessProposal(txs=[proof], height=height), None).status == 2
    assert value.check.known(request_of(value, proof)) is None and replays == []
    # Once the bundle arrives, the next block's background judging replays it.
    held.add('aa' * 32)
    finalize(value, [])
    drained(value)
    assert replays == ['aa' * 32]
    height = value.state['height'] + 1
    assert value.CheckTx(pb.RequestCheckTx(tx=proof), None).code == 0
    assert value.ProcessProposal(pb.RequestProcessProposal(txs=[proof], height=height), None).status == 1
    assert replays == ['aa' * 32]


def test_a_checker_failure_is_no_verdict_and_is_judged_afresh(tmp_path):
    from neuroshard.demo import abci_pb2 as pb
    from neuroshard.inference import optimistic_app as app

    calls = []

    def check(state, request):
        calls.append(request['proof_root'])
        if len(calls) == 1:
            raise OSError('store unreadable')
        return True

    value, people = application(tmp_path, check)
    keys, job_id = committed_job(value, people)
    proof = opened_challenge(value, people, keys, job_id)
    # The background replay failed, so nothing is cached and admission judges the proof afresh.
    assert calls == ['aa' * 32] and value.check.known(request_of(value, proof)) is None
    assert value.CheckTx(pb.RequestCheckTx(tx=proof), None).code == 0 and len(calls) == 2
    with pytest.raises(ledger.NoVerdict):
        app.shard_checker({})(value.state, {'proof_root': 'aa' * 32})


def test_validators_with_and_without_the_bundle_execute_a_committed_challenge_alike(tmp_path):
    from neuroshard.demo import abci_pb2 as pb

    def lacking(state, request):
        raise ledger.ProofUnavailable('bundle never delivered')

    holder, people = application(tmp_path, lambda state, request: True, 'holder.sqlite')
    other, _ = application(tmp_path, lacking, 'other.sqlite', people)
    for value in (holder, other):
        for person in people.values():
            person.nonce = 0
        keys, job_id = committed_job(value, people)
        proof = opened_challenge(value, people, keys, job_id)
    height = holder.state['height'] + 1
    assert holder.CheckTx(pb.RequestCheckTx(tx=proof), None).code == 0
    assert holder.ProcessProposal(pb.RequestProcessProposal(txs=[proof], height=height), None).status == 1
    # The other validator would not vote for the block, but executes it exactly once it is committed.
    assert other.ProcessProposal(pb.RequestProcessProposal(txs=[proof], height=height), None).status == 2
    assert [r.code for r in finalize(holder, [proof])] == [r.code for r in finalize(other, [proof])] == [0]
    assert ledger.root(holder.state) == ledger.root(other.state)
    assert other.state['results'][job_id]['status'] == 'fraud' and other.state['owners'][keys[1]]['status'] == 'slashed'


def test_a_slow_proof_replay_runs_outside_the_state_lock(tmp_path):
    import threading

    from neuroshard.demo import abci_pb2 as pb

    free, threads = [], []

    def check(state, request):
        probe = threading.Thread(target=lambda: free.append(value.lock.acquire(timeout=2) and
                                                            (value.lock.release() or True)))
        probe.start()
        probe.join()
        threads.append(threading.current_thread().name)
        return True

    value, people = application(tmp_path, check)
    keys, job_id = committed_job(value, people)
    proof = opened_challenge(value, people, keys, job_id)
    assert free == [True] and value.CheckTx(pb.RequestCheckTx(tx=proof), None).code == 0
    finalize(value, [proof])
    assert value.state['results'][job_id]['status'] == 'fraud' and len(threads) == 1
