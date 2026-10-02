import hashlib

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from neuroshard.core.crypto.ecdsa import derive_keypair_from_token, ecdsa_sign
from neuroshard.inference import optimistic as ledger

CHAIN = 'neuroshard-optimistic-test'
MODEL = 'ab' * 32
PARAMS = {**ledger.PARAMS, 'challenge_blocks': 3, 'job_blocks': 6}


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


def commit(account, name, job_id, statement):
    key, _ = log_identity(name)
    signature = key.sign(ledger.commitment_message(CHAIN, job_id, statement)).hex()
    return account.sign('log_commit', job_id=job_id, statement_root=statement, log_signature=signature)


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


def opened(state, people, keys, price=1_000_000):
    envelope = people['user'].sign('serve_open', model_root=MODEL, owners=[keys[1], keys[2]],
                                   request_root='cd' * 32, price=price)
    return ledger.transition(state, envelope, None), ledger.transaction_id(envelope)


def committed(state, people, job_id, statements=('11' * 32, '22' * 32)):
    for shard, statement in zip((1, 2), statements):
        state = ledger.transition(state, commit(people[f'owner-{shard}'], f'log-{shard}', job_id, statement), None)
    return state


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
    assert state['results'][job_id]['status'] == 'settled' and state['results'][job_id]['paid_each'] == 500_000
    for shard in (1, 2):
        assert state['accounts'][people[f'owner-{shard}'].public]['balance'] == before[f'owner-{shard}'] + 500_000
    assert state['accounts'][people['user'].public]['balance'] == before['user']


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
    refuse(state, people['user'].sign('serve_open', model_root=MODEL, owners=[keys[1], keys[2]], request_root='cd' * 32,
                                      price=1_000_000), 'active and bonded')
    people['user'].nonce -= 1
    refuse(state, people['owner-1'].sign('owner_withdraw', log_key=keys[1]), 'still exposed')
    people['owner-1'].nonce -= 1
    for _ in range(PARAMS['challenge_blocks']):
        state = ledger.advance(state, state['height'] + 1)
    balance = state['accounts'][people['owner-1'].public]['balance']
    state = ledger.transition(state, people['owner-1'].sign('owner_withdraw', log_key=keys[1]), None)
    assert state['owners'][keys[1]]['status'] == 'withdrawn'
    assert state['accounts'][people['owner-1'].public]['balance'] == balance - PARAMS['fee'] + PARAMS['owner_bond_minimum']


def test_commitments_must_come_from_a_named_owner_and_be_signed_by_its_log_key(market):
    state, people, keys = market
    state, job_id = opened(state, people, keys)
    refuse(state, commit(people['auditor'], 'log-1', job_id, '11' * 32), 'no uncommitted owner')
    refuse(state, commit(people['owner-1'], 'log-2', job_id, '11' * 32), 'not signed')
    people['owner-1'].nonce -= 1
    state = ledger.transition(state, commit(people['owner-1'], 'log-1', job_id, '11' * 32), None)
    refuse(state, commit(people['owner-1'], 'log-1', job_id, '33' * 32), 'no uncommitted owner')
    assert state['jobs'][job_id]['deadline'] is None and state['jobs'][job_id]['commits'] == {keys[1]: '11' * 32}


def test_a_verified_fraud_proof_slashes_the_owner_pays_the_auditor_and_refunds_every_affected_user(market):
    state, people, keys = market
    state, first = opened(state, people, keys)
    state, second = opened(state, people, keys, price=2_000_000)
    state = committed(state, people, first)
    seen = []

    def execute(previous, request):
        seen.append(request)
        return True

    user = state['accounts'][people['user'].public]['balance']
    auditor = state['accounts'][people['auditor'].public]['balance']
    challenge = people['auditor'].sign('challenge', job_id=first, log_key=keys[1], proof_root='ee' * 32)
    state = ledger.transition(state, challenge, execute)
    assert seen == [{'job_id': first, 'shard': 1, 'log_key': keys[1], 'statement_root': '11' * 32, 'proof_root': 'ee' * 32}]
    reward = PARAMS['owner_bond_minimum'] * PARAMS['auditor_share_ppm'] // 1_000_000
    assert state['owners'][keys[1]] == {**state['owners'][keys[1]], 'amount': 0, 'status': 'slashed'}
    assert state['accounts'][people['auditor'].public]['balance'] == auditor - PARAMS['fee'] + reward
    assert state['accounts'][people['user'].public]['balance'] == user + 1_000_000 + 2_000_000
    assert state['results'][first]['status'] == 'fraud' and state['results'][second]['status'] == 'voided'
    assert not state['jobs']
    refuse(state, people['user'].sign('serve_open', model_root=MODEL, owners=[keys[1], keys[2]], request_root='cd' * 32,
                                      price=1_000_000), 'active and bonded')


def test_challenges_that_fail_or_come_late_change_nothing(market):
    state, people, keys = market
    state, job_id = opened(state, people, keys)
    called = []
    refuse(state, people['auditor'].sign('challenge', job_id=job_id, log_key=keys[1], proof_root='ee' * 32),
           'No committed log', lambda *a: called.append(a) or True)
    people['auditor'].nonce -= 1
    state = committed(state, people, job_id)
    refuse(state, people['auditor'].sign('challenge', job_id=job_id, log_key=keys[1], proof_root='ee' * 32),
           'does not verify', lambda previous, request: False)
    people['auditor'].nonce -= 1
    assert not called
    for _ in range(PARAMS['challenge_blocks']):
        state = ledger.advance(state, state['height'] + 1)
    late = people['auditor'].sign('challenge', job_id=job_id, log_key=keys[1], proof_root='ee' * 32)
    state = ledger.advance(state, state['height'] + 1)
    refuse(state, late, 'No committed log', lambda *a: True)


def test_a_job_its_owners_never_finish_committing_expires_with_a_full_refund(market):
    state, people, keys = market
    user = state['accounts'][people['user'].public]['balance']
    state, job_id = opened(state, people, keys)
    state = ledger.transition(state, commit(people['owner-1'], 'log-1', job_id, '11' * 32), None)
    for _ in range(PARAMS['job_blocks'] + 1):
        state = ledger.advance(state, state['height'] + 1)
    assert state['results'][job_id] == {**state['results'][job_id], 'status': 'expired', 'committed': [keys[1]]}
    assert state['accounts'][people['user'].public]['balance'] == user - PARAMS['fee']


def test_independent_replays_of_the_same_blocks_reach_the_same_root(market):
    state, people, keys = market
    envelope = people['user'].sign('serve_open', model_root=MODEL, owners=[keys[1], keys[2]], request_root='cd' * 32,
                                   price=1_000_000)
    job_id = ledger.transaction_id(envelope)
    blocks = [[envelope], [commit(people['owner-1'], 'log-1', job_id, '11' * 32)],
              [commit(people['owner-2'], 'log-2', job_id, '22' * 32)], [], [], [], []]
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
    probe = 'import sys, neuroshard.inference.optimistic; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', probe], check=True, cwd=root, env={'PYTHONPATH': str(root / 'src')})


def test_the_abci_application_admits_proposes_and_finalizes_only_transactions_that_apply(tmp_path):
    import json

    from neuroshard.demo import abci_pb2 as pb
    from neuroshard.inference import optimistic_app as app

    people = {name: Account(name) for name in ('user', 'owner-1', 'owner-2', 'auditor')}
    terms = {'model_root': MODEL, 'shards': 3, 'allocations': {p.public: 20_000_000 for p in people.values()},
             'params': PARAMS}
    replays = []

    def check(state, request):
        replays.append(request['proof_root'])
        return request['proof_root'] == 'aa' * 32

    application = app.Application(tmp_path / 'settlement.sqlite', check)
    application.InitChain(pb.RequestInitChain(chain_id=CHAIN, app_state_bytes=json.dumps(terms).encode(),
                                              initial_height=1), None)
    raw = lambda envelope: ledger.canonical(envelope)

    def block(txs):
        height = application.state['height'] + 1
        assert application.ProcessProposal(pb.RequestProcessProposal(txs=txs, height=height), None).status == 1
        results = application.FinalizeBlock(pb.RequestFinalizeBlock(txs=txs, height=height), None).tx_results
        application.Commit(pb.RequestCommit(), None)
        return results

    keys = {}
    for shard, name in ((1, 'owner-1'), (2, 'owner-2')):
        keys[shard], envelope = bond(people[name], shard, f'log-{shard}')
        assert application.CheckTx(pb.RequestCheckTx(tx=raw(envelope)), None).code == 0
        block([raw(envelope)])
    opened = people['user'].sign('serve_open', model_root=MODEL, owners=[keys[1], keys[2]], request_root='cd' * 32,
                                 price=1_000_000)
    job_id = ledger.transaction_id(opened)
    commits = [raw(commit(people[f'owner-{s}'], f'log-{s}', job_id, f'{s}{s}' * 32)) for s in (1, 2)]
    # A proposal keeps only transactions that apply in order; one that does not apply is dropped.
    proposed = application.PrepareProposal(pb.RequestPrepareProposal(txs=commits + [raw(opened)] + commits,
                                                                     height=application.state['height'] + 1,
                                                                     max_tx_bytes=app.MAX_TX_BYTES), None).txs
    assert list(proposed) == [raw(opened)] + commits
    assert application.ProcessProposal(pb.RequestProcessProposal(txs=commits, height=application.state['height'] + 1),
                                       None).status == 2
    block(list(proposed))
    bad = people['auditor'].sign('challenge', job_id=job_id, log_key=keys[1], proof_root='ee' * 32)
    rejected = application.CheckTx(pb.RequestCheckTx(tx=raw(bad)), None)
    assert rejected.code == 1 and rejected.log == 'Fraud proof does not verify'
    application.CheckTx(pb.RequestCheckTx(tx=raw(bad)), None)
    assert replays == ['ee' * 32]
    while job_id in application.state['jobs']:
        block([])
    assert application.state['results'][job_id]['status'] == 'settled'
    reloaded = app.Application(tmp_path / 'settlement.sqlite', check)
    assert ledger.root(reloaded.state) == ledger.root(application.state)
    queried = json.loads(application.Query(pb.RequestQuery(path='/state'), None).value)
    assert queried['root'] == ledger.root(application.state)
    assert application.Info(pb.RequestInfo(), None).last_block_app_hash == bytes.fromhex(queried['root'])
