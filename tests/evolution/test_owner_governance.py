import pytest

from neuroshard.inference import optimistic as ledger
from neuroshard.inference.optimistic_app import promotion_or_fraud
from test_optimistic_serving import CHAIN, PARAMS, Account, bond

MODEL, NEXT, QUALITY = 'ab' * 32, 'cd' * 32, 'ef' * 32
MODEL_ROOT = None


def network():
    steward, alice, bob, carol, user = (Account(n) for n in ('steward', 'alice', 'bob', 'carol', 'user'))
    state = ledger.genesis(CHAIN, MODEL, 3, {a.public: 40_000_000 for a in (steward, alice, bob, carol, user)}, PARAMS)
    checker = promotion_or_fraud({'promotion_authority': steward.public,
                                  'approved_models': {NEXT: {'quality_root': QUALITY}}}, None)
    keys = {}
    for account, shard, name, amount in ((alice, 1, 'alice-log', 6_000_000), (bob, 2, 'bob-log', 5_000_000),
                                         (carol, 1, 'carol-log', 5_000_000)):
        key, envelope = bond_at(account, shard, name, amount, state)
        state = ledger.transition(state, envelope, checker)
        keys[name] = key
    return state, checker, (steward, alice, bob, carol, user), keys


def bond_at(account, shard, name, amount, state):
    from test_optimistic_serving import log_identity

    key, public = log_identity(name)
    possession = key.sign(ledger.possession_message(CHAIN, account.public, public, shard, amount, account.nonce)).hex()
    return public, account.sign('owner_bond', model_root=state['model_root'], shard=shard, log_key=public,
                                amount=amount, possession=possession)


def step(state, envelope, checker, blocks=0):
    for _ in range(blocks):
        state = ledger.advance(state, state['height'] + 1)
    return ledger.transition(state, envelope, checker)


def handed(state, checker, steward):
    return step(state, steward.sign('governance_handover', approval_ppm=666_667, vote_blocks=20), checker)


def test_only_the_steward_hands_over_once_and_then_cannot_promote():
    state, checker, (steward, alice, *_), _ = network()
    with pytest.raises(ValueError, match='steward key'):
        step(state, alice.sign('governance_handover', approval_ppm=666_667, vote_blocks=20), checker)
    with pytest.raises(ValueError, match='Integer outside'):
        step(state, steward.sign('governance_handover', approval_ppm=400_000, vote_blocks=20), checker)
    steward.nonce = state['accounts'][steward.public]['nonce']
    governed = handed(state, checker, steward)
    assert ledger.governed(governed) and not ledger.governed(state)
    with pytest.raises(ValueError, match='propose it instead'):
        step(governed, steward.sign('model_promote', model_root=NEXT, quality_root=QUALITY), checker)
    steward.nonce = governed['accounts'][steward.public]['nonce']
    with pytest.raises(ValueError, match='already govern'):
        handed(governed, checker, steward)


def test_a_bond_weighted_supermajority_of_owners_promotes_and_the_record_names_them():
    state, checker, (steward, alice, bob, carol, user), keys = network()
    state = handed(state, checker, steward)
    with pytest.raises(ValueError, match='not been handed'):
        step(ledger.genesis(CHAIN, MODEL, 3, {user.public: 40_000_000}, PARAMS),
             user.sign('model_propose', model_root=NEXT, quality_root=QUALITY), checker)
    user.nonce = state['accounts'][user.public]['nonce']
    proposal = user.sign('model_propose', model_root=NEXT, quality_root=QUALITY)
    pid = ledger.transaction_id(proposal)
    state = step(state, proposal, checker)
    assert state['governance']['proposals'][pid]['weights'] == {
        keys['alice-log']: 6_000_000, keys['bob-log']: 5_000_000, keys['carol-log']: 5_000_000}
    ledger.invariant(state)
    with pytest.raises(ValueError, match='bonded when the proposal opened'):
        step(state, user.sign('model_vote', proposal_id=pid, log_key=keys['alice-log'], approve=True), checker)
    user.nonce = state['accounts'][user.public]['nonce']
    state = step(state, alice.sign('model_vote', proposal_id=pid, log_key=keys['alice-log'], approve=True), checker)
    with pytest.raises(ValueError, match='One yes-or-no'):
        step(state, alice.sign('model_vote', proposal_id=pid, log_key=keys['alice-log'], approve=False), checker)
    alice.nonce = state['accounts'][alice.public]['nonce']
    with pytest.raises(ValueError, match='neither passed nor closed'):
        step(state, user.sign('model_enact', proposal_id=pid), checker)  # 6 of 16 is below two thirds
    user.nonce = state['accounts'][user.public]['nonce']
    state = step(state, bob.sign('model_vote', proposal_id=pid, log_key=keys['bob-log'], approve=True), checker)
    before = state['accounts'][user.public]['balance']
    state = step(state, carol.sign('model_enact', proposal_id=pid), checker)  # 11 of 16 passes; anyone may enact
    assert state['model_root'] == NEXT and not state['governance']['proposals']
    record = state['model_history'][-1]
    assert record['authority'] == 'owners' and (record['approving_weight'], record['total_weight']) == (11_000_000,
                                                                                                    16_000_000)
    assert state['accounts'][user.public]['balance'] == before + ledger.GOVERNANCE['proposal_deposit']
    ledger.invariant(state)


def test_a_rejected_proposal_forfeits_its_deposit_after_the_window_and_jobs_block_enactment():
    state, checker, (steward, alice, bob, carol, user), keys = network()
    state = handed(state, checker, steward)
    proposal = user.sign('model_propose', model_root=NEXT, quality_root=QUALITY)
    pid = ledger.transaction_id(proposal)
    state = step(state, proposal, checker)
    burned = state['burned']
    state = step(state, alice.sign('model_vote', proposal_id=pid, log_key=keys['alice-log'], approve=False), checker)
    with pytest.raises(ValueError, match='No open model proposal'):
        step(state, bob.sign('model_vote', proposal_id=pid, log_key=keys['bob-log'], approve=True), checker, blocks=21)
    bob.nonce = state['accounts'][bob.public]['nonce']
    closed = step(state, carol.sign('model_enact', proposal_id=pid), checker, blocks=21)
    assert closed['model_root'] == MODEL and closed['burned'] == burned + ledger.GOVERNANCE['proposal_deposit'] + PARAMS['fee'] * 2
    ledger.invariant(closed)
    state = closed  # a model may be proposed again once its earlier proposal has closed
    for account in (user, alice, bob, carol):
        account.nonce = state['accounts'][account.public]['nonce']
    second = user.sign('model_propose', model_root=NEXT, quality_root=QUALITY)
    sid = ledger.transaction_id(second)
    state = step(state, second, checker)
    for account, name in ((alice, 'alice-log'), (bob, 'bob-log')):
        state = step(state, account.sign('model_vote', proposal_id=sid, log_key=keys[name], approve=True), checker)
    state['jobs']['open'] = {'price': 0, 'challenges': {}, 'deadline': None, 'expires': 10 ** 9}
    # An open job no longer blocks an enacted version: it waits for the job, and new jobs pause meanwhile.
    pending = step(state, carol.sign('model_enact', proposal_id=sid), checker)
    assert pending['model_root'] == MODEL and pending['governance']['pending_model']['model_root'] == NEXT
    del pending['jobs']['open']
    switched = ledger.advance(pending, pending['height'] + 1)
    assert switched['model_root'] == NEXT and 'pending_model' not in switched['governance']


def test_ungoverned_state_root_is_unchanged_by_this_feature():
    state, *_ = network()
    assert 'governance' not in state


def test_chain_reads_verify_roots_and_cross_check_every_rpc(monkeypatch):
    from neuroshard.assistant import network

    state = ledger.genesis(CHAIN, MODEL, 3, {Account('x').public: 10}, PARAMS)
    honest = {**state, 'root': ledger.root(state)}
    forged_body = {**state, 'model_root': NEXT}
    forged = {**forged_body, 'root': ledger.root(forged_body)}
    replies = {'http://a': honest, 'http://b': honest}
    monkeypatch.setattr(network.chain_network, 'state', lambda address: replies[address])
    chain = network.Chain(['http://a', 'http://b'], CHAIN)
    assert chain.state()['model_root'] == MODEL
    replies['http://b'] = forged
    with pytest.raises(network.Divergence, match='disagree at height 0'):
        chain.state()
    replies['http://b'] = {**forged_body, 'root': ledger.root(state)}
    with pytest.raises(network.Divergence, match='does not match its own root'):
        chain.state()

    def down(address):
        if address == 'http://a':
            raise OSError('unreachable')
        return honest
    monkeypatch.setattr(network.chain_network, 'state', down)
    assert chain.state()['height'] == 0 and chain.rpc == 'http://b'
    assert network.endpoints({'rpc': 'http://a', 'rpcs': ['http://b', 'http://a']}) == ['http://a', 'http://b']
    with pytest.raises(ValueError):
        network.endpoints({'rpc': []})


def test_any_mirror_serves_pinned_files_and_a_wrong_one_is_refused(tmp_path):
    import hashlib
    import urllib.request

    from neuroshard.assistant import network

    content = b'unit weights' * 1000
    pin = {'asset': 'unit.safetensors', 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
    descriptor = {'version': 'v', 'files': {'L7/trainable.safetensors': pin}, 'assets': 'http://127.0.0.1:1/'}
    holder = tmp_path / 'holder' / 'v' / 'stage-2' / 'L7'
    holder.mkdir(parents=True)
    (holder / 'trainable.safetensors').write_bytes(content)
    server, offered = network.serve_artifacts(descriptor, tmp_path / 'holder', '127.0.0.1', 0)
    try:
        port = server.server_address[1]
        assert offered == ['unit.safetensors']
        evil = tmp_path / 'evil' / 'v' / 'stage-2' / 'L7'
        evil.mkdir(parents=True)
        (evil / 'trainable.safetensors').write_bytes(b'x' * len(content))
        bad_server, bad_offered = network.serve_artifacts(descriptor, tmp_path / 'evil', '127.0.0.1', 0)
        assert bad_offered == []  # a holder never offers bytes that differ from the pin
        bad_server.shutdown()

        def opener(url, timeout):
            if url.startswith('http://lying/'):
                class Lying:
                    def __enter__(self):
                        return self

                    def __exit__(self, *a):
                        return False

                    def read(self, n, done=[False]):
                        if done[0]:
                            return b''
                        done[0] = True
                        return b'y' * len(content)
                return Lying()
            return urllib.request.urlopen(url, timeout=timeout)

        mirrored = {**descriptor, 'mirrors': ['http://lying/', f'http://127.0.0.1:{port}/']}
        target = tmp_path / 'client' / 'trainable.safetensors'
        used = network.fetch(mirrored, 'L7/trainable.safetensors', target, progress=lambda *_: None, opener=opener)
        assert used == f'http://127.0.0.1:{port}/' and target.read_bytes() == content
    finally:
        server.shutdown()


def test_owners_admit_and_retire_validators_and_finalize_block_emits_the_updates(tmp_path):
    from neuroshard.demo import abci_pb2 as pb
    from neuroshard.inference.optimistic_app import Application, validator_updates

    state, checker, (steward, alice, bob, carol, user), keys = network()
    with pytest.raises(ValueError, match='not been handed'):
        step(state, user.sign('validator_propose', consensus_key='aa' * 32, power=1), checker)
    user.nonce = state['accounts'][user.public]['nonce']
    state = handed(state, checker, steward)
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    new_key = Ed25519PrivateKey.generate().public_key().public_bytes_raw().hex()
    proposal = user.sign('validator_propose', consensus_key=new_key, power=10)
    pid = ledger.transaction_id(proposal)
    state = step(state, proposal, checker)
    for account, name in ((alice, 'alice-log'), (bob, 'bob-log')):
        state = step(state, account.sign('model_vote', proposal_id=pid, log_key=keys[name], approve=True), checker)
    enacted = step(state, carol.sign('model_enact', proposal_id=pid), checker)
    assert enacted['validators'] == {new_key: 10} and enacted['validator_history'][-1]['power'] == 10
    assert enacted['model_root'] == MODEL
    ledger.invariant(enacted)
    updates = validator_updates(state.get('validators', {}), enacted['validators'])
    assert [(u.pub_key.ed25519.hex(), u.power) for u in updates] == [(new_key, 10)]
    assert validator_updates(enacted['validators'], enacted['validators']) == []
    retire = user.sign('validator_propose', consensus_key=new_key, power=0)
    rid = ledger.transaction_id(retire)
    pending = step(enacted, retire, checker)
    for account, name in ((alice, 'alice-log'), (bob, 'bob-log')):
        pending = step(pending, account.sign('model_vote', proposal_id=rid, log_key=keys[name], approve=True), checker)
    with pytest.raises(ValueError, match='at least one admitted validator'):
        step(pending, carol.sign('model_enact', proposal_id=rid), checker)

    app = Application(tmp_path / 'chain.sqlite', checker)
    app.state = state
    raw = ledger.canonical(carol.sign('model_enact', proposal_id=pid))
    carol.nonce = state['accounts'][carol.public]['nonce']
    raw = ledger.canonical(carol.sign('model_enact', proposal_id=pid))
    try:
        response = app.FinalizeBlock(pb.RequestFinalizeBlock(height=state['height'] + 1, txs=[raw]), None)
        assert response.tx_results[0].code == 0
        assert [(u.pub_key.ed25519.hex(), u.power) for u in response.validator_updates] == [(new_key, 10)]
    finally:
        app.check.worker.shutdown(wait=True)
        app.db.close()
