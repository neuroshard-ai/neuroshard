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
    carol.nonce = state['accounts'][carol.public]['nonce']
    second = user.sign('model_propose', model_root=NEXT, quality_root=QUALITY)
    sid = ledger.transaction_id(second)
    state = step(state, second, checker)
    for account, name in ((alice, 'alice-log'), (bob, 'bob-log')):
        state = step(state, account.sign('model_vote', proposal_id=sid, log_key=keys[name], approve=True), checker)
    state['jobs']['open'] = {'price': 0, 'challenges': {}}
    with pytest.raises(ValueError, match='Drain'):
        step(state, carol.sign('model_enact', proposal_id=sid), checker)


def test_ungoverned_state_root_is_unchanged_by_this_feature():
    state, *_ = network()
    assert 'governance' not in state
