"""Regression tests for the October 18, 2026 beta-readiness audit (.neuroshard/beta-audit-20261018).

Each test reproduces one reported failure and asserts the safe outcome instead.
"""
import copy
import hashlib
import io

import pytest
import torch

from neuroshard.evolution import growth_replay as g
from neuroshard.evolution.growth_model import model_factory

from test_network_growth import RECIPE, tiny

torch.set_num_threads(1)


def _honest_step():
    model = tiny()
    trainable, optimizer = g.start(model, RECIPE)
    g.train_segment(model, trainable, optimizer, RECIPE, 0, 1)
    before_root, before = g.state_root(trainable, optimizer), g.save(trainable, optimizer)
    after_root = g.train_segment(model, trainable, optimizer, RECIPE, 1, 2)[0]
    return before_root, before, after_root


def _verdict(before, before_root, claimed_root, recipe=RECIPE, make=tiny):
    bundle = g.bundle(before, 1)
    return g.judge(make, recipe, bundle, {'step': 1, 'before_root': before_root, 'claimed_root': claimed_root,
                                          'proof_root': hashlib.sha256(bundle).hexdigest()})


def _rewrite(before, change):
    saved = torch.load(io.BytesIO(before), weights_only=True)
    change(saved)
    buffer = io.BytesIO()
    torch.save(saved, buffer)
    return buffer.getvalue()


# Finding 1: uncommitted optimizer settings must not turn honest work into fraud.

@pytest.mark.parametrize('key,value', [('weight_decay', 100.0), ('eps', 1.0), ('betas', (0.1, 0.2)),
                                       ('amsgrad', True), ('maximize', True), ('foreach', True)])
def test_altered_optimizer_settings_cannot_prove_an_honest_step_wrong(key, value):
    before_root, before, after_root = _honest_step()
    assert _verdict(before, before_root, after_root) is False
    def change(saved):
        saved['optimizer']['param_groups'][0][key] = value
    forged = _rewrite(before, change)
    assert g.bytes_root(forged) == before_root  # the setting is outside the committed root
    assert _verdict(forged, before_root, after_root) is False


def test_a_reordered_or_regrouped_state_proves_nothing():
    before_root, before, after_root = _honest_step()
    reordered = _rewrite(before, lambda saved: saved['order'].reverse())
    regrouped = _rewrite(before, lambda saved: saved['optimizer']['param_groups'].append(
        dict(saved['optimizer']['param_groups'][0], params=[])))
    assert _verdict(reordered, before_root, after_root) is False
    assert _verdict(regrouped, before_root, after_root) is False


def test_a_real_forgery_is_still_proven():
    before_root, before, after_root = _honest_step()
    assert _verdict(before, before_root, 'ff' * 32) is True


# Finding 3: a recipe root binds everything that affects execution.

def test_the_base_model_seed_and_configuration_are_bound_by_the_recipe_root():
    recipe = {**copy.deepcopy(RECIPE), 'base': 'tiny-granite', 'base_seed': 0, 'model_root': 'ab' * 32,
              'base_config': {'vocab_size': 64, 'hidden_size': 32, 'intermediate_size': 64, 'num_hidden_layers': 2,
                              'num_attention_heads': 4, 'num_key_value_heads': 2, 'tie_word_embeddings': True}}
    root = g.recipe_root(recipe, recipe['model_root'])
    assert g.recipe_root({**recipe, 'base_seed': 1}, recipe['model_root']) != root
    assert g.recipe_root({**recipe, 'base_config': {**recipe['base_config'], 'hidden_size': 64}},
                         recipe['model_root']) != root
    # A parent recipe carrying a base configuration or seed would ignore them, so it is refused outright.
    with pytest.raises(ValueError):
        g.recipe_root({**recipe, 'base': 'granite-parent'}, recipe['model_root'])


def test_parent_recipe_roots_are_unchanged_and_extra_fields_are_not_executed():
    parent = {'arm': 'addition', 'spec': {'seed': 1}, 'sources': {'experience': []}, 'model_root': 'cd' * 32}
    legacy = hashlib.sha256(__import__('json').dumps(
        {'model_root': 'cd' * 32, 'arm': 'addition', 'spec': {'seed': 1}, 'sources': {'experience': []}},
        sort_keys=True).encode()).hexdigest()
    assert g.recipe_root(parent, parent['model_root']) == legacy
    assert g.recipe_root({**parent, 'base': 'granite-parent', 'route': 'documents', 'kind': 'x'},
                         parent['model_root']) == legacy
    assert set(g.execution({**parent, 'route': 'documents', 'whatever': 1})) == {'model_root', 'arm', 'spec', 'sources'}


def test_validators_judge_only_the_pinned_parent_unless_configured():
    with pytest.raises(ValueError):
        model_factory({'base': 'tiny-granite', 'base_config': {}}, allowed={'granite-parent'})
    model_factory({'base': 'granite-parent'}, allowed={'granite-parent'})


# Finding 2: free faucet tokens must not buy governance.

from unittest.mock import patch

from neuroshard.assistant import network
from neuroshard.inference import optimistic as ledger
from test_optimistic_serving import Account, CHAIN, MODEL, bond


def _governed_chain(honest=3):
    steward = Account('audit-steward')
    owners = [Account(f'audit-owner-{i}') for i in range(honest)]
    s = ledger.genesis(CHAIN, MODEL, 3, {steward.public: 1_000_000_000, **{a.public: 20_000_000 for a in owners}},
                       network.PARAMS)
    keys = []
    for i, a in enumerate(owners):
        key, e = bond(a, 1 if i == 0 else 2, f'audit-owner-log-{i}')
        keys.append(key)
        s = ledger.transition(s, e, None)
    s = ledger.transition(s, steward.sign('governance_handover', approval_ppm=666_667, vote_blocks=20),
                          lambda *_: True)
    return s, steward, owners, keys


def _vote_through(s, proposer, kind, voters, **fields):
    proposal = proposer.sign(kind, **fields)
    pid = ledger.transaction_id(proposal)
    s = ledger.transition(s, proposal, None)
    for account, key in voters:
        s = ledger.transition(s, account.sign('model_vote', proposal_id=pid, log_key=key, approve=True), None)
    return ledger.transition(s, proposer.sign('model_enact', proposal_id=pid), None), pid


def test_faucet_funded_bonds_serve_but_cannot_vote():
    s, steward, owners, keys = _governed_chain()
    assert set(s['governance']['members']) == set(keys)
    attackers = [Account(f'audit-attacker-{i}') for i in range(6)]
    attack_keys = []
    for i, a in enumerate(attackers):  # far more bonded weight than every honest member together
        s = ledger.transition(s, steward.sign('transfer', to=a.public, amount=40_000_000), None)
        key, e = bond(a, 1 + i % 2, f'audit-attacker-log-{i}', 30_000_000)
        attack_keys.append(key)
        s = ledger.transition(s, e, None)
    assert all(s['owners'][k]['status'] == 'active' for k in attack_keys)  # they may serve
    proposal = attackers[0].sign('model_propose', model_root='ee' * 32, quality_root='dd' * 32)
    pid = ledger.transaction_id(proposal)
    s = ledger.transition(s, proposal, None)
    assert set(s['governance']['proposals'][pid]['weights']) == set(keys)
    with pytest.raises(ValueError, match='Only an owner bonded when the proposal opened votes'):
        ledger.transition(s, attackers[0].sign('model_vote', proposal_id=pid, log_key=attack_keys[0], approve=True), None)
    with pytest.raises(ValueError, match='neither passed nor closed'):
        ledger.transition(s, attackers[2].sign('model_enact', proposal_id=pid), None)
    validator = attackers[1].sign('validator_propose', consensus_key='cc' * 32, power=50)
    s = ledger.transition(s, validator, None)
    assert set(s['governance']['proposals'][ledger.transaction_id(validator)]['weights']) == set(keys)


def test_members_admit_and_remove_owners():
    s, steward, owners, keys = _governed_chain()
    newcomer = Account('audit-newcomer')
    s = ledger.transition(s, steward.sign('transfer', to=newcomer.public, amount=20_000_000), None)
    key, e = bond(newcomer, 2, 'audit-newcomer-log')
    s = ledger.transition(s, e, None)
    assert key not in ledger.voting_weights(s)
    s, _ = _vote_through(s, owners[0], 'member_propose', list(zip(owners, keys)), log_key=key, admit=True)
    assert key in s['governance']['members'] and key in ledger.voting_weights(s)
    assert s['governance']['member_history'][-1]['admit'] is True
    s, _ = _vote_through(s, owners[0], 'member_propose', list(zip(owners, keys)), log_key=key, admit=False)
    assert key not in ledger.voting_weights(s)


def test_a_minority_of_members_cannot_admit_anyone():
    s, steward, owners, keys = _governed_chain()
    outsider = Account('audit-outsider')
    s = ledger.transition(s, steward.sign('transfer', to=outsider.public, amount=20_000_000), None)
    key, e = bond(outsider, 1, 'audit-outsider-log')
    s = ledger.transition(s, e, None)
    proposal = outsider.sign('member_propose', log_key=key, admit=True)
    pid = ledger.transaction_id(proposal)
    s = ledger.transition(s, proposal, None)
    s = ledger.transition(s, owners[0].sign('model_vote', proposal_id=pid, log_key=keys[0], approve=True), None)
    with pytest.raises(ValueError, match='neither passed nor closed'):
        ledger.transition(s, outsider.sign('model_enact', proposal_id=pid), None)


def test_membership_starts_at_the_upgrade_height_and_earlier_blocks_replay_unchanged():
    with patch.dict(ledger.UPGRADE_HEIGHTS, {CHAIN: None}):
        s, steward, owners, keys = _governed_chain()
        assert 'members' not in s['governance']
        legacy = ledger.advance(s, s['height'] + 1)
        assert 'members' not in legacy['governance']
    with patch.dict(ledger.UPGRADE_HEIGHTS, {CHAIN: s['height'] + 2}):
        before = ledger.advance(s, s['height'] + 1)
        assert ledger.root(before) == ledger.root(legacy)  # identical until the upgrade height
        after = ledger.advance(before, before['height'] + 1)
        assert set(after['governance']['members']) == set(keys)


# Finding 5: cheap abandoned growth jobs must not block growth.

def _spam_jobs(s, account, count, reward=1):
    out = []
    for i in range(count):
        e = account.sign('growth_open', recipe_root=f'{i + 1:064x}', start_root='ff' * 32, steps=1, segment_steps=1,
                         reward=reward)
        s = ledger.transition(s, e, None)
        out.append(ledger.transaction_id(e))
    return s, out


def test_the_audit_spam_is_refused_and_sponsored_jobs_cannot_take_every_slot():
    spammer, cheap, more = (Account(f'audit-growth-spam-{n}') for n in ('a', 'b', 'c'))
    s = ledger.genesis(CHAIN, MODEL, 3, {a.public: 100_000_000 for a in (spammer, cheap, more)}, network.PARAMS)
    with pytest.raises(ValueError, match='minimum reward per step'):
        _spam_jobs(s, cheap, 1)
    s, _ = _spam_jobs(s, spammer, 4, reward=1000)
    with pytest.raises(ValueError, match='at capacity'):
        _spam_jobs(s, more, 1, reward=1000)
    s['growth_queue'] = [{'recipe_root': 'aa' * 32, 'start_root': 'bb' * 32, 'steps': 1, 'segment_steps': 1,
                          'reward': 1, 'admitted': 0, 'proposal_id': 'cc' * 32}]
    s['growth_pool'] = 1
    s['initial_supply'] += 1
    s = ledger.advance(s, s['height'] + 1)
    assert sum(1 for job in s['growth'].values() if job['sponsor'] is None) == 1  # the pool's job opened


def test_abandoned_jobs_expire_and_refund():
    from neuroshard.inference import growth

    spammer = Account('audit-growth-abandon')
    s = ledger.genesis(CHAIN, MODEL, 3, {spammer.public: 100_000_000}, network.PARAMS)
    s, ids = _spam_jobs(s, spammer, 4, reward=1000)
    balance = s['accounts'][spammer.public]['balance']
    idle = growth.PARAMS['growth_idle_blocks']
    for _ in range(idle):
        s = ledger.advance(s, s['height'] + 1)
    assert len(s['growth']) == 4  # not yet idle long enough
    s = ledger.advance(s, s['height'] + 1)
    assert 'growth' not in s
    assert all(s['results'][i]['status'] == 'growth_expired' for i in ids)
    assert s['accounts'][spammer.public]['balance'] == balance + 4 * 1000  # rewards back, deposits burned
    _spam_jobs(s, spammer, 4, reward=1000)  # capacity is free again


# Finding 6: a fresh node can rebuild the served model after long history and several route changes.

def test_grown_units_survive_routine_history():
    from neuroshard.assistant.audit import named_recipe, named_state

    with patch.dict(ledger.UPGRADE_HEIGHTS, {CHAIN: 1}):
        s = ledger.genesis(CHAIN, MODEL, 3, {}, network.PARAMS)
        ledger.remember(s, 'grown', {'status': 'grown', 'recipe_root': 'aa' * 32, 'final_root': 'bb' * 32})
        s = ledger.advance(s, 1)  # the upgrade carries recent grown results into the registry
        for i in range(network.PARAMS['max_results'] + 10):
            s['height'] += 1
            ledger.remember(s, f'result-{i}', {'status': 'expired'})
        assert 'grown' not in s['results']
        assert named_recipe(s, 'aa' * 32) and named_state(s, 'bb' * 32)
        assert [r['final_root'] for r in ledger.grown(s)] == ['bb' * 32]


def test_a_fresh_node_rebuilds_a_version_that_changed_two_routes():
    from neuroshard.evolution import growth_runtime

    desc = {'format': 'neuroshard-assistant-network/1', 'version': 'v0', 'plan_sha256': 'aa' * 32, 'files': {},
            'policies': {'documents': {}, 'chat': {}},
            'routing': {'documents': {'addition': 'D'}, 'chat': {'addition': 'C'}}}
    after_documents = growth_runtime.candidate(desc, 'D', '11' * 32, b'doc-manifest', b'doc-tensors')
    after_both = growth_runtime.candidate(after_documents, 'C', '22' * 32, b'chat-manifest', b'chat-tensors')
    state = {'model_root': network.model_root(after_both), 'results': {}, 'grown_units': {
        '11' * 32: {'status': 'grown', 'recipe_root': '33' * 32, 'final_root': '11' * 32, 'height': 5},
        '22' * 32: {'status': 'grown', 'recipe_root': '44' * 32, 'final_root': '22' * 32, 'height': 9}}}
    recipes = {'33' * 32: {'route': 'documents'}, '44' * 32: {'route': 'chat'}}

    def unit_files(_, recipe, root):
        return (b'doc-manifest', b'doc-tensors') if recipe['route'] == 'documents' else (b'chat-manifest', b'chat-tensors')
    with patch.object(growth_runtime, 'unit_files', side_effect=unit_files):
        fresh = growth_runtime.upgrade(desc, state, recipes, lambda *_: b'ignored')
        assert fresh is not None and network.model_root(fresh[0]) == state['model_root']
        assert growth_runtime.upgrade(after_documents, state, recipes, lambda *_: b'ignored') is not None
