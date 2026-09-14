"""Native weighted replay quorum, one-fault liveness and funded rejection."""
import copy

import pytest

from neuroshard.evolution import auditing, settlement as state
from neuroshard.evolution.audit_worker import replay
from test_settlement import scenario, tx, claim, blocks
from test_auditing import forged_case


@pytest.fixture
def case(scenario):
    s, owners, store, record, artifacts = scenario
    state.account(s, owners[0].public_key)['balance'] += s['assignment']['bond']
    s['assignment'] = None
    s['manifest']['auditing'] = {**auditing.QUORUM_PROFILE, 'commit_blocks': 16, 'reveal_blocks': 16}
    auditing.initialize(s)
    return s, owners, store, record, artifacts


def send(s, owner, kind, **fields):
    return state.transition(s, tx(s, owner, kind, **fields))


def offer(case, accepted=(0, 1, 2, 3)):
    s, owners, *_ = case
    s = send(s, owners[0], 'fund_audit', publisher=owners[0].public_key,
        auditors=sorted({o.public_key for o in owners}), stage_limit=2, expires_in=64)
    key = next(iter(s['auditing']['budgets']))
    for index in accepted:
        s = send(s, owners[index], 'accept_audit', budget_id=key)
    return s, key


def submit(case, accepted=(0, 1, 2, 3)):
    _, owners, store, record, _ = case
    s, key = offer(case, accepted)
    s = send(s, owners[0], 'reserve', parent=s['model_root'], round=0,
        workers=[o.public_key for o in owners[:2]], audit_budget=key)
    return state.transition(s, claim(s, owners, store, record)), key


def verdicts(s, votes):
    c = copy.deepcopy(s['candidate'])
    coverage = auditing.coverage(c)
    for owner, valid in votes:
        value = auditing.verdict_commitment(s['chain_id'], c['id'], owner.public_key, coverage, 'd'*64, valid)
        s = send(s, owner, 'audit_commit', claim_id=c['id'], commitment=value)
    s = blocks(s, c['audit_commit_end']-s['height']+1 if len(votes) < 3 else 1)
    for owner, valid in votes:
        s = send(s, owner, 'audit_verdict', claim_id=c['id'], coverage_root=coverage, salt='d'*64, valid=valid)
    return s


def test_three_honest_replays_settle_with_one_nonparticipating_validator(case):
    _, owners, store, _, _ = case
    s, key = submit(case, accepted=(0, 1, 2))
    for _ in owners[:3]:
        result = replay(store, s['candidate'])
        assert result['valid'] and len(result['stages']) == 2
    s = verdicts(s, [(o, True) for o in owners[:3]])
    s = blocks(s, state.PARAMS['challenge_blocks']+1)
    assert s['issued'] == state.PARAMS['reward_atoms']
    assert s['auditing']['paid_services'] == 3
    assert s['auditing']['paid_atoms'] == 3*2*auditing.PROFILE['price_per_stage']
    assert s['auditing']['history'][-1]['refunded_atoms'] == 2*auditing.PROFILE['price_per_stage']
    assert key not in s['auditing']['budgets']
    state.invariant(s)


def test_a_minority_can_hash_a_forgery_but_cannot_authorize_issuance(case):
    broken = forged_case(case)
    _, owners, store, _, _ = broken
    s, _ = submit(broken)
    assert not replay(store, s['candidate'])['valid']
    s = verdicts(s, [(owners[3], True)])
    assert not auditing.complete(s, s['candidate'])
    s = blocks(s, s['candidate']['audit_reveal_end']-s['height']+1)
    assert s['issued'] == 0 and s['candidate'] is None
    state.invariant(s)


def test_honest_invalid_verdicts_are_paid_without_a_second_dispute_bond(case):
    broken = forged_case(case)
    _, owners, store, _, _ = broken
    s, _ = submit(broken)
    result = replay(store, s['candidate'])
    assert not result['valid'] and len(result['stages']) == 2
    s = verdicts(s, [(owners[0], False), (owners[1], False), (owners[2], False), (owners[3], True)])
    assert auditing.rejected(s, s['candidate']) and s['candidate']['challenge'] is None
    s = blocks(s, state.PARAMS['challenge_blocks']+1)
    assert s['issued'] == 0 and s['candidate'] is None
    history = s['auditing']['history'][-1]
    assert history['reason'] == 'native audit quorum rejected execution'
    assert history['paid_atoms'] == 3*2*auditing.PROFILE['price_per_stage']
    assert history['slashed_atoms'] == auditing.PROFILE['auditor_bond']
    state.invariant(s)


def test_sponsor_cannot_choose_keys_and_two_thirds_is_not_a_quorum(case):
    s, owners, *_ = case
    with pytest.raises(ValueError, match='all active bonded owners'):
        send(s, owners[0], 'fund_audit', publisher=owners[0].public_key,
            auditors=[owners[3].public_key], stage_limit=2, expires_in=64)
    with pytest.raises(ValueError, match='must accept'):
        submit(case, accepted=(0, 1))
    s, key = offer(case)
    budget = s['auditing']['budgets'][key]
    powers = budget['voting_snapshot']['owners']
    # Six units split among three owners. Four units are exactly two thirds.
    keys = list(powers)
    powers.update({keys[0]: 2, keys[1]: 2, keys[2]: 1, keys[3]: 1})
    for index, owner in enumerate(keys):
        budget['auditors'][owner]['revealed'] = index < 2
    assert not auditing.enough(budget, lambda a: a['revealed'])


def test_multiple_consensus_keys_do_not_manufacture_owner_weight(case):
    s, owners, *_ = case
    keys = list(s['validators'])
    # Move an existing key to the first owner: weight is summed, never counted
    # once per signature or once per identity string.
    s['validators'][keys[1]]['owner'] = owners[0].public_key
    snap = auditing.snapshot(s)
    assert len(snap['owners']) == 3 and snap['owners'][owners[0].public_key] == 2
    assert sum(snap['owners'].values()) == 4


def test_stale_weight_snapshot_and_verdict_substitution_fail(case):
    s, key = offer(case)
    _, owners, *_ = case
    validator = next(iter(s['validators']))
    s['validators'][validator]['history'].append([1, 2])
    with pytest.raises(ValueError, match='weights changed'):
        send(s, owners[0], 'reserve', parent=s['model_root'], round=0,
            workers=[o.public_key for o in owners[:2]], audit_budget=key)
    s, _ = submit(case)
    c = s['candidate']
    root = auditing.coverage(c)
    for owner in owners[:3]:
        s = send(s, owner, 'audit_commit', claim_id=c['id'], commitment=auditing.verdict_commitment(
            s['chain_id'], c['id'], owner.public_key, root, 'd'*64, False))
    s = blocks(s, 1)
    with pytest.raises(ValueError, match='differs from its commitment'):
        send(s, owners[0], 'audit_verdict', claim_id=c['id'], coverage_root=root, salt='d'*64, valid=True)


def test_threshold_collusion_remains_an_explicit_security_boundary(case):
    """This is BFT validity under honest bonded weight, not a ZK proof."""
    broken = forged_case(case)
    _, owners, store, forged, _ = broken
    s, _ = submit(broken)
    assert not replay(store, s['candidate'])['valid']
    s = verdicts(s, [(o, True) for o in owners[:3]])
    s = blocks(s, state.PARAMS['challenge_blocks']+1)
    assert s['issued'] == state.PARAMS['reward_atoms'] and s['model_root'] == forged['model_root']
    state.invariant(s)


def test_snapshot_bonds_cannot_be_withdrawn_while_the_obligation_is_live(case):
    s, key = offer(case)
    _, owners, *_ = case
    consensus = next(k for k, v in s['validators'].items() if v['owner'] == owners[3].public_key)
    assert not auditing.held(s, consensus)
    s = send(s, owners[0], 'reserve', parent=s['model_root'], round=0,
        workers=[o.public_key for o in owners[:2]], audit_budget=key)
    assert auditing.held(s, consensus)
    # Even after the ordinary evidence cooldown, the snapshot collateral must
    # remain until this bounded audit obligation has completed or expired.
    s['validators'][consensus].update(status='cooldown', release_height=0, release_time_ns=-1)
    with pytest.raises(ValueError, match='native audit obligation'):
        send(s, owners[3], 'withdraw', consensus_key=consensus)
    s = blocks(s, s['assignment']['expires']-s['height']+1)
    assert not auditing.held(s, consensus)


def test_one_false_rejection_cannot_veto_three_honest_positive_replays(case):
    _, owners, store, _, _ = case
    s, _ = submit(case)
    assert replay(store, s['candidate'])['valid']
    s = verdicts(s, [(owners[0], True), (owners[1], True), (owners[2], True), (owners[3], False)])
    s = blocks(s, state.PARAMS['challenge_blocks']+1)
    assert s['issued'] == state.PARAMS['reward_atoms'] and s['auditing']['paid_services'] == 3
    state.invariant(s)


def test_native_funding_derives_owners_and_silence_cannot_slash_honest_auditors(case):
    s, owners, *_ = case
    funded = send(s, owners[0], 'fund_audit', publisher=owners[0].public_key,
        auditors=[], stage_limit=2, expires_in=64)
    budget = next(iter(funded['auditing']['budgets'].values()))
    assert set(budget['auditors']) == {o.public_key for o in owners}
    assert budget['funds'] == 8*auditing.PROFILE['price_per_stage']
    claimed, _ = submit(case)
    result = blocks(claimed, claimed['candidate']['audit_reveal_end']+1)
    assert result['issued'] == 0 and result['auditing']['history'][-1]['slashed_atoms'] == 0


def test_native_fee_budget_does_not_increase_when_bonds_are_split_between_owners(case):
    s, owners, *_ = case
    keys = list(s['validators'])
    s['validators'][keys[1]]['owner'] = owners[0].public_key
    before = send(s, owners[0], 'fund_audit', publisher=owners[0].public_key,
        auditors=[], stage_limit=2, expires_in=64)
    s['validators'][keys[1]]['owner'] = owners[1].public_key
    after = send(s, owners[0], 'fund_audit', publisher=owners[0].public_key,
        auditors=[], stage_limit=2, expires_in=64)
    assert next(iter(before['auditing']['budgets'].values()))['funds'] == next(iter(after['auditing']['budgets'].values()))['funds']
