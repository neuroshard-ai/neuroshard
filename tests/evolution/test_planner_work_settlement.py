"""Planner windows reuse funded audits and conserve native issuance."""
import copy
import hashlib
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from neuroshard.demo import protocol
from neuroshard.evolution import auditing, planner_window, planner_work, settlement as state
from neuroshard.evolution.reference_data import identity
from test_planner_window import claim
from test_settlement import tx, blocks
from test_native_audit_quorum import verdicts


def network(profile):
    owners = [protocol.Identity('planner-native-'+str(i)) for i in range(4)]
    validators = []
    for index, owner in enumerate(owners):
        key = Ed25519PrivateKey.from_private_bytes(hashlib.sha256(('planner-validator-'+str(index)).encode()).digest())
        validators.append({'owner': owner.public_key, 'consensus_key': key.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw).hex(), 'bond': state.PARAMS['bond_unit'], 'liquid': 1_000_000_000})
    manifest = {'params': state.PARAMS, 'initial_model_root': profile['graph'], 'data_root': 'a'*64,
        'auditing': {**auditing.QUORUM_PROFILE, 'commit_blocks': 16, 'reveal_blocks': 16},
        'planner_work': profile}
    return state.genesis('planner-native-test', validators, manifest), owners


def reserve(s, owners, stages=1):
    before = set(s['auditing']['budgets'])
    s = state.transition(s, tx(s, owners[0], 'fund_audit', publisher=owners[0].public_key,
        auditors=sorted(owner.public_key for owner in owners), stage_limit=stages, expires_in=256))
    budget, = set(s['auditing']['budgets'])-before
    for owner in owners[:3]:
        s = state.transition(s, tx(s, owner, 'accept_audit', budget_id=budget))
    return state.transition(s, tx(s, owners[0], 'reserve_planner',
        input_checkpoint=identity(s['planner_work']['checkpoint']),
        workers=[owner.public_key for owner in owners[:3]], audit_budget=budget))


def submit(s, owners, window):
    receipts = [owner.sign({**planner_work.receipt(s['chain_id'], s['assignment'], window), 'rank': rank})
                for rank, owner in enumerate(owners[:3])]
    def forbidden(*args, **kwargs):
        raise AssertionError('Consensus must not execute the neural model')
    return state.transition(s, tx(s, owners[0], 'claim_planner', window=window, workers=receipts), referee=forbidden)


def finish(s, owners, values=(True, True, True)):
    s = verdicts(s, list(zip(owners, values)))
    return blocks(s, s['candidate']['deadline']-s['height']+1)


def test_complete_funded_window_pays_three_roles_once_and_preserves_serving(claim):
    profile, before, window = claim
    s, owners = network(profile)
    s = submit(reserve(s, owners), owners, window)
    assert s['issued'] == 0 and s['planner_work']['checkpoint'] == before
    assert not auditing.complete(s, s['candidate'])
    s = json.loads(json.dumps(s))  # Persist and restart before native verdicts.
    voted = verdicts(s, [(owner, True) for owner in owners[:3]])
    balances = [state.account(voted, owner.public_key)['balance'] for owner in owners[:3]]
    accepted = blocks(voted, voted['candidate']['deadline']-voted['height']+1)
    reward = state.PARAMS['reward_atoms']
    assert accepted['issued'] == reward and len(accepted['paid_work']) == 1
    assert accepted['planner_work']['checkpoint'] == window['checkpoints'][-1]
    assert accepted['serving_root'] == profile['graph']
    # Audit fees and returned bonds are accounted separately from the shares.
    audit_fee = auditing.QUORUM_PROFILE['price_per_stage']
    audit_bond = auditing.QUORUM_PROFILE['auditor_bond']
    unused_budget = 4*audit_fee-3*audit_fee
    changes = [state.account(accepted, owner.public_key)['balance']-balance
               for owner, balance in zip(owners[:3], balances)]
    assert changes == [reward//3+(i == 0)+audit_fee+audit_bond+
                       (state.PARAMS['claim_bond']+unused_budget if i == 0 else 0) for i in range(3)]
    state.invariant(accepted)
    with pytest.raises(ValueError, match='current planner checkpoint'):
        state.transition(accepted, tx(accepted, owners[0], 'reserve_planner',
            input_checkpoint=identity(before), workers=[o.public_key for o in owners[:3]], audit_budget='f'*64))
    with pytest.raises(ValueError, match='consecutive window'):
        submit(reserve(accepted, owners), owners, window)


@pytest.mark.parametrize('result', ['rejected', 'missing', 'minority'])
def test_negative_missing_and_minority_audits_never_pay(claim, result):
    profile, before, window = claim
    s, owners = network(profile)
    s = submit(reserve(s, owners), owners, window)
    if result == 'rejected':
        s = finish(s, owners, (False, False, False))
    else:
        if result == 'minority':
            s = verdicts(s, [(owners[0], True)])
        s = blocks(s, s['candidate']['audit_reveal_end']-s['height']+1)
    assert s['candidate'] is None and s['issued'] == 0 and s['paid_work'] == {}
    assert s['planner_work']['checkpoint'] == before and s['serving_root'] == profile['graph']
    state.invariant(s)


def test_wrong_worker_partial_receipts_and_duplicate_numerical_work_fail(claim):
    profile, _, window = claim
    s, owners = network(profile)
    reserved = reserve(s, owners)
    with pytest.raises(ValueError, match='Worker receipt differs'):
        submit(reserved, [owners[0], owners[2], owners[1], owners[3]], window)
    with pytest.raises(ValueError, match='all participating'):
        state.transition(reserved, tx(reserved, owners[0], 'claim_planner', window=window, workers=[]))
    s = copy.deepcopy(reserved)
    key, = planner_window.validate(profile, profile['initial'], window)['work_ids']
    s['paid_work'][key] = 'b'*64
    with pytest.raises(ValueError, match='already been paid'):
        submit(s, owners, window)
    claimed = submit(reserved, owners, window)
    with pytest.raises(ValueError, match='weighted replay verdict'):
        state.transition(claimed, tx(claimed, owners[0], 'challenge', claim_id=claimed['candidate']['id'],
            stage=0, challenge_kind='fraud', object_root=None))
    with pytest.raises(ValueError, match='prescribed planner'):
        state.transition(reserved, tx(reserved, owners[0], 'reserve_shards',
            input_checkpoint='a'*64, workers=[owners[0].public_key], audit_budget='b'*64))


def test_replay_backend_cannot_change_binding_coverage_or_boolean_verdict(claim):
    profile, _, window = claim
    s, owners = network(profile)
    candidate = submit(reserve(s, owners), owners, window)['candidate']
    report = {'format': planner_work.FORMAT+'/replay', 'claim_id': candidate['id'],
        'record_root': candidate['record_root'], 'binding': planner_work.binding(candidate),
        'stages': [{'stage': 0, 'valid': True}]}
    assert planner_work.replay_report(candidate, report)['valid']
    for change in ('input', 'coverage', 'boolean'):
        altered = copy.deepcopy(report)
        if change == 'input': altered['binding']['input_checkpoint'] = 'f'*64
        elif change == 'coverage': altered['stages'] = []
        else: altered['stages'][0]['valid'] = 1
        with pytest.raises(ValueError):
            planner_work.replay_report(candidate, altered)
