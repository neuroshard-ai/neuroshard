"""Settle actual expert replay records through the native weighted quorum."""
import copy
import hashlib
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from neuroshard.demo import protocol
from neuroshard.evolution import auditing, expert_work, settlement as state
from neuroshard.evolution.reference_data import identity
from test_expert_window_claim import records
from test_native_audit_quorum import verdicts
from test_settlement import tx, blocks


@pytest.fixture
def network(records):
    parent, window, checkpoints, inputs = records
    owners = [protocol.Identity('expert-native-'+str(i)) for i in range(4)]
    validators = []
    for i, owner in enumerate(owners):
        key = Ed25519PrivateKey.from_private_bytes(hashlib.sha256(('expert-validator-'+str(i)).encode()).digest())
        validators.append({'owner': owner.public_key, 'consensus_key': key.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw).hex(), 'bond': state.PARAMS['bond_unit'], 'liquid': 1_000_000_000})
    manifest = {'params': state.PARAMS, 'initial_model_root': parent['state_root'], 'data_root': 'a'*64,
        'auditing': {**auditing.QUORUM_PROFILE, 'commit_blocks': 16, 'reveal_blocks': 16},
        'expert_work': {'format': expert_work.FORMAT, 'parent': parent, 'checkpoint': checkpoints[0],
            **inputs,
            'batch_roots': [row['batch'] for row in window['steps']], 'schedule': list(range(4)),
            'numerical_profile': 'b'*64}}
    return state.genesis('expert-native-test', validators, manifest), owners, window, checkpoints


def fund(s, owners, stages):
    s = state.transition(s, tx(s, owners[0], 'fund_audit', publisher=owners[0].public_key,
        auditors=sorted(owner.public_key for owner in owners), stage_limit=stages, expires_in=256))
    budget = next(iter(s['auditing']['budgets']))
    for owner in owners[:3]:
        s = state.transition(s, tx(s, owner, 'accept_audit', budget_id=budget))
    return s, budget


def features(s, owners, produced=None):
    s, budget = fund(s, owners, 336)
    s = state.transition(s, tx(s, owners[0], 'reserve_expert_inputs',
        workers=[owner.public_key for owner in owners[:3]], audit_budget=budget))
    output = produced if produced is not None else s['manifest']['expert_work']['feature_root']
    transcript = 'e'*64
    receipts = [owner.sign(expert_work.receipt(s['chain_id'], s['assignment'], output, transcript, rank))
                for rank, owner in enumerate(owners[:3])]
    values = produced if produced is not None else {'feature_root': output}
    kind = 'claim_expert_prefix' if produced is not None else 'claim_expert_inputs'
    return state.transition(s, tx(s, owners[0], kind,
        **values, transcript_root=transcript, workers=receipts), referee=forbidden_referee)


def forbidden_referee(*args, **kwargs):
    raise AssertionError('Native consensus must not execute a neural model')


def test_fresh_prefix_is_unknown_at_genesis_and_only_accepted_bytes_enable_training(network):
    original, owners, window, checkpoints = network
    manifest = copy.deepcopy(original['manifest'])
    profile = manifest['expert_work']
    produced = {key: profile.pop(key) for key in ('feature_root', 'batch_roots')}
    profile.update(format=expert_work.PROSPECTIVE, batch_count=len(produced['batch_roots']))
    assert produced['feature_root'] not in json.dumps(manifest)
    validators = [{'owner': row['owner'], 'consensus_key': key, 'bond': row['amount'],
                   'liquid': original['accounts'][row['owner']]['balance']}
                  for key, row in original['validators'].items()]
    s = state.genesis('fresh-expert-native-test', validators, manifest)
    pending = features(s, owners, produced)
    with pytest.raises(ValueError, match='prefix execution audit'):
        expert_work.execution_profile(pending)
    rejected = finish(pending, owners, False)
    assert rejected['expert_work']['feature_root'] is None and rejected['issued'] == 0
    accepted = finish(pending, owners)
    assert expert_work.execution_profile(accepted) == original['manifest']['expert_work']
    settled = finish(training(accepted, owners, window, checkpoints), owners)
    assert settled['issued'] == 4 * state.PARAMS['reward_atoms']
    assert settled['expert_work']['checkpoint'] == checkpoints[-1]
    assert settled['serving_root'] == s['serving_root'] and settled['manifest'] == manifest


def finish(s, owners, valid=True):
    s = verdicts(s, [(owner, valid) for owner in owners[:3]])
    return blocks(s, s['candidate']['deadline'] - s['height'] + 1)


def training(s, owners, window, checkpoints):
    s, budget = fund(s, owners, 4)
    s = state.transition(s, tx(s, owners[0], 'reserve_expert', input_checkpoint=checkpoints[0]['checkpoint'],
        worker=owners[0].public_key, audit_budget=budget))
    receipt = owners[0].sign(expert_work.receipt(s['chain_id'], s['assignment'],
        window['output']['checkpoint'], identity(window), 0))
    return state.transition(s, tx(s, owners[0], 'claim_expert', window=window,
        intermediates=checkpoints, worker=receipt), referee=forbidden_referee)


def test_prefix_quorum_then_bounded_update_quorum_issues_without_serving_promotion(network):
    s, owners, window, checkpoints = network
    initial_serving = s['serving_root']
    with pytest.raises(ValueError, match='prefix execution audit'):
        state.transition(s, tx(s, owners[0], 'reserve_expert', input_checkpoint=checkpoints[0]['checkpoint'],
            worker=owners[0].public_key, audit_budget='f'*64))
    s = features(s, owners)
    assert s['issued'] == 0 and s['expert_work']['feature_claim'] is None
    s = finish(s, owners)
    assert s['expert_work']['feature_claim'] and s['issued'] == 0
    s = training(s, owners, window, checkpoints)
    assert s['issued'] == 0 and s['expert_work']['checkpoint'] == checkpoints[0]
    s = finish(s, owners)
    assert s['issued'] == 4 * state.PARAMS['reward_atoms'] and s['training_round'] == 4
    assert s['expert_work']['checkpoint'] == checkpoints[-1]
    assert s['model_root'] == checkpoints[-1]['state_root'] and s['serving_root'] == initial_serving
    assert len(s['paid_work']) == 4
    with pytest.raises(ValueError, match='current expert checkpoint'):
        state.transition(s, tx(s, owners[0], 'reserve_expert', input_checkpoint=checkpoints[0]['checkpoint'],
            worker=owners[0].public_key, audit_budget='f'*64))
    state.invariant(s)


def test_missing_or_negative_prefix_audits_never_enable_training(network):
    s, owners, window, checkpoints = network
    claimed = features(s, owners)
    for result in (finish(claimed, owners, False), blocks(claimed, claimed['candidate']['audit_reveal_end']+1)):
        assert result['issued'] == 0 and result['expert_work']['feature_claim'] is None
        with pytest.raises(ValueError, match='prefix execution audit'):
            state.transition(result, tx(result, owners[0], 'reserve_expert',
                input_checkpoint=checkpoints[0]['checkpoint'], worker=owners[0].public_key, audit_budget='f'*64))
        state.invariant(result)


def test_negative_or_missing_update_audits_never_pay_or_advance(network):
    s, owners, window, checkpoints = network
    s = finish(features(s, owners), owners)
    claimed = training(s, owners, window, checkpoints)
    for result in (finish(claimed, owners, False),
                   blocks(claimed, claimed['candidate']['audit_reveal_end']-claimed['height']+1)):
        assert result['issued'] == 0 and result['expert_work']['checkpoint'] == checkpoints[0]
        assert result['serving_root'] == s['serving_root']
        state.invariant(result)


def test_window_relabeling_or_missing_states_cannot_enter_a_claim(network):
    s, owners, window, checkpoints = network
    s = finish(features(s, owners), owners)
    changed = copy.deepcopy(window)
    changed['steps'][0]['work_identity'] = 'f'*64
    with pytest.raises(ValueError, match='actual consumed'):
        training(s, owners, changed, checkpoints)
    with pytest.raises(ValueError, match='every intermediate'):
        training(s, owners, window, checkpoints[:2]+checkpoints[3:])
    with pytest.raises(ValueError, match='prepared expert'):
        state.transition(s, tx(s, owners[0], 'reserve_shards', input_checkpoint='a'*64,
            workers=[owners[0].public_key], audit_budget='b'*64))
    assert s['issued'] == 0 and s['assignment'] is None


def test_auditor_backend_cannot_omit_stages_or_bind_a_different_input(network):
    s, owners, window, checkpoints = network
    s = finish(features(s, owners), owners)
    claim = training(s, owners, window, checkpoints)['candidate']
    report = {'format': expert_work.FORMAT+'/replay', 'claim_id': claim['id'],
        'record_root': claim['record_root'], 'binding': {
            'parent': identity(claim['parent_checkpoint']), 'prepared': claim['prepared'],
            'input_checkpoint': checkpoints[0]['checkpoint'], 'output_root': checkpoints[-1]['checkpoint'],
            'feature_root': claim['feature_root'], 'feature_claim': claim['feature_claim'],
            'numerical_profile': 'b'*64}, 'stages': [{'stage': i, 'valid': True} for i in range(4)]}
    assert expert_work.replay_report(claim, report)['valid']
    damaged = copy.deepcopy(report)
    damaged['stages'].pop()
    with pytest.raises(ValueError, match='complete claimed'):
        expert_work.replay_report(claim, damaged)
    damaged = copy.deepcopy(report)
    damaged['binding']['input_checkpoint'] = 'f'*64
    with pytest.raises(ValueError, match='complete claimed'):
        expert_work.replay_report(claim, damaged)
    report['stages'][1]['valid'] = False
    assert not expert_work.replay_report(claim, report)['valid']
