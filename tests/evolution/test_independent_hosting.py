import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from neuroshard.demo import protocol
from neuroshard.evolution import auditing, expert_lifecycle as life, expert_work, hosting, settlement as state
from neuroshard.evolution.independent_hosting import (
    ACCEPTED_GRAPH, CONTRACT_IDENTITY, FORMAT, ITEM, LEARNED_INTEGRATION, METHOD_FORMAT,
    PREVIOUS_CONTRACT, administrator_powers, bind_cpu_genesis, bind_method_freeze,
    bind_soak_administration, bind_spec, load_spec, method_freeze, refuse_launch,
    share_is_concentrated, validator_powers,
)
from neuroshard.evolution.reference_data import identity, sha256
from test_expert_lifecycle import send
from test_serving_graph import FIXTURE, graphs
from test_settlement import tx


ROOT = Path(__file__).resolve().parents[2]


def spec():
    return json.loads((ROOT / 'config/experiments/independent-hosting.json').read_text())


def validators_for(owners, bonds):
    rows = []
    for i, owner in enumerate(owners):
        key = Ed25519PrivateKey.from_private_bytes(
            hashlib.sha256(('independent-hosting-validator-' + str(i)).encode()).digest())
        rows.append({
            'owner': owner.public_key,
            'consensus_key': key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex(),
            'bond': bonds[i],
            'liquid': 10**10,
        })
    return rows


def genesis_for(graphs, owners, bonds, *, hosting_enabled=False):
    previous, candidate = graphs
    initial = json.loads(FIXTURE.read_bytes())['initial_expert']
    profile = {
        'format': expert_work.FORMAT, 'parent': candidate['parent'], 'checkpoint': initial,
        'prepared': 'a' * 64, 'feature_root': 'b' * 64, 'feature_stages': 336,
        'batch_roots': ['c' * 64], 'schedule': [0] * 560,
        'numerical_profile': candidate['numerical_profile'],
    }
    manifest = {
        'params': state.PARAMS,
        'initial_model_root': candidate['parent']['state_root'],
        'data_root': 'a' * 64,
        'auditing': {**auditing.QUORUM_PROFILE, 'commit_blocks': 16, 'reveal_blocks': 16},
        'expert_work': profile,
        'expert_lifecycle': {
            'format': life.FORMAT, 'serving_graph': previous, 'candidate_graph': candidate,
            'quality': {'policy_root': 'd' * 64, 'prepared': 'e' * 64, 'stages': 96},
            'price_per_token': 101, 'max_tokens': 64,
        },
    }
    if hosting_enabled:
        manifest['hosting'] = {**hosting.PROFILE, 'prepare_blocks': 4, 'execution_blocks': 8,
                               'cooldown_blocks': 2}
    return state.genesis('independent-hosting-test', validators_for(owners, bonds), manifest)


def test_frozen_contract_identity_is_pinned():
    current = spec()
    assert identity(current) == CONTRACT_IDENTITY
    assert current['format'] == FORMAT
    assert current['item'] == ITEM
    assert current['accepted_graph'] == ACCEPTED_GRAPH
    assert current['parent']['learned_integration'] == LEARNED_INTEGRATION
    assert current['parent']['independent_hosting'] == PREVIOUS_CONTRACT
    assert current['revision']['replaces'] == PREVIOUS_CONTRACT
    assert current['independent_soak']['minimum_independent_operators'] == 4
    assert current['independent_soak']['voting_share_aggregates_by'] == 'administrator'
    assert bind_spec(current) == {
        'independent_hosting': CONTRACT_IDENTITY,
        'gpu_launch_authorized': False,
        'admission_evidence': False,
        'independent_administration': False,
    }
    assert identity(load_spec()) == CONTRACT_IDENTITY


def test_bind_rejects_gpu_admission_and_public_upgrade(monkeypatch):
    from neuroshard.evolution import independent_hosting as module

    def reject(field, value, match):
        current = spec()
        current[field] = value
        monkeypatch.setattr(module, 'CONTRACT_IDENTITY', identity(current))
        with pytest.raises(ValueError, match=match):
            bind_spec(current)

    reject('gpu_launch_authorized', True, 'does not authorize a GPU launch')
    reject('upgrade_public_0_4_0', True, 'does not upgrade the public 0.4.0 chain')
    reject('aws_under_one_account_does_not_satisfy', False, 'do not satisfy item 4')
    reject('reopen_learned_integration_confirmation', True, 'confirmation stays closed')
    reject('independent_soak', {**spec()['independent_soak'], 'authorized': True},
           'soak is not authorized')
    reject('independent_soak', {**spec()['independent_soak'], 'minimum_independent_operators': 3},
           'four independently administered operators')
    reject('independent_soak', {**spec()['independent_soak'],
                                'voting_share_aggregates_by': 'key'},
           'by administrator, not key')


def test_four_equal_validators_pass_the_cpu_genesis_bound(graphs):
    owners = [protocol.Identity('independent-host-owner-' + str(i)) for i in range(4)]
    network = genesis_for(graphs, owners, [state.PARAMS['bond_unit']] * 4)
    powers = bind_cpu_genesis(network)
    assert len(powers) == 4
    assert len(set(powers.values())) == 1
    assert share_is_concentrated(powers) is False
    assert max(powers.values()) * 3 < sum(powers.values())


def test_three_equal_validators_are_concentrated_and_rejected(graphs):
    owners = [protocol.Identity('independent-host-three-' + str(i)) for i in range(3)]
    network = genesis_for(graphs, owners, [state.PARAMS['bond_unit']] * 3)
    powers = validator_powers(network)
    assert share_is_concentrated(powers) is True
    with pytest.raises(ValueError, match='four validators'):
        bind_cpu_genesis(network)


def test_a_majority_bond_is_concentrated_even_with_four_keys(graphs):
    owners = [protocol.Identity('independent-host-unequal-' + str(i)) for i in range(4)]
    bonds = [state.PARAMS['bond_unit'] * 2, state.PARAMS['bond_unit'],
             state.PARAMS['bond_unit'], state.PARAMS['bond_unit']]
    network = genesis_for(graphs, owners, bonds)
    with pytest.raises(ValueError, match='equal validator bonds'):
        bind_cpu_genesis(network)
    assert share_is_concentrated(validator_powers(network)) is True


def test_a_stranger_key_can_register_as_a_provider_without_genesis_membership(graphs):
    owners = [protocol.Identity('independent-host-market-' + str(i)) for i in range(4)]
    stranger = protocol.Identity('independent-host-stranger')
    network = genesis_for(graphs, owners, [state.PARAMS['bond_unit']] * 4, hosting_enabled=True)
    bind_cpu_genesis(network)
    assert stranger.public_key not in {row['owner'] for row in network['validators'].values()}
    collateral = 50 * hosting.PROFILE['lease_bond']
    network = state.transition(network, tx(
        network, owners[0], 'transfer', to=stranger.public_key, amount=collateral + 10**6))
    network = send(network, stranger, 'register_provider',
                   endpoint='https://stranger.example:8443', certificate='1' * 64,
                   collateral=collateral)
    network = send(network, stranger, 'offer_expert', graph=network['serving_root'],
                   rank=0, fee=100, capacity=1, expires_in=10000)
    assert stranger.public_key in network['hosting']['providers']
    assert any(offer['owner'] == stranger.public_key for offer in network['hosting']['offers'].values())


def test_one_provider_key_still_cannot_hold_the_complete_backbone(graphs):
    owners = [protocol.Identity('independent-host-backbone-' + str(i)) for i in range(4)]
    network = genesis_for(graphs, owners, [state.PARAMS['bond_unit']] * 4, hosting_enabled=True)
    bind_cpu_genesis(network)
    network = send(network, owners[0], 'register_provider',
                   endpoint='https://owner-0.example:8443', certificate='1' * 64,
                   collateral=50 * hosting.PROFILE['lease_bond'])
    offers = {}
    for rank in range(3):
        before = set(network['hosting']['offers'])
        network = send(network, owners[0], 'offer_expert', graph=network['serving_root'],
                       rank=rank, fee=100, capacity=1, expires_in=10000)
        offers[str(rank)] = (set(network['hosting']['offers']) - before).pop()
    with pytest.raises(ValueError, match='complete backbone'):
        hosting.select(network, network['expert_lifecycle']['serving_graph'],
                       {'0', '1', '2'}, offers, network['height'] + 32, 20000)


def test_four_administrators_with_one_validator_each_pass_the_soak_bound(graphs):
    owners = [protocol.Identity('independent-host-admin-' + str(i)) for i in range(4)]
    network = genesis_for(graphs, owners, [state.PARAMS['bond_unit']] * 4)
    powers = bind_cpu_genesis(network)
    ownership = {key: 'operator-' + str(i) for i, key in enumerate(sorted(powers))}
    admins = bind_soak_administration(powers, ownership)
    assert len(admins) == 4
    assert share_is_concentrated(admins) is False


def test_three_administrators_cannot_satisfy_the_strict_one_third_bound(graphs):
    owners = [protocol.Identity('independent-host-shared-admin-' + str(i)) for i in range(4)]
    network = genesis_for(graphs, owners, [state.PARAMS['bond_unit']] * 4)
    powers = bind_cpu_genesis(network)
    keys = sorted(powers)
    ownership = {keys[0]: 'operator-a', keys[1]: 'operator-a',
                 keys[2]: 'operator-b', keys[3]: 'operator-c'}
    admins = administrator_powers(powers, ownership)
    assert len(admins) == 3
    assert share_is_concentrated(admins) is True
    with pytest.raises(ValueError, match='four independently administered operators'):
        bind_soak_administration(powers, ownership)


def test_method_freeze_refuses_gpu_and_soak():
    current = spec()
    freeze = method_freeze()
    saved = json.loads((ROOT / 'config/experiments/independent-hosting-method.json').read_text())
    assert freeze['format'] == METHOD_FORMAT
    assert freeze['gpu_launch_authorized'] is False
    assert freeze['independent_soak_authorized'] is False
    assert freeze['neural_execution'] is False
    assert saved == freeze
    assert identity(saved) == '5d44fdd0523e0e168a5bd0fca9f0b975c7ef71bf8851ca29fb9e3a65d07c047c'
    digest = bind_method_freeze(freeze, current)
    assert digest == identity(saved)
    with pytest.raises(ValueError, match='does not authorize a GPU launch or independent-operator soak'):
        refuse_launch(current, freeze)
    for name, digest in freeze['files'].items():
        assert sha256(ROOT / name) == digest
