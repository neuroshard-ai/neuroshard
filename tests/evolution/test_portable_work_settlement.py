"""Connect actual portable checkpoint bytes to the existing native ledger."""
import copy
import hashlib
import json

import pytest

from neuroshard.demo import protocol
from neuroshard.evolution import auditing, portable_work, settlement as state
from neuroshard.evolution.reference_data import identity
from test_adaptive_shards import run
from test_native_audit_quorum import verdicts
from test_settlement import tx, blocks


@pytest.fixture(scope='module')
def checkpoints(tmp_path_factory):
    home = tmp_path_factory.mktemp('portable-native')
    child = run(home, 'baseline', [0, 3, 6], 0, 1)
    initial = json.loads((home/'baseline/rank-0/commit-000000.json').read_bytes())
    traces = [json.loads((home/'baseline'/f'rank-{rank}'/'transcript/transcript.json').read_bytes()) for rank in range(2)]
    from neuroshard.evolution.sharded.transcript import validate
    return initial, child, validate(traces)


@pytest.fixture
def network(checkpoints):
    initial, child, transcript = checkpoints
    owners = [protocol.Identity(f'portable-native-{i}') for i in range(4)]
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    validators = [{'owner': owner.public_key, 'consensus_key': Ed25519PrivateKey.from_private_bytes(
        hashlib.sha256(str(i).encode()).digest()).public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex(),
        'bond': state.PARAMS['bond_unit'], 'liquid': 1_000_000_000} for i, owner in enumerate(owners)]
    manifest = {'params': state.PARAMS, 'initial_model_root': initial['state_root'], 'data_root': 'a'*64,
        'auditing': {**auditing.QUORUM_PROFILE, 'commit_blocks': 16, 'reveal_blocks': 16},
        'portable_work': {'format': portable_work.FORMAT, 'checkpoint': initial,
                         'prepared': 'b'*64, 'max_step': 4, 'max_window_steps': 4}}
    s = state.genesis('portable-native-test', validators, manifest)
    s = state.transition(s, tx(s, owners[0], 'fund_audit', publisher=owners[0].public_key,
        auditors=sorted(o.public_key for o in owners), stage_limit=2, expires_in=64))
    budget = next(iter(s['auditing']['budgets']))
    for owner in owners[:3]:
        s = state.transition(s, tx(s, owner, 'accept_audit', budget_id=budget))
    s = state.transition(s, tx(s, owners[0], 'reserve_shards', input_checkpoint=identity(initial),
        workers=[o.public_key for o in owners[:2]], audit_budget=budget))
    return s, owners, initial, child, transcript


def claim(s, owners, child, transcript):
    receipts = [owners[rank].sign(portable_work.receipt(s['chain_id'], s['assignment'], child, transcript, rank))
                for rank in range(2)]
    return state.transition(s, tx(s, owners[0], 'claim_shards', output=child,
                                  transcript_root=transcript, workers=receipts))


def test_portable_work_mints_only_after_native_quorum_and_never_twice(network):
    s, owners, initial, child, transcript = network
    s = claim(s, owners, child, transcript)
    assert s['issued'] == 0 and s['model_root'] == initial['state_root']
    s = verdicts(s, [(o, True) for o in owners[:3]])
    s = blocks(s, state.PARAMS['challenge_blocks']+1)
    assert s['model_root'] == child['state_root'] and s['issued'] == state.PARAMS['reward_atoms']
    assert s['serving_root'] == initial['state_root']
    assert s['portable_work']['checkpoint'] == child
    assert len(s['paid_work']) == 1 and s['auditing']['paid_services'] == 3
    with pytest.raises(ValueError, match='current portable checkpoint'):
        state.transition(s, tx(s, owners[0], 'reserve_shards', input_checkpoint=identity(initial),
            workers=[o.public_key for o in owners[:2]], audit_budget='d'*64))
    state.invariant(s)


def test_quorum_rejection_and_missing_reports_never_advance_the_portable_model(network):
    s, owners, initial, child, transcript = network
    claimed = claim(s, owners, child, transcript)
    absent = blocks(claimed, claimed['candidate']['audit_reveal_end']+1)
    assert absent['issued'] == 0 and absent['model_root'] == initial['state_root']
    rejected = verdicts(claimed, [(o, False) for o in owners[:3]])
    rejected = blocks(rejected, state.PARAMS['challenge_blocks']+1)
    assert rejected['issued'] == 0 and rejected['model_root'] == initial['state_root']
    assert rejected['auditing']['paid_services'] == 3
    state.invariant(absent)
    state.invariant(rejected)


def test_receipts_parent_and_parameter_ages_are_bound_to_the_window(network):
    s, owners, _, child, transcript = network
    damaged = copy.deepcopy(child)
    damaged['parent'] = 'c'*64
    with pytest.raises(ValueError, match='replace its parent'):
        claim(s, owners, damaged, transcript)
    damaged = copy.deepcopy(child)
    damaged['tensors'][next(iter(damaged['tensors']))]['born'] = 1
    damaged['state_root'] = identity({k: damaged[k] for k in ['format', 'job', 'step', 'config', 'optimizer', 'tensors']})
    with pytest.raises(ValueError, match='reset parameter ages'):
        claim(s, owners, damaged, transcript)
    receipts = [owners[rank].sign(portable_work.receipt(s['chain_id'], s['assignment'], child, 'e'*64, rank))
                for rank in range(2)]
    with pytest.raises(ValueError, match='reserved shard window'):
        state.transition(s, tx(s, owners[0], 'claim_shards', output=child, transcript_root=transcript, workers=receipts))
    with pytest.raises(ValueError, match='only its prepared portable execution'):
        state.transition(s, tx(s, owners[0], 'reserve', parent=s['model_root'], round=0,
            workers=[o.public_key for o in owners[:2]], audit_budget='d'*64))


def test_a_validator_backend_must_bind_and_cover_the_entire_gpu_window(network):
    s, owners, before, after, transcript = network
    s = claim(s, owners, after, transcript)
    c = s['candidate']
    binding = {'job': before['job'], 'prepared': c['prepared'], 'input': identity(before),
               'output': identity(after), 'start': before['step'], 'end': after['step'],
               'boundaries': before['boundaries'], 'reference': identity(None)}
    rows = [{'rank': rank, 'valid': True, 'transcript_root': transcript, 'binding': binding} for rank in range(2)]
    assert portable_work.replay_report(c, rows)['valid']
    with pytest.raises(ValueError, match='every shard'):
        portable_work.replay_report(c, rows[:1])
    wrong = copy.deepcopy(rows)
    wrong[1]['binding']['input'] = 'f'*64
    with pytest.raises(ValueError, match='stale, incomplete'):
        portable_work.replay_report(c, wrong)
    wrong = copy.deepcopy(rows)
    for row in wrong:
        row['binding']['reference'] = 'a'*64
    with pytest.raises(ValueError, match='stale, incomplete'):
        portable_work.replay_report(c, wrong)
    rows[0]['valid'] = False
    assert not portable_work.replay_report(c, rows)['valid']


@pytest.mark.parametrize('group', [1, [], None, 'adam'])
def test_malformed_optimizer_group_is_an_invalid_transaction_not_an_abci_error(network, group):
    import threading
    from neuroshard.demo import abci_pb2 as pb
    from neuroshard.dataflow.store import canonical
    from neuroshard.evolution.app import Application

    s, owners, _, child, transcript = network
    damaged = copy.deepcopy(child)
    # Preserve inventory indices and group count to reach group validation.
    damaged['optimizer'] = [group for _ in child['optimizer']]
    damaged['state_root'] = identity({k: damaged[k] for k in
        ['format', 'job', 'step', 'config', 'optimizer', 'tensors']})
    receipts = [owners[rank].sign(portable_work.receipt(
        s['chain_id'], s['assignment'], damaged, transcript, rank)) for rank in range(2)]
    raw = canonical(tx(s, owners[0], 'claim_shards', output=damaged,
                       transcript_root=transcript, workers=receipts))
    application = Application.__new__(Application)
    application.lock, application.state, application.artifacts = threading.RLock(), s, None
    result = application.CheckTx(pb.RequestCheckTx(tx=raw), None)
    assert result.code == 1
    assert 'Adam group' in result.log
