"""Admission and paid serving cannot bypass reserved computation or quality."""
import copy

import pytest

from neuroshard.demo import protocol
from neuroshard.evolution import portable_lifecycle as life, portable_work, settlement as state
from neuroshard.evolution.reference_data import identity
from test_native_audit_quorum import verdicts
from test_portable_work_settlement import checkpoints, network, claim
from test_settlement import blocks, tx


def send(s, who, kind, **fields):
    result = state.transition(s, tx(s, who, kind, **fields))
    state.invariant(result)
    return result


@pytest.fixture
def ready(network):
    s, owners, initial, child, transcript = network
    # End the old fixture reservation and enable only through a new genesis.
    manifest = copy.deepcopy(s['manifest'])
    manifest['portable_lifecycle'] = {
        'format': life.FORMAT, 'serving_checkpoint': initial, 'tokenizer_root': 'e'*64,
        'eos_id': 2, 'price_per_token': 100, 'max_prompt_tokens': 32, 'max_new_tokens': 8,
        'proposal_blocks': 64, 'job_blocks': 10000,
        'executor_root': '6'*64,
    }
    validators = [{'owner': v['owner'], 'consensus_key': key, 'bond': v['amount'],
                   'liquid': 1_000_000_000} for key, v in s['validators'].items()]
    fresh = state.genesis('portable-life-test', validators, manifest)
    job = {'parent': identity(initial), 'job': 'c'*64, 'prepared': 'b'*64,
        'reference_root': identity(None), 'executor_root': '6'*64, 'max_step': 1, 'max_window_steps': 1,
        'quality': {'policy_root': 'f'*64, 'baseline_checkpoint': identity(initial), 'stages': 2}}
    # Ledger tests exercise commitments; the numerical tests check these bytes.
    output = copy.deepcopy(child)
    output['job'] = job['job']
    output['state_root'] = identity({k: output[k] for k in
        ('format', 'job', 'step', 'config', 'optimizer', 'tensors')})
    return fresh, owners, initial, output, transcript, job


def activate(case):
    s, owners, initial, child, transcript, job = case
    s = send(s, owners[0], 'propose_shard_job', job=job)
    proposal = s['portable_lifecycle']['proposal']['id']
    for owner in owners[:3]:
        s = send(s, owner, 'vote_shard_job', proposal_id=proposal, approve=True)
    s = blocks(s, state.PARAMS['activation_blocks'])
    assert s['portable_lifecycle']['active']['job'] == job
    assert s['model_root'] == initial['state_root'] and s['issued'] == 0
    return s


def fund(s, owners, stages):
    prior = set(s['auditing']['budgets'])
    s = send(s, owners[0], 'fund_audit', publisher=owners[0].public_key,
             auditors=[], stage_limit=stages, expires_in=256)
    key = (set(s['auditing']['budgets']) - prior).pop()
    for owner in owners[:3]:
        s = send(s, owner, 'accept_audit', budget_id=key)
    return s, key


def train(case):
    _, owners, initial, child, transcript, _ = case
    s, budget = fund(activate(case), owners, 2)
    s = send(s, owners[0], 'reserve_shards', input_checkpoint=identity(initial),
             workers=[o.public_key for o in owners[:2]], audit_budget=budget)
    s = claim(s, owners, child, transcript)
    assert s['candidate']['execution_job'] == child['job']
    s = verdicts(s, [(owner, True) for owner in owners[:3]])
    return blocks(s, state.PARAMS['challenge_blocks'] + 1)


def quality(s, owners, passed=True):
    active = s['portable_lifecycle']['active']
    report = {'format': life.FORMAT + '/quality',
        'policy_root': active['job']['quality']['policy_root'],
        'baseline_checkpoint': active['job']['quality']['baseline_checkpoint'],
        'candidate_checkpoint': identity(s['portable_work']['checkpoint']),
        'prepared': active['job']['prepared'], 'passed': passed, 'results_root': 'd'*64}
    s, budget = fund(s, owners, 2)
    return send(s, owners[0], 'quality_shards', job_id=active['id'], report=report,
                transcript_root='9'*64, audit_budget=budget)


def approve(s, owners):
    return blocks(verdicts(s, [(o, True) for o in owners[:3]]), state.PARAMS['challenge_blocks'] + 1)


def request(s, owners):
    previous = set(s['portable_lifecycle']['jobs'])
    s = send(s, owners[3], 'infer_shards', checkpoint=identity(s['portable_lifecycle']['serving_checkpoint']),
        workers=[o.public_key for o in owners[:2]], prompt_ids=[1, 3], max_tokens=3,
        max_price=1000, expires_in=2048)
    key = (set(s['portable_lifecycle']['jobs']) - previous).pop()
    return s, key


def response(s, owners, key, tokens=None):
    tokens = [4, 2] if tokens is None else tokens
    job = s['portable_lifecycle']['jobs'][key]
    s, budget = fund(s, owners, len(tokens))
    receipts = [owners[r].sign(life.inference_receipt(s['chain_id'], job, tokens, '8'*64, r))
                for r in range(2)]
    return send(s, owners[0], 'respond_shards', job_id=key, token_ids=tokens,
                transcript_root='8'*64, workers=receipts, audit_budget=budget)


def test_job_requires_native_admission_and_cannot_import_or_repay_a_checkpoint(ready):
    s, owners, initial, child, transcript, job = ready
    with pytest.raises(ValueError, match='Activate a live'):
        send(s, owners[0], 'reserve_shards', input_checkpoint=identity(initial),
             workers=[o.public_key for o in owners[:2]], audit_budget='a'*64)
    wrong = {**job, 'parent': identity(child)}
    with pytest.raises(ValueError, match='currently settled'):
        send(s, owners[0], 'propose_shard_job', job=wrong)
    s = send(s, owners[0], 'propose_shard_job', job=job)
    proposal = s['portable_lifecycle']['proposal']['id']
    for owner in owners[:2]:
        s = send(s, owner, 'vote_shard_job', proposal_id=proposal, approve=True)
    s = blocks(s, state.PARAMS['activation_blocks'])
    assert s['portable_lifecycle']['active'] is None
    expired = blocks(s, 65)
    assert expired['portable_lifecycle']['proposal'] is None and expired['issued'] == 0
    s = train(ready)
    assert s['issued'] == state.PARAMS['reward_atoms']
    assert s['serving_root'] == initial['state_root']
    with pytest.raises(ValueError, match='exhausted'):
        send(s, owners[0], 'reserve_shards', input_checkpoint=identity(child),
             workers=[o.public_key for o in owners[:2]], audit_budget='a'*64)


def test_quality_pass_promotes_only_after_separate_quorum_without_issuance(ready):
    _, owners, initial, child, _, _ = ready
    s = quality(train(ready), owners)
    assert s['serving_root'] == initial['state_root']
    before = s['issued']
    s = approve(s, owners)
    assert s['serving_root'] == child['state_root'] and s['issued'] == before
    assert s['portable_lifecycle']['active']['closed']
    with pytest.raises(ValueError, match='Activate a live'):
        quality(s, owners)


def test_failed_quality_preserves_serving_and_already_earned_training_rewards(ready):
    _, owners, initial, child, _, _ = ready
    s = approve(quality(train(ready), owners, passed=False), owners)
    assert s['serving_root'] == initial['state_root']
    assert s['model_root'] == child['state_root'] and s['issued'] == state.PARAMS['reward_atoms']
    assert s['portable_lifecycle']['active']['closed']


def test_unsettled_or_substituted_quality_cannot_promote(ready):
    _, owners, *_ = ready
    s = activate(ready)
    with pytest.raises(ValueError, match='Settle every'):
        quality(s, owners)
    claimed = quality(train(ready), owners)
    missing = blocks(claimed, claimed['candidate']['audit_reveal_end'] + 1)
    assert missing['serving_root'] == ready[2]['state_root']
    with pytest.raises(ValueError, match='originally committed'):
        quality(missing, owners, False)


def test_paid_inference_pins_serving_checkpoint_and_refunds_unused_tokens(ready):
    _, owners, initial, child, *_ = ready
    s = train(ready)
    s, key = request(s, owners)
    # The in-flight request still refers to the old checkpoint after promotion.
    s = approve(quality(s, owners), owners)
    assert s['serving_root'] == child['state_root']
    s = response(s, owners, key)
    assert s['candidate']['input_checkpoint'] == initial
    issued = s['issued']
    s = approve(s, owners)
    result = s['portable_lifecycle']['results'][key]
    assert result['checkpoint'] == identity(initial)
    assert result['paid_atoms'] == sum(result['payments']) == 200
    assert result['refunded_atoms'] == 800 and s['issued'] == issued
    with pytest.raises(KeyError):
        response(s, owners, key)


def test_missing_or_rejected_inference_replay_never_pays_and_request_expires(ready):
    s, owners, *_ = ready
    s, key = request(s, owners)
    before = s['issued']
    s = response(s, owners, key)
    s = blocks(s, s['candidate']['audit_reveal_end'] + 1)
    assert key in s['portable_lifecycle']['jobs']
    job = s['portable_lifecycle']['jobs'][key]
    s = blocks(s, job['expires'] - s['height'] + 1)
    result = s['portable_lifecycle']['results'][key]
    assert result['status'] == 'expired' and result['refunded_atoms'] == 1000 and s['issued'] == before


@pytest.mark.parametrize('tokens', [[2, 4], [4], [True, 2], [999, 2]])
def test_serving_rejects_invalid_tokens_and_stopping_rules(ready, tokens):
    s, owners, *_ = ready
    s, key = request(s, owners)
    with pytest.raises(ValueError):
        response(s, owners, key, tokens)


def test_service_audit_must_bind_policy_model_request_and_every_partition(ready):
    _, owners, *_ = ready
    s = quality(train(ready), owners)
    claim = s['candidate']
    binding = {'statement': identity(life.service_statement(claim))}
    rows = [{'rank': rank, 'binding': binding, 'transcript_root': claim['record_root'], 'valid': True}
            for rank in range(2)]
    assert portable_work.replay_report(claim, rows)['valid']
    with pytest.raises(ValueError, match='every portable'):
        portable_work.replay_report(claim, rows[:1])
    mutated = copy.deepcopy(claim)
    mutated['report']['passed'] = False
    with pytest.raises(ValueError, match='complete obligation'):
        portable_work.replay_report(mutated, rows)
    rows[0]['valid'] = False
    assert not portable_work.replay_report(claim, rows)['valid']


def test_later_job_preserves_paid_cursor_and_cannot_reactivate_same_recipe(ready):
    _, owners, _, child, _, job = ready
    s = approve(quality(train(ready), owners), owners)
    later = {**job, 'parent': identity(child), 'max_step': 2,
             'quality': {**job['quality'], 'baseline_checkpoint': identity(child)}}
    with pytest.raises(ValueError, match='only once'):
        send(s, owners[0], 'propose_shard_job', job=later)
    later['job'] = '7'*64
    s = send(s, owners[0], 'propose_shard_job', job=later)
    assert len(s['paid_work']) == 1 and s['training_round'] == 1
