"""The released growth agent reaches validators that can judge a proof, not only the first RPC that answers.

Review of the gate harness (2026-10-24): the agent sent ``growth_prove`` through ``Chain.submit``, which stops at the
first reachable RPC even if it answers "no verdict" (code 3). A validator lacking the bundle then stood between the
proof and four validators that would admit it, on every retry. These run the real ``Agent.check`` and ``Chain``;
only the RPC replies are stubbed.
"""
import base64
import hashlib
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from neuroshard.assistant import network
from neuroshard.evolution import growth_agent
from neuroshard.inference import optimistic as ledger

RPCS = [f'http://validator-{i}' for i in range(5)]


def setup(tmp):
    chain = network.Chain(RPCS, 'isolated-test')
    signed = []

    def sign(chain_id, nonce, kind, **fields):
        envelope = {'body': {'kind': kind, 'chain_id': chain_id, 'nonce': nonce, **fields}, 'public_key': 'aa',
                    'signature': 'sig'}
        signed.append(envelope)
        return envelope
    account = SimpleNamespace(public='challenger', sign=sign)
    job = {'recipe_root': 'recipe', 'commits': [], 'challenges': {'challenge': {'opened': 5}}}
    state = {'height': 6, 'accounts': {'challenger': {'nonce': 4}}, 'growth': {'job': job}}
    agent = growth_agent.Agent(chain, account, tmp, recipes={'recipe': {'present': True}}, trainer=object())
    agent.memory['actions'] = [{'kind': 'challenge', 'job_id': 'job', 'challenge_id': 'challenge', 'segment': 0,
                                'seen': True}]
    return chain, agent, state, job, signed


def committed_rpc(address, method, params, timeout=60):
    assert method == 'tx'
    return {'height': '9', 'tx_result': {'code': 0}}


def test_a_validator_without_the_bundle_does_not_block_the_proof():
    calls = []

    def broadcast(address, envelope):
        calls.append((address, ledger.canonical(envelope)))
        if address.endswith('-0'):  # reachable, preferred, but cannot judge
            return {'code': 3, 'log': 'No verdict on this validator yet: growth recipe or proof bundle not held'}
        return {'code': 0, 'log': '', 'hash': ''}

    with tempfile.TemporaryDirectory() as home:
        chain, agent, state, job, signed = setup(home)
        with patch.object(chain, 'state', return_value=state), \
                patch.object(network.chain_network, 'broadcast', broadcast), \
                patch.object(network.chain_network, 'rpc', committed_rpc):
            out = agent.check(state, 'job', job)
    assert [o['kind'] for o in out] == ['prove']
    assert {a for a, _ in calls} - {RPCS[0]}  # validators that can judge were asked
    assert len(signed) == 1 and len({body for _, body in calls}) == 1  # one envelope, one nonce, everywhere


def test_when_no_validator_can_judge_yet_the_agent_retries_the_same_challenge_later():
    def broadcast(address, envelope):
        return {'code': 3, 'log': 'No verdict on this validator yet: growth recipe or proof bundle not held'}

    with tempfile.TemporaryDirectory() as home:
        chain, agent, state, job, signed = setup(home)
        with patch.object(chain, 'state', return_value=state), \
                patch.object(network.chain_network, 'broadcast', broadcast):
            first = agent.check(state, 'job', job)
            second = agent.check(state, 'job', job)
    assert [o['kind'] for o in first + second] == ['prove-retry', 'prove-retry']
    assert 'no validator admitted' in first[0]['error']
    assert all(e['body']['nonce'] == 4 for e in signed)  # retries reuse the account's committed nonce


def test_unreachable_validators_are_skipped_and_the_admitting_one_is_used_to_confirm():
    def broadcast(address, envelope):
        if address.endswith(('-0', '-1')):
            raise OSError('connection refused')
        return {'code': 3 if address.endswith('-2') else 0, 'log': 'No verdict' if address.endswith('-2') else ''}

    chain = network.Chain(RPCS, 'isolated-test')
    account = SimpleNamespace(public='x', sign=lambda chain_id, nonce, kind, **f: {
        'body': {'kind': kind, 'chain_id': chain_id, 'nonce': nonce, **f}, 'public_key': 'aa', 'signature': 's'})
    with patch.object(chain, 'nonce', return_value=0), patch.object(network.chain_network, 'broadcast', broadcast), \
            patch.object(network.chain_network, 'rpc', committed_rpc):
        envelope = chain.submit_everywhere(account, 'growth_prove', job_id='j', challenge_id='c')
    assert chain.rpc in RPCS[3:]
    key = base64.b64encode(hashlib.sha256(ledger.canonical(envelope)).digest()).decode()
    assert key  # the committed transaction is the one envelope that was sent


def test_a_committed_proof_that_failed_execution_is_reported_not_retried_silently():
    def broadcast(address, envelope):
        return {'code': 0, 'log': ''}

    def failed(address, method, params, timeout=60):
        return {'height': '9', 'tx_result': {'code': 1, 'log': 'Growth proof does not verify'}}

    chain = network.Chain(RPCS, 'isolated-test')
    account = SimpleNamespace(public='x', sign=lambda chain_id, nonce, kind, **f: {
        'body': {'kind': kind, 'chain_id': chain_id, 'nonce': nonce, **f}, 'public_key': 'aa', 'signature': 's'})
    with patch.object(chain, 'nonce', return_value=0), patch.object(network.chain_network, 'broadcast', broadcast), \
            patch.object(network.chain_network, 'rpc', failed):
        with pytest.raises(network.chain_network.Rejected, match='does not verify'):
            chain.submit_everywhere(account, 'growth_prove', job_id='j', challenge_id='c')
