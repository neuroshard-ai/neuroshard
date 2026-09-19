"""Exercise the deployment controller without allocating cloud resources.

Numerical reports below are explicit oracle fixtures. This checks real signed
native transitions and cross-thread durable journals, not model correctness.
"""
from concurrent.futures import ThreadPoolExecutor
import copy
import importlib.util
from pathlib import Path
import sys

import pytest

from neuroshard.evolution import expert_lifecycle, settlement
from neuroshard.evolution.reference_data import identity
from neuroshard.evolution.transactions import Outbox
from test_expert_lifecycle import network
from test_hosted_customer import Chain
from test_provider_hosting import graphs, market, lease, ready, response
from test_settlement import blocks


def driver():
    scripts = Path(__file__).resolve().parents[2]/'scripts'
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location('provider_llm_driver', scripts/'probe_provider_llm.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def test_customer_keys_are_created_before_allocation_and_never_rotated(tmp_path):
    script = driver()
    keys = script.customer_wallets(tmp_path)
    before = [key.path.read_bytes() for key in keys]
    assert len({key.public_key for key in keys}) == 2
    assert all(key.path.stat().st_mode & 0o777 == 0o600 for key in keys)
    assert [key.path.read_bytes() for key in script.customer_wallets(tmp_path)] == before


class Native:
    def __init__(self, home, state, owners):
        self.home, self.chain, self.owners = home, Chain(state), owners
        self.urls, self.genesis = ['local'], {'chain_id': state['chain_id']}
        self.outboxes = [Outbox(home/f'outbox-{i}.sqlite', 'local', state['chain_id'], owner,
            rpc=self.chain.rpc, query=lambda _url, path, data: self.chain.query(path, data))
            for i, owner in enumerate(owners)]

    def query(self, path='/status'):
        if path == '/candidate':
            return copy.deepcopy(self.chain.state['candidate'])
        self.chain.state = blocks(self.chain.state, 1)
        return {'height': self.chain.state['height']}

    def settled(self, claim_id, **kwargs):
        candidate = self.chain.state['candidate']
        self.chain.state = blocks(self.chain.state, candidate['deadline']-self.chain.state['height']+1)
        settlement.invariant(self.chain.state)
        row = next(row for row in self.chain.state['settled'] if row['id'] == claim_id)
        return {'settlement': row, 'status': {'issued': self.chain.state['issued']}}


class Oracle:
    def __init__(self, *, invalid=False):
        self.calls, self.invalid = [], invalid

    def query(self, service, request, **kwargs):
        self.calls.append(request['id'])
        claim = request['claim']
        report = {'format': expert_lifecycle.FORMAT+'/replay',
            'statement': identity(expert_lifecycle.service_statement(claim)),
            'stages': [{'stage': i, 'valid': not self.invalid} for i in range(claim['stages'])]}
        return {'status': 'completed', 'report': report}


@pytest.mark.parametrize('invalid', [False, True])
def test_audit_thread_uses_its_own_connection_to_the_same_durable_journal(tmp_path, market, invalid):
    state, owners, offers = market
    state, key = lease(state, owners, offers)
    state = response(ready(state, owners, key), owners, key)
    claim = copy.deepcopy(state['candidate'])
    native, oracle = Native(tmp_path, state, owners), Oracle(invalid=invalid)
    script = driver()
    try:
        with ThreadPoolExecutor(1) as worker:
            result = worker.submit(script.settle, oracle, native, {}, claim, tmp_path/'case')
            if invalid:
                with pytest.raises(ValueError, match='complete numerical replay'):
                    result.result(timeout=20)
                assert not native.chain.submissions
                assert not native.chain.state['settled']
            else:
                assert len(result.result(timeout=20)) == 3
                assert len(set(oracle.calls)) == 3
                assert len(native.chain.submissions) == 6
                assert native.chain.state['expert_lifecycle']['results'][key]['status'] == 'completed'
                assert native.chain.state['issued'] == 0
                # Original connections remain usable in their main thread and
                # observe the same audit operations; no parallel second journal.
                for box in native.outboxes[:3]:
                    assert box.recorded(claim['id']+'/commit')
                    assert box.recorded(claim['id']+'/reveal')
                    assert box.pending() is None
    finally:
        for box in native.outboxes:
            box.close()
