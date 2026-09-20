"""Controller payments and negative/partial reports against native transitions.

Numerical truth is deliberately a fixture here; the committed native preflight
separately executes actual model shards and three full numerical replays.
"""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import operated_alpha_auditor as controller
from neuroshard.evolution import auditing, expert_lifecycle, settlement
from neuroshard.evolution.reference_data import identity
from neuroshard.evolution.transactions import Outbox
from test_service_admission import capacity, market, graphs, network, admit, ready, response
from test_hosted_customer import Chain
from test_settlement import blocks


class Native(Chain):
    def query(self, path='/status', data=None):
        if path == '/status':
            return {'chain_id': self.chain_id, 'height': self.state['height'], 'issued': self.state['issued']}
        if path == '/candidate':
            return copy.deepcopy(self.state['candidate'])
        if path == '/service_admission':
            return copy.deepcopy(self.state['service_admission'])
        return super().query(path, data)


@pytest.mark.parametrize('verdict', [True, False, 'incomplete'])
def test_complete_reports_settle_once_but_partial_coverage_never_signs(tmp_path, capacity, monkeypatch, verdict):
    state, owners, offers = capacity
    state, key = admit(state, owners, offers)
    state = response(ready(state, owners, key), owners, key)
    native = Native(state)
    native.owners = owners
    native.genesis = {'chain_id': native.chain_id}
    boxes = [Outbox(tmp_path/f'outbox-{i}.sqlite', 'local', native.chain_id, owner,
        rpc=native.rpc, query=lambda _url, path, data: native.query(path, data)) for i, owner in enumerate(owners[:3])]
    native.outboxes = boxes
    claim = copy.deepcopy(state['candidate'])
    calls = []
    def replay(_service, request, **_options):
        calls.append(request['id'])
        stages = [] if verdict == 'incomplete' else [{'stage': i, 'valid': verdict}
                                                   for i in range(auditing.stages(claim))]
        return {'status': 'completed', 'report': {'format': expert_lifecycle.FORMAT+'/replay',
            'statement': identity(expert_lifecycle.service_statement(claim)), 'stages': stages}}
    cloud = SimpleNamespace(query=replay)
    monkeypatch.setattr(controller, 'committed', lambda *_args: copy.deepcopy(native.state))
    try:
        if verdict == 'incomplete':
            with pytest.raises(ValueError, match='entire expert service obligation'):
                controller.tick(tmp_path, cloud, native, {})
            assert not native.submissions and all(box.pending() is None for box in boxes)
            return
        # The first audit commits before its RPC response is lost. Resumption
        # must recover that exact transaction without buying another nonce.
        native.drop = 'audit_commit'
        original = boxes[0].send
        def short(*args, **kwargs):
            kwargs['timeout'] = .01
            return original(*args, **kwargs)
        boxes[0].send = short
        with pytest.raises(TimeoutError):
            controller.tick(tmp_path, cloud, native, {})
        assert controller.tick(tmp_path, cloud, native, {})['phase'] == 'reconciled'
        boxes[0].send = original
        native.drop = None
        for _ in range(5):
            native.state = blocks(native.state, 1)
            outcome = controller.tick(tmp_path, cloud, native, {})
            if outcome['phase'] == 'full_replays_reported':
                break
        assert outcome['phase'] == 'full_replays_reported'
        assert outcome['valid'] is verdict
        assert len(calls) == len(set(calls)) == 3
        assert len(native.submissions) == 6
        native.state = blocks(native.state, native.state['candidate']['deadline']-native.state['height']+1)
        result = native.state['expert_lifecycle']['results'][key]
        assert result['status'] == ('completed' if verdict else 'verification_failed')
        assert native.state['issued'] == 0 and not native.state['auditing']['budgets']
        settlement.invariant(native.state)
    finally:
        for box in boxes:
            box.close()
