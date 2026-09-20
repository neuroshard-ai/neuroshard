"""Provider startup waits for its own funding receipt, never a remote balance."""
import base64
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from operated_alpha_bootstrap import wait_funding
from neuroshard.client import wire


@pytest.mark.parametrize('mode', ['delayed', 'wrong-chain', 'wrong-manifest', 'wrong-receipt', 'failed-funding'])
def test_local_provider_funding_barrier(mode):
    manifest = {'test': 'bootstrap'}
    config = {'node_rpc': 'http://127.0.0.1:26657', 'chain_id': 'test-alpha',
              'manifest_root': wire.digest(manifest)}
    receipt = {'hash': 'AB'*32, 'height': '5', 'tx_result': {'code': 0}}
    clock = [0.]
    calls = []

    def rpc(_url, method, params=None, **_kwargs):
        calls.append(method)
        if mode == 'delayed' and clock[0] < .5:
            raise OSError('observer not listening yet')
        if method == 'abci_query':
            value = {} if mode == 'wrong-manifest' else manifest
            return {'response': {'value': base64.b64encode(wire.canonical(value)).decode()}}
        if method == 'status':
            return {'node_info': {'network': 'wrong' if mode == 'wrong-chain' else 'test-alpha'},
                    'sync_info': {'catching_up': mode == 'delayed' and clock[0] < 1.}}
        assert method == 'tx'
        assert base64.b64decode(params['hash']).hex().upper() == receipt['hash']
        if clock[0] < 1.5 and mode == 'delayed':
            raise wire.Rejected('tx not found')
        return {**receipt, 'height': '4' if mode == 'wrong-receipt' else '5'}

    def sleep(seconds):
        clock[0] += seconds

    if mode == 'failed-funding':
        receipt['tx_result']['code'] = 1
    options = dict(rpc=rpc, clock=lambda: clock[0], sleep=sleep, seconds=3)
    if mode == 'delayed':
        result = wait_funding(config, receipt, **options)
        assert result['height'] == 5 and clock[0] == 1.5
    else:
        error = {'wrong-receipt': RuntimeError, 'failed-funding': ValueError}.get(mode, TimeoutError)
        with pytest.raises(error):
            wait_funding(config, receipt, **options)
        if mode in ('wrong-chain', 'wrong-manifest', 'failed-funding'):
            assert 'tx' not in calls
