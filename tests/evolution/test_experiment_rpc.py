import base64
import hashlib
import importlib.util
from http.client import RemoteDisconnected
from pathlib import Path

import pytest

from neuroshard.dataflow.store import canonical
from neuroshard.demo.client import Rejected


spec=importlib.util.spec_from_file_location('native_rpc',Path(__file__).resolve().parents[2]/'scripts/native_rpc.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
broadcast_finalized=module.broadcast_finalized
ENVELOPE={'signed':'immutable fixture'}
HASH=hashlib.sha256(canonical(ENVELOPE)).hexdigest().upper()


def receipt(code=0):
    return {'hash':HASH,'height':'42','tx_result':{'code':code,'log':'final rejection'}}


class Clock:
    def __init__(self):self.now=0.
    def __call__(self):return self.now
    def sleep(self,seconds):self.now+=seconds


def test_lost_submission_response_is_confirmed_without_resigning_or_resubmitting():
    calls=[]
    def rpc(url,method,params,timeout):
        calls.append((method,params))
        if method=='broadcast_tx_sync':raise RemoteDisconnected('lost acknowledgment')
        if len(calls)==2:raise Rejected('tx not found')
        return receipt()
    clock=Clock()
    assert broadcast_finalized('node',ENVELOPE,rpc=rpc,clock=clock,sleep=clock.sleep)['height']=='42'
    assert [method for method,_ in calls]==['broadcast_tx_sync','tx','tx']
    assert base64.b64decode(calls[-1][1]['hash']).hex().upper()==HASH


@pytest.mark.parametrize('mode',['check_rejection','final_rejection','wrong_hash'])
def test_submission_does_not_confuse_checktx_or_unrelated_results_with_final_acceptance(mode):
    def rpc(url,method,params,timeout):
        if method=='broadcast_tx_sync':return {'code':1 if mode=='check_rejection' else 0}
        return {**receipt(1 if mode=='final_rejection' else 0),'hash':'0'*64 if mode=='wrong_hash' else HASH}
    with pytest.raises(Rejected):
        broadcast_finalized('node',ENVELOPE,rpc=rpc)


def test_unconfirmed_transaction_reports_unknown_outcome_with_original_hash():
    def rpc(url,method,params,timeout):
        if method=='broadcast_tx_sync':return {'code':0}
        raise Rejected('tx not found')
    clock=Clock()
    with pytest.raises(TimeoutError,match=f'outcome unknown.*{HASH}'):
        broadcast_finalized('node',ENVELOPE,timeout=1,rpc=rpc,clock=clock,sleep=clock.sleep)
