import base64,hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from neuroshard.demo import work,protocol,client
from neuroshard.inference import settlement
from neuroshard.inference.pool import Worker,authorized_request
from neuroshard.publicnet.pool import signed


def test_unknown_broadcast_is_recovered_by_exact_hash_without_resubmission(monkeypatch):
    envelope=protocol.Identity('test-settlement').sign({'nonce':0,'kind':'test'})
    expected=base64.b64encode(hashlib.sha256(work.canonical(envelope)).digest()).decode()
    calls=[]
    def rpc(url,method,params,**kwargs):
        calls.append(method)
        if method=='broadcast_tx_sync':raise OSError('connection closed after acceptance')
        assert params=={'hash':expected,'prove':False}
        if calls.count('tx')==1:raise client.Rejected('not indexed yet')
        return {'height':'42','tx_result':{'code':0}}
    monkeypatch.setattr(settlement.wire,'rpc',rpc)
    monkeypatch.setattr(settlement.time,'sleep',lambda _:None)
    assert settlement.submit('http://127.0.0.1',envelope)['height']=='42'
    assert calls==['broadcast_tx_sync','tx','tx']


def test_worker_rejects_changed_batch_and_reuses_durable_receipt(tmp_path,monkeypatch):
    worker=Worker.__new__(Worker);worker.home=tmp_path;worker.stage_index=0
    worker.identity=protocol.Identity('worker');sponsor=protocol.Identity('sponsor')
    worker.config={'chain_id':'transport-test'};worker.rpc='local';worker.path=tmp_path/'worker.json'
    worker.journal={'task_id':None,'operations':{}}
    worker.engine=SimpleNamespace(features=lambda ids:{'fixture':ids})
    lease={'task_kind':'train','owner':sponsor.public_key,'workers':[worker.identity.public_key,worker.identity.public_key],
           'expires':30,'task_id':'task-1'}
    task={'chain_id':'transport-test','lease':lease,'input_ids':[1,2],'weights':{'test':0},'model_root':'a'*64}
    summary={'chain_id':'transport-test','height':20,'lease':lease}
    monkeypatch.setattr('neuroshard.inference.pool.wire.query',lambda rpc,path:task if path=='/task' else summary)
    request={'task_id':'task-1','input_ids':[1,2],'weights':{'test':0},'operation':'features'}
    envelope=signed(sponsor,'transport-test','assignment',worker=worker.identity.public_key,stage=0,request=request)
    first=worker.compute(envelope)
    assert worker.path.is_file()
    worker.engine=SimpleNamespace(features=lambda _:pytest.fail('A repeated delivery must reuse its receipt'))
    assert worker.compute(envelope)==first
    request['input_ids']=[9,9]
    with pytest.raises(ValueError,match='differs'):
        worker.compute(signed(sponsor,'transport-test','assignment',worker=worker.identity.public_key,stage=0,request=request))
