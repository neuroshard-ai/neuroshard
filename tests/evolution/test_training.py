import copy
import math

import pytest

from neuroshard.evolution.model import Shard, grow, place, torch
from neuroshard.evolution.pipeline import LocalEndpoint, Pipeline, validate_record
from neuroshard.evolution.worker import Worker, replay_trace

BATCH = [[1, 12, 9, 18, 6, 2], [1, 13, 7, 17, 5, 2]]


def endpoints(tmp_path, store, count):
    return [LocalEndpoint(Worker(tmp_path/f'worker{i}', store, 48000000)) for i in range(count)]


def test_partitioned_update_matches_unpartitioned_autograd_and_all_components_change(seed, tmp_path):
    store, root, model = seed
    cap = 6000
    remote = endpoints(tmp_path, store, len(place(model, [cap]*3)))
    pipe = Pipeline(store, root, remote, [cap]*3, 'training')
    result = pipe.train(BATCH)
    assert validate_record(store, result['record_root'])['valid']
    for trace in result['traces']:
        assert replay_trace(store, trace)['valid']
    oracle = Shard(model, place(model, [48000000])[0], store)
    ids = torch.tensor(BATCH)
    loss = oracle.loss(oracle(ids, ids=True), ids)
    loss.backward()
    scale = min(1., 1. / (math.sqrt(oracle.gradient_squared_norm()) + 1e-6))
    oracle.update(.003, scale)
    expected = oracle.save(store)
    actual = store.json(result['model_root'])['components']
    # Exact gradient accumulation can differ across graph cuts at the tied
    # embedding. Numerical agreement with a central oracle is tested separately
    # from exact replay of the *specified* partitioned execution.
    for name, component in actual.items():
        assert component['root'] != model['components'][name]['root']
        for key, tensor in store.tensors(component['root']).items():
            torch.testing.assert_close(tensor, store.tensors(expected[name]['root'])[key], rtol=1e-6, atol=1e-8)
    assert float.fromhex(result['loss_hex']) == float(loss.detach())
    pipe.close()


def test_growth_adds_trainable_capacity_preserves_logits_and_learns(seed, tmp_path):
    store, root, model = seed
    old = Shard(model, place(model,[48000000])[0], store)
    grown_root, grown = grow(root, store, 2)
    new = Shard(grown, place(grown,[48000000])[0], store)
    ids = torch.tensor(BATCH)
    assert torch.equal(old.logits(old(ids,ids=True)), new.logits(new(ids,ids=True)))
    assert grown['parameters'] > model['parameters']
    with pytest.raises(ValueError, match='Insufficient'):
        place(grown, [6000,6000])
    assert len(place(grown, [6000]*3)) == 3
    pipe = Pipeline(store, grown_root, endpoints(tmp_path,store,3),[6000]*3,'growth')
    trained = store.json(pipe.train(BATCH)['model_root'])
    for name in ('block_004','block_005'):
        tensors = store.tensors(trained['components'][name]['root'])
        assert torch.count_nonzero(tensors['self_attn.o_proj.weight']) > 0
        assert torch.count_nonzero(tensors['mlp.down_proj.weight']) > 0
    pipe.close()


def test_corrupt_gradient_is_detected_by_bounded_referee(seed, tmp_path):
    store, root, model = seed
    pipe = Pipeline(store, root, endpoints(tmp_path,store,2),[6000]*2,'fraud')
    record = pipe.train(BATCH)
    original = store.json(record['traces'][1])
    corrupt = copy.deepcopy(original)
    tensor = store.tensors(corrupt['gradient_out'])['value']
    corrupt['gradient_out'] = store.put_tensors({'value':tensor*2})
    # The output is valid but recomputed gradients and updates no longer match.
    assert replay_trace(store,store.put_json(corrupt))['mismatch'] == 'backward gradient'
    changed = store.json(record['record_root'])
    changed['traces'][1] = store.put_json(corrupt)
    with pytest.raises(ValueError,match='Backward gradient substitution'):
        validate_record(store,store.put_json(changed))
    pipe.close()


def test_worker_and_coordinator_restart_during_step(seed, tmp_path):
    store, root, model = seed
    workers = endpoints(tmp_path,store,2)
    journal = tmp_path/'coordinator.json'
    pipe = Pipeline(store, root, workers,[6000]*2,'recover',journal=journal)
    original = workers[1].operation
    failed = False
    def interrupted(session, operation):
        nonlocal failed
        result = original(session,operation)
        if operation['phase'] == 'backward' and not failed:
            failed = True
            raise ConnectionError('Lost acknowledgment after durable backward')
        return result
    workers[1].operation = interrupted
    with pytest.raises(ConnectionError):
        pipe.train(BATCH)
    pipe.close()
    for endpoint in workers:
        endpoint.worker.db.close()
    recovered = Pipeline(store,root,endpoints(tmp_path,store,2),[6000]*2,'recover',journal=journal)
    with pytest.raises(ValueError,match='pending batch'):
        recovered.train([[1,2,3]])
    result = recovered.train(BATCH)
    assert validate_record(store,result['record_root'])['valid']
    second = recovered.train(BATCH)
    assert second['parent'] == result['model_root'] and second['step'] == 1
    recovered.close()


def test_declared_small_shard_cannot_load_large_or_wrong_shape_tensor(seed):
    store, root, model = seed
    assignment = place(model,[6000]*2)[0]
    corrupt = copy.deepcopy(assignment)
    corrupt['parameters'] = 1
    with pytest.raises(ValueError,match='capacity'):
        Shard(model,corrupt,store)
    corrupt_model = copy.deepcopy(model)
    corrupt_model['components']['embed']['root'] = store.put_tensors({'weight':torch.zeros(100,100)})
    with pytest.raises(ValueError,match='shape or dtype'):
        Shard(corrupt_model,assignment,store)


def test_native_entry_sets_cpu_profile_before_legacy_genesis_imports():
    import os,subprocess,sys
    environment=dict(os.environ)
    environment.pop('ATEN_CPU_CAPABILITY',None)
    environment.pop('MKL_ENABLE_INSTRUCTIONS',None)
    environment['PYTHONPATH']='src'
    code='from neuroshard.evolution import app; from neuroshard.demo.work import make_model; make_model(); import torch; assert torch.backends.cpu.get_cpu_capability()=="DEFAULT"; assert torch.get_num_threads()==1; assert not torch.backends.mkldnn.enabled'
    subprocess.run([sys.executable,'-c',code],env=environment,check=True,capture_output=True)


def test_layers_remain_in_numeric_order_after_layer_999(seed):
    store,root,model=seed
    model=copy.deepcopy(model)
    component=model['components']['block_000']
    for i in range(4,1002):model['components'][f'block_{i:03}']=component
    model['config']['num_hidden_layers']=1002
    model['parameters']=sum(c['parameters'] for c in model['components'].values())
    assigned=place(model,[48000000])
    assert assigned[0]['components'][-4:]==['block_998','block_999','block_1000','block_1001']


def test_assistant_only_targets_match_central_autograd_and_replay(seed,tmp_path):
    store,root,model=seed
    labels=[[-100,-100,-100,*row[3:]] for row in BATCH]
    pipe=Pipeline(store,root,endpoints(tmp_path,store,2),[6000]*2,'masked')
    masked={'input_ids':BATCH,'labels':labels}
    before=pipe.evaluate(masked)
    result=pipe.train(masked)
    oracle=Shard(model,place(model,[48000000])[0],store)
    ids=torch.tensor(BATCH)
    loss=oracle.loss(oracle(ids,ids=True),ids,torch.tensor(labels))
    assert float(loss.detach())==float.fromhex(before['loss_hex'])
    assert float(loss.detach())==float.fromhex(result['loss_hex'])
    loss.backward()
    scale=min(1.,1./(math.sqrt(oracle.gradient_squared_norm())+1e-6))
    oracle.update(.003,scale)
    expected=oracle.save(store)
    for name,component in store.json(result['model_root'])['components'].items():
        for key,value in store.tensors(component['root']).items():
            torch.testing.assert_close(value,store.tensors(expected[name]['root'])[key],atol=1e-8,rtol=1e-6)
    for trace in result['traces']:assert replay_trace(store,trace)['valid']
    pipe.close()


def test_labels_cannot_invent_targets_or_request_unbounded_logits():
    from neuroshard.evolution.batches import unpack
    with pytest.raises(ValueError,match='Targets'):
        unpack({'input_ids':[[1,2,3]],'labels':[[-100,4,3]]},64)
    with pytest.raises(ValueError,match='Targets'):
        unpack({'input_ids':[[1,2,3]],'labels':[[-100,-100,-100]]},64)
    with pytest.raises(ValueError,match='bounds'):
        unpack([[1]*256]*4,64)
