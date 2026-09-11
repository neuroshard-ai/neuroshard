import base64
import copy

import pytest

from neuroshard.dataflow.store import canonical
from neuroshard.evolution import update_witness as witness
from neuroshard.evolution import settlement as ledger
from neuroshard.evolution.model import torch
from neuroshard.evolution.pipeline import Pipeline, LocalEndpoint
from neuroshard.evolution.worker import Worker
from neuroshard.evolution.verification import Metadata, bundle, validate_record, audit
from test_settlement import tx, claim, blocks, scenario


@pytest.mark.parametrize('length',[1, 15, 1023, 1024, 1025, 3073, 65537])
def test_merkle_openings_bind_coordinates_tails_and_canonical_padding(length):
    torch.manual_seed(9)
    value = torch.randn(length)
    descriptor = witness.commit(value)
    for index in {0, (length-1)//witness.CHUNK_ELEMENTS}:
        actual, opening = witness.open_chunk(value,index)
        assert actual == descriptor
        raw = witness._opening(descriptor,index,opening)
        expected = value[index*1024:(index+1)*1024].numpy().tobytes()
        assert raw == expected
        changed = copy.deepcopy(opening)
        corrupt = bytearray(base64.b64decode(changed['data']))
        corrupt[0] ^= 1
        changed['data'] = base64.b64encode(corrupt).decode()
        with pytest.raises(ValueError,match='signed stage commitment'):
            witness._opening(descriptor,index,changed)
        if length > 1024:
            with pytest.raises(ValueError):
                witness._opening(descriptor,1-index if index in (0,1) else 0,opening)


@pytest.fixture
def indexed(scenario,tmp_path):
    state,owners,store,old,artifacts = scenario
    parent = store.json(old['parent'])
    parent['update_witnesses'] = witness.FORMAT
    parent_root = store.put_json(parent)
    pipe = Pipeline(store,parent_root,[LocalEndpoint(Worker(tmp_path/f'indexed{i}',store)) for i in range(2)],
                    [6000]*2,'indexed')
    record = pipe.train(store.json(old['batch']))
    pipe.close()
    # A fresh profile in this fixture, before any work was accepted or paid.
    state['model_root'] = state['serving_root'] = parent_root
    state['manifest']['initial_model_root'] = parent_root
    return state,owners,store,record,artifacts


def corrupt_update(store,record):
    forged = store.json(record['record_root'])
    trace = store.json(forged['traces'][0])
    index = next(i for i,item in enumerate(trace['updates']['tensors'])
                 if item['component']=='embed' and item['tensor']=='weight')
    entry = trace['updates']['tensors'][index]
    changed = store.tensors(trace['components']['embed']['root'])
    changed['weight'].view(-1)[0] += 1
    trace['components']['embed']['root'] = store.put_tensors(changed)
    entry['after'] = witness.commit(changed['weight'])
    forged['traces'][0] = store.put_json(trace)
    model = store.json(forged['model_root'])
    model['components'].update(trace['components'])
    forged['model_root'] = store.put_json(model)
    forged['record_root'] = store.put_json(forged)
    return forged,index


def test_real_sgd_witness_matches_full_stage_and_false_accusation_does_not_extend_deadline(indexed):
    state,owners,store,record,artifacts = indexed
    metadata = Metadata(bundle(store,record['record_root']))
    assert validate_record(metadata,record['record_root'])['valid']
    assert audit(store,metadata,record['record_root'],0)['valid']
    proof = witness.prepare(store,record['traces'][0],0,0)
    assert witness.check(metadata,record['record_root'],0,0,proof)['valid']
    state = ledger.transition(state,claim(state,owners,store,record))
    deadline = state['candidate']['deadline']
    burned = state['burned']
    state = ledger.transition(state,tx(state,owners[2],'refute_update',claim_id=state['candidate']['id'],
                              stage=0,tensor_index=0,witness=proof))
    assert state['candidate']['deadline']==deadline and state['issued']==0
    assert state['update_check_count']==1 and state['audit_count']==0
    assert state['burned']-burned==ledger.PARAMS['challenge_bond']+ledger.PARAMS['fee']
    state = blocks(state,ledger.PARAMS['challenge_blocks']+1)
    assert state['issued']==1_000_000 and state['model_root']==record['model_root']


def test_forged_update_refuted_without_artifact_reads_or_full_referee(indexed):
    state,owners,store,record,_ = indexed
    forged,index = corrupt_update(store,record)
    metadata = Metadata(bundle(store,forged['record_root']))
    assert validate_record(metadata,forged['record_root'])['valid']
    assert not audit(store,metadata,forged['record_root'],0)['valid']
    proof = witness.prepare(store,forged['traces'][0],index,0)
    assert len(canonical(proof)) <= witness.MAX_WITNESS_BYTES
    assert not witness.check(metadata,forged['record_root'],0,index,proof)['valid']
    state = ledger.transition(state,claim(state,owners,store,forged))
    before = state['accounts'][owners[2].public_key]['balance']
    def forbidden(*_):raise AssertionError('Compact proof must not call the neural referee')
    state = ledger.transition(state,tx(state,owners[2],'refute_update',claim_id=state['candidate']['id'],
                              stage=0,tensor_index=index,witness=proof),None,False,forbidden)
    assert state['candidate'] is None and state['issued']==0 and state['training_round']==0
    assert state['model_root']==record['parent']
    assert state['accounts'][owners[2].public_key]['balance']-before==ledger.PARAMS['claim_bond']//2-ledger.PARAMS['fee']
    assert 'inconsistent committed SGD chunk' in state['settled'][-1]['reason']
    ledger.invariant(state)


def test_consistent_invented_gradient_cannot_replace_full_training_audit(indexed):
    _,_,store,record,_ = indexed
    forged = store.json(record['record_root'])
    trace = store.json(forged['traces'][0])
    index = next(i for i,item in enumerate(trace['updates']['tensors']) if item['component']=='embed')
    parent = store.json(record['parent'])
    before = store.tensors(parent['components']['embed']['root'])['weight']
    gradient = torch.zeros_like(before)
    trace['components']['embed'] = parent['components']['embed']
    entry = trace['updates']['tensors'][index]
    proof = {'chunk':0}
    for field,value in (('before',before),('gradient',gradient),('after',before)):
        entry[field],proof[field] = witness.open_chunk(value,0)
    forged['traces'][0] = store.put_json(trace)
    model = store.json(record['model_root'])
    model['components']['embed'] = parent['components']['embed']
    forged['model_root'] = store.put_json(model)
    root = store.put_json(forged)
    metadata = Metadata(bundle(store,root))
    assert validate_record(metadata,root)['valid']
    assert witness.check(metadata,root,0,index,proof)['valid']
    assert not audit(store,metadata,root,0)['valid']
    with pytest.raises(ValueError,match='actual stage'):
        witness.prepare(store,forged['traces'][0],index,0)


def test_noninteger_shape_spelling_is_rejected_before_consensus_arithmetic(indexed):
    _,_,store,record,_ = indexed
    value = store.json(record['record_root'])
    trace = store.json(value['traces'][0])
    shape = trace['updates']['tensors'][0]['before']['shape']
    shape[0] = float(shape[0])
    value['traces'][0] = store.put_json(trace)
    root = store.put_json(value)
    with pytest.raises(ValueError,match='shape differs'):
        validate_record(Metadata(bundle(store,root)),root)


def test_fake_paths_wrong_tensor_and_oversized_witness_cannot_slash(indexed):
    state,owners,store,record,_ = indexed
    proof = witness.prepare(store,record['traces'][0],0,0)
    state = ledger.transition(state,claim(state,owners,store,record))
    initial = copy.deepcopy(state)
    for bad,index in (({**proof,'chunk':99999},0),
                      ({**proof,'before':{**proof['before'],'data':'A'*40000}},0),
                      (proof,1)):
        with pytest.raises(ValueError):
            ledger.transition(state,tx(state,owners[2],'refute_update',claim_id=state['candidate']['id'],
                              stage=0,tensor_index=index,witness=bad))
        assert state==initial


def test_false_gradient_or_tensor_index_still_needs_full_audit(indexed):
    _,_,store,record,_ = indexed
    altered = store.json(record['record_root'])
    trace = store.json(altered['traces'][0])
    trace['updates']['tensors'][0]['gradient']['root'] = 'e'*64
    altered['traces'][0] = store.put_json(trace)
    root = store.put_json(altered)
    metadata = Metadata(bundle(store,root))
    # These declarations are optimistic assertions, not an authenticated
    # decomposition of a SHA-256 safetensors file. Full replay checks the link.
    assert validate_record(metadata,root)['valid']
    result = audit(store,metadata,root,0)
    assert not result['valid'] and result['mismatch']=='optimizer tensor commitments'
    with pytest.raises(ValueError,match='actual stage'):
        witness.prepare(store,altered['traces'][0],0,0)


def test_legacy_profile_cannot_silently_enable_compact_refutation(scenario):
    state,owners,store,record,_ = scenario
    state = ledger.transition(state,claim(state,owners,store,record))
    with pytest.raises(ValueError,match='no compact update'):
        ledger.transition(state,tx(state,owners[2],'refute_update',claim_id=state['candidate']['id'],
                          stage=0,tensor_index=0,witness={'chunk':0,'before':{},'gradient':{},'after':{}}))
