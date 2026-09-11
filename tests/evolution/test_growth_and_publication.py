import copy

import pytest

from neuroshard.evolution.model import grow,torch
from neuroshard.evolution.verification import Metadata,validate_growth,audit_growth
from neuroshard.evolution.data import Corpus,publish_window
from neuroshard.dataflow.store import LocalStore
from test_data_and_evaluation import Tokenizer,source,example


def test_growth_referee_rejects_plausible_but_nonidentity_new_weights(seed):
    store,parent,model = seed
    candidate,value = grow(parent,store,2)
    metadata = Metadata({parent:model,candidate:value})
    assert audit_growth(store,metadata,parent,candidate)['valid']
    forged = copy.deepcopy(value)
    component = copy.deepcopy(forged['components']['block_004'])
    tensors = store.tensors(component['root'])
    tensors['mlp.down_proj.weight'] = torch.ones_like(tensors['mlp.down_proj.weight'])
    component['root'] = store.put_tensors(tensors)
    for name in ('block_004','block_005'):
        forged['components'][name] = component
    key = store.put_json(forged)
    metadata = Metadata({parent:model,key:forged})
    assert validate_growth(metadata,parent,key)['valid']
    assert not audit_growth(store,metadata,parent,key)['valid']


def test_training_replica_contains_provenance_but_refuses_protected_data(seed,tmp_path):
    store,_,_ = seed
    corpus = Corpus(tmp_path/'corpus',store,Tokenizer(),32)
    train = corpus.register(source())
    window = corpus.collect(train,8,rows=[example(i) for i in range(8)])
    replica = LocalStore(tmp_path/'replica')
    result = publish_window(corpus,window['root'],replica)
    # Eight documents, two windows per document, manifest and source.
    assert result['read_back_verified'] and result['objects']==26
    for key in window['sequences']:
        assert replica.get(key)==store.get(key)
    assert replica.get(train)==store.get(train)
    protected = corpus.register(source('heldout'))
    window = corpus.collect(protected,8,rows=[example(i) for i in range(8,16)])
    with pytest.raises(ValueError,match='training windows only'):
        publish_window(corpus,window['root'],replica)
