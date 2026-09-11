import random

import pytest
from neuroshard.evolution.data import Corpus
from neuroshard.evolution.evaluation import decide


class Tokenizer:
    def apply_chat_template(self,messages,**_):
        import hashlib
        raw=''.join(m['content'] for m in messages)
        return [int(c,16) for c in hashlib.sha256(raw.encode()).hexdigest()]


def source(role='train'):
    return {'repo':'test/corpus','revision':'a'*40,'split':'train','license':'Apache-2.0','role':role}


def example(i):
    random.seed(i)
    return {'messages':[{'role':'user','content':' '.join(str(random.randrange(1000000)) for _ in range(80))}]}


def test_new_windows_resume_and_exclude_evaluation_documents(seed,tmp_path):
    store,_,_=seed
    corpus=Corpus(tmp_path/'data',store,Tokenizer(),32)
    heldout=corpus.register(source('heldout'))
    reserved=corpus.collect(heldout,64,rows=[example(i) for i in range(64)])
    train=corpus.register(source())
    rejected=corpus.collect(train,64,rows=[example(i) for i in range(64)])
    assert not rejected['sequences'] and rejected['rejected']['duplicate']==64
    first=corpus.collect(train,64,rows=[example(i) for i in range(64,128)])
    assert first['start']==64 and first['end']==128 and first['sequences']
    corpus.db.close()
    resumed=Corpus(tmp_path/'data',store,Tokenizer(),32)
    second=resumed.collect(train,64,rows=[example(i) for i in range(128,192)])
    assert second['start']==128 and second['end']==192
    batches=resumed.training(second['root'],32,'epoch-2',replay_fraction=.25)
    assert sum(k in first['sequences'] for k in batches)==8
    assert set(batches).isdisjoint(reserved['sequences'])


def test_evaluation_examples_are_consumed_once_and_candidates_cannot_train_on_them(seed,tmp_path):
    store,root,_=seed
    corpus=Corpus(tmp_path/'data',store,Tokenizer(),32)
    key=corpus.register(source('fresh'))
    window=corpus.collect(key,64,rows=[example(i) for i in range(64)])
    first=store.json(corpus.reserve_evaluation(root,'fresh',32,'after-commit-block-A'))
    second=store.json(corpus.reserve_evaluation(root,'fresh',32,'after-commit-block-B'))
    assert set(first['sequences']).isdisjoint(second['sequences'])
    with pytest.raises(ValueError,match='Insufficient'):
        corpus.reserve_evaluation(root,'fresh',32,'C')
    with pytest.raises(ValueError,match='training window'):
        corpus.training(window['root'],32,'bad')


def test_gate_rejects_regression_and_noise_and_accepts_measured_gain():
    base=[3.+i/100 for i in range(64)]
    better=[v-.03 for v in base]
    assert decide(base,better,base,better)['promote']
    assert not decide(base,[v+.04 for v in base],base,better)['promote']
    noisy=[v+(.1 if i%2 else -.101) for i,v in enumerate(base)]
    assert not decide(base,base,base,noisy)['promote']
