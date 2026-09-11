import copy

import pytest

from neuroshard.evolution.controller import Epochs


class Corpus:
    def __init__(self,store):
        self.store=store
        self.collections=0
        self.sequence=store.put_json({'tokens':list(range(32)),'role':'train'})
        self.selection_calls=[]
    def collect(self,source,count):
        self.collections+=1
        value={'sequences':[self.sequence],'start':self.collections-1,'end':self.collections}
        return {'root':self.store.put_json(value),**value}
    def training(self,window,count,seed):
        return [self.sequence]*count
    def reserve_evaluation(self,candidate,role,count,beacon):
        self.selection_calls.append((candidate,role,beacon))
        return self.store.put_json({'sequences':[self.sequence]*count})


class Factory:
    def __init__(self,store,initial,regress=False,interrupt=False):
        self.store,self.initial,self.regress,self.interrupt=store,initial,regress,interrupt
        self.training_calls=0
    def __call__(self,root,name,journal):
        factory=self
        class Pipe:
            def __init__(self):self.model_root=root;self.step=0
            def train(self,batch):
                factory.training_calls+=1
                model=copy.deepcopy(factory.store.json(root))
                model['test_revision']=model.get('test_revision',0)+1
                self.model_root=factory.store.put_json(model)
                self.step+=1
                return {'model_root':self.model_root}
            def evaluate(self,batch):
                if factory.interrupt and name.endswith('candidate'):
                    factory.interrupt=False
                    raise ConnectionError('Interrupted during candidate evaluation')
                version=factory.store.json(self.model_root).get('test_revision',0)
                loss=3.+(.1 if factory.regress else -.1)*version
                return {'loss_hex':float(loss).hex()}
            def close(self):pass
        return Pipe()


def test_rejected_revision_never_replaces_accepted_model(seed,tmp_path):
    store,root,_=seed
    corpus=Corpus(store)
    factory=Factory(store,root,regress=True)
    epochs=Epochs(tmp_path/'epochs',store,corpus,factory,root,'train','heldout',steps=1,examples=32)
    result=epochs.run_once()
    assert result['status']=='rejected' and epochs.accepted_root==root
    assert factory.training_calls==1


def test_restart_keeps_committed_candidate_and_evaluation_selection(seed,tmp_path):
    store,root,_=seed
    corpus=Corpus(store)
    factory=Factory(store,root,interrupt=True)
    epochs=Epochs(tmp_path/'epochs',store,corpus,factory,root,'train','heldout',steps=1,examples=32,epochs_per_day=1,clock=lambda:1.)
    with pytest.raises(ConnectionError):epochs.run_once()
    selection=list(corpus.selection_calls)
    assert epochs.accepted_root==root
    epochs.db.close()
    resumed=Epochs(tmp_path/'epochs',store,corpus,factory,root,'train','heldout',steps=1,examples=32,epochs_per_day=1,clock=lambda:1.)
    result=resumed.run_once()
    assert result['status']=='accepted'
    assert resumed.accepted_root==result['candidate']
    assert corpus.selection_calls==selection and factory.training_calls==1
    assert resumed.run_once()['status']=='period_budget_complete'
    resumed.clock=lambda:86401.
    second=resumed.run_once()
    assert second['status']=='accepted' and factory.training_calls==2
    assert second['baseline']==result['candidate']
