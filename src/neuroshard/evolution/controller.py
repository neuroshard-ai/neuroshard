"""Durable research epochs with fresh data, replay, optional growth and rejection.

The local registry records experimental decisions. It is deliberately distinct
from the opt-in native lifecycle and never changes the public serving registry.
"""
import fcntl
import json
import secrets
import sqlite3
import time
from pathlib import Path

from neuroshard.dataflow.store import canonical
from .evaluation import evaluate_reservation,decide
from .model import grow,place
from .schema import integer
from .batches import from_windows


class Epochs:
    def __init__(self,home,store,corpus,pipeline_factory,initial_root,train_source,heldout_source,
                 steps=32,examples=64,documents=128,epochs_per_day=2,capacities=(48000000,)*3,
                 growth_layers=0,clock=time.time,progress=None):
        self.home,self.store,self.corpus = Path(home),store,corpus
        self.factory,self.clock = pipeline_factory,clock
        self.progress=progress
        self.train_source,self.heldout_source = train_source,heldout_source
        self.options = {'steps':integer(steps,1,4096),'examples':integer(examples,32,256),
                        'documents':integer(documents,1,1024),'epochs_per_day':integer(epochs_per_day,1,24),
                        'growth_layers':integer(growth_layers,0,16),'capacities':list(capacities)}
        if getattr(corpus,'codec',None) is not None:
            corpus.codec.check_model(store.json(initial_root))
            self.options['tokenizer_root']=corpus.tokenizer_root
        self.home.mkdir(parents=True,exist_ok=True)
        self.db = sqlite3.connect(self.home/'epochs.sqlite')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS registry (id INTEGER PRIMARY KEY, model TEXT);
            CREATE TABLE IF NOT EXISTS epochs (id INTEGER PRIMARY KEY, day INTEGER, status TEXT, value BLOB);
        ''')
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO registry VALUES (1,?)',(initial_root,))
        if getattr(corpus,'codec',None) is not None:
            corpus.codec.check_model(store.json(self.accepted_root))

    @property
    def accepted_root(self):
        return self.db.execute('SELECT model FROM registry WHERE id=1').fetchone()[0]

    def save(self,epoch):
        with self.db:
            self.db.execute('UPDATE epochs SET status=?,value=? WHERE id=?',
                            (epoch['status'],canonical(epoch),epoch['id']))
        if self.progress:
            self.progress({'epoch':epoch['id'],'phase':epoch['status']})

    def run_once(self):
        with (self.home/'coordinator.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            active = self.db.execute("SELECT value FROM epochs WHERE status NOT IN ('accepted','rejected','no_data') ORDER BY id LIMIT 1").fetchone()
            if active:
                epoch = json.loads(active[0])
                if epoch['options']!=self.options:
                    raise ValueError('Finish the active epoch before changing its parameters')
            else:
                day = int(self.clock()//86400)
                count = self.db.execute('SELECT count(*) FROM epochs WHERE day=?',(day,)).fetchone()[0]
                if count>=self.options['epochs_per_day']:
                    return {'status':'period_budget_complete','accepted_root':self.accepted_root,'next_day':day+1}
                with self.db:
                    cursor = self.db.execute("INSERT INTO epochs(day,status,value) VALUES (?,'collecting',?)",(day,b'{}'))
                    epoch = {'id':cursor.lastrowid,'day':day,'status':'collecting','options':self.options,
                             'baseline':self.accepted_root,'train_source':self.train_source,'heldout_source':self.heldout_source}
                    self.db.execute('UPDATE epochs SET value=? WHERE id=?',(canonical(epoch),epoch['id']))
            if epoch['status']=='collecting':
                window = self.corpus.collect(epoch['train_source'],self.options['documents'])
                epoch['window'] = window['root']
                if not window['sequences']:
                    epoch['status'] = 'no_data'
                    self.save(epoch)
                    return epoch
                epoch['initial'] = epoch['baseline']
                if self.options['growth_layers']:
                    proposed,_ = grow(epoch['baseline'],self.store,self.options['growth_layers'])
                    try:
                        place(self.store.json(proposed),self.options['capacities'])
                    except ValueError:
                        epoch['growth'] = 'insufficient_capacity'
                    else:
                        epoch['initial'],epoch['growth'] = proposed,'candidate'
                epoch['batches'] = self.corpus.training(window['root'],2*self.options['steps'],f'epoch:{epoch["id"]}')
                epoch['status'] = 'training'
                self.save(epoch)
            if epoch['status']=='training':
                pipe = self.factory(epoch['initial'],f'epoch-{epoch["id"]}-train',self.home/f'epoch-{epoch["id"]}-train.json')
                try:
                    for step in range(pipe.step,self.options['steps']):
                        batch = from_windows(self.store,epoch['batches'][2*step:2*step+2],
                                             getattr(self.corpus,'tokenizer_root',None))
                        result = pipe.train(batch)
                        if self.progress:
                            self.progress({'epoch':epoch['id'],'phase':'training','step':step+1,
                                           'steps':self.options['steps'],'loss':float.fromhex(result['loss_hex'])})
                        with (self.home/f'epoch-{epoch["id"]}-steps.jsonl').open('ab') as log:
                            log.write(canonical(result)+b'\n')
                    epoch['candidate'] = pipe.model_root
                finally:
                    pipe.close()
                # Persist the candidate before creating an evaluation selector.
                epoch['status'] = 'selecting_evaluation'
                self.save(epoch)
            if epoch['status']=='selecting_evaluation':
                if 'beacon' not in epoch:
                    epoch['beacon'] = secrets.token_hex(32)
                    self.save(epoch)
                epoch.setdefault('evaluation',{})
                for role in ('retention','fresh'):
                    if role in epoch['evaluation']:
                        continue
                    for attempt in range(8):
                        try:
                            reservation = self.corpus.reserve_evaluation(epoch['candidate'],role,self.options['examples'],epoch['beacon'])
                            break
                        except ValueError as exc:
                            if 'Insufficient unused' not in str(exc):
                                raise
                            collected = self.corpus.collect(epoch['heldout_source'],256)
                            if collected['start']==collected['end']:
                                raise ValueError('Held-out source exhausted; add an independently curated source') from exc
                    else:
                        raise ValueError('Evaluation collection budget exhausted for this attempt')
                    epoch['evaluation'][role] = reservation
                    self.save(epoch)
                epoch['status'] = 'evaluating'
                self.save(epoch)
            if epoch['status']=='evaluating':
                epoch.setdefault('measurements',{})
                for name in ('baseline','candidate'):
                    if name in epoch['measurements']:
                        continue
                    pipe = self.factory(epoch[name],f'epoch-{epoch["id"]}-{name}',None)
                    try:
                        values = {role:evaluate_reservation(pipe,self.store,reservation)
                                  for role,reservation in epoch['evaluation'].items()}
                    finally:
                        pipe.close()
                    epoch['measurements'][name] = values
                    self.save(epoch)
                before,after = epoch['measurements']['baseline'],epoch['measurements']['candidate']
                epoch['decision'] = decide(before['retention'],after['retention'],before['fresh'],after['fresh'])
                epoch['status'] = 'accepted' if epoch['decision']['promote'] else 'rejected'
                with self.db:
                    if self.accepted_root!=epoch['baseline']:
                        raise ValueError('Accepted model changed while an epoch was in flight')
                    if epoch['status']=='accepted':
                        self.db.execute('UPDATE registry SET model=? WHERE id=1',(epoch['candidate'],))
                    self.db.execute('UPDATE epochs SET status=?,value=? WHERE id=?',(epoch['status'],canonical(epoch),epoch['id']))
            return epoch
