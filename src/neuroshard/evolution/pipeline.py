"""Model-parallel training with bounded activations and atomic revision commitments."""
import copy
import math
import time
import json
import os
from pathlib import Path

from neuroshard.dataflow.store import canonical
from .objects import digest
from .model import place
from . import schema


class LocalEndpoint:
    def __init__(self,worker):self.worker=worker
    def open(self,*args):return self.worker.open(*args)
    def operation(self,*args):return self.worker.operation(*args)
    def release(self,*args):return self.worker.release(*args)
    def evaluate(self,*args):return self.worker.evaluate(*args)


class Pipeline:
    def __init__(self, store, model_root, endpoints, capacities, session_id, learning_rate=.003, clip_norm=1.0, journal=None,start_step=0):
        schema.integer(start_step,0,2**53-1)
        self.start_step = start_step
        self.store,self.model_root,self.endpoints = store,model_root,endpoints
        self.session_id,self.learning_rate,self.clip_norm = session_id,learning_rate,clip_norm
        if not math.isfinite(learning_rate) or not 0 < learning_rate <= .1 or not math.isfinite(clip_norm) or not 0 < clip_norm <= 10:
            raise ValueError('Invalid optimizer parameters')
        self.journal = Path(journal) if journal else None
        self.model = store.json(model_root)
        self.partitions = place(self.model,capacities)
        if len(self.partitions) != len(endpoints):
            raise ValueError('Each assigned partition needs one worker endpoint')
        for endpoint,partition in zip(endpoints,self.partitions):
            endpoint.open(session_id,model_root,partition,*([start_step] if start_step else []))
        self.step = start_step
        self.pending_batch = None
        self.initial_root = model_root
        if self.journal and self.journal.exists():
            saved = json.loads(self.journal.read_bytes())
            if saved['initial_root'] != model_root or saved['session_id'] != session_id or saved['partitions'] != self.partitions or saved['learning_rate'] != learning_rate or saved['clip_norm'] != clip_norm or saved.get('start_step',0)!=start_step:
                raise ValueError('Coordinator journal differs from assignment')
            self.model_root, self.step, self.pending_batch = saved['model_root'], saved['step'], saved['pending_batch']
            self.model = store.json(self.model_root)

    def persist(self):
        if self.journal:
            self.journal.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.journal.with_suffix('.pending')
            with temporary.open('wb') as f:
                f.write(canonical({k:getattr(self,k) for k in ('initial_root','session_id','partitions','learning_rate','clip_norm','model_root','start_step','step','pending_batch')}))
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary,self.journal)
            directory = os.open(self.journal.parent,os.O_RDONLY|os.O_DIRECTORY)
            try: os.fsync(directory)
            finally: os.close(directory)

    def train(self,batch):
        before = time.monotonic()
        batch_root = self.store.put_json(batch)
        if self.pending_batch is not None and self.pending_batch != batch_root:
            raise ValueError('Recover the pending batch before assigning another')
        self.pending_batch = batch_root
        self.persist()
        common = {'step':self.step,'parent':self.model_root}
        def call(index,phase,**fields):
            return self.endpoints[index].operation(self.session_id,{**common,'phase':phase,**fields})
        forward = call(0,'begin',input=batch_root)
        for index in range(1,len(self.endpoints)):
            forward = call(index,'forward',input=forward['output'])
        loss = call(0,'loss',input=forward['output'])
        gradient = loss['gradient']
        norms = [None]*len(self.endpoints)
        for index in reversed(range(len(self.endpoints))):
            backward = call(index,'backward',gradient=gradient)
            gradient = backward['gradient']
            norms[index] = backward['norm_squared_hex']
        squared = math.fsum(float.fromhex(n) for n in norms)
        scale = min(1.0,self.clip_norm/(math.sqrt(squared)+1e-6))
        results = [call(index,'update',learning_rate_hex=float(self.learning_rate).hex(),scale_hex=scale.hex())
                   for index in range(len(self.endpoints))]
        candidate = copy.deepcopy(self.model)
        candidate['parent'] = self.model_root
        for result in results:
            candidate['components'].update(result['components'])
        new_root = self.store.put_json(candidate)
        record = {'parent':self.model_root,'model_root':new_root,'batch':batch_root,'step':self.step,
                  'traces':[r['trace'] for r in results],'norms':norms,'clip_norm_hex':float(self.clip_norm).hex(),
                  'learning_rate_hex':float(self.learning_rate).hex(),'scale_hex':scale.hex(),'loss_hex':loss['loss_hex']}
        record_root = self.store.put_json(record)
        self.model_root,self.model = new_root,candidate
        self.step += 1
        self.pending_batch = None
        self.persist()
        return {**record,'record_root':record_root,'elapsed_seconds':time.monotonic()-before}

    def evaluate(self, batch):
        batch_root = self.store.put_json(batch)
        current = batch_root
        for endpoint in self.endpoints:
            current = endpoint.evaluate(self.session_id, {'phase':'forward','input':current})['output']
        return self.endpoints[0].evaluate(self.session_id, {'phase':'head','input':current,'batch':batch_root})

    def generate(self, token_ids, max_tokens=32, eos_ids=()):
        schema.integer(max_tokens, 1, 64)
        if not 2 <= len(token_ids) <= 192:
            raise ValueError('Prompt must contain 2–192 tokens')
        ids, output = list(token_ids), []
        for _ in range(max_tokens):
            token = self.evaluate([ids])['next_ids'][0]
            output.append(token)
            ids.append(token)
            if token in eos_ids:
                break
        return {'model_root':self.model_root,'token_ids':output}

    def close(self):
        for endpoint in self.endpoints:endpoint.release(self.session_id)



from .verification import validate_record
