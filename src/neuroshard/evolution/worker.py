"""Durable full-model shard operations and independently replayable step transcripts."""
import copy
import json
import math
import os
import sqlite3
import time
from pathlib import Path

from neuroshard.dataflow.store import canonical
from .objects import digest
from .model import Shard, torch
from . import schema
from .batches import unpack


def tensor_root(store, value):
    return store.put_tensors({'value':value})


def tensor(store, root):
    value = store.tensors(root)
    if set(value) != {'value'}:
        raise ValueError('Expected a single tensor')
    result = value['value']
    if result.dtype != torch.float32 or result.ndim != 3 or not 1 <= result.shape[0] <= 4 or not 2 <= result.shape[1] <= 256 or result.shape[0]*result.shape[1]>512 or not 8 <= result.shape[2] <= 4096 or not torch.isfinite(result).all():
        raise ValueError('Activation exceeds bounds or is nonfinite')
    return result


class Session:
    def __init__(self, store, model_root, partition, checkpoint=None):
        self.store, self.model_root, self.partition = store, model_root, partition
        self.model = store.json(model_root)
        schema.partition(self.model, partition)
        if checkpoint:
            self.model['components'].update(checkpoint)
        self.shard = Shard(self.model, partition, store)
        self.phase = 'idle'
        self.trace = None
        self.input = self.output = self.ids = self.labels = None

    def run(self, operation):
        kind = operation['phase']
        if kind not in ('begin', 'forward') and self.trace is not None:
            if operation.get('parent', self.trace['parent']) != self.trace['parent'] or operation.get('step', self.trace['step']) != self.trace['step']:
                raise ValueError('Operation belongs to a different training step')
        if kind in ('begin','forward'):
            if self.phase != 'idle':
                raise ValueError('Previous step has not committed')
            parent = self.store.json(operation['parent'])
            for name in self.partition['components']:
                if parent['components'][name] != self.model['components'][name]:
                    raise ValueError('Step parent differs from the worker checkpoint')
            self.shard.zero_grad(set_to_none=True)
            self.trace = {'parent':operation['parent'],'partition':self.partition,'step':operation['step'],
                          'input':operation['input'],'phase':kind}
            if kind == 'begin':
                if 'embed' not in self.partition['components']:
                    raise ValueError('Only the embedding shard starts a batch')
                ids,labels = unpack(self.store.json(operation['input']),self.model['config']['vocab_size'])
                self.ids = torch.tensor(ids,dtype=torch.long)
                self.labels = torch.tensor(labels,dtype=torch.long) if labels is not None else None
                self.output = self.shard(self.ids,ids=True)
            else:
                self.input = tensor(self.store,operation['input']).requires_grad_(True)
                if self.input.shape[-1] != self.model['config']['hidden_size']:
                    raise ValueError('Forward activation width differs from architecture')
                self.output = self.shard(self.input)
            self.trace['output'] = tensor_root(self.store,self.output)
            self.phase = 'forward'
            return {'output':self.trace['output']}
        if kind == 'loss':
            if self.phase != 'forward' or self.ids is None:
                raise ValueError('Loss requires the first shard forward graph')
            hidden = tensor(self.store,operation['input']).requires_grad_(True)
            if hidden.shape != (*self.ids.shape,self.model['config']['hidden_size']):
                raise ValueError('Head input shape differs from the training batch')
            loss = self.shard.loss(hidden,self.ids,self.labels)
            loss.backward()
            self.trace.update(head_input=operation['input'],head_gradient=tensor_root(self.store,hidden.grad),
                              loss_hex=float(loss.detach()).hex())
            self.phase = 'head_backward'
            return {'gradient':self.trace['head_gradient'],'loss_hex':self.trace['loss_hex']}
        if kind == 'backward':
            required = 'head_backward' if self.ids is not None else 'forward'
            if self.phase != required:
                raise ValueError('Backward phase has no matching forward graph')
            gradient = tensor(self.store,operation['gradient'])
            if gradient.shape != self.output.shape:
                raise ValueError('Boundary gradient shape mismatch')
            self.output.backward(gradient)
            norm = self.shard.gradient_squared_norm()
            if not math.isfinite(norm):
                raise ValueError('Nonfinite gradient norm')
            self.trace.update(gradient_out=operation['gradient'],gradient_in=None if self.input is None else tensor_root(self.store,self.input.grad),
                              norm_squared_hex=norm.hex())
            self.phase = 'backward'
            return {'gradient':self.trace['gradient_in'],'norm_squared_hex':self.trace['norm_squared_hex']}
        if kind == 'update':
            if self.phase != 'backward':
                raise ValueError('Update requires complete gradients')
            rate, scale = float.fromhex(operation['learning_rate_hex']),float.fromhex(operation['scale_hex'])
            if not math.isfinite(rate) or not 0 < rate <= .1 or not math.isfinite(scale) or not 0 < scale <= 1:
                raise ValueError('Invalid bounded optimizer parameters')
            commitments = None
            if self.model.get('update_witnesses'):
                from .update_witness import capture_before, capture_after
                commitments = capture_before(self.shard)
            self.shard.update(rate,scale)
            if commitments is not None:
                self.trace['updates'] = capture_after(self.shard, commitments)
            components = self.shard.save(self.store)
            self.trace.update(learning_rate_hex=operation['learning_rate_hex'],scale_hex=operation['scale_hex'],components=components)
            trace_root = self.store.put_json(self.trace)
            self.model['components'].update(components)
            self.phase = 'idle'
            self.input = self.output = self.ids = self.labels = None
            self.shard.zero_grad(set_to_none=True)
            return {'components':components,'trace':trace_root,'resident_parameters':self.shard.resident_parameters}
        raise ValueError('Unknown shard operation')


def replay_trace(store, trace_root):
    """Exact bounded-stage referee; no other stage's weights are loaded."""
    trace = store.json(trace_root)
    if trace['partition']['parameters'] > 48_000_000:
        raise ValueError('Referee shard exceeds the 48M-parameter bound')
    session = Session(store,trace['parent'],trace['partition'])
    common = {'parent':trace['parent'],'step':trace['step']}
    result = session.run({**common,'phase':trace['phase'],'input':trace['input']})
    if result['output'] != trace['output']:
        return {'valid':False,'mismatch':'forward output'}
    if trace['phase'] == 'begin':
        result = session.run({'phase':'loss','input':trace['head_input']})
        if result != {'gradient':trace['head_gradient'],'loss_hex':trace['loss_hex']}:
            return {'valid':False,'mismatch':'loss or head gradient'}
    result = session.run({'phase':'backward','gradient':trace['gradient_out']})
    if result != {'gradient':trace['gradient_in'],'norm_squared_hex':trace['norm_squared_hex']}:
        return {'valid':False,'mismatch':'backward gradient'}
    result = session.run({'phase':'update','learning_rate_hex':trace['learning_rate_hex'],'scale_hex':trace['scale_hex']})
    if result['components'] != trace['components']:
        return {'valid':False,'mismatch':'optimizer update'}
    if session.trace.get('updates') != trace.get('updates'):
        return {'valid':False,'mismatch':'optimizer tensor commitments'}
    return {'valid':True,'trace':trace_root,'resident_parameters':session.shard.resident_parameters}


class Worker:
    """A SQLite intent log recovers graphs after interruption and reuses committed outputs."""
    def __init__(self, home, store, capacity=48_000_000):
        self.home, self.store, self.capacity = Path(home),store,capacity
        self.home.mkdir(parents=True,exist_ok=True)
        self.db = sqlite3.connect(self.home/'worker.sqlite')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, initial BLOB, checkpoint BLOB);
            CREATE TABLE IF NOT EXISTS operations (id TEXT PRIMARY KEY, session TEXT, step INTEGER, position INTEGER, input BLOB, result BLOB);
        ''')
        self.sessions = {}

    def open(self, session_id, model_root, partition, start_step=0):
        schema.integer(start_step,0,2**53-1)
        if partition['capacity'] > self.capacity:
            raise ValueError('Assignment exceeds advertised worker capacity')
        initial = {'model_root':model_root,'partition':partition}
        if start_step:
            initial['start_step'] = start_step
        row = self.db.execute('SELECT initial,checkpoint FROM sessions WHERE id=?',(session_id,)).fetchone()
        if row:
            if json.loads(row[0]) != initial:
                raise ValueError('Conflicting assignment for an existing epoch')
        else:
            with self.db:
                self.db.execute('INSERT INTO sessions VALUES (?,?,NULL)',(session_id,canonical(initial)))
        if session_id not in self.sessions:
            if self.sessions:
                raise ValueError('Worker already holds another resident training session')
            checkpoint = json.loads(row[1]) if row and row[1] else None
            session = Session(self.store,model_root,partition,checkpoint)
            # The last committed update already lives in checkpoint. Recreate only
            # the unfinished graph, including any operation interrupted after intent.
            last = self.db.execute("SELECT max(step) FROM operations WHERE session=? AND result IS NOT NULL AND json_extract(input,'$.phase')='update'",(session_id,)).fetchone()[0]
            pending = self.db.execute('SELECT input,result FROM operations WHERE session=? AND step>? ORDER BY step,position',
                                      (session_id,-1 if last is None else last)).fetchall()
            for raw,result in pending:
                if result is not None:
                    actual = session.run(json.loads(raw))
                    if actual != json.loads(result):
                        raise ValueError('Recovered operation differs from its durable commitment')
            self.sessions[session_id] = session
        return {'session':session_id,'resident_parameters':self.sessions[session_id].shard.resident_parameters}

    def operation(self, session_id, operation):
        schema.integer(operation['step'], 0, 2**53-1)
        op_id = digest(canonical({'session':session_id,'operation':operation}))
        old = self.db.execute('SELECT result FROM operations WHERE id=?',(op_id,)).fetchone()
        if old and old[0] is not None:
            return json.loads(old[0])
        if session_id not in self.sessions:
            row = self.db.execute('SELECT initial FROM sessions WHERE id=?', (session_id,)).fetchone()
            if row is None:
                raise ValueError('Unknown worker session')
            initial = json.loads(row[0])
            self.open(session_id, initial['model_root'], initial['partition'])
        session = self.sessions[session_id]
        last = self.db.execute("SELECT max(step) FROM operations WHERE session=? AND result IS NOT NULL AND position=3", (session_id,)).fetchone()[0]
        initial = json.loads(self.db.execute('SELECT initial FROM sessions WHERE id=?',(session_id,)).fetchone()[0])
        if operation['step'] != (initial.get('start_step',0) if last is None else last+1):
            raise ValueError('Training steps must be consecutive')
        position = {'begin':0,'forward':0,'loss':1,'backward':2,'update':3}[operation['phase']]
        collision = self.db.execute('SELECT id FROM operations WHERE session=? AND step=? AND position=?',
                                    (session_id,operation['step'],position)).fetchone()
        if collision and collision[0] != op_id:
            raise ValueError('Conflicting operation at an already assigned step')
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO operations VALUES (?,?,?,?,?,NULL)',
                            (op_id,session_id,operation['step'],position,canonical(operation)))
        try:
            result = session.run(operation)
            with self.db:
                self.db.execute('UPDATE operations SET result=? WHERE id=?',(canonical(result),op_id))
                if operation['phase'] == 'update':
                    self.db.execute('UPDATE sessions SET checkpoint=? WHERE id=?',(canonical(result['components']),session_id))
        except Exception:
            self.release(session_id)
            raise
        return result

    def evaluate(self, session_id, request):
        return evaluate_session(self.sessions[session_id], self.store, request)

    def release(self, session_id):
        self.sessions.pop(session_id,None)
        import gc
        gc.collect()


def evaluate_session(session, store, request):
    """Shared worker/referee forward execution, without a database dependency."""
    if session.phase != 'idle':
        raise ValueError('Evaluation cannot interrupt a training step')
    with torch.no_grad():
        if request['phase'] == 'forward':
            first = 'embed' in session.partition['components']
            value = torch.tensor(unpack(store.json(request['input']),session.model['config']['vocab_size'])[0], dtype=torch.long) if first else tensor(store, request['input'])
            if not first and value.shape[-1] != session.model['config']['hidden_size']:
                raise ValueError('Forward activation width differs from architecture')
            if first and (value.ndim != 2 or not 1 <= value.shape[0] <= 16 or not 2 <= value.shape[1] <= 256 or
                          value.min() < 0 or value.max() >= session.model['config']['vocab_size']):
                raise ValueError('Evaluation input exceeds execution bounds')
            return {'output': tensor_root(store, session.shard(value, ids=first))}
        if request['phase'] == 'head':
            hidden = tensor(store, request['input'])
            ids,labels = unpack(store.json(request['batch']),session.model['config']['vocab_size'])
            ids = torch.tensor(ids, dtype=torch.long)
            if hidden.shape != (*ids.shape,session.model['config']['hidden_size']):
                raise ValueError('Evaluation hidden state differs from the assigned batch shape')
            targets = torch.tensor(labels,dtype=torch.long) if labels is not None else ids
            logits = session.shard.logits(hidden)
            flat = logits[:, :-1].reshape(-1,session.model['config']['vocab_size'])
            loss = torch.nn.functional.cross_entropy(flat,targets[:, 1:].reshape(-1))
            losses = torch.nn.functional.cross_entropy(flat,targets[:, 1:].reshape(-1),reduction='none').reshape(ids.shape[0],-1)
            per_example = losses.sum(-1)/(targets[:, 1:]!=-100).sum(-1)
            return {'loss_hex': float(loss).hex(),
                    'losses_hex':[float(value).hex() for value in per_example],
                    'next_ids': logits[:, -1:].argmax(-1).flatten().tolist()}
        raise ValueError('Unknown evaluation phase')
