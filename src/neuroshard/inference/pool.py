"""Outbound workers for frozen-backbone features and trainable adapter updates."""
import json,os,threading,time
from pathlib import Path

from neuroshard.demo import client as wire,protocol,work
from neuroshard.lab import client
from neuroshard.publicnet import pool as transport
from .engine import Engine
from .settlement import submit


def authorized_request(envelope,identity,stage,task,summary):
    body,sponsor=transport.authenticate(envelope,task['chain_id'],'assignment');lease=task['lease']
    if (not lease or lease['task_kind']!='train' or sponsor!=lease['owner'] or body.get('worker')!=identity.public_key
        or body.get('stage')!=stage or lease['workers'][stage]!=identity.public_key
        or summary['chain_id']!=task['chain_id'] or summary['lease']!=lease or summary['height']>lease['expires']):
        raise ValueError('Assignment is not authorized by the local finalized lease')
    request=body['request'];expected={'task_id','input_ids','weights','operation'}|({'activation'} if stage==1 else set())
    if (set(request)!=expected or request['task_id']!=lease['task_id']
        or request['operation']!=('features' if stage==0 else 'update')
        or request['input_ids']!=task['input_ids'] or request['weights']!=task['weights']):
        raise ValueError('Assignment differs from the local checkpoint, batch, or stage')
    return request


class Worker(transport.Worker):
    def __init__(self,home,stage):
        self.home,self.stage_index=Path(home),stage
        self.config=json.loads((self.home/'node.json').read_text())
        self.identity=protocol.Identity.load_or_create(self.home/'account.key')
        self.rpc=f'http://127.0.0.1:{self.config["base_port"]+1}'
        spec=json.loads((self.home/'config/genesis.json').read_text())['app_state']['manifest']
        self.engine=Engine(self.config['model_dir'],spec['model'])
        self.path=self.home/f'llm-worker-stage-{stage}.json'
        self.journal=json.loads(self.path.read_text()) if self.path.exists() else {'task_id':None,'operations':{}}

    def save(self):
        temporary=self.path.with_suffix('.tmp')
        with temporary.open('wb') as f:
            f.write(work.canonical(self.journal));f.flush();os.fsync(f.fileno())
        os.replace(temporary,self.path)
        directory=os.open(self.path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)

    def compute(self,envelope):
        task,summary=wire.query(self.rpc,'/task'),wire.query(self.rpc,'/summary')
        if task['chain_id']!=self.config['chain_id']:raise ValueError('Local chain identity changed')
        request=authorized_request(envelope,self.identity,self.stage_index,task,summary)
        task_id,operation=request['task_id'],request['operation']
        if self.journal['task_id']!=task_id:self.journal={'task_id':task_id,'operations':{}}
        root=work.digest(request);old=self.journal['operations'].get(operation)
        if old:
            if old['digest']!=root:raise ValueError('Sponsor changed the assigned operation')
            if 'result' not in old:raise ValueError('Interrupted computation requires a new lease')
            return old['result']
        self.journal['operations'][operation]={'digest':root};self.save()
        features=self.engine.features(request['input_ids']) if self.stage_index==0 else request['activation']
        common={'task_id':task_id,'model_root':task['model_root'],'input_root':work.digest(task['input_ids']),
                'feature_root':work.digest(features)}
        if self.stage_index==0:
            result={'activation':features,'receipt':self.identity.sign({**common,'stage':0})}
        else:
            value=self.engine.update(task['weights'],task['input_ids'],features)
            receipt={**common,'stage':1,'result_root':work.digest(value['weights']),
                     'gradient_root':value['gradient_root'],'loss_hex':value['loss_hex']}
            result={'weights':value['weights'],'receipt':self.identity.sign(receipt)}
        self.journal['operations'][operation]['result']=result;self.save();return result

    def run(self,url,stop=None,max_tasks=0):
        url=transport.coordinator_url(url);stop=stop or threading.Event();delivered=set();last_error=None
        while not stop.is_set():
            try:
                envelope=transport.signed(self.identity,self.config['chain_id'],'poll',stage=self.stage_index)
                assignment=wire.http(url+'/work/poll',envelope,timeout=10).get('assignment')
                if assignment:
                    result=self.compute(assignment)
                    wire.http(url+'/work/result',transport.signed(self.identity,self.config['chain_id'],'result',
                        job_id=work.digest(assignment['body']),result=result),timeout=15)
                    task_id=assignment['body']['request']['task_id']
                    if task_id not in delivered:
                        print(json.dumps({'returned_task':task_id,'stage':self.stage_index,
                                          'payment':'Pending native verification and settlement'}),flush=True)
                    delivered.add(task_id)
                    if max_tasks and len(delivered)>=max_tasks:return
                last_error=None
            except (OSError,ValueError,KeyError,TypeError) as exc:
                if str(exc)!=last_error:print(json.dumps({'worker_status':str(exc)}),flush=True)
                last_error=str(exc)
            stop.wait(1)


class Coordinator(transport.Coordinator):
    def __init__(self,home):
        super().__init__(home)
        services=self.home/'services.json'
        self.fallback_workers=set(json.loads(services.read_text()).get('fallback_workers',[])) if services.exists() else set()

    def mine(self,wait_seconds=180):
        deadline=time.monotonic()+wait_seconds
        # A restarted sponsor must let an earlier finalized lease settle or expire
        # before attempting another reservation. The attempt budget never resets.
        while wire.query(self.rpc,'/summary')['lease'] is not None:
            if wait_seconds and time.monotonic()>deadline:raise TimeoutError('Waiting for the existing native lease')
            time.sleep(1)
        while True:
            with self.lock:
                # Preserve arrival order within each group; operator fallback workers yield
                # to public workers. Selected workers rejoin at the back after settlement.
                ordered=sorted(self.workers.items(),key=lambda item:item[0][0] in self.fallback_workers)
                selected=[next((k for (k,s),seen in ordered if s==stage and time.monotonic()-seen<10),None)
                          for stage in (0,1)]
            if all(selected):break
            if wait_seconds and time.monotonic()>deadline:raise TimeoutError('Waiting for one worker per stage')
            time.sleep(.5)
        status=wire.query(self.rpc,'/summary')
        submit(self.rpc,client.transaction(self.rpc,self.identity,'reserve',task_kind='train',
            parent=status['model_root'],round=status['round'],workers=selected,request={},price=0))
        task=wire.query(self.rpc,'/task');lease=task['lease']
        if not lease or lease['owner']!=self.identity.public_key:raise ValueError('Reservation no longer available')
        common={'task_id':lease['task_id'],'input_ids':task['input_ids'],'weights':task['weights']}
        first=self.compute(selected[0],0,{**common,'operation':'features'})
        second=self.compute(selected[1],1,{**common,'operation':'update','activation':first['activation']})
        result=submit(self.rpc,client.transaction(self.rpc,self.identity,'submit',task_id=lease['task_id'],
            result_root=work.digest(second['weights']),receipts=[first['receipt'],second['receipt']]))
        with self.lock:
            for key in list(self.workers):
                if key[0] in selected:self.workers.pop(key)
        return result
