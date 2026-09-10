"""Native ABCI application for bounded model training and paid inference."""
import argparse,json,logging,signal,sqlite3,threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
from neuroshard.demo import abci_pb2 as pb,protocol,work
from neuroshard.demo.app import Application as BaseApplication,INVALID
from neuroshard.lab import abci_pb2 as lab_pb,state as ledger
from neuroshard.lab.app import register,metadata,check_native_parameters
from . import profile,state
from .engine import Engine


class Application(BaseApplication):
    def __init__(self,home):
        self.home=Path(home);config=json.loads((self.home/'node.json').read_text())
        genesis=json.loads((self.home/'config/genesis.json').read_text())
        self.spec=genesis['app_state']['manifest']
        self.data=json.loads((self.home/'dataset.json').read_text())
        profile.check(self.spec,config['model_dir'],self.data)
        self.engine=Engine(config['model_dir'],self.spec['model'])
        self.lock=threading.RLock()
        self.db=sqlite3.connect(str(self.home/'candidate.sqlite'),check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY,value BLOB)')
        saved=self.db.execute('SELECT value FROM state WHERE id=1').fetchone()
        self.state=json.loads(saved[0]) if saved else None
        if self.state and self.state['manifest']!=self.spec:raise ValueError('Saved execution profile differs from genesis')
        self.pending=None;self.cache=OrderedDict()

    def InitChain(self,request,context):
        with self.lock:
            supplied=protocol.parse_json(request.app_state_bytes)
            if supplied['manifest']!=self.spec or request.initial_height not in (0,1):raise ValueError('Incompatible genesis')
            check_native_parameters(request.consensus_params,self.spec['native_consensus'])
            expected={v['consensus_key']:v['bond']//self.spec['params']['bond_unit'] for v in supplied['validators']}
            if expected!={v.pub_key.ed25519.hex():v.power for v in request.validators} or len(expected)!=len(request.validators):
                raise ValueError('Bonded voting power differs from native genesis')
            if self.state is None:
                self.state=state.genesis(request.chain_id,supplied['validators'],self.spec,self.engine.initial(),
                                        float.fromhex(self.spec['initial_validation_loss_hex']))
                self.persist(self.state)
            elif self.state['chain_id']!=request.chain_id:raise ValueError('Wrong chain')
            return pb.ResponseInitChain(app_hash=self.app_hash())

    def execute(self,candidate,task):
        if task and 'validate_request' in task:
            request=task['validate_request']
            ids=self.engine.tokenizer.apply_chat_template([{'role':'user','content':request['prompt']}],
                                                  add_generation_prompt=True,tokenize=True)
            if len(ids)>self.spec['model']['max_input_tokens']:raise ValueError('Prompt exceeds token limit')
            return None
        key=('infer',task['job_id']) if task else ('train',candidate['lease']['task_id'])
        if key in self.cache:
            self.cache.move_to_end(key);return self.cache[key]
        if task:
            result=self.engine.infer(task['job']['weights'],task['job']['request'])
        else:
            ids=profile.batch(self.data,candidate['round']);lease=candidate['lease']
            value=self.engine.train(candidate['weights'],ids)
            root=work.digest(value['weights'])
            common={'task_id':lease['task_id'],'model_root':candidate['model_root'],
                    'input_root':work.digest(ids),'feature_root':value['feature_root']}
            result={**value,'result_root':root,'loss':float.fromhex(value['loss_hex']),
                'validation_loss_hex':self.engine.evaluate(value['weights'],self.data['validation']).hex(),
                'receipts':[{**common,'stage':0},{**common,'stage':1,'result_root':root,
                    'gradient_root':value['gradient_root'],'loss_hex':value['loss_hex']}]}
        self.cache[key]=result
        while len(self.cache)>40:self.cache.popitem(last=False)
        return result

    def apply_to(self,candidate,raw):
        if len(raw)>ledger.PARAMS['max_tx_bytes']:raise ValueError('Oversized transaction')
        return state.transition(candidate,protocol.parse_json(raw),self.execute)

    def CheckTx(self,request,context):
        with self.lock:
            try:
                if self.state is None:raise ValueError('Node is initializing')
                projected,_=state.advance(self.state,self.state['height']+1,self.state['time_ns'])
                self.apply_to(projected,request.tx)
                return pb.ResponseCheckTx(gas_wanted=1)
            except INVALID as exc:return pb.ResponseCheckTx(code=1,log=str(exc))

    def PrepareProposal(self,request,context):
        with self.lock:
            projected,_=state.advance(self.state,*metadata(request))
            for raw in request.txs:
                if len(raw)>request.max_tx_bytes:continue
                try:
                    self.apply_to(projected,raw);return pb.ResponsePrepareProposal(txs=[raw])
                except INVALID:pass
            return pb.ResponsePrepareProposal()

    def ProcessProposal(self,request,context):
        with self.lock:
            try:
                if len(request.txs)>1:raise ValueError('At most one transaction per block')
                projected,_=state.advance(self.state,*metadata(request))
                for raw in request.txs:self.apply_to(projected,raw)
                return pb.ResponseProcessProposal(status=1)
            except INVALID:return pb.ResponseProcessProposal(status=2)

    def FinalizeBlock(self,request,context):
        with self.lock:
            if len(request.txs)>1:raise ValueError('Too many transactions')
            candidate,updates=state.advance(self.state,*metadata(request));results=[]
            for raw in request.txs:
                try:
                    candidate=self.apply_to(candidate,raw);results.append(lab_pb.ExecTxResult())
                except INVALID as exc:results.append(lab_pb.ExecTxResult(code=1,log=str(exc)))
            self.pending=candidate
            return lab_pb.ResponseFinalizeBlock(tx_results=results,app_hash=self.app_hash(candidate),
                validator_updates=[lab_pb.ValidatorUpdate(pub_key=lab_pb.PublicKey(ed25519=bytes.fromhex(k)),power=v)
                                   for k,v in sorted(updates.items())])

    def Query(self,request,context):
        with self.lock:
            try:
                s=self.state
                if s is None or request.prove or request.height not in (0,s['height']):
                    raise ValueError('Only current-state queries without proofs are supported')
                options=protocol.parse_json(request.data) if request.data else {}
                if not isinstance(options,dict):raise ValueError('Query options must be an object')
                if request.path in ('/summary','/status'):
                    value={k:s[k] for k in ('chain_id','height','time_ns','round','model_root','serving_root',
                        'issued','burned','initial_supply','last_training_loss_hex','validation_loss_hex','serving_loss_hex',
                        'last_training_evaluation','lease')}
                    powers=ledger.voting_power(s,s['height'])
                    value.update(validator_count=len(powers),total_voting_power=sum(powers.values()),account_count=len(s['accounts']),
                        params=ledger.parameters(s),profile=self.spec['profile'],manifest_hash=work.digest(self.spec),
                        app_hash=self.app_hash().hex(),pending_inference=len(s['jobs']))
                elif request.path=='/manifest':value=self.spec
                elif request.path=='/task':
                    value={k:s[k] for k in ('chain_id','round','model_root','weights','lease')}
                    value['input_ids']=profile.batch(self.data,s['round'])
                elif request.path=='/account':
                    owner=ledger.public_key(options['public_key']);a=s['accounts'].get(owner,{'balance':0,'nonce':0})
                    value={'public_key':owner,'balance':state.available(s,owner),'locked':state.locked(s,owner),
                           'total':a['balance'],'nonce':a['nonce']}
                elif request.path=='/validators':
                    limit=ledger.integer(options.get('limit',100),1,100);after=options.get('after','')
                    if not isinstance(after,str) or len(after) not in (0,64):raise ValueError('Invalid cursor')
                    powers=sorted(ledger.voting_power(s,s['height']).items());page=[(k,p) for k,p in powers if k>after][:limit]
                    value={'validators':[{'public_key':k,'power':p,'owner':s['validators'][k]['owner'],
                        'bond':s['validators'][k]['amount']} for k,p in page], 'total':len(powers),
                        'next_after':page[-1][0] if len(page)==limit else None}
                elif request.path=='/jobs':
                    provider=ledger.public_key(options['provider'])
                    value={'jobs':[{'id':k,**{f:v[f] for f in ('owner','provider','model_root','request','expires','price')}}
                        for k,v in sorted(s['jobs'].items()) if v['provider']==provider]}
                elif request.path=='/job':
                    key=options['id']
                    if not isinstance(key,str) or len(key)!=64:raise ValueError('Invalid request id')
                    if key in s['jobs']:value={'id':key,'status':'pending',**s['jobs'][key]}
                    elif key in s['results']:value={'id':key,**s['results'][key]}
                    else:raise ValueError('Request not found in current state; inspect its native blocks')
                else:raise ValueError('Unknown query')
                return pb.ResponseQuery(value=work.canonical(value),height=s['height'])
            except INVALID as exc:return pb.ResponseQuery(code=1,log=str(exc))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--home',type=Path,required=True)
    parser.add_argument('--port',type=int,required=True);args=parser.parse_args()
    app=Application(args.home)
    server=grpc.server(ThreadPoolExecutor(max_workers=8),options=[('grpc.so_reuseport',0),
        ('grpc.max_send_message_length',work.MAX_MESSAGE_BYTES),('grpc.max_receive_message_length',work.MAX_MESSAGE_BYTES)])
    register(app,server);server.add_insecure_port(f'127.0.0.1:{args.port}')
    signal.signal(signal.SIGTERM,lambda *_:server.stop(1));server.start()
    print(f'Native LLM application on {args.port}',flush=True);server.wait_for_termination();app.db.close()


if __name__=='__main__':
    logging.basicConfig(level=logging.WARNING);main()
