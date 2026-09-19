"""Experimental NeuroShard-native ABCI settlement for optimistic full-model work."""
import argparse
import binascii
import hashlib
import json
import signal
import sqlite3
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc

from neuroshard.dataflow.store import canonical
from neuroshard.demo import abci_pb2 as pb, protocol
from neuroshard.demo.app import Application as Base, INVALID
from neuroshard.lab import abci_pb2 as lab_pb, state as ledger
from neuroshard.lab.app import register,metadata,check_native_parameters
from . import settlement as state
from .objects import Objects,digest
from .verification import audit
from .model import configure

configure()

ERRORS = (*INVALID,binascii.Error)


def code_hash():
    # Pin every repository Python dependency, including imported legacy helpers
    # and package initializers. This intentionally also binds unused modules;
    # the research profile prefers excess coverage over an incomplete allowlist.
    package = Path(__file__).resolve().parent.parent
    return digest(canonical({p.relative_to(package).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(package.rglob('*.py'))}))


class Application(Base):
    def __init__(self,home):
        from .runtime import check
        check()
        self.home = Path(home)
        genesis = json.loads((self.home/'config/genesis.json').read_bytes())
        self.spec = genesis['app_state']['manifest']
        if self.spec['code_hash'] != code_hash():
            raise ValueError('Experimental consensus source differs from genesis')
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.home/'evolution.sqlite',check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY,value BLOB)')
        saved = self.db.execute('SELECT value FROM state WHERE id=1').fetchone()
        self.state = json.loads(saved[0]) if saved else None
        if self.state and self.state['manifest'] != self.spec:
            raise ValueError('Saved state differs from genesis manifest')
        self.pending = None
        self.artifacts = Objects(self.home/'proof-objects')
        self.referee_cache = OrderedDict()

    def InitChain(self,request,context):
        with self.lock:
            supplied = protocol.parse_json(request.app_state_bytes)
            if supplied['manifest'] != self.spec or request.initial_height not in (0,1):
                raise ValueError('Incompatible native genesis')
            check_native_parameters(request.consensus_params,self.spec['native_consensus'])
            expected = {v['consensus_key']:v['bond']//self.spec['params']['bond_unit'] for v in supplied['validators']}
            if expected != {v.pub_key.ed25519.hex():v.power for v in request.validators} or len(expected)!=len(request.validators):
                raise ValueError('Native voting power differs from bonded genesis')
            if self.state is None:
                self.state = state.genesis(request.chain_id,supplied['validators'],self.spec)
                self.persist(self.state)
            elif self.state['chain_id']!=request.chain_id:
                raise ValueError('Wrong native chain')
            return pb.ResponseInitChain(app_hash=self.app_hash())

    def referee(self,store,bundle,root,stage):
        key = (root,stage)
        if key not in self.referee_cache:
            self.referee_cache[key] = audit(store,bundle,root,stage)
            while len(self.referee_cache)>16:
                self.referee_cache.popitem(last=False)
        return self.referee_cache[key]

    def apply_to(self,candidate,raw,commit=False):
        if len(raw)>state.MAX_TX_BYTES:
            raise ValueError('Transaction exceeds native byte limit')
        return state.transition(candidate,protocol.parse_json(raw),self.artifacts,commit,self.referee)

    def CheckTx(self,request,context):
        with self.lock:
            try:
                if self.state is None:
                    raise ValueError('Node is initializing')
                projected,_ = state.advance(self.state,self.state['height']+1,self.state['time_ns'])
                self.apply_to(projected,request.tx)
                return pb.ResponseCheckTx(gas_wanted=len(request.tx))
            except ERRORS as exc:
                return pb.ResponseCheckTx(code=1,log=str(exc))

    def PrepareProposal(self,request,context):
        with self.lock:
            projected,_ = state.advance(self.state,*metadata(request))
            for raw in request.txs:
                if len(raw)>request.max_tx_bytes:
                    continue
                try:
                    self.apply_to(projected,raw)
                    return pb.ResponsePrepareProposal(txs=[raw])
                except ERRORS:
                    pass
            return pb.ResponsePrepareProposal()

    def ProcessProposal(self,request,context):
        with self.lock:
            try:
                if len(request.txs)>1:
                    raise ValueError('One transaction per block in the bounded execution profile')
                projected,_ = state.advance(self.state,*metadata(request))
                for raw in request.txs:
                    self.apply_to(projected,raw)
                return pb.ResponseProcessProposal(status=1)
            except ERRORS:
                return pb.ResponseProcessProposal(status=2)

    def FinalizeBlock(self,request,context):
        with self.lock:
            if len(request.txs)>1:
                raise ValueError('Too many transactions')
            candidate,updates = state.advance(self.state,*metadata(request))
            results = []
            for raw in request.txs:
                try:
                    candidate = self.apply_to(candidate,raw,True)
                    results.append(lab_pb.ExecTxResult())
                except ERRORS as exc:
                    results.append(lab_pb.ExecTxResult(code=1,log=str(exc)))
            self.pending = candidate
            return lab_pb.ResponseFinalizeBlock(tx_results=results,app_hash=self.app_hash(candidate),
                validator_updates=[lab_pb.ValidatorUpdate(pub_key=lab_pb.PublicKey(ed25519=bytes.fromhex(k)),power=v) for k,v in sorted(updates.items())])

    def Query(self,request,context):
        if request.path == '/hosting/quote':
            # Committed states are replaced, never mutated in place. Capture
            # one atomic view, then keep market matching outside consensus's
            # state lock so discovery cannot hold up block finalization.
            with self.lock:
                s = self.state
            try:
                if s is None or request.prove or request.height not in (0, s['height']):
                    raise ValueError('Only current-state queries without proofs are supported')
                from .provider_quotes import quote
                options = protocol.parse_json(request.data)
                value = quote(s, options['question'], options['max_tokens'],
                    provider_ceiling=options.get('provider_ceiling', 2**60), publisher=options.get('publisher'))
                return pb.ResponseQuery(value=canonical(value), height=s['height'])
            except ERRORS as exc:
                return pb.ResponseQuery(code=1, log=str(exc))
        with self.lock:
            try:
                s = self.state
                if s is None or request.prove or request.height not in (0,s['height']):
                    raise ValueError('Only current-state queries without proofs are supported')
                options = protocol.parse_json(request.data) if request.data else {}
                if request.path in ('/status','/summary'):
                    fields = ('chain_id','height','time_ns','model_root','serving_root','training_round','issued','burned','initial_supply','period','period_steps','period_growths','audit_count','update_check_count','settled','assignment')
                    value = {k:s[k] for k in fields}
                    value['candidate'] = {k:v for k,v in s['candidate'].items() if k!='metadata'} if s['candidate'] else None
                    value['app_hash'] = self.app_hash().hex()
                    value['validators'] = ledger.voting_power(s,s['height'])
                elif request.path=='/candidate':
                    value = s['candidate']
                elif request.path=='/auditing':
                    value = s.get('auditing')
                elif request.path=='/portable_work':
                    value = s.get('portable_work')
                elif request.path=='/expert_work':
                    value = s.get('expert_work')
                elif request.path=='/planner_work':
                    value = s.get('planner_work')
                elif request.path=='/expert_lifecycle':
                    value = s.get('expert_lifecycle')
                elif request.path=='/hosting':
                    value = s.get('hosting')
                elif request.path=='/hosting/job':
                    from .hosting import snapshot
                    value = snapshot(s, options['job_id'])
                elif request.path=='/portable_lifecycle':
                    value = s.get('portable_lifecycle')
                elif request.path=='/lifecycle':
                    life = s.get('lifecycle')
                    value = None if life is None else {k:v for k,v in life.items()
                        if k not in ('seen_documents','seen_batches','active','proposal')}
                    if value is not None:
                        value['active'] = {k:v for k,v in life['active'].items() if k!='batches'} if life['active'] else None
                        value['proposal'] = {k:v for k,v in life['proposal'].items() if k!='metadata'} if life['proposal'] else None
                        value['consumed_documents'] = len(life['seen_documents'])
                elif request.path=='/data':
                    value = s.get('lifecycle',{}).get('active')
                elif request.path=='/data/proposal':
                    value = s.get('lifecycle',{}).get('proposal')
                elif request.path=='/evaluation':
                    value = s.get('lifecycle',{}).get('evaluation')
                elif request.path=='/inference':
                    life = s.get('lifecycle',{})
                    key = options.get('id')
                    value = (life.get('jobs',{}).get(key) or life.get('results',{}).get(key)) if key else {
                        'jobs':life.get('jobs',{}), 'results':life.get('results',{})}
                elif request.path=='/manifest':
                    value = self.spec
                elif request.path=='/work':
                    from .schema import root
                    identity=root(options['identity'])
                    value={'identity':identity,'paid_claim':s['paid_work'].get(identity)}
                elif request.path=='/account':
                    owner = ledger.public_key(options['public_key'])
                    value = {'public_key':owner,**s['accounts'].get(owner,{'balance':0,'nonce':0})}
                else:
                    raise ValueError('Unknown query')
                return pb.ResponseQuery(value=canonical(value),height=s['height'])
            except ERRORS as exc:
                return pb.ResponseQuery(code=1,log=str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home',type=Path,required=True)
    parser.add_argument('--port',type=int,required=True)
    args = parser.parse_args()
    app = Application(args.home)
    server = grpc.server(ThreadPoolExecutor(max_workers=8),options=[('grpc.so_reuseport',0),
        ('grpc.max_send_message_length',8*1024*1024),('grpc.max_receive_message_length',8*1024*1024)])
    register(app,server)
    server.add_insecure_port(f'127.0.0.1:{args.port}')
    signal.signal(signal.SIGTERM,lambda *_:server.stop(1))
    server.start()
    server.wait_for_termination()
    app.db.close()


if __name__=='__main__':
    main()
