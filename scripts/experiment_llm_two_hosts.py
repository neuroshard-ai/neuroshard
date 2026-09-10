"""Disposable two-host native LLM integration; never changes the deployed v2 chain."""
import os
os.environ.update(ATEN_CPU_CAPABILITY='default',MKL_ENABLE_INSTRUCTIONS='SSE4_2',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse,json,subprocess,sys,time,shlex,hashlib,uuid
from pathlib import Path
from neuroshard.demo import client as wire,protocol,work
from neuroshard.publicnet import bootstrap
from neuroshard.inference import node

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--ssh',required=True,help='An already authorized SSH host; no secrets in arguments')
parser.add_argument('--model-dir',type=Path,required=True)
parser.add_argument('--remote-model-dir',required=True)
parser.add_argument('--remote-python',required=True,help='Absolute path to a matching wheel-installed CPU runtime')
parser.add_argument('--remote-engine',required=True)
parser.add_argument('--remote-work',required=True,help='Absolute directory for disposable run homes')
parser.add_argument('--engine',type=Path,required=True)
parser.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[1]/'networks/neuroshard-llm-testnet-1/dataset.json')
parser.add_argument('--output',type=Path,default=Path.cwd()/'.neuroshard/two-host-experiments')
args=parser.parse_args()
if not args.remote_work.startswith('/home/') or not args.remote_python.startswith('/'):
 raise ValueError('Use absolute remote paths and a /home/... work directory')
root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
run_id='llm-'+uuid.uuid4().hex[:8];chain_id='neuroshard-'+run_id
local=root/run_id;local.mkdir();remote=Path(args.remote_work)
rbase=remote/run_id;ssh=['ssh','-o','BatchMode=yes',args.ssh];python=args.remote_python
renv={'ATEN_CPU_CAPABILITY':'default','MKL_ENABLE_INSTRUCTIONS':'SSE4_2','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
engine=args.engine.resolve();rengine=Path(args.remote_engine)
children=[];report={'chain_id':chain_id,'run_directory':str(local),'ssh_host':args.ssh}

def remote_cmd(args):return ' '.join([*(f'{k}={shlex.quote(v)}' for k,v in renv.items()),*[shlex.quote(str(a)) for a in args]])
def rrun(code):return subprocess.check_output(ssh+[remote_cmd([python,'-c',code])],text=True)
def start(name,args,rem=False):
 out=(local/(name+'.log')).open('w');child=subprocess.Popen(ssh+[remote_cmd(args)] if rem else args,stdout=out,stderr=out,start_new_session=True);out.close();children.append(child)
def wait_for(fn,timeout=180):
 deadline=time.monotonic()+timeout;last=None
 while time.monotonic()<deadline:
  try:
   result=fn()
   if result:return result
  except (OSError,ValueError,KeyError,subprocess.CalledProcessError) as exc:last=exc
  if any(p.poll() is not None for p in children):raise RuntimeError('A child stopped; inspect '+str(local))
  time.sleep(1)
 raise TimeoutError(str(last))
try:
 declarations=[]
 for i in range(2):
  home=local/f'node{i}';d=bootstrap.declaration(home,chain_id,2500000,20000000,engine);declarations.append(d)
 for i in range(2):
  code=f'from neuroshard.publicnet.bootstrap import declaration;import json;print(json.dumps(declaration({str(rbase/f"node{i}")!r},{chain_id!r},2500000,20000000,{str(rengine)!r})))'
  declarations.append(json.loads(rrun(code)))
 bundle=node.make_genesis(chain_id,declarations,local/'network',args.model_dir,args.data)
 print(json.dumps({'phase':'genesis','bundle':bundle}),flush=True)
 subprocess.run(['rsync','-az',str(local/'network')+'/',f'{args.ssh}:{rbase}/network/'],check=True)
 ids=[subprocess.check_output([str(engine),'show-node-id','--home',str(local/f'node{i}')],text=True).strip() for i in range(2)]
 rids=[rrun(f'import subprocess;print(subprocess.check_output([{str(rengine)!r},"show-node-id","--home",{str(rbase/f"node{i}")!r}],text=True).strip())').strip() for i in range(2)]
 for i in range(2):
  peers=[f'{ids[1-i]}@127.0.0.1:{43656+(1-i)*10}',f'{rids[0]}@127.0.0.1:45656']
  node.initialize(local/f'node{i}',local/'network/genesis.json',bundle['genesis_sha256'],peers,engine,args.model_dir,args.data,43656+i*10,True)
 for i in range(3):
  peers=[f'{rids[1 if i==0 else 0]}@127.0.0.1:{44666 if i==0 else 44656}',f'{ids[0]}@127.0.0.1:45666']
  code=f'from neuroshard.inference.node import initialize;initialize({str(rbase/f"node{i}")!r},{str(rbase/"network/genesis.json")!r},{bundle["genesis_sha256"]!r},{peers!r},{str(rengine)!r},{args.remote_model_dir!r},{str(rbase/"network/dataset.json")!r},{44656+i*10},True)'
  rrun(code)
 print(json.dumps({'phase':'nodes_initialized'}),flush=True)
 start('tunnel',ssh[:-1]+['-N','-o','ExitOnForwardFailure=yes','-L','127.0.0.1:45656:127.0.0.1:44656','-L','127.0.0.1:45679:127.0.0.1:44677','-R','127.0.0.1:45666:127.0.0.1:43656','-R','127.0.0.1:43660:127.0.0.1:43660',ssh[-1]])
 for i in range(2):start(f'local-node{i}',[sys.executable,'-m','neuroshard.inference.node','run','--home',str(local/f'node{i}')])
 for i in range(3):start(f'remote-node{i}',[python,'-m','neuroshard.inference.node','run','--home',str(rbase/f'node{i}')],True)
 rpc='http://127.0.0.1:43657';rrpc='http://127.0.0.1:45679'
 wait_for(lambda:wire.query(rpc,'/summary')['height']>=2)
 wait_for(lambda:wire.query(rrpc,'/summary')['height']>=2)
 worker_key=json.loads(rrun(f'import json;from pathlib import Path;print(json.dumps(json.loads(Path({str(rbase/"node2/node.json")!r}).read_text())["public_key"]))'))
 report['worker_public_key']=worker_key;report['worker_balance_before']=wire.query(rpc,'/account',{'public_key':worker_key})['balance']
 start('sponsor',[sys.executable,'-m','neuroshard.publicnet.worker_cli','sponsor','--home',str(local/'node0'),'--tasks','3','--port','43660','--wait-seconds','0','--interval-seconds','2'])
 start('stage0',[sys.executable,'-m','neuroshard.publicnet.worker_cli','worker','--home',str(local/'node0'),'--stage','0','--coordinator','http://127.0.0.1:43660'])
 start('stage1',[python,'-m','neuroshard.publicnet.worker_cli','worker','--home',str(rbase/'node2'),'--stage','1','--coordinator','http://127.0.0.1:43660'],True)
 print(json.dumps({'phase':'workers_started','worker':worker_key}),flush=True)
 # Sponsor intentionally exits after its bounded three tasks; remove that process from liveness checks.
 sponsor=children[-3];children.remove(sponsor)
 try:wait_for(lambda:wire.query(rpc,'/summary')['round']==3,420)
 finally:children.append(sponsor)
 report['training_summary']=wire.query(rpc,'/summary');report['worker_balance_earned']=wire.query(rpc,'/account',{'public_key':worker_key})['balance']
 assert report['worker_balance_before']==0 and report['worker_balance_earned']==1200000
 children.remove(sponsor)
 print(json.dumps({'phase':'training_settled','round':3,'worker_balance':report['worker_balance_earned']}),flush=True)
 provider_key=json.loads(rrun(f'import json;from pathlib import Path;print(json.dumps(json.loads(Path({str(rbase/"node1/node.json")!r}).read_text())["public_key"]))'))
 start('provider',[python,'-m','neuroshard.inference.provider','--home',str(rbase/'node1')],True)
 code=f'''import json
from pathlib import Path
from neuroshard.demo import client as wire,protocol
rpc="http://127.0.0.1:44677"
identity=protocol.Identity.load_or_create(Path({str(rbase/'node2/account.key')!r}))
s=wire.query(rpc,"/summary");nonce=wire.query(rpc,"/account",{{"public_key":identity.public_key}})["nonce"]
tx=identity.sign({{"kind":"infer","chain_id":s["chain_id"],"nonce":nonce,"provider":{provider_key!r},"model_root":s["serving_root"],"request":{{"prompt":"What is the capital of France?","max_tokens":32}},"price":32000,"expires":s["height"]+120}})
result=wire.broadcast(rpc,tx)
print(json.dumps({{"request_id":protocol.transaction_id(tx),"submitted":result}}))'''
 submitted=json.loads(rrun(code));report['request']=submitted
 def completed():
  result=wire.query(rpc,'/job',{'id':submitted['request_id']});return result if result['status']=='completed' else None
 report['inference']=wait_for(completed,180)
 report['worker_balance_after_inference']=wire.query(rpc,'/account',{'public_key':worker_key})['balance']
 assert report['worker_balance_after_inference']==1167000
 assert 'Paris' in report['inference']['output']['text']
 assert report['inference']['model_root']==report['training_summary']['serving_root']
 assert report['training_summary']['last_training_evaluation']['promoted']
 assert report['training_summary']['serving_root']!=json.loads((local/'network/genesis.json').read_text())['app_state']['manifest']['initial_model_root']
 common=min(wire.query(rpc,'/summary')['height'],wire.query(rrpc,'/summary')['height'])
 a=wire.rpc(rpc,'block',{'height':str(common)});b=wire.rpc(rrpc,'block',{'height':str(common)})
 assert a['block_id']['hash']==b['block_id']['hash']
 report['common_block']={'height':common,'hash':a['block_id']['hash']}
 report['passed']=True
 print(json.dumps({'phase':'passed','inference':report['inference'],'common_block':report['common_block']}),flush=True)
finally:
 import signal
 # Stop the exact remote run explicitly; closing an SSH channel is not process supervision.
 pattern='^'+python+' -m neuroshard\\.(inference\\.(node|provider)|publicnet\\.worker_cli) .*--home '+str(rbase)+'/'
 subprocess.run(ssh+['pkill -TERM -f '+shlex.quote(pattern)+' || true'],timeout=30)
 for child in reversed(children):
  if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
 for child in children:
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 (local/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 (root/'latest-network-run.txt').write_text(str(local)+'\n')
