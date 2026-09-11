#!/usr/bin/env python3
"""Reproduce the bounded response-from-seed experiment on three worker endpoints.

Uses a fixed 32-step plan, immutable source revisions and protected examples.
A historical selection repeats old evidence; omit it for a new exploratory draw.
"""
import argparse,json,secrets,time
from pathlib import Path
from neuroshard.evolution.data import Corpus
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.transport import Endpoint
from neuroshard.evolution.pipeline import Pipeline,validate_record
from neuroshard.evolution.batches import from_windows
from neuroshard.evolution.evaluation import decide,comparison
from neuroshard.evolution.worker import replay_trace
from transformers import AutoTokenizer
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--home',type=Path,required=True)
parser.add_argument('--model-dir',type=Path,required=True)
parser.add_argument('--workers-config',type=Path,required=True)
parser.add_argument('--plan',type=Path,required=True)
parser.add_argument('--upstream-cache',type=Path)
parser.add_argument('--historical-selection',type=Path)
args=parser.parse_args()
home=args.home.resolve();home.mkdir(parents=True,exist_ok=True)
folder=home/'response-from-seed';folder.mkdir(exist_ok=True)
plan=json.loads(args.plan.read_bytes())
if (plan['steps'],plan['batch_rows'],plan['sequence_length'],plan['response_tokens'],plan['context_tokens'],
    plan['source_train_documents'],plan['source_test_documents'],plan['evaluation_examples_per_role'],
    plan['clip_norm'],plan['fresh_min_gain'],plan['retention_margin'],plan['z'],plan['roles']) !=     (32,2,128,64,64,384,768,128,1.,.001,.02,2.576,['retention','fresh','test']):
    raise ValueError('This script reproduces the recorded bounded experiment; use the epoch controller for other designs')
saved=home/'response-from-seed-plan.json'
if saved.exists() and json.loads(saved.read_bytes())!=plan:
    raise ValueError('An existing experiment cannot change its plan')
saved.write_text(json.dumps(plan,indent=2)+'\n')
store=Objects(home/'objects')
from neuroshard.evolution.model import from_pretrained
initial,_=from_pretrained(args.model_dir,store)
if initial!=plan['initial_model_root']:raise ValueError('Plan differs from pinned initial model')
corpus_home=folder/'corpus';corpus_home.mkdir(exist_ok=True)
if args.upstream_cache and not (corpus_home/'upstream').exists():(corpus_home/'upstream').symlink_to(args.upstream_cache.resolve(),target_is_directory=True)
tokenizer=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True,trust_remote_code=False)
corpus=Corpus(corpus_home,store,tokenizer,128,target_mode='response')
source={'repo':'HuggingFaceTB/smol-smoltalk','revision':'f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc','license':'Apache-2.0'}
train=corpus.register({**source,'split':'train','role':'train'},initial_cursor=plan['source_train_start'])
heldout=corpus.register({**source,'split':'test','role':'heldout'},initial_cursor=plan['source_test_start'])
data_path=folder/'data.json'
if not data_path.exists():
 replay=corpus.collect(train,128);fresh=corpus.collect(train,256);protected=corpus.collect(heldout,768)
 windows=corpus.training(fresh['root'],64,'neuroshard/response-from-seed/v1')
 data_path.write_text(json.dumps({'replay_window':replay['root'],'fresh_window':fresh['root'],'heldout_window':protected['root'],'training_windows':windows},indent=2)+'\n')
data=json.loads(data_path.read_bytes())
workers=json.loads(args.workers_config.read_bytes())['workers'][:3]
if len(workers)!=3:raise ValueError('Three worker endpoints are required')
endpoints=[Endpoint(w['url'],(args.workers_config.resolve().parent/Path(w['token_file']).expanduser()).read_text().strip(),store) for w in workers]
pipe=Pipeline(store,plan['initial_model_root'],endpoints,[48000000]*3,'response-from-seed',learning_rate=plan['learning_rate'],clip_norm=plan['clip_norm'],journal=folder/'journal.json')
for i in range(pipe.step,plan['steps']):
 result=pipe.train(from_windows(store,data['training_windows'][i*2:i*2+2]));validate_record(store,result['record_root'])
 with (folder/'steps.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
 print('train',i+1,float.fromhex(result['loss_hex']),result['elapsed_seconds'],result['model_root'],flush=True)
candidate=pipe.model_root;pipe.close()
(folder/'candidate.json').write_text(json.dumps({'model_root':candidate,'initial':plan['initial_model_root'],'plan':plan},indent=2)+'\n')
last=json.loads((folder/'steps.jsonl').read_text().splitlines()[-1]);audits=[]
for trace in last['traces']:
 started=time.monotonic();result=replay_trace(store,trace);result['elapsed_seconds']=time.monotonic()-started;audits.append(result);assert result['valid'];print('audit',result,flush=True)
(folder/'audits.json').write_text(json.dumps(audits,indent=2)+'\n')
selection_path=folder/'selection.json'
if not selection_path.exists():
 selection=json.loads(args.historical_selection.read_bytes()) if args.historical_selection else {'candidate':candidate,'baseline':plan['initial_model_root'],'beacon':secrets.token_hex(32)}
 if selection['candidate']!=candidate or selection['baseline']!=plan['initial_model_root']:
  raise ValueError('Historical selection differs from computed checkpoints')
 # Reconstruct reservations locally from the recorded beacon.
 selection.pop('reservations',None)
 selection_path.write_text(json.dumps(selection,indent=2)+'\n')
selection=json.loads(selection_path.read_bytes());selection.setdefault('reservations',{})
if selection['candidate']!=candidate or selection['baseline']!=plan['initial_model_root']:
 raise ValueError('Saved selection belongs to another pair of models')
if args.historical_selection:
 historical=json.loads(args.historical_selection.read_bytes())
 if any(selection[k]!=historical[k] for k in ('candidate','baseline','beacon')):
  raise ValueError('Historical selection differs from the saved experiment')
for role in plan['roles']:
 if role not in selection['reservations']:
  selection['reservations'][role]=corpus.reserve_evaluation(candidate,role,plan['evaluation_examples_per_role'],selection['beacon'])
  selection_path.write_text(json.dumps(selection,indent=2)+'\n')
values={}
for name,root in [('baseline',plan['initial_model_root']),('candidate',candidate)]:
 path=folder/f'evaluation-{name}.json'
 if path.exists():
  values[name]=json.loads(path.read_bytes())
  if values[name]['model_root']!=root:raise ValueError('Saved evaluation belongs to another model')
  continue
 pipe=Pipeline(store,root,endpoints,[48000000]*3,'response-from-seed-eval-'+name)
 measured={'model_root':root}
 for role,reservation in selection['reservations'].items():
  sequences=store.json(reservation)['sequences'];losses=[]
  for offset in range(0,len(sequences),4):
   result=pipe.evaluate(from_windows(store,sequences[offset:offset+4]));losses.extend(float.fromhex(v) for v in result['losses_hex'])
   if (offset+4)%16==0:print('evaluation',name,role,offset+4,len(sequences),flush=True)
  measured[role]=losses;print('MEAN',name,role,sum(losses)/len(losses),flush=True)
 path.write_text(json.dumps(measured,indent=2)+'\n');values[name]=measured;pipe.close()
a,b=values['baseline'],values['candidate']
decision=decide(a['retention'],b['retention'],a['fresh'],b['fresh'],retention_margin=plan['retention_margin'],min_gain=plan['fresh_min_gain'])
decision['test']=comparison(a['test'],b['test'])
(folder/'decision.json').write_text(json.dumps(decision,indent=2)+'\n');print('DECISION',json.dumps(decision),flush=True)
