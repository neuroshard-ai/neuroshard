#!/usr/bin/env python3
"""Test question matching on frozen owned shards, without answers or training.

This development probe is not the ordinary serving path. The scorer keeps
intent labels and answers locally; workers receive candidate questions only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

from neuroshard.evolution.reference_data import identity, save, sha256

ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = ('Match question meanings. Select the candidate asking for the same fact as the query. '
    'Match the subject, quantity, action and condition, not just shared words. '
    'If none asks for the same fact, select 0. Do not answer any question. '
    'Return ONLY the candidate number, with no explanation.')


def messages(question, choices):
    def payload(q, candidates):
        return json.dumps({'query': q, 'candidates': {str(i+1): c for i,c in enumerate(candidates)}}, ensure_ascii=False)
    examples = [
        ('What is the highest permitted temperature for the cooling unit?',
         ['What is the minimum temperature of the cooling unit?', 'What is the maximum operating temperature of the cooler?'], '2'),
        ('What color is the cooling unit?',
         ['What is the minimum temperature of the cooling unit?', 'What is the maximum operating temperature of the cooler?'], '0'),
        ('Which command records a reviewers approval?',
         ['Which command creates a review request?', 'Which command submits a review vote?'], '2')]
    result = [{'role':'system','content':INSTRUCTION}]
    for q,c,a in examples:
        result.extend([{'role':'user','content':payload(q,c)},{'role':'assistant','content':a}])
    return result+[{'role':'user','content':payload(question,choices)}]


def prepare(home, campaign, diagnostic, pilot):
    import numpy as np
    home.mkdir(parents=True,exist_ok=False)
    model=json.loads((pilot/'model.json').read_bytes())
    facts=json.loads((campaign/'compiled/source-catalog.json').read_bytes())['cohorts']['admission']
    canonical={f['id']:f['training'][0] for f in facts}
    names=sorted(canonical)
    w=np.asarray([model['classifier']['weights'][n] for n in names],dtype=np.int64)
    bias=np.asarray([model['classifier']['biases'][n] for n in names],dtype=np.int64)
    measured=json.loads((diagnostic/'answers.json').read_bytes())['result']
    metadata={r['id']:r for p in (diagnostic/'inputs').iterdir() for line in p.read_bytes().splitlines() if (r:=json.loads(line))}
    executions=measured['executions']+[r for rs in measured['retention']['roles'].values() for r in rs]
    probes, labels, seen=[],[],set()
    for row in executions:
        spec=metadata[row['id']]; body=row['after']['answering']; routing=body['routing']
        if len(spec['messages'])!=2 or len(routing)!=len(spec['topics']):continue
        if len(routing)>1 and body['planning']['path']!='explicit':continue
        for observed,topic in zip(routing,spec['topics']):
            key=identity(observed['question'])
            if key in seen:continue
            seen.add(key)
            logits=w@np.asarray(observed['features'],dtype=np.int64)+bias*16384
            chosen=sorted(range(len(names)),key=lambda i:(-int(logits[i]),names[i]))[:6]
            candidates=[names[i] for i in chosen]
            expected=topic if topic in canonical else 'parent'
            probes.append({'id':key,'model':'interpreter','purpose':'planning',
                'messages':messages(observed['question'],[canonical[n] for n in candidates]),'max_tokens':8})
            labels.append({'id':key,'question':observed['question'],'candidates':candidates,'expected':expected,
                'retrieved':expected=='parent' or expected in candidates})
    # Separate forced canonical controls measure the neural knowledge independently.
    for fact in facts:
        key=identity({'canonical':fact['id']})
        probes.append({'id':key,'model':'admission','purpose':'answer','max_tokens':96,
            'messages':[{'role':'user','content':'NeuroShard research protocol: '+canonical[fact['id']]+' Provide only the answer.'}]})
        labels.append({'id':key,'canonical':fact['id'],'expected':fact['answer']})
    plan={'format':'neuroshard-question-match-probe-v1','source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        'driver':sha256(__file__),'graph':identity(json.loads((diagnostic/'after-graph.json').read_bytes())),
        'profile':identity(json.loads((diagnostic/'profile.json').read_bytes())),
        'classifier':identity(model),'probes':identity(probes),'labels':identity(labels),'max_seconds':1200,
        'new_final':False,'training':False,'native_promotion':False,'scope':'Opened development diagnostic; question-only matching plus separate forced knowledge controls.'}
    for name,value in [('plan',plan),('probes',probes),('labels',labels)]:save(home/(name+'.json'),value)
    print(json.dumps({'probes':len(probes),'selection':len(labels)-len(facts),'retrieval_misses':sum(not r.get('retrieved',True) for r in labels),'plan':identity(plan)}),flush=True)


def worker(config):
    import torch.distributed as dist
    from neuroshard.evolution.sharded.graph_execution import GraphNetwork
    from neuroshard.evolution.sharded.planned_graph import PlannedGraphNetwork
    read=lambda p:json.loads(Path(p).read_bytes())
    plan,probes=read(config['plan']),read(config['probes'])
    graph,profile=read(config['graph']),read(config['profile'])
    if (sha256(__file__)!=plan['driver'] or identity(probes)!=plan['probes']
            or identity(graph)!=plan['graph'] or identity(profile)!=plan['profile']):
        raise ValueError('Probe source or numerical inputs changed')
    rank=int(os.environ['RANK']);output=Path(config['output']);output.mkdir(parents=True,exist_ok=False)
    dist.init_process_group('gloo',timeout=timedelta(seconds=300))
    started=time.monotonic()
    try:
        net=GraphNetwork(graph,profile,objects=Path(config['objects']),interpreter=Path(config['interpreter']),
            seed=Path(config['seed']),source_home=Path(config['source_home']),rank=rank)
        # Deliberately call a known model for diagnosis, not automatic serving.
        service=object.__new__(PlannedGraphNetwork);service.net=net;service.config={}
        results=[]
        for probe in probes:
            if time.monotonic()-started>plan['max_seconds']:raise TimeoutError('Question matching probe exceeded its deadline')
            service.trace=[]
            text=service.call(probe['model'],probe['messages'],probe['max_tokens'],probe['purpose'])
            results.append({'id':probe['id'],'text':text,'outputs':service.trace})
            save(output/'responses.json',results)
        service.trace=[];p=probes[0]
        repeated=service.call(p['model'],p['messages'],p['max_tokens'],p['purpose'])
        exact={'id':p['id'],'text':repeated,'outputs':service.trace}==results[0]
        net.verify_unchanged()
        value={'plan':identity(plan),'responses':identity(results),'exact_replay':exact}
        if net.all_owners.exchange(value)!=[value]*net.world_size:raise ValueError('Probe owners disagreed')
        save(output/'result.json',{**value,'seconds':time.monotonic()-started})
    finally:dist.destroy_process_group()


def run(home,campaign,diagnostic):
    from ordinary_cloud import Cloud,REMOTE,REPO
    read=lambda name:json.loads((home/(name+'.json')).read_bytes())
    plan,probes,labels=map(read,('plan','probes','labels'))
    if sha256(__file__)!=plan['driver'] or identity(labels)!=plan['labels']:raise ValueError('Commit the frozen probe before execution')
    cloud=Cloud(campaign);previous=json.loads((diagnostic/'service.json').read_bytes())
    key=uuid.uuid4().hex;folder=REMOTE+'/question-match/'+key
    units=['neuroshard-match-'+key[:16]+'-'+str(rank) for rank in range(len(previous['placement']))]
    installed=[]
    try:
        for rank,physical in enumerate(previous['placement']):
            config=json.loads(cloud.read(physical,previous['folder']+'/config-'+str(rank)+'.json'))
            config.update(plan=folder+'/plan.json',probes=folder+'/probes.json',output=folder+'/owner-'+str(rank))
            for name,value in [('plan',plan),('probes',probes),('config-'+str(rank),config)]:cloud.put(physical,folder+'/'+name+'.json',value)
            installed.append((rank,physical))
        with ThreadPoolExecutor(max_workers=7) as pool:
            list(pool.map(lambda rp:cloud.start(rp[1],units[rp[0]],
                [REPO+'/scripts/run_question_match_probe.py','--worker',folder+'/config-'+str(rp[0])+'.json'],
                rank=rp[0],world=len(installed),port=31013),installed))
        save(home/'resources.json',{'units':units,'placement':previous['placement'],'folder':folder})
        deadline=time.monotonic()+min(plan['max_seconds']+120,cloud.remaining())
        while time.monotonic()<deadline:
            try:
                result=json.loads(cloud.read(0,folder+'/owner-0/result.json'));break
            except subprocess.CalledProcessError:
                if not cloud.active(0,units[0]):raise RuntimeError('Question matching worker stopped') from None
                time.sleep(2)
        else:raise TimeoutError('Question matching probe did not finish')
        responses=json.loads(cloud.read(0,folder+'/owner-0/responses.json'))
        lookup={r['id']:r for r in responses};scores=[]
        for label in labels:
            raw=lookup[label['id']]['text'].strip()
            if 'canonical' in label:
                scores.append({**label,'text':raw,'correct':raw.rstrip('.').casefold()==str(label['expected']).rstrip('.').casefold()})
            else:
                valid=re.fullmatch(r'[0-6]',raw) is not None
                selected=int(raw) if valid else -1
                topic='parent' if selected==0 else label['candidates'][selected-1] if selected>0 else 'invalid'
                scores.append({**label,'text':raw,'selected':topic,'correct':topic==label['expected']})
        summary={**result,'canonical':{'correct':sum(r['correct'] for r in scores if 'canonical'in r),'count':sum('canonical'in r for r in scores)},
            'selection':{'correct':sum(r['correct'] for r in scores if 'canonical'not in r),'count':sum('canonical'not in r for r in scores)},
            'false_positives':sum(r.get('expected')=='parent' and not r['correct'] for r in scores),'new_final':False,'automatic_serving':False}
        for name,value in [('responses',responses),('scores',scores),('result',summary)]:save(home/(name+'.json'),value)
        print(json.dumps(summary),flush=True)
    finally:
        for rank,physical in installed:cloud.command(physical,['sudo','systemctl','stop',units[rank]],timeout=45)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',type=Path);p.add_argument('--home',type=Path);p.add_argument('--campaign',type=Path)
    p.add_argument('--diagnostic',type=Path);p.add_argument('--pilot',type=Path);p.add_argument('--prepare',action='store_true')
    a=p.parse_args()
    if a.worker:worker(json.loads(a.worker.read_bytes()))
    elif a.prepare:prepare(a.home,a.campaign,a.diagnostic,a.pilot)
    else:run(a.home,a.campaign,a.diagnostic)
