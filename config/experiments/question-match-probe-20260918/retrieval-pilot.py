import json, math, time
from pathlib import Path
import numpy as np
from neuroshard.evolution import expert_router as er
from neuroshard.evolution.reference_data import identity, save
P=Path('/home/ubuntu/neuroshard/.neuroshard/question-intent-pilot-20260918')

def fit_fast(samples, prototype, epochs=24, balance=True):
    rows=sorted(samples,key=lambda x:x['id']); names=sorted(prototype['prototypes']); d=prototype['dimensions']
    assert identity(rows)==prototype['training_root']
    if balance:
        groups={name:[r for r in rows if r['route']==name] for name in names}
        rows=[groups[n][i%len(groups[n])] for i in range(max(map(len,groups.values()))) for n in names]
    steps=epochs*len(rows)
    assert steps*er.SCALE**2*(d+1)<2**63 and steps**2*er.SCALE<2**63
    xs=np.asarray([r['features']+[er.SCALE] for r in rows],dtype=np.int64)
    ys=[names.index(r['route']) for r in rows]
    w=np.zeros((len(names),d+1),dtype=np.int64); a=np.zeros_like(w); step=0
    for epoch in range(epochs):
        for x,target in zip(xs,ys):
            step+=1; scores=w@x; scores[target]=np.iinfo(np.int64).min; rival=int(np.argmax(scores))
            if w[target]@x > scores[rival]+8388608:continue
            w[target]+=x;w[rival]-=x;a[target]+=(step-1)*x;a[rival]-=(step-1)*x
        if len(samples)>100:print('epoch',epoch+1,flush=True)
    avg=(step*w-a).tolist(); denom=math.isqrt(max(sum(v*v for v in row) for row in avg)<<64)
    scaled=[[er.rounded_ratio(v*(er.SCALE<<32),denom) for v in row] for row in avg]
    classifier={'method':'integer-balanced-averaged-margin-perceptron-v1' if balance else 'integer-averaged-margin-perceptron-v1','epochs':epochs,'training_margin':8388608,'weights':{n:r[:-1] for n,r in zip(names,scaled)},'biases':{n:r[-1] for n,r in zip(names,scaled)}}
    return er.validate({**prototype,'format':er.LINEAR_FORMAT,'classifier':classifier})

def prototypes_fast(samples, embedding, tokenizer):
    names=sorted({r['route'] for r in samples}); prototypes={}
    for name in names:
        rows=sorted([r for r in samples if r['route']==name],key=lambda r:r['id'])
        x=np.asarray([r['features'] for r in rows],dtype=np.int64); centers=[x[0]]
        def distances():return np.stack([np.sum((x-c)**2,axis=1) for c in centers],axis=1)
        for _ in range(min(4,len(rows))-1):
            dist=np.min(distances(),axis=1); best=int(np.flatnonzero(dist==dist.max())[-1])
            if any(np.array_equal(x[best],c) for c in centers):break
            centers.append(x[best])
        for _ in range(12):
            chosen=np.argmin(distances(),axis=1); updated=[]
            for i,c in enumerate(centers):
                cluster=x[chosen==i]; pooled=cluster.sum(axis=0)
                updated.append(np.asarray(er.normalize(pooled.tolist())) if len(cluster) and np.any(pooled) else c)
            if all(np.array_equal(a,b) for a,b in zip(centers,updated)):break
            centers=updated
        prototypes[name]=[c.tolist() for c in centers]
    radius=max(int(np.min(np.stack([np.sum((np.asarray([r['features'] for r in samples if r['route']==name],dtype=np.int64)-np.asarray(c))**2,axis=1) for c in centers],axis=1),axis=1).max()) for name,centers in prototypes.items())
    return er.validate({'format':er.FORMAT,'embedding_root':embedding,'tokenizer_root':tokenizer,'training_root':identity(sorted(samples,key=lambda r:r['id'])),'dimensions':len(samples[0]['features']),'fallback':'parent','minimum_margin':0,'maximum_distance':(radius*5+3)//4,'prototypes':prototypes})

# Check the optimized offline fit against the reference, including unequal class sizes.
small=[{'id':identity([n,i]),'features':[100+i*3,90-i, -100 if n=='x' else 100,40*i], 'route':n} for n,k in [('parent',7),('x',5),('y',3)] for i in range(k)]
proto=er.fit(small,embedding_root='a'*64,tokenizer_root='b'*64)
for balanced in (False,True):
 assert fit_fast(small,proto,epochs=4,balance=balanced)==er.fit_classifier(small,proto,epochs=4,balance_classes=balanced)
assert prototypes_fast(small,'a'*64,'b'*64)==er.calibrate_support(small,proto)
print('optimized reference equivalence passed',flush=True)
samples=json.loads((P/'samples.json').read_bytes())
guard=json.loads(Path('/home/ubuntu/neuroshard/.neuroshard/general-interface-repair-20260917b/guard.json').read_bytes())
start=time.monotonic()
if (P/'prototype.json').exists():prototype=json.loads((P/'prototype.json').read_bytes())
else:
 prototype=prototypes_fast(samples,guard['embedding_root'],guard['tokenizer_root'])
 save(P/'prototype.json',prototype)
print('prototypes complete',round(time.monotonic()-start,1),flush=True)
model=fit_fast(samples,prototype);save(P/'model.json',model)
q=Path('/home/ubuntu/neuroshard/.neuroshard/literal-question-diagnostic-20260918'); measured=json.loads((q/'answers.json').read_bytes())['result']
metadata={r['id']:r for p in (q/'inputs').iterdir() for line in p.read_bytes().splitlines() if (r:=json.loads(line))}
executions=measured['executions']+[r for rs in measured['retention']['roles'].values() for r in rs]
rows=[]
for row in executions:
 routing=row['after']['answering']['routing']; topics=metadata[row['id']]['topics']
 if len(routing)!=len(topics):continue
 for observed,topic in zip(routing,topics):
  decision=er.select(model,observed['features']); expected=topic if topic in model['prototypes'] else 'parent'
  rows.append({'id':row['id'],'question':observed['question'],'expected':expected,'chosen':decision['route'],'predicted':decision.get('predicted'),'confident':decision['confident'],'correct':expected==decision['route']})
result={'seconds':time.monotonic()-start,'model':identity(model),'admission':{'correct':sum(r['correct'] for r in rows if r['expected']!='parent'),'count':sum(r['expected']!='parent' for r in rows)},'other':{'false_positives':sum(not r['correct'] for r in rows if r['expected']=='parent'),'count':sum(r['expected']=='parent' for r in rows)},'scope':'Opened development only; train-only classifier; no new final; optimized exact integer offline fitting exceeds reference fitting operation cap.'}
save(P/'diagnostic.json',rows);save(P/'result.json',result);print(json.dumps(result),flush=True)
