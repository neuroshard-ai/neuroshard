"""Fit only training inputs; compare two declared selectors on opened questions."""
import argparse
import ast
import json
import math
from pathlib import Path
import time
import numpy as np
from neuroshard.evolution import expert_router as er
from neuroshard.evolution.reference_data import identity,save,sha256


def run(home, training, labels, reference):
    namespace={'np':np,'er':er,'identity':identity,'math':math}
    tree=ast.parse(reference.read_text())
    functions=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'prototypes_fast','fit_fast'}],type_ignores=[])
    exec(compile(functions,str(reference),'exec'),namespace)
    features={r['id']:r['features'] for r in json.loads((home/'features.json').read_bytes())}
    samples=[{**r,'features':features[r['id']]} for r in json.loads(training.read_bytes())]
    observations=[r for r in json.loads(labels.read_bytes()) if 'canonical' not in r]
    profile=json.loads((home/'profile.json').read_bytes())
    recipe={'format':'neuroshard-semantic-selector-fit-v1','training':identity(samples),'observations':identity(observations),
        'encoder':identity(profile),'epochs':24,'balance_classes':True,'prototypes_per_class':4,'iterations':12,
        'support_numerator':5,'support_denominator':4,'training_margin':8388608,
        'comparison':['nearest-training-question','averaged-margin-perceptron'],
        'reference':sha256(reference),'driver':sha256(__file__),'new_final':False,'threshold_search':False}
    save(home/'fit-prescription.json',recipe);start=time.monotonic()
    prototype=namespace['prototypes_fast'](samples,identity(profile),identity({k:v for k,v in profile['files'].items() if k.startswith('tokenizer') or k in {'vocab.txt','special_tokens_map.json'}}))
    model=namespace['fit_fast'](samples,prototype);save(home/'classifier.json',model)
    x=np.asarray([r['features'] for r in samples],dtype=np.int64);names=sorted(model['prototypes'])
    weights=np.asarray([model['classifier']['weights'][n] for n in names]);biases=np.asarray([model['classifier']['biases'][n] for n in names])
    rows=[]
    for row in observations:
        f=np.asarray(features[row['id']],dtype=np.int64)
        distance=((x-f)**2).sum(axis=1);nearest=min(range(len(samples)),key=lambda i:(int(distance[i]),samples[i]['id']))
        logits=weights@f+biases*er.SCALE;order=sorted(range(len(names)),key=lambda i:(-int(logits[i]),names[i]));best=names[order[0]]
        support=min(int(np.sum((f-np.asarray(center))**2)) for center in model['prototypes'][best])
        confident=support<=model['maximum_distance'] and logits[order[0]]-logits[order[1]]>model['minimum_margin']
        rows.append({'id':row['id'],'question':row['question'],'expected':row['expected'],'nearest':samples[nearest]['route'],
            'nearest_training_id':samples[nearest]['id'],'classifier':best if confident else 'parent'})
    result={method:{'new_correct':sum(r[method]==r['expected'] for r in rows if r['expected']!='parent'),
        'new_count':sum(r['expected']!='parent' for r in rows),
        'false_positives':sum(r[method]!='parent' for r in rows if r['expected']=='parent'),
        'other_count':sum(r['expected']=='parent' for r in rows)} for method in ('nearest','classifier')}
    result.update(seconds=time.monotonic()-start,recipe=identity(recipe),new_final=False)
    save(home/'fit-scores.json',rows);save(home/'fit-result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('home','training','labels','reference'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.home,a.training,a.labels,a.reference)
