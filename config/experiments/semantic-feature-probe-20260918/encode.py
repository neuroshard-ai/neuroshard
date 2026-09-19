"""Frozen inference-only feature extraction; no intent labels or answers."""
import argparse
import json
from pathlib import Path
import time

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer
from neuroshard.evolution.expert_router import normalize
from neuroshard.evolution.reference_data import identity,save,sha256

REVISION='5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'
MODEL='BAAI/bge-small-en-v1.5'


def run(home):
    plan=json.loads((home/'plan.json').read_bytes());rows=json.loads((home/'questions.json').read_bytes())
    if plan['driver']!=sha256(__file__) or plan['questions']!=identity(rows):raise ValueError('Feature probe source or inputs changed')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=snapshot_download(MODEL,revision=REVISION,cache_dir=home/'cache',
        allow_patterns=['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','model.safetensors','README.md'])
    tokenizer=AutoTokenizer.from_pretrained(source,trust_remote_code=False)
    model=AutoModel.from_pretrained(source,trust_remote_code=False,use_safetensors=True,attn_implementation='eager',torch_dtype=torch.float32).eval().to('cuda')
    profile={'model':MODEL,'revision':REVISION,'license':'MIT','files':{p.name:sha256(p) for p in Path(source).iterdir() if p.is_file()},
        'parameters':sum(p.numel() for p in model.parameters()),'pool':'CLS','quantization':'round-hidden-times-4096-then-integer-normalize-16384',
        'max_tokens':512,'instruction':'none','dtype':'float32','attention':'eager','batch':32,
        'torch':torch.__version__,'device':torch.cuda.get_device_name(0)}
    save(home/'profile.json',profile);outputs=[];started=time.monotonic()
    with torch.inference_mode():
        for start in range(0,len(rows),32):
            batch=rows[start:start+32];encoded=tokenizer([r['question'] for r in batch],padding=True,truncation=False,return_tensors='pt')
            if encoded['input_ids'].shape[1]>512:raise ValueError('A whole query exceeds the feature budget')
            hidden=model(**{k:v.to('cuda') for k,v in encoded.items()}).last_hidden_state[:,0,:]
            vectors=torch.round(hidden.cpu()*4096).to(torch.int64).tolist()
            outputs.extend({'id':r['id'],'features':normalize(v)} for r,v in zip(batch,vectors))
            save(home/'progress.json',{'done':len(outputs),'count':len(rows)})
    save(home/'features.json',outputs)
    save(home/'result.json',{'profile':identity(profile),'features':identity(outputs),'seconds':time.monotonic()-started,'count':len(outputs)})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--home',type=Path,required=True);run(p.parse_args().home)
