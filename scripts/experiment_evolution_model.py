#!/usr/bin/env python3
"""Check the real seed, identity growth and a fully trainable model-parallel step.

Requires the pinned numerical dependencies. Local workers share one process;
--workers-config selects separately running authenticated worker endpoints.
Every invocation needs a new home, so results and worker operation IDs cannot
silently mix with an earlier run. This does not change a public network.
"""
import argparse
import gc
import json
import secrets
import time
from pathlib import Path

from neuroshard.evolution.model import Shard,from_pretrained,grow,place,torch
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.pipeline import Pipeline,LocalEndpoint,validate_record
from neuroshard.evolution.worker import Worker,replay_trace
from neuroshard.evolution.runtime import check


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--workers-config',type=Path)
    args=parser.parse_args()
    profile=check()
    args.home.mkdir(parents=True,exist_ok=False)
    store=Objects(args.home/'objects')
    root,model=from_pretrained(args.model_dir,store)
    ids=torch.tensor([[1,100,200,300,400,500,2]])
    def logits(manifest):
        shards=[Shard(manifest,p,store) for p in place(manifest,[48000000]*4)]
        with torch.no_grad():
            hidden=shards[0](ids,ids=True)
            for shard in shards[1:]:hidden=shard(hidden)
            result=shards[0].logits(hidden)
        del shards
        gc.collect()
        return result
    actual=logits(model)
    from transformers import AutoModelForCausalLM
    reference=AutoModelForCausalLM.from_pretrained(args.model_dir,local_files_only=True,
        trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').eval()
    with torch.no_grad():expected=reference(ids).logits
    if not torch.equal(actual,expected):
        raise AssertionError('Custom seed logits differ from the pinned reference')
    del reference,expected
    gc.collect()
    grown_root,grown=grow(root,store,4)
    if not torch.equal(actual,logits(grown)):
        raise AssertionError('Identity growth changed the tested initial logits')
    try:place(grown,[48000000]*3)
    except ValueError:pass
    else:raise AssertionError('Expected growth to require a fourth declared worker allocation')
    config=json.loads(args.workers_config.read_bytes()) if args.workers_config else None
    if config:
        from neuroshard.evolution.transport import Endpoint
        entries=config['workers'][:3]
        endpoints=[Endpoint(w['url'],(args.workers_config.resolve().parent/Path(w['token_file']).expanduser()).read_text().strip(),store) for w in entries]
    else:
        endpoints=[LocalEndpoint(Worker(args.home/f'worker{i}',store)) for i in range(3)]
    pipe=Pipeline(store,root,endpoints,[48000000]*3,'conformance-'+secrets.token_hex(8),journal=args.home/'training.json')
    try:
        record=pipe.train(ids.tolist())
        start=time.monotonic()
        compact=validate_record(store,record['record_root'])
        compact['elapsed_seconds']=time.monotonic()-start
    finally:pipe.close()
    audits=[]
    for trace in record['traces']:
        start=time.monotonic()
        result=replay_trace(store,trace)
        if not result['valid']:raise AssertionError(result)
        result['elapsed_seconds']=time.monotonic()-start
        audits.append(result)
    result={'profile':profile,'seed_root':root,'grown_root':grown_root,
            'seed_parameters':model['parameters'],'grown_parameters':grown['parameters'],
            'reference_logits_equal':True,'growth_logits_equal':True,
            'worker_transport':'http' if config else 'local-single-process',
            'training':record,'compact_check':compact,'audits':audits}
    (args.home/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
