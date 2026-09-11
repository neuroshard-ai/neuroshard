#!/usr/bin/env python3
"""Real-model forward/generation conformance, optionally across worker hosts.

The hand-written examples test execution and text identity, not model quality.
Use --verify-result on a second host with the copied content-addressed objects.
"""
import argparse
import json
import secrets
import time
from pathlib import Path

from neuroshard.dataflow.store import canonical
from neuroshard.evolution import forward
from neuroshard.evolution.app import code_hash
from neuroshard.evolution.batches import from_windows
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.pipeline import Pipeline,LocalEndpoint
from neuroshard.evolution.model import place
from neuroshard.evolution.runtime import check
from neuroshard.evolution.text import TextCodec
from neuroshard.evolution.verification import Metadata,audit
from neuroshard.evolution.worker import Worker


def verify(store,result):
    if result['source_hash'] != code_hash():
        raise ValueError('Conformance source changed')
    runtime = check()
    codec = TextCodec.load(store,result['tokenizer_root'])
    codec.check_model(store.json(result['model_root']))
    replays = []
    for name in ('evaluation','generation'):
        key = result[name]['record_root']
        metadata = Metadata(forward.bundle(store,key))
        if name == 'evaluation':forward.validate(metadata,key)
        else:forward.validate_generation(metadata,key)
        for stage in range(len(forward.trace_roots(metadata,key))):
            started = time.monotonic()
            verdict = audit(store,metadata,key,stage)
            if not verdict['valid']:
                raise ValueError('Forward replay mismatch: '+verdict['mismatch'])
            replays.append({'kind':name,'stage':stage,'seconds':time.monotonic()-started,'valid':True})
        print(json.dumps({'phase':'replayed','kind':name,'stages':sum(r['kind']==name for r in replays)}),flush=True)
    return {'source_hash':code_hash(),'runtime':runtime,'replays':replays,'all_valid':True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home',type=Path,required=True)
    parser.add_argument('--objects',type=Path,required=True)
    parser.add_argument('--model-root')
    parser.add_argument('--workers-config',type=Path)
    parser.add_argument('--verify-result',type=Path)
    args = parser.parse_args()
    args.home.mkdir(parents=True,exist_ok=False)
    if args.verify_result:
        store = Objects(args.objects)
        result = verify(store,json.loads(args.verify_result.read_bytes()))
        (args.home/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        return
    if not args.model_root:
        raise ValueError('Supply --model-root for a new conformance run')
    source = Objects(args.objects)
    store = Objects(args.home/'objects')
    def fetch(key):
        try:return source.get(key)
        except FileNotFoundError:return None
    store.fetchers.append(fetch)
    codec = TextCodec.load(store,store.json(args.model_root)['tokenizer_root'])
    model = store.json(args.model_root)
    codec.check_model(model)
    capacities = [48000000]*3
    count = len(place(model,capacities))
    if args.workers_config:
        from neuroshard.evolution.transport import Endpoint
        workers = json.loads(args.workers_config.read_bytes())['workers'][:count]
        if len(workers) != count:raise ValueError('Not enough worker endpoints')
        endpoints = [Endpoint(w['url'],(args.workers_config.resolve().parent/Path(w['token_file']).expanduser()).read_text().strip(),store) for w in workers]
    else:
        endpoints = [LocalEndpoint(Worker(args.home/f'worker{i}',store)) for i in range(count)]
    windows = []
    for question,answer in [('What is the capital of France?','Paris.'),('What is 2 + 2?','4.'),
                            ('Translate hello into French.','Bonjour.'),('Name a primary color.','Red.')]:
        prepared = codec.response_windows([{'role':'user','content':question},{'role':'assistant','content':answer}])
        if prepared['truncated'] or len(prepared['windows']) != 1:
            raise ValueError('Conformance example unexpectedly requires multiple windows')
        windows.append(store.put_json(prepared['windows'][0]))
    result = {'source_hash':code_hash(),'model_root':args.model_root,'tokenizer_root':codec.root,
              'parameters':model['parameters'],'runtime':check(),'quality_improvement_claimed':False,
              'worker_transport':'http' if args.workers_config else 'local-single-process'}
    pipe = Pipeline(store,args.model_root,endpoints,capacities,'forward-profile-'+secrets.token_hex(8))
    try:
        before = time.monotonic()
        result['evaluation'] = pipe.evaluate_record(from_windows(store,windows,codec.root))
        result['evaluation_seconds'] = time.monotonic()-before
        before = time.monotonic()
        prompt = codec.prompt([{'role':'user','content':'What is the capital of France?'}])
        result['generation'] = pipe.generate_record(prompt,8,[codec.tokenizer.eos_token_id])
        result['generation_seconds'] = time.monotonic()-before
        result['text'] = codec.tokenizer.decode(result['generation']['token_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)
    finally:
        pipe.close()
    # Fetch all graph JSON before writing a portable result and replay inputs.
    for name in ('evaluation','generation'):
        metadata = Metadata(forward.bundle(store,result[name]['record_root']))
        result[name+'_metadata_bytes'] = len(canonical(metadata.values))
    result['local_verification'] = verify(store,result)
    (args.home/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'phase':'complete','parameters':result['parameters'],'text':result['text'],
                      'evaluation_seconds':result['evaluation_seconds'],'generation_seconds':result['generation_seconds'],
                      'replay_stages':len(result['local_verification']['replays'])}),flush=True)


if __name__ == '__main__':main()
