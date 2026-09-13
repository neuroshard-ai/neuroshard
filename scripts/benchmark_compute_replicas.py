#!/usr/bin/env python3
"""Measure a fixed inference workload across one or two private GPU replicas."""
import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

from neuroshard.evolution import reference_data as data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home',type=Path,required=True)
    parser.add_argument('--endpoint',action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--model-digest',required=True,help='Expected parameter digest from the selected training result')
    parser.add_argument('--requests',type=int,default=32)
    parser.add_argument('--concurrency',type=int,default=2)
    parser.add_argument('--timeout',type=float,default=60)
    parser.add_argument('--attempts',type=int,default=2)
    parser.add_argument('--allow-unavailable',action='store_true')
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve previous benchmark output')
    if not 1<=len(args.endpoint)<=2 or not 1<=args.concurrency<=8 or not 1<=args.attempts<=2:
        raise ValueError('Benchmark exceeds bounded replica/concurrency/retry limits')
    prepared=json.loads((args.home/'prepared.json').read_bytes())
    records=data.read_records(args.home/'inputs/dev.jsonl',prepared['roles']['dev']['sha256'])
    if not 1<=args.requests<=len(records) or not 0<args.timeout<=60:raise ValueError('Invalid benchmark bounds')
    healthy=[];unavailable=[]
    for endpoint in args.endpoint:
        try:
            with urllib.request.urlopen(endpoint+'/health',timeout=min(5,args.timeout)) as r:healthy.append(json.load(r))
        except (OSError,ValueError) as error:
            unavailable.append({'endpoint':endpoint,'error':type(error).__name__})
            if not args.allow_unavailable:raise
    if not healthy or len({h['model_digest'] for h in healthy})!=1:raise ValueError('Replicas must expose one identical model')
    digest=healthy[0]['model_digest']
    if digest!=args.model_digest:raise ValueError('Serving model differs from the selected candidate')
    def request(index):
        payload={'request_id':f'{args.run_id}-{index}','task_id':records[index]['id']}
        attempts=[];started=time.monotonic()
        for attempt in range(args.attempts):
            endpoint=args.endpoint[(index+attempt)%len(args.endpoint)]
            try:
                req=urllib.request.Request(endpoint+'/infer',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(req,timeout=args.timeout) as response:answer=json.load(response)
                if (answer['model_digest']!=digest or answer['task_id']!=payload['task_id']
                        or answer['request_id']!=payload['request_id']):raise ValueError('Response identity mismatch')
                return {**payload,'success':True,'seconds':time.monotonic()-started,'endpoint':endpoint,
                        'attempts':attempts+[{'endpoint':endpoint,'success':True}],'answer':answer}
            except (OSError,ValueError) as error:
                attempts.append({'endpoint':endpoint,'success':False,'error':type(error).__name__})
        return {**payload,'success':False,'seconds':time.monotonic()-started,'attempts':attempts}
    started=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:results=list(pool.map(request,range(args.requests)))
    seconds=time.monotonic()-started
    latencies=sorted(r['seconds'] for r in results)
    report={'run_id':args.run_id,'model_digest':digest,'endpoints':args.endpoint,'unavailable':unavailable,
            'requests':args.requests,'concurrency':args.concurrency,'seconds':seconds,
            'successful':sum(r['success'] for r in results),'requests_per_second':sum(r['success'] for r in results)/seconds,
            'median_latency_seconds':statistics.median(latencies),'p95_latency_seconds':latencies[max(0,__import__('math').ceil(.95*len(latencies))-1)],
            'results':results,'scope':'Fixed committed tasks, trusted operated replicas; no payments or permissionless admission'}
    data.save(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='results'}))


if __name__=='__main__':main()
