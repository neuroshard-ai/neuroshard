#!/usr/bin/env python3
"""Execute the committed fresh assistant screen once, before new cohorts."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import uuid

from neuroshard.evolution import ordinary_quality
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.reference_data import identity, save, sha256
from ordinary_cloud import Cloud

ROOT = Path(__file__).resolve().parents[1]


def run(campaign, diagnostic, plan_file, home):
    relative = str(plan_file.resolve().relative_to(ROOT))
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    if (subprocess.check_output(['git', 'show', source+':'+relative], cwd=ROOT) != plan_file.read_bytes()
            or subprocess.check_output(['git', 'show', source+':scripts/run_general_holdout.py'], cwd=ROOT)
            != Path(__file__).read_bytes()):
        raise ValueError('Commit the exact screen and driver before opening it')
    previous = json.loads((diagnostic/'diagnostic-result.json').read_bytes())
    if previous['development_passed'] is not True:
        raise ValueError('Repair the exposed interface before opening this prospective screen')
    plan = json.loads(plan_file.read_bytes())
    graph = json.loads((diagnostic/'after-graph.json').read_bytes())
    baseline = json.loads((diagnostic/'before-graph.json').read_bytes())
    profile = json.loads((diagnostic/'profile.json').read_bytes())
    if (identity(graph) != plan['service_graph']
            or any(sha256(ROOT/path) != digest for path, digest in profile['sources'].items())):
        raise ValueError('The prospective screen requires the frozen complete method')
    for rows in plan['roles'].values():
        ordinary_quality.validate_rows(rows)
    home.mkdir(parents=True, exist_ok=False)
    store = Objects(campaign/'compiled/objects')
    initial = json.loads((campaign/'initial-job.json').read_bytes())
    quality = store.json(initial['lifecycle']['quality']['policy_root'])
    inputs = {spec['file']: store.get(spec['sha256']) for spec in quality['roles'].values()}
    cloud = Cloud(campaign)
    service = cloud.service(uuid.uuid4().hex, graph, baseline, profile, quality, inputs,
                            Objects(diagnostic/'objects'), slot=13)
    save(home/'service.json', service)
    rows, first = [], None
    try:
        save(home/'opened.json', {'source': source, 'driver': sha256(Path(__file__)),
            'plan': identity(plan), 'graph': identity(graph), 'selection': 'Every committed case once; exact first replay.'})
        deadline = time.monotonic()+600
        for role, cases in plan['roles'].items():
            for case in cases:
                if time.monotonic() >= deadline:
                    raise TimeoutError('The frozen assistant screen reached its generation bound')
                request = {'id': identity({'plan':identity(plan),'case':case['id'],'run':service['key']}),
                    'kind': 'generate', 'graph': identity(graph),
                    'question': case['messages'][:-1], 'max_tokens': plan['generation']}
                actual = cloud.query(service, request, timeout=min(180,int(deadline-time.monotonic())))
                if actual['status'] != 'completed':
                    raise ValueError('Prospective serving execution did not complete')
                result = actual['result']
                correct = ordinary_quality.correct(case, result)
                rows.append({'id':case['id'],'role':role,'correct':correct,'result':result})
                save(home/'answers.json',rows)
                print(json.dumps({'role':role,'done':len(rows),'correct':correct}),flush=True)
                if first is None:
                    first = request,result
        request, result = first
        replay = cloud.query(service,{**request,'id':identity({'replay':request['id']})},timeout=180)
        exact = replay['status']=='completed' and replay['result']==result
        counts = {role:{'correct':sum(row['correct'] for row in rows if row['role']==role),
                       'count':len(cases)} for role,cases in plan['roles'].items()}
        passed = exact and all(value['correct']/value['count']>=plan['minimum_accuracy'] for value in counts.values())
        report = {'plan':identity(plan),'graph':identity(graph),'counts':counts,'passed':passed,
            'exact_replay':exact,'source':source,'seconds':600-(deadline-time.monotonic()),
            'neural_training':False,'new_cohorts_admitted':0,'native_promotion':False,
            'scope':'Prospective initial-assistant screen. Repeated useful native learning remains separate.'}
        save(home/'result.json',report)
        print(json.dumps(report),flush=True)
    finally:
        cloud.stop(service)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','diagnostic','plan','home'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    run(args.campaign,args.diagnostic,args.plan,args.home)
