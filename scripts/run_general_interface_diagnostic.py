#!/usr/bin/env python3
"""Use the existing bounded owners to measure the exposed assistant repair.

This is a diagnostic on already opened regressions. It cannot promote, issue,
train neural weights or count as an independent final. The ordinary native
request queue and complete-policy executor perform the actual inference.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import uuid

from neuroshard.evolution import answering, ordinary_quality
from neuroshard.evolution.data import document_identity
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.reference_data import identity, save, sha256
from neuroshard.evolution.sharded import learned_graph, planned_graph
from ordinary_cloud import Cloud
from prepare_general_interface import GENERAL_INSTRUCTION, FORMAT
from neuroshard.evolution.request_planning import ASSISTANT_POLICY

ROOT = Path(__file__).resolve().parents[1]


def assemble(campaign, home):
    old = Objects(campaign/'compiled/objects')
    operation = json.loads((campaign/'operation.json').read_bytes())
    old_graph = old.json(operation['baseline_graph'])
    previous = answering.load(old_graph, old)
    graph = copy.deepcopy(answering.core(old_graph))
    profile = copy.deepcopy(old.json(operation['executor']))
    profile['sources'] = {path: sha256(ROOT/path) for path in profile['sources']}
    graph['executor_root'] = identity(profile)
    store = Objects(home/'objects')
    router = json.loads((home/'router.json').read_bytes())
    configs, graphs = {}, {}
    for name in ('before', 'after'):
        learned = learned_graph.configuration(graph,
            previous['learned']['router'] if name == 'before' else router,
            previous['learned']['feature_profile'], ROOT,
            previous['learned'].get('route_models'), compose=True)
        options = {key: previous[key] for key in ('route_scopes', 'planner_weights', 'composer',
                   'answer_policy', 'request_policy') if key in previous}
        if name == 'after':
            options['request_policy'] = ASSISTANT_POLICY
        configs[name] = planned_graph.configuration(graph, learned, previous['planner'], ROOT,
            previous['expert_prompts'], previous['general_instruction'] if name == 'before' else GENERAL_INSTRUCTION,
            **options)
        graphs[name] = answering.attach(graph, configs[name], store)
        save(home/(name+'-graph.json'), graphs[name])
    rows = {role:[{**row,'id':document_identity(row['messages'])} for row in values]
            for role,values in json.loads((campaign/'compiled/anchors.json').read_bytes()).items()}
    initial = json.loads((campaign/'initial-job.json').read_bytes())
    quality = old.json(initial['lifecycle']['quality']['policy_root'])
    # Only file installation uses this policy; no new quality evaluation or
    # unopened test file is dispatched by the diagnostic driver.
    inputs = {spec['file']: old.get(spec['sha256']) for spec in quality['roles'].values()}
    save(home/'profile.json', profile)
    save(home/'cases.json', rows)
    save(home/'diagnostic.json', {'format': FORMAT+'/diagnostic',
        'source': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        'driver': sha256(Path(__file__)), 'graphs': {name: identity(value) for name,value in graphs.items()},
        'profile': identity(profile), 'cases': identity(rows), 'max_tokens': 64,
        'gates': {'no_previously_correct_lost': True, 'skills': .75, 'conversation': .75, 'knowledge': .75},
        'frozen_neural_weights': True, 'new_final_opened': False, 'scope': 'Exposed development regressions only'})
    return graphs, profile, rows, quality, inputs, store


def run(campaign, home):
    frozen = json.loads((home/'diagnostic.json').read_bytes())
    if sha256(Path(__file__)) != frozen['driver']:
        raise ValueError('Diagnostic driver differs from its committed prescription')
    graphs = {name: json.loads((home/(name+'-graph.json')).read_bytes()) for name in ('before','after')}
    profile = json.loads((home/'profile.json').read_bytes())
    cases = json.loads((home/'cases.json').read_bytes())
    if (identity(profile) != frozen['profile'] or identity(cases) != frozen['cases']
            or any(identity(value) != frozen['graphs'][name] for name,value in graphs.items())
            or any(sha256(ROOT/path) != digest for path,digest in profile['sources'].items())):
        raise ValueError('Diagnostic models, cases or numerical source changed')
    old = Objects(campaign/'compiled/objects')
    initial = json.loads((campaign/'initial-job.json').read_bytes())
    quality = old.json(initial['lifecycle']['quality']['policy_root'])
    inputs = {spec['file']: old.get(spec['sha256']) for spec in quality['roles'].values()}
    cloud = Cloud(campaign)
    service = cloud.service(uuid.uuid4().hex, graphs['after'], graphs['before'], profile,
        quality, inputs, Objects(home/'objects'), slot=12)
    save(home/'service.json', service)
    rows, first = [], None
    try:
        prior_path = next(campaign.glob('jobs/*/quality-0.json'))
        prior = json.loads(prior_path.read_bytes())['result']['retention']['roles']
        for role, examples in cases.items():
            baseline = {row['id']:row for row in prior[role]}
            for row in examples:
                request = {'id':identity({'diagnostic':identity(frozen),'row':row['id'],'run':service['key']}),
                    'kind':'generate','graph':identity(graphs['after']),
                    'question':row['messages'][:-1],'max_tokens':64}
                actual = cloud.query(service, request, timeout=180)
                if actual['status'] != 'completed':
                    raise ValueError('The actual ordinary diagnostic execution was unavailable')
                result = actual['result']
                correct = ordinary_quality.correct(row, result)
                before = baseline[row['id']]['before_correct']
                rows.append({'role':role,'id':row['id'],'before_correct':before,'after_correct':correct,
                             'lost_correct':before and not correct,'result':result})
                save(home/'answers.json',rows)
                print(json.dumps({'role':role,'done':len(rows),'correct':correct,'lost':before and not correct}),flush=True)
                if first is None:
                    first = request, result
        request, first_result = first
        replay = cloud.query(service,{**request,'id':identity({'replay':request['id']})},timeout=180)
        exact = replay['status']=='completed' and replay['result']==first_result
        counts = {role:{'correct':sum(row['after_correct'] for row in rows if row['role']==role),
                       'count':sum(row['role']==role for row in rows)} for role in cases}
        passed = exact and not any(row['lost_correct'] for row in rows) and all(
            row['correct']/row['count'] >= .75 for row in counts.values())
        result={'format':FORMAT+'/diagnostic-result','diagnostic':identity(frozen),'counts':counts,
            'lost_correct':sum(row['lost_correct'] for row in rows),'exact_replay':exact,'development_passed':passed,
            'tokens_issued':0,'native_promoted':False,'new_final_opened':False}
        save(home/'diagnostic-result.json',result)
        print(json.dumps(result),flush=True)
    finally:
        cloud.stop(service)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--home',type=Path,required=True)
    parser.add_argument('--assemble',action='store_true')
    args=parser.parse_args()
    (assemble if args.assemble else run)(args.campaign,args.home)
