#!/usr/bin/env python3
"""Measure literal question preservation on an already opened failed cohort.

Both policies use the exact same trained checkpoint. This inference-only
diagnostic cannot promote, issue tokens or count as a fresh learning cohort.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import uuid

from neuroshard.evolution import answering, expert_lifecycle as life, request_planning
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.reference_data import identity, save, sha256
from neuroshard.evolution.sharded import graph_quality, learned_graph, planned_graph
from ordinary_cloud import Cloud

ROOT = Path(__file__).resolve().parents[1]


def neural_response(value):
    body = value['answering']
    return identity({key:body[key] for key in ('plan','answers','outputs','text','status','error')})


def assemble(campaign, job_home, home, semantic_policy=None,
             request_policy=request_planning.LOSSLESS_POLICY, checkpoint=None, question_reranker=None):
    if not (job_home/'quality-0.json').exists():
        raise ValueError('Use only a previously opened quality result')
    old = Objects(campaign/'compiled/objects')
    job = json.loads((job_home/'job.json').read_bytes())
    measured = json.loads((job_home/'quality-0.json').read_bytes())['result']
    if measured['decision']['passed'] is not False:
        raise ValueError('This diagnostic targets a recorded failed cohort')
    home.mkdir(parents=True, exist_ok=False)
    store = Objects(home/'objects')
    operation = json.loads((campaign/'operation.json').read_bytes())
    profile = copy.deepcopy(old.json(operation['executor']))
    profile['sources'] = {name: sha256(ROOT/name) for name in profile['sources']}
    if semantic_policy is not None:
        for name in ('src/neuroshard/evolution/semantic_questions.py',
                     'src/neuroshard/evolution/sharded/semantic_features.py'):
            profile['sources'][name] = sha256(ROOT/name)
    if question_reranker is not None:
        for name in ('src/neuroshard/evolution/question_reranking.py',
                     'src/neuroshard/evolution/sharded/question_reranker.py'):
            profile['sources'][name] = sha256(ROOT/name)
    original = job['lifecycle']['candidate_template']
    previous = answering.load(original, old)
    if checkpoint is None:
        checkpoint = json.loads((job_home/'produce-124-actor-0.json').read_bytes())['result']['window']['output']
    template = copy.deepcopy(answering.core(original))
    template['executor_root'] = identity(profile)

    def bind(core, policy, semantics=None, reranker=None):
        learned = learned_graph.configuration(core, previous['learned']['router'],
            previous['learned']['feature_profile'], ROOT,
            previous['learned'].get('route_models'), compose=True)
        options = {key: previous[key] for key in ('route_scopes', 'planner_weights', 'composer',
            'answer_policy', 'general_answer_policy') if key in previous}
        config = planned_graph.configuration(core, learned, previous['planner'], ROOT,
            previous['expert_prompts'], previous['general_instruction'], request_policy=policy,
            semantic_questions=semantics, question_reranker=reranker, **options)
        return answering.attach(core, config, store)

    template = bind(template, request_policy, semantic_policy, question_reranker)
    after = life.materialize_graph(template, checkpoint)
    before = bind(answering.core(after), previous['request_policy'], previous.get('semantic_questions'),
                  previous.get('question_reranker'))
    if answering.core(before) != answering.core(after):
        raise ValueError('Question preservation must not change neural weights or topology')
    quality = copy.deepcopy(old.json(job['lifecycle']['quality']['policy_root']))
    quality.update(baseline_graph=identity(before), candidate_template=template)
    graph_quality.validate_policy(quality)
    inputs = home/'inputs'
    inputs.mkdir()
    for spec in quality['roles'].values():
        (inputs/spec['file']).write_bytes(old.get(spec['sha256']))
    for name, value in (('before-graph',before), ('after-graph',after), ('profile',profile), ('quality',quality)):
        save(home/(name+'.json'),value)
    previous_neural = {row['id']:neural_response(row['after']) for row in measured['executions']}
    previous_neural.update({row['id']:neural_response(row['after'])
        for rows in measured['retention']['roles'].values() for row in rows})
    save(home/'previous-neural.json',previous_neural)
    previous_encoder = previous.get('semantic_questions', {}).get('encoder')
    candidate_encoder = None if semantic_policy is None else semantic_policy['encoder']
    plan = {'format':('neuroshard-semantic-question-diagnostic-v1' if semantic_policy is not None
                     else 'neuroshard-literal-question-diagnostic-v1'),
        'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        'driver':sha256(Path(__file__)), 'before':identity(before), 'after':identity(after),
        'profile':identity(profile), 'quality':identity(quality), 'prior_result':identity(measured),
        'previous_neural':identity(previous_neural),
        'neural_weights_unchanged':(previous_encoder == candidate_encoder
            and previous.get('question_reranker') == question_reranker), 'answering_expert_weights_unchanged':True,
        'question_reranker': None if question_reranker is None else identity(question_reranker),
        'semantic_encoder_added': (identity(candidate_encoder)
            if candidate_encoder is not None and previous_encoder != candidate_encoder else None),
        'new_final_opened':False, 'native_promotion':False,
        'max_seconds':1800, 'scope':'Exposed development diagnostic; original failed result remains failed.'}
    save(home/'diagnostic.json',plan)
    return plan


def run(campaign, home):
    read = lambda name: json.loads((home/(name+'.json')).read_bytes())
    plan, before, after, profile, quality = [read(name) for name in
        ('diagnostic','before-graph','after-graph','profile','quality')]
    if (plan['driver'] != sha256(Path(__file__))
            or any(plan[name] != identity(value) for name,value in
                   (('before',before),('after',after),('profile',profile),('quality',quality)))
            or any(sha256(ROOT/name) != value for name,value in profile['sources'].items())):
        raise ValueError('Require the committed diagnostic inputs and source')
    previous_neural = read('previous-neural')
    if identity(previous_neural) != plan['previous_neural']:
        raise ValueError('The comparison lost its previous numerical transcript commitments')
    inputs = {spec['file']:(home/'inputs'/spec['file']).read_bytes() for spec in quality['roles'].values()}
    cloud = Cloud(campaign)
    service = cloud.service(uuid.uuid4().hex, after, before, profile, quality, inputs,
                            Objects(home/'objects'), slot=12)
    save(home/'service.json',service)
    try:
        request = {'id':identity({'diagnostic':identity(plan),'run':service['key']}),'kind':'evaluate'}
        response = cloud.query(service,request,timeout=plan['max_seconds'])
        if response['status'] != 'completed':
            raise ValueError('The diagnostic answering system failed to execute')
        save(home/'answers.json',response)
        result = response['result']
        old = {row['id']:neural_response(row['before']) for row in result['executions']}
        old.update({row['id']:neural_response(row['before'])
            for rows in result['retention']['roles'].values() for row in rows})
        old_exact = old == previous_neural
        rows = [json.loads(line) for line in inputs[quality['roles']['test']['file']].splitlines()]
        first = next(row for row in rows if row['stratum']=='composed')
        expected = next(row['after'] for row in result['executions'] if row['id']==first['id'])
        replay = cloud.query(service,{'id':identity({'replay':request['id']}),'kind':'generate',
            'graph':identity(after),'question':first['messages'][:-1],'max_tokens':quality['generation']['new']},timeout=180)
        exact = replay['status']=='completed' and replay['result']==expected
        metrics = result['decision']['metrics']
        summary = {'diagnostic':identity(plan),'metrics':metrics,
            'retained_lost':result['retention']['lost_correct'],
            'retention_accuracy':result['retention']['accuracy'],'exact_replay':exact,
            'previous_behavior_reproduced':old_exact,
            'development_absolute_gates_passed':bool(exact and old_exact and result['retention']['passed']
                and metrics['single']['accuracy']>=quality['gates']['single_accuracy']
                and metrics['composed']['accuracy']>=quality['gates']['composed_accuracy']),
            'original_frozen_trial_passed':False,'neural_training':False,
            'new_final_opened':False,'tokens_issued':0,'native_promoted':False,
            'seconds':response['seconds']}
        save(home/'result.json',summary)
        print(json.dumps(summary),flush=True)
    finally:
        cloud.stop(service)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--home',type=Path,required=True)
    parser.add_argument('--job',type=Path)
    parser.add_argument('--assemble',action='store_true')
    args=parser.parse_args()
    if args.assemble:
        if args.job is None:parser.error('--assemble requires --job')
        print(json.dumps(assemble(args.campaign,args.job,args.home)))
    else:run(args.campaign,args.home)
