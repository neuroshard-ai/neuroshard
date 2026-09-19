#!/usr/bin/env python3
"""Carry unopened cohorts into the repaired complete answering system.

This does not select new neural hyperparameters or refit domain gates. Every
training/final byte remains equal to the previous prescription. The independent
assistant screen joins retention, and the existing allocation keeps its deadline.
The ordinary freezer must subsequently bind and commit the renewed operation.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil

from neuroshard.evolution import answering, expert_router, ordinary_cohorts, ordinary_quality
from neuroshard.evolution.data import document_identity
from neuroshard.evolution.expert_preparation import record_set
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.reference_data import identity, save, sha256
from prepare_ordinary_cohorts import configuration


def renew(previous, diagnostic, holdout_plan, holdout, home):
    read = lambda path: json.loads(path.read_bytes())
    development, prospective = read(diagnostic/'diagnostic-result.json'), read(holdout/'result.json')
    plan = read(holdout_plan)
    accepted = read(diagnostic/'after-graph.json')
    if (development['development_passed'] is not True or prospective['passed'] is not True
            or prospective['plan'] != identity(plan) or prospective['graph'] != identity(accepted)
            or plan['service_graph'] != identity(accepted) or prospective['exact_replay'] is not True):
        raise ValueError('Require the repaired complete interface and its prospective assistant pass')
    measured = {row['id']:row for row in read(holdout/'answers.json')}
    for role, cases in plan['roles'].items():
        correct = [ordinary_quality.correct(case, measured[case['id']]['result']) for case in cases]
        if (sum(correct)/len(correct) < plan['minimum_accuracy']
                or any(ok != measured[case['id']]['correct'] for case,ok in zip(cases,correct))):
            raise ValueError('The prospective result differs from its actual complete replies')
    initial = answering.load(accepted, Objects(diagnostic/'objects'))
    repaired = initial['learned']['router']
    home.mkdir(parents=True, exist_ok=False)
    shutil.copytree(previous/'compiled', home/'compiled')
    for name in ('allocation.json', 'known_hosts', 'runtime.json'):
        (home/name).symlink_to((previous/name).resolve())
    allocation = read(previous/'allocation.json')
    save(home/'allocation-reference.json', {'owner':str(previous.resolve()),
        'created':allocation['created'],'deadline':allocation['deadline'],
        'instances':[row['InstanceId'] for row in allocation['instances']],
        'additional_instances':0,'budget_reset':False})
    compiled = home/'compiled'
    store = Objects(compiled/'objects')
    inputs = read(compiled/'inputs.json')
    prior_inputs = identity(inputs)
    original_policies = {name:store.json(root) for name,root in inputs['policies'].items()}
    core = read(compiled/'baseline-core.json')
    # The freezer replaces numerical source/runtime references consistently.
    # Baseline neural tensors, tokenizer and ownership remain exact.
    if any(core[key] != answering.core(accepted)[key] for key in core
           if key not in ('executor_root','numerical_profile')):
        raise ValueError('This renewal cannot replace accepted neural weights or topology')
    topology = copy.deepcopy(core)
    for name in ('baseline', *ordinary_cohorts.ORDER):
        router = copy.deepcopy(original_policies[name]['configuration']['learned']['router'])
        if (router['base'] != repaired['base']
                or router['additions'][:len(repaired['additions'])] != repaired['additions']):
            raise ValueError('Preserve all accepted and prospective domain gate weights')
        router['fallback_guard'] = copy.deepcopy(repaired['fallback_guard'])
        expert_router.validate(router)
        if name != 'baseline':
            topology = ordinary_cohorts.extend(topology, name, core['experts']['planner'])
        bound = answering.attach(topology, configuration(topology, router, initial), store)
        inputs['policies'][name] = bound['answering']['policy_root']
        if name != 'baseline':
            inputs['cohorts'][name]['answering_policy'] = bound['answering']['policy_root']
            inputs['cohorts'][name]['router'] = store.put_json(router)
            # Domain fitting was already measured. The new fallback guard was
            # separately measured on all specialized training inputs as negatives.
            inputs['cohorts'][name]['inherited_selector_evidence'] = prior_inputs
    anchors = read(compiled/'anchors.json')
    for role, rows in plan['roles'].items():
        existing = {document_identity(row['messages']) for row in anchors[role]}
        if existing & {row['id'] for row in rows}:
            raise ValueError('The fresh assistant screen repeats an existing anchor')
        anchors[role].extend(copy.deepcopy(rows))
    specs = {role:record_set(store, role,
        [{**row,'id':document_identity(row['messages'])} for row in rows]) for role,rows in anchors.items()}
    rules = read(compiled/'quality-rule.json')
    rules['retention_anchors'] = specs
    inputs.update(anchors=specs, quality_rule=identity(rules))
    save(compiled/'anchors.json', anchors)
    save(compiled/'quality-rule.json', rules)
    save(compiled/'inputs.json', inputs)
    for name in ordinary_cohorts.ORDER:
        for filename in ('training.jsonl','final.jsonl','training-annotations.json'):
            if sha256(previous/'compiled'/name/filename) != sha256(compiled/name/filename):
                raise ValueError('Renewal changed the frozen neural training or final inputs')
    save(home/'renewal.json', {'previous':str(previous.resolve()),'prior_inputs':prior_inputs,
        'previous_operation':identity(read(previous/'operation.json')),
        'development':identity(development),'prospective':identity(prospective),'plan':identity(plan),
        'unchanged_neural_inputs':True,'fresh_cases_added_to_retention':sum(map(len,plan['roles'].values())),
        'reused_domain_gates':True,'fallback_guard':identity(repaired['fallback_guard']),
        'source_binding':'Pending ordinary freezer and a new source commitment; no training authorized by this compiler alone.'})
    print(json.dumps({'compiled':str(compiled),'anchors':{role:spec['count'] for role,spec in specs.items()}}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('previous','diagnostic','holdout-plan','holdout','home'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    renew(args.previous,args.diagnostic,args.holdout_plan,args.holdout,args.home)
