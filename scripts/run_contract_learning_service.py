#!/usr/bin/env python3
"""Evaluate the prescribed terminal C repair in the unchanged answering policy."""
import argparse
import json
from pathlib import Path

from neuroshard.evolution import expert_checkpoint, expert_data, serving_graph
from neuroshard.evolution.reference_data import identity, save, sha256
from run_ordinary_access_trial import run


def prepare(home, source):
    read = lambda name: json.loads((home/'inputs'/name).read_bytes())
    freeze, plan, prepared, parent = [read(name+'.json') for name in ('freeze', 'plan', 'prepared', 'parent')]
    for name, digest in freeze['sources'].items():
        if sha256(source/name) != digest:
            raise ValueError('The frozen learning-to-serving source changed')
    if (identity(plan) != freeze['plan'] or identity(prepared) != freeze['prepared']
            or sha256(home/'inputs/retention.json') != freeze['retention']):
        raise ValueError('The complete training prescription or retention inventory changed')
    checkpoint = read('terminal-checkpoint.json')
    expert_checkpoint.unpack(parent, checkpoint)
    if (checkpoint['step'] != plan['training']['steps'] or checkpoint['recipe'] != plan['training']
            or checkpoint['job'] != expert_data.job_identity(plan, prepared)):
        raise ValueError('Evaluate only the preselected terminal training checkpoint')
    graph, profile, config = [read('serving-'+name+'.json') for name in ('graph', 'profile', 'planned')]
    if plan['seed_expert']['checkpoint'] != graph['experts']['planner']:
        raise ValueError('The repaired C expert did not start from the measured serving graph')
    original = identity(graph['descriptor'])
    graph['experts']['planner'] = checkpoint
    graph['descriptor']['previous_graph'] = original
    for expert in graph['descriptor']['experts']:
        if expert['id'] == 'planner':
            expert['checkpoint'] = checkpoint['checkpoint']
    profile['sources'] = {name: sha256(source/name) for name in profile['sources']}
    graph['executor_root'] = identity(profile)
    serving_graph.validate(graph)
    config['graph'] = config['learned']['graph'] = identity(graph)
    for value in (config, config['learned']):
        value['sources'] = {name: sha256(source/name) for name in value['sources']}
    trial = read('serving-access-trial.json')
    trial.update(service=identity(config), router=identity(config['learned']['router']),
        expert_checkpoints={key: value['checkpoint'] for key, value in graph['experts'].items()},
        sources=freeze['sources'], previous_automatic_passes=[row['id'] for row in read('before-access.json')['rows'] if row['passed']],
        retention={'sha256': freeze['retention'], 'root': identity(read('retention.json'))})
    target = home/'serving'
    (target/'inputs').mkdir(parents=True, exist_ok=False)
    for name, value in (('graph', graph), ('profile', profile), ('planned', config), ('access-trial', trial),
                        ('plan', read('serving-plan.json')), ('retention', read('retention.json'))):
        save(target/'inputs'/(name+'.json'), value)
    for name in ('objects', 'interpreter', 'seed'):
        (target/name).symlink_to((home/name).resolve(), target_is_directory=True)
    save(home/'derived-service.json', {'training_freeze': identity(freeze), 'checkpoint': checkpoint['checkpoint'],
                                      'service': identity(config), 'graph': identity(graph),
                                      'training_updates': checkpoint['step']})
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    run(prepare(args.home, args.source), args.source)
