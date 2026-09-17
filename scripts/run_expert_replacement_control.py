#!/usr/bin/env python3
"""Compare retaining an expert with replacing it using identical trained weights.

This is a training-matched capacity control, not a claim of equal lifetime
storage or inference costs. It executes only the original number of experts;
the new and old learned routes share the terminal replacement. No answer labels
enter routing, and the trained checkpoint is never selected using control scores.
"""
import argparse
import copy
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import time

import torch.distributed as dist

from neuroshard.evolution import cohort_questions, expert_router, serving_graph
from neuroshard.evolution.reference_data import identity, read_records, save, sha256
from neuroshard.evolution.sharded.graph_execution import GraphNetwork
from neuroshard.evolution.sharded import learned_graph, portable
from neuroshard.evolution.sharded.router_features import EmbeddingFeatures

FORMAT = 'neuroshard-training-matched-replacement-control-v1'


def execution_profile(original, contract, source):
    """Bind the already declared routing/client changes, preserving kernels."""
    allowed = {'src/neuroshard/evolution/audit_worker.py',
               'src/neuroshard/evolution/transactions.py',
               'src/neuroshard/evolution/sharded/learned_graph.py'}
    updates = contract.get('executor_source_updates', {})
    if set(updates) - (allowed & set(original['sources'])):
        raise ValueError('Replacement control cannot change numerical execution sources')
    profile = copy.deepcopy(original)
    for name, previous in profile['sources'].items():
        expected = updates.get(name, previous)
        if sha256(source/name) != expected:
            raise ValueError('Control executor source was not prospectively committed')
        profile['sources'][name] = expected
    return profile


def replacement(baseline, terminal, name):
    """Replace one equal-shaped tail, preserving the fixed neural capacity."""
    old = baseline['experts'][name]
    if ({key: value['shape'] for key, value in old['tensors'].items()}
            != {key: value['shape'] for key, value in terminal['tensors'].items()}):
        raise ValueError('The replacement must fit the same neural parameter budget')
    graph = copy.deepcopy(baseline)
    graph['experts'][name] = copy.deepcopy(terminal)
    graph['descriptor']['previous_graph'] = identity(baseline['descriptor'])
    for row in graph['descriptor']['experts']:
        if row['id'] == name:
            row['checkpoint'] = terminal['checkpoint']
    serving_graph.validate(graph)
    return graph


def run(home, source, phase):
    inputs = home/'inputs'
    read = lambda name: json.loads((inputs/name).read_bytes())
    contract, config, freeze = read('replacement-control.json'), read('serving-trial.json'), read('freeze.json')
    if contract['format'] != FORMAT or contract['primary_freeze'] != identity(freeze):
        raise ValueError('The replacement control must bind the unchanged primary experiment')
    for name, expected in contract['sources'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or sha256(source/path) != expected:
            raise ValueError('Replacement control source differs from its prospective freeze')
    baseline, expanded = read('graph.json'), read('candidate-graph.json')
    terminal = expanded['experts'][config['expert']]
    plan = read('plan.json')
    if (identity(baseline) != config['baseline_graph'] or terminal['recipe'] != plan['training']
            or terminal['step'] != plan['training']['steps']
            or plan['seed_expert']['checkpoint'] != baseline['experts'][contract['replace_expert']]):
        raise ValueError('Reuse the identical continued training, initialization and terminal weights')
    from neuroshard.evolution import expert_data
    if terminal['job'] != expert_data.job_identity(plan, read('prepared.json')):
        raise ValueError('Replacement came from another numerical job')
    graph = replacement(baseline, terminal, contract['replace_expert'])
    profile = execution_profile(read('graph-profile.json'), contract, source)
    graph['executor_root'] = identity(profile)
    rank = int(os.environ['RANK'])
    dist.init_process_group('gloo', timeout=timedelta(seconds=1800))
    output = home/('replacement-'+phase)
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    try:
        net = GraphNetwork(graph, profile, objects=home/'objects',
            interpreter=home/'interpreter', seed=home/'seed', source_home=source, rank=rank)
        features, packet = None, None
        if rank == 0:
            embedding = graph['interpreter_assets']['partitions']['0']['tensors']['model.embed_tokens.weight']['sha256']
            features = EmbeddingFeatures(portable.tensor_path(home/'interpreter', embedding), embedding,
                                         net.tokenizer, graph['tokenizer']['root'])
            base, fitting = config['router'], []
            for role, route in (('train', config['expert']), ('replay', contract['replace_expert'])):
                spec = config['routing_inputs'][role]
                for row in read_records(inputs/spec['file'], spec['sha256']):
                    fitting.append({'id': row['id'], 'route': route,
                                    'features': features(row['messages'][0]['content'])})
            for route, centers in base['prototypes'].items():
                for index, center in enumerate(centers):
                    fitting.append({'id': identity(['accepted-prototype', identity(base), route, index]),
                                    'route': route, 'features': center})
            model = expert_router.append_route(base, fitting, config['expert'], **config['gate_fitting'])
            aliases = {route: route for route in model['prototypes']}
            aliases[config['expert']] = contract['replace_expert']
            packet = learned_graph.configuration(graph, model, features.profile, source, aliases, compose=True)
        packet = net.all_owners.exchange(packet)[0]
        service = learned_graph.LearnedGraphNetwork(net, packet, source_home=source, features=features)
        save(output/'service.json', packet)
        executions = {}
        roles = ('dev', 'retained') if phase == 'dev' else ('test', 'retained')
        if phase == 'test':
            opened = json.loads((home/'open-final.json').read_bytes())
            if opened['freeze'] != identity(freeze) or opened['checkpoint'] != terminal['checkpoint']:
                raise ValueError('Final requires the primary published terminal decision')
        for role in roles:
            spec = config['evaluation'][role]
            values = read_records(inputs/spec['file'], spec['sha256'])
            if len(values) != spec['count'] or identity([row['id'] for row in values]) != spec['ids']:
                raise ValueError('The control cannot change sealed evaluation identities')
            cohort_questions.validate_rows(values, release_scope=spec['release_scope'])
            results = []
            for index, row in enumerate(values):
                if time.monotonic()-started > contract['max_seconds']:
                    raise TimeoutError('Replacement control exceeded its frozen time allowance')
                answer = service.answer(row['messages'][0]['content'], config['max_tokens'])
                results.append({'id': row['id'], 'response': answer,
                    'correct': cohort_questions.correct(row, answer['text'], release_scope=spec['release_scope'])})
                if rank == 0:
                    print(json.dumps({'role': role, 'done': index+1, 'count': len(values)}), flush=True)
            executions[role] = results
            save(output/(role+'.json'), results)
        result = {'format': FORMAT+'/result', 'contract': identity(contract), 'phase': phase,
            'service': service.root, 'graph': identity(graph), 'checkpoint': terminal['checkpoint'],
            'router': identity(packet['router']), 'neural_parameters': graph['descriptor']['total_parameters'],
            'additional_parameters_in_primary': sum(math.prod(spec['shape']) for spec in terminal['tensors'].values()),
            'logical_owners': net.world_size, 'additional_training_updates': 0,
            'execution_roots': {role: identity(rows) for role, rows in executions.items()},
            'correct': {role: sum(row['correct'] for row in rows) for role, rows in executions.items()},
            'seconds': time.monotonic()-started, 'native_activated': False, 'tokens_issued': 0}
        save(output/'result.json', result)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--phase', choices=('dev', 'test'), required=True)
    args = parser.parse_args()
    run(args.home, args.source, args.phase)
