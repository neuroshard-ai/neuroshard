#!/usr/bin/env python3
"""Paired answer evaluation of a frozen expert addition and retained router.

Each explicit two-question request is routed one subquestion at a time. This
tests cross-expert execution, not arbitrary conversational planning. All answer
labels remain inside scoring; fitting sees only committed training prompts.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch.distributed as dist

from neuroshard.evolution import cohort_questions, expert_router, serving_graph
from neuroshard.evolution.reference_data import identity, read_records, save, sha256
from neuroshard.evolution.sharded.graph_execution import GraphNetwork
from neuroshard.evolution.sharded import learned_graph, portable
from neuroshard.evolution.sharded.router_features import EmbeddingFeatures

FORMAT = 'neuroshard-isolated-expert-serving-trial-v1'


def run(home, source, phase):
    inputs = home/'inputs'
    read = lambda name: json.loads((inputs/name).read_bytes())
    config, freeze = read('serving-trial.json'), read('freeze.json')
    if config['format'] != FORMAT or identity(config) != freeze['serving_trial']:
        raise ValueError('Freeze routing, answers, resources and generation before training')
    for path, expected in freeze['sources'].items():
        if Path(path).is_absolute() or '..' in Path(path).parts or sha256(source/path) != expected:
            raise ValueError('Frozen experiment source changed')
    baseline, graph = read('graph.json'), read('candidate-graph.json')
    if identity(baseline) != config['baseline_graph']:
        raise ValueError('The paired baseline graph changed')
    if (any(graph[key] != baseline[key] for key in ('parent', 'interpreter_assets', 'interpreter_prompt',
            'tokenizer', 'numerical_profile', 'executor_root'))
            or set(graph['experts']) != {*baseline['experts'], config['expert']}
            or any(graph['experts'][name] != value for name, value in baseline['experts'].items())
            or graph['experts'][config['expert']]['recipe'] != read('plan.json')['training']
            or graph['experts'][config['expert']]['step'] != read('plan.json')['training']['steps']):
        raise ValueError('The candidate must preserve every accepted expert and finish the frozen job')
    from neuroshard.evolution import expert_data
    if graph['experts'][config['expert']]['job'] != expert_data.job_identity(read('plan.json'), read('prepared.json')):
        raise ValueError('The serving tail came from a different training job')
    rank = int(os.environ['RANK'])
    dist.init_process_group('gloo', timeout=timedelta(seconds=1800))
    started = time.monotonic()
    output = home/('serving-'+phase)
    output.mkdir(exist_ok=False)
    try:
        net = GraphNetwork(graph, read('graph-profile.json'), objects=home/'objects',
            interpreter=home/'interpreter', seed=home/'seed', source_home=source, rank=rank)
        features, packet = None, None
        if rank == 0:
            embedding = graph['interpreter_assets']['partitions']['0']['tensors']['model.embed_tokens.weight']['sha256']
            features = EmbeddingFeatures(portable.tensor_path(home/'interpreter', embedding), embedding,
                                         net.tokenizer, graph['tokenizer']['root'])
            base = config['router']
            if features.root != base['embedding_root']:
                raise ValueError('Serving gate changed the frozen input features')
            fitting = []
            for role, route in (('train', config['expert']), ('replay', 'protocol')):
                spec = config['routing_inputs'][role]
                rows = read_records(inputs/spec['file'], spec['sha256'])
                for row in rows:
                    fitting.append({'id': row['id'], 'route': route,
                                    'features': features(row['messages'][0]['content'])})
            # Previously accepted prototypes summarize earlier router training,
            # not held-out retention questions. Their origin is explicit.
            for route, centers in base['prototypes'].items():
                for index, center in enumerate(centers):
                    fitting.append({'id': identity(['accepted-prototype', identity(base), route, index]),
                                    'route': route, 'features': center})
            candidate = expert_router.append_route(base, fitting, config['expert'], **config['gate_fitting'])
            packet = {'baseline': learned_graph.configuration(baseline, base, features.profile, source, compose=True),
                      'candidate': learned_graph.configuration(graph, candidate, features.profile, source, compose=True),
                      'training': identity(fitting)}
        packet = net.all_owners.exchange(packet)[0]
        save(output/'services.json', packet)
        before = learned_graph.LearnedGraphNetwork(net, packet['baseline'], source_home=source,
                                                    features=features, graph=baseline)
        after = learned_graph.LearnedGraphNetwork(net, packet['candidate'], source_home=source, features=features)
        if phase == 'test':
            decision = json.loads((home/'open-final.json').read_bytes())
            if decision != {'freeze': identity(freeze), 'candidate': after.root,
                            'checkpoint': graph['experts'][config['expert']]['checkpoint']}:
                raise ValueError('Publish the exact terminal service before opening its final set')
        results, roles = {}, ('dev', 'retained') if phase == 'dev' else ('test', 'retained')
        for role in roles:
            spec = config['evaluation'][role]
            values = read_records(inputs/spec['file'], spec['sha256'])
            if len(values) != spec['count'] or identity([row['id'] for row in values]) != spec['ids']:
                raise ValueError('The complete sealed evaluation inventory changed')
            cohort_questions.validate_rows(values, release_scope=spec['release_scope'])
            executions, old_rows, new_rows, lost = [], [], [], []
            for index, row in enumerate(values):
                if time.monotonic()-started > config['max_seconds']:
                    raise TimeoutError('The frozen serving comparison deadline expired')
                question = row['messages'][0]['content']
                old, new = before.answer(question, config['max_tokens']), after.answer(question, config['max_tokens'])
                old_correct = cohort_questions.correct(row, old['text'], release_scope=spec['release_scope'])
                new_correct = cohort_questions.correct(row, new['text'], release_scope=spec['release_scope'])
                executions.append({'id': row['id'], 'before': old, 'after': new,
                                   'before_correct': old_correct, 'after_correct': new_correct})
                old_rows.append({'id': row['id'], 'text': old['text']})
                new_rows.append({'id': row['id'], 'text': new['text']})
                if old_correct and not new_correct:
                    lost.append(row['id'])
                if rank == 0:
                    print(json.dumps({'role': role, 'done': index+1, 'count': len(values)}), flush=True)
            decision = cohort_questions.decision(values, old_rows, new_rows, config['gates'],
                                                 release_scope=spec['release_scope'])
            results[role] = {'decision': decision, 'lost_correct': lost, 'executions': executions}
            save(output/(role+'.json'), results[role])
        decision = results[roles[0]]['decision']
        decision['checks']['retained_answers'] = not results['retained']['lost_correct']
        decision['passed'] = all(decision['checks'].values())
        result = {'format': FORMAT+'/result', 'freeze': identity(freeze), 'phase': phase,
            'baseline_service': before.root, 'candidate_service': after.root, 'graph': identity(graph),
            'checkpoint': graph['experts'][config['expert']]['checkpoint'], 'decision': decision,
            'retained': results['retained']['decision'], 'lost_correct': results['retained']['lost_correct'],
            'execution_roots': {role: identity(value) for role, value in results.items()},
            'tokens_issued': 0, 'native_activated': False, 'seconds': time.monotonic()-started}
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
