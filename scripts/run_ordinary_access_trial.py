#!/usr/bin/env python3
"""One frozen inference allocation: automatic access and forced-expert controls."""
import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch.distributed as dist

from neuroshard.evolution import serving_diagnosis as diagnosis
from neuroshard.evolution.reference_data import identity, save, sha256
from neuroshard.evolution.sharded.graph_execution import GraphNetwork
from neuroshard.evolution.sharded.planned_graph import PlannedGraphNetwork
from neuroshard.evolution.sharded.portable import tensor_path
from neuroshard.evolution.sharded.router_features import EmbeddingFeatures


def forced(service, atom, maximum):
    """Oracle route is diagnostic-only; the expert's served prompt is unchanged."""
    question, route = atom['gold_question'], atom['expert']
    service.trace = []
    answer, argument, error = service.answer_atom(route, question, question,
        [{'role': 'user', 'content': question}], maximum, whole_request=True)
    result = {'question': question, 'forced_route': route, 'answer': answer,
              'argument': argument, 'error': error, 'outputs': copy.deepcopy(service.trace),
              'correct': error is None and diagnosis.atomic_correct(answer['text'], atom)}
    service.net.verify_unchanged()
    if service.net.all_owners.exchange(identity(result)) != [identity(result)] * service.net.world_size:
        raise ValueError('Owners disagree on a forced-expert control')
    return result


def run(home, source):
    read = lambda name: json.loads((home/'inputs'/name).read_bytes())
    trial, diagnostic, graph, profile, config = [read(name+'.json') for name in
        ('access-trial', 'plan', 'graph', 'profile', 'planned')]
    if (trial['no_neural_training'] is not True or trial['final_opened'] is not False
            or trial['diagnostic'] != identity(diagnostic) or trial['service'] != identity(config)
            or trial['router'] != identity(config['learned']['router'])
            or trial['expert_checkpoints'] != {key: value['checkpoint'] for key, value in graph['experts'].items()}):
        raise ValueError('Require the exact frozen service, experts and diagnostic inventory')
    for name, digest in trial['sources'].items():
        if sha256(Path(source)/name) != digest:
            raise ValueError('Frozen access trial source changed')
    facts = json.loads((Path(source)/'config/experiments/continual-expert-facts.json').read_bytes())
    diagnosis.validate(diagnostic, facts)
    output = home/'results'
    output.mkdir(exist_ok=False)
    dist.init_process_group('gloo', timeout=timedelta(seconds=1800))
    started = time.monotonic()
    try:
        net = GraphNetwork(graph, profile, objects=home/'objects', interpreter=home/'interpreter',
            seed=home/'seed', source_home=source, rank=int(os.environ['RANK']))
        feature = config['learned']['feature_profile']
        features = (EmbeddingFeatures(tensor_path(home/'interpreter', feature['embedding_sha256']),
            feature['embedding_sha256'], net.tokenizer, graph['tokenizer']['root']) if net.rank == 0 else None)
        service = PlannedGraphNetwork(net, config, source_home=source, features=features)
        save(output/'ready.json', {'trial': identity(trial), 'service': service.root, 'runtime': net.runtime})
        rows, traces, gold_cache, forced_cache, first, first_forced = [], [], {}, {}, None, None
        for case in diagnostic['cases']:
            if time.monotonic()-started > trial['max_seconds']:
                raise TimeoutError('The frozen access inference deadline expired')
            ordinary = service.answer(case['messages'], case['max_tokens'])
            gold, controls = [], []
            for atom in case['atoms']:
                key = identity([atom['expert'], atom['gold_question'], case['max_tokens']])
                if key not in gold_cache:
                    gold_cache[key] = service.answer([{'role': 'user', 'content': atom['gold_question']}],
                                                     case['max_tokens'])
                    forced_cache[key] = forced(service, atom, case['max_tokens'])
                gold.append(gold_cache[key])
                controls.append(forced_cache[key])
                if first_forced is None:
                    first_forced = (copy.deepcopy(atom), case['max_tokens'], forced_cache[key])
            row = diagnosis.diagnose(case, ordinary, gold)
            # Automatic pass is never replaced by success with a supplied route.
            row['forced_correct'] = [value['correct'] for value in controls]
            row['forced_errors'] = [value['error'] for value in controls]
            rows.append(row)
            traces.append({'id': case['id'], 'response': ordinary, 'gold_responses': gold,
                           'forced_controls': controls})
            if first is None:
                first = ordinary
            save(output/'traces.json', traces)
        valid, _ = service.replay(first)
        atom, maximum, previous = first_forced
        forced_replay = forced(service, atom, maximum) == previous
        save(output/'replay.json', {'service': service.root, 'automatic': valid,
                                   'forced': forced_replay})
        old_passes = set(trial['previous_automatic_passes'])
        passed_ids = {row['id'] for row in rows if row['passed']}
        selected_gold = [diagnosis.routes_of(response) == [atom['expert']]
                        for case, trace in zip(diagnostic['cases'], traces)
                        for atom, response in zip(case['atoms'], trace['gold_responses'])]
        checks = {'automatic_gold_selection': all(selected_gold),
                  'retained_previous_passes': old_passes <= passed_ids,
                  'automatic_replay': valid, 'forced_replay': forced_replay,
                  'ordinary_answering': len(passed_ids) == len(rows)}
        planning_checks = None
        if 'request_policy' in config:
            from neuroshard.evolution import planned_metering
            tariff = {'prompt_atom_price': 1, 'output_atom_price': 1,
                      'context': graph['tokenizer']['max_context']}
            receipts = [planned_metering.meter(config, graph, response, tariff)
                        for response in [*(trace['response'] for trace in traces), *gold_cache.values()]]
            save(output/'metering.json', {'tariff': tariff, 'ledger_payment': False, 'receipts': receipts})
            planning_checks = {
                'ordinary_access': all(row['ordinary'] not in ('selection', 'decomposition') for row in rows),
                'gold_access': all(mode not in ('selection', 'decomposition') for row in rows for mode in row['gold']),
                'retained_previous_passes': old_passes <= passed_ids,
                'automatic_replay': valid, 'forced_replay': forced_replay,
                'complete_call_metering': True}
        result = {'format': 'neuroshard-ordinary-access-trial-v1/result', 'trial': identity(trial),
            'service': service.root, 'rows': rows, 'summary': diagnosis.summarize(rows),
            'checks': checks, 'passed': all(checks.values()),
            'automatic_gold_routes': {'correct': sum(selected_gold), 'count': len(selected_gold)},
            'replay': {'automatic': valid, 'forced': forced_replay},
            'forced_unique': {'count': len(forced_cache),
                              'correct': sum(row['correct'] for row in forced_cache.values()),
                              'input_failures': sum(row['error'] is not None for row in forced_cache.values())},
            'traces': identity(traces), 'neural_updates': 0, 'tokens_issued': 0,
            'native_activated': False, 'final_opened': False, 'seconds': time.monotonic()-started}
        if planning_checks is not None:
            result['planning_checks'] = planning_checks
            result['planning_passed'] = all(planning_checks.values())
        save(output/'result.json', result)
        net.verify_unchanged()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    run(args.home, args.source)
