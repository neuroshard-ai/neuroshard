#!/usr/bin/env python3
"""Inference-only ordinary questions through the planned serving path.

Gold standalone questions run on the same frozen service. Labels stay in the
scorer. This driver never trains, never opens a new final and never accepts the
explicit two-question grammar.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch.distributed as dist

from neuroshard.evolution import serving_diagnosis as diagnosis
from neuroshard.evolution.reference_data import identity, save
from neuroshard.evolution.sharded.graph_execution import GraphNetwork
from neuroshard.evolution.sharded.planned_graph import PlannedGraphNetwork
from neuroshard.evolution.sharded.portable import tensor_path
from neuroshard.evolution.sharded.router_features import EmbeddingFeatures


def run(home, source):
    read = lambda name: json.loads((home/'inputs'/name).read_bytes())
    plan, graph, profile, config = [read(name+'.json') for name in ('plan', 'graph', 'profile', 'planned')]
    facts = json.loads((Path(source)/'config/experiments/continual-expert-facts.json').read_bytes())
    diagnosis.validate(plan, facts, source)
    if any(marker in message['content'] for case in plan['cases'] for message in case['messages']
            for marker in diagnosis.GRAMMAR):
        raise ValueError('Ordinary diagnostic cases cannot use the explicit two-question grammar')
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
        service = PlannedGraphNetwork(net, config, source_home=source, features=features,
                                      planner_weights_home=home/'planner')
        save(output/'ready.json', {'plan': identity(plan), 'service': service.root, 'runtime': net.runtime})
        rows, first = [], None
        for case in plan['cases']:
            if time.monotonic()-started > plan['resources']['max_seconds']:
                raise TimeoutError('The frozen ordinary-serving diagnostic deadline expired')
            ordinary = service.answer(case['messages'], case['max_tokens'])
            gold = [service.answer([{'role': 'user', 'content': atom['gold_question']}], case['max_tokens'])
                    for atom in case['atoms']]
            rows.append(diagnosis.diagnose(case, ordinary, gold))
            if first is None:
                first = ordinary
            save(output/'conversations.json', rows)
        valid, replayed = service.replay(first)
        result = {'format': diagnosis.FORMAT+'/result', 'plan': identity(plan),
                  'service': service.root, 'rows': rows, 'summary': diagnosis.summarize(rows),
                  'replay': {'valid': valid}, 'tokens_issued': 0, 'native_activated': False,
                  'seconds': time.monotonic()-started}
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
