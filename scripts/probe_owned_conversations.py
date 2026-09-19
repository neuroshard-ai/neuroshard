#!/usr/bin/env python3
"""Exercise an installed conversation executor on precommitted complete inputs."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time
import torch.distributed as dist
from neuroshard.evolution.reference_data import identity, save
from neuroshard.evolution.sharded.graph_execution import GraphNetwork
from neuroshard.evolution.sharded.planned_graph import PlannedGraphNetwork
from neuroshard.evolution.sharded.portable import tensor_path
from neuroshard.evolution.sharded.router_features import EmbeddingFeatures


def run(home, source):
    read = lambda name: json.loads((home/'inputs'/name).read_bytes())
    plan, graph, profile, config = [read(name+'.json') for name in ('plan', 'graph', 'profile', 'planned')]
    output = home/'results'
    output.mkdir(exist_ok=False)
    dist.init_process_group('gloo', timeout=timedelta(seconds=180))
    try:
        net = GraphNetwork(graph, profile, objects=home/'objects', interpreter=home/'interpreter',
            seed=home/'seed', source_home=source, rank=int(os.environ['RANK']))
        feature = config['learned']['feature_profile']
        features = (EmbeddingFeatures(tensor_path(home/'interpreter', feature['embedding_sha256']),
            feature['embedding_sha256'], net.tokenizer, graph['tokenizer']['root']) if net.rank == 0 else None)
        service = PlannedGraphNetwork(net, config, source_home=source, features=features)
        save(output/'ready.json', {'plan': identity(plan), 'service': service.root, 'runtime': net.runtime})
        rows = []
        for case in plan['cases']:
            started = time.monotonic()
            result = service.answer(case['messages'], case['max_tokens'])
            targets = case['expected_answers']
            passed = (result['status'] == 'completed' and len(result['answers']) == len(targets)
                and all(row['expert'] == target['expert']
                    and all(term in row['text'].casefold() for term in target.get('contains', []))
                    and ('json' not in target or json_answer(row['text']) == target['json'])
                    for row, target in zip(result['answers'], targets)))
            rows.append({'id': case['id'], 'passed': passed, 'seconds': time.monotonic()-started,
                         'response': result})
            save(output/'conversations.json', rows)
        valid, replay = service.replay(rows[plan['replay_index']]['response'])
        save(output/'replay.json', {'valid': valid, 'response': replay})
        net.verify_unchanged()
        save(output/'result.json', {'plan': identity(plan), 'service': service.root,
            'count': len(rows), 'complete': sum(row['passed'] for row in rows),
            'replay': valid, 'passed': valid and all(row['passed'] for row in rows), 'scope': plan['scope']})
    finally:
        dist.destroy_process_group()


def json_answer(text):
    try:
        return json.loads(text)
    except ValueError:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    run(args.home, args.source)
