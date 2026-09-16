#!/usr/bin/env python3
"""Bounded, answer-blind review of every draft in a recorded planner run."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch.distributed as dist

from neuroshard.evolution.reference_data import identity, save, sha256
from neuroshard.evolution.sharded.graph_execution import GraphNetwork
from neuroshard.evolution.sharded.planned_graph import PlannedGraphNetwork
from neuroshard.evolution.sharded.portable import tensor_path
from neuroshard.evolution.sharded.router_features import EmbeddingFeatures
from probe_automatic_composition import score


def run(home, source, policy_path):
    read = lambda name: json.loads((home/'inputs'/name).read_bytes())
    policy = json.loads(policy_path.read_bytes())
    plan, graph, profile, config = [read(name+'.json') for name in ('plan', 'graph', 'profile', 'planned')]
    drafts = json.loads((home/'results/planner.json').read_bytes())
    if (identity(plan) != policy['plan'] or sha256(Path(__file__)) != policy['script_sha256']
            or len(drafts) != len(plan['planner_cases'])
            or [row['id'] for row in drafts] != [case['id'] for case in plan['planner_cases']]):
        raise ValueError('The diagnostic changed its frozen inputs or implementation')
    output = home/'review-results'
    output.mkdir(exist_ok=False)
    dist.init_process_group('gloo', timeout=timedelta(seconds=180))
    try:
        net = GraphNetwork(graph, profile, objects=home/'objects', interpreter=home/'interpreter',
            seed=home/'seed', source_home=source, rank=int(os.environ['RANK']))
        # This diagnostic calls only the interpreter; no routing features or
        # expected answer/plan labels enter any model input.
        feature = config['learned']['feature_profile']
        features = (EmbeddingFeatures(tensor_path(home/'interpreter', feature['embedding_sha256']),
            feature['embedding_sha256'], net.tokenizer, graph['tokenizer']['root']) if net.rank == 0 else None)
        service = PlannedGraphNetwork(net, config, source_home=source, features=features)
        rows = []
        for case, draft in zip(plan['planner_cases'], drafts):
            messages = [{'role': 'system', 'content': policy['instruction']},
                {'role': 'user', 'content': json.dumps({'conversation': case['messages'],
                    'draft': draft['text']}, ensure_ascii=False)}]
            service.trace = []
            started = time.monotonic()
            raw = service.call('interpreter', messages, policy['max_tokens'], 'plan_review')
            rows.append({'id': case['id'], 'draft': draft['text'], 'text': raw,
                'seconds': time.monotonic()-started, 'calls': service.trace,
                **score(raw, case['required_terms'])})
            save(output/'planner.json', rows)
        answers = []
        for question in policy['general_questions']:
            service.trace = []
            raw = service.call('interpreter', [{'role': 'system', 'content': config['general_instruction']},
                {'role': 'user', 'content': question}], 64, 'general_answer')
            answers.append({'question': question, 'text': raw, 'calls': service.trace})
        save(output/'general.json', answers)
        net.verify_unchanged()
        save(output/'result.json', {'policy': identity(policy), 'graph': identity(graph),
            'draft_complete': sum(row['complete'] for row in drafts),
            'review_complete': sum(row['complete'] for row in rows), 'count': len(rows),
            'all_coreferences': all(row['complete'] for row in rows
                if row['id'].startswith(('pronoun-', 'prior-'))),
            'scope': 'Exposed development regression; no training or admission; every draft reviewed.'})
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('home', 'source', 'policy'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.home, args.source, args.policy)
