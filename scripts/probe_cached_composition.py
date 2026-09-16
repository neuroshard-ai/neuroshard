#!/usr/bin/env python3
"""Run the prospectively frozen conversation probe on actual owned models."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch.distributed as dist

from neuroshard.evolution.reference_data import identity, save
from neuroshard.evolution.sharded.cached_inference import generate_branch_cached
from neuroshard.evolution.sharded.graph_execution import GraphNetwork
from neuroshard.evolution.sharded.planned_graph import PlannedGraphNetwork
from neuroshard.evolution.sharded.portable import tensor_path
from neuroshard.evolution.sharded.router_features import EmbeddingFeatures
from probe_automatic_composition import score


def run(home, source):
    read = lambda name: json.loads((home/'inputs'/name).read_bytes())
    plan, graph, profile, config = [read(name + '.json') for name in ('plan', 'graph', 'profile', 'planned')]
    rank = int(os.environ['RANK'])
    output = home/'results'
    output.mkdir(exist_ok=False)
    dist.init_process_group('gloo', timeout=timedelta(seconds=180))
    try:
        net = GraphNetwork(graph, profile, objects=home/'objects', interpreter=home/'interpreter',
                           seed=home/'seed', source_home=source, rank=rank)
        feature = config['learned']['feature_profile']
        features = (EmbeddingFeatures(tensor_path(home/'interpreter', feature['embedding_sha256']),
            feature['embedding_sha256'], net.tokenizer, graph['tokenizer']['root']) if rank == 0 else None)
        service = PlannedGraphNetwork(net, config, source_home=source, features=features)
        save(output/'ready.json', {'rank': rank, 'service': service.root, 'graph': identity(graph),
            'plan': identity(plan), 'runtime': net.runtime, 'owned_parameters': net.shard.resident_parameters,
            'preserved_parameters': net.preserved.shard.resident_parameters if net.preserved else 0})
        probes = read('kernel-questions.json')
        kernel = []
        for probe in probes:
            selected, question = probe['model'], probe['question']
            expert = selected in graph['experts']
            network = (net.preserved if selected == 'interpreter' else
                next(iter(net.net.networks.values())) if selected == 'parent' and rank < 3 else
                net.net.networks.get(selected))
            dist.barrier()
            record = None
            if network is not None:
                wire = network.wire if expert else network.parent_wire
                before, started = wire.sent_tensor_bytes, time.monotonic()
                original = network.generate(question, 32, expert)['ids']
                seconds, sent = time.monotonic() - started, wire.sent_tensor_bytes - before
                ids = net.tokenizer.apply_chat_template([{'role': 'user', 'content': question}],
                    tokenize=True, add_generation_prompt=True)
                observation = {}
                cached = generate_branch_cached(network, ids, 32, expert, observation)
                record = {'original_ids': original, 'cached_ids': cached, 'equal': original == cached,
                          'uncached_seconds': seconds, 'uncached_bytes': sent, 'cached': observation}
            observed = net.all_owners.exchange(record)
            kernel.append({'model': selected, 'owners': observed})
            save(output/'kernels.json', kernel)
        planner = []
        for case in plan['planner_cases']:
            service.trace = []
            at = time.monotonic()
            # Only the conversation reaches neural execution. Expected terms
            # and case IDs are supplied exclusively to the scorer afterwards.
            raw = service.call('interpreter', service.prefix + case['messages'],
                               config['planner']['max_tokens'], 'planning')
            planner.append({'id': case['id'], 'text': raw, 'calls': service.trace,
                            'seconds': time.monotonic() - at, **score(raw, case['required_terms'])})
            save(output/'planner.json', planner)
        conversations = []
        for case in plan['end_to_end_cases']:
            at = time.monotonic()
            result = service.answer(case['messages'], case['max_tokens'])
            answers, expected = result['answers'], case['expected_answers']
            # Score actual answer tokens, never the question headings or prompt.
            passed = (result['status'] == 'completed' and len(answers) == len(expected)
                and all(actual['expert'] == target['expert']
                    and all(term in actual['text'].casefold() for term in target['contains'])
                    for actual, target in zip(answers, expected)))
            conversations.append({'id': case['id'], 'passed': passed,
                                  'seconds': time.monotonic() - at, 'response': result})
            save(output/'conversations.json', conversations)
        valid, replay = service.replay(conversations[2]['response'])
        save(output/'replay.json', {'valid': valid, 'response': replay})
        owners = [row for item in kernel for row in item['owners'] if row is not None]
        ratio = sum(row['cached']['sent_tensor_bytes'] for row in owners) / sum(row['uncached_bytes'] for row in owners)
        checks = {'cached_token_equality': all(row['equal'] for row in owners),
            'cached_tensor_bytes': ratio <= plan['pass_rule']['cached_tensor_bytes_ratio_at_most'],
            'planner': sum(row['complete'] for row in planner) >= plan['pass_rule']['planner_at_least'],
            'coreferences': all(row['complete'] for row in planner if row['id'].startswith(('pronoun-', 'prior-'))),
            'answers': sum(row['passed'] for row in conversations) >= plan['pass_rule']['end_to_end_at_least'],
            'mixed_answers': all(row['passed'] for row in conversations if 'mixed' in row['id'] or row['id'] == 'general-and-directory'),
            'replay': valid}
        net.verify_unchanged()
        save(output/'result.json', {'plan': identity(plan), 'service': service.root, 'checks': checks,
            'passed': all(checks.values()), 'cached_tensor_bytes_ratio': ratio,
            'planner_complete': sum(row['complete'] for row in planner),
            'answers_complete': sum(row['passed'] for row in conversations), 'scope': plan['scope']})
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    run(args.home, args.source)
