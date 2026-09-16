#!/usr/bin/env python3
"""Isolate input framing and output-head compatibility of frozen shard knowledge.

This diagnostic deliberately names the source being tested. It is not evidence
of automatic routing. No parameters are trained, no final-set examples are used,
and every input/head condition and answer rule is committed before execution.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch.distributed as dist

from neuroshard.evolution.fusion_data import correct
from neuroshard.evolution.reference_data import identity, save, sha256
from neuroshard.evolution.sharded.cached_inference import generate_branch_cached
from neuroshard.evolution.sharded.graph_execution import GraphNetwork


def run(home, source):
    read = lambda name: json.loads((home/'inputs'/name).read_bytes())
    plan, graph, profile, freeze = [read(name+'.json') for name in ('plan', 'graph', 'profile', 'freeze')]
    if (identity(plan) != freeze['plan'] or identity(graph) != freeze['graph']
            or sha256(source/'scripts/probe_fusion_sources.py') != freeze['driver']
            or sha256(source/'config/experiments/fusion-source-interface-probe.json') != sha256(home/'inputs/plan.json')):
        raise ValueError('Commit the complete source-interface diagnostic before execution')
    rank = int(os.environ['RANK'])
    output = home/'results'
    output.mkdir(exist_ok=False)
    dist.init_process_group('gloo', timeout=timedelta(seconds=1200))
    started = time.monotonic()
    try:
        net = GraphNetwork(graph, profile, objects=home/'objects', interpreter=home/'interpreter',
                           seed=home/'seed', source_home=source, rank=rank)
        records = []
        for number, case in enumerate(plan['cases']):
            network = net.net.networks.get(case['source'])
            for framing, messages in case['inputs'].items():
                tokens = net.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                net.check_context(tokens, plan['max_new_tokens'])
                for head in ('native', 'preserved'):
                    observation, ids, original = {}, None, None
                    if rank == 0:
                        original = net.shard.logits
                        if head == 'preserved':
                            # Only the tied output head changes. Embedding and
                            # all source decoder weights stay the same.
                            net.shard.logits = net.preserved.shard.logits
                    try:
                        if network is not None:
                            ids = generate_branch_cached(network, tokens, plan['max_new_tokens'], True, observation)
                    finally:
                        if rank == 0:
                            net.shard.logits = original
                    observed = net.all_owners.exchange({'ids': ids, 'observation': observation} if network is not None else None)
                    ids = observed[0]['ids']
                    if any(value is not None and value['ids'] != ids for value in observed):
                        raise ValueError('Source owners disagree on generated tokens')
                    text = net.tokenizer.decode(ids, skip_special_tokens=True).strip()
                    value = {'id': case['id'], 'source': case['source'], 'framing': framing,
                             'head': head, 'ids': ids, 'text': text, 'correct': correct(text, case['scoring']),
                             'owners': observed}
                    records.append(value)
                    save(output/'responses.json', records)
                    save(output/'progress.json', {'case': number+1, 'cases': len(plan['cases']),
                         'calls': len(records), 'seconds': time.monotonic()-started})
        summary = {}
        for source_name in sorted({row['source'] for row in records}):
            summary[source_name] = {}
            for framing in plan['framings']:
                summary[source_name][framing] = {}
                for head in ('native', 'preserved'):
                    selected = [row for row in records if row['source'] == source_name
                                and row['framing'] == framing and row['head'] == head]
                    summary[source_name][framing][head] = {'correct': sum(row['correct'] for row in selected),
                                                          'total': len(selected)}
        net.verify_unchanged()
        result = {'plan': identity(plan), 'graph': identity(graph), 'summary': summary,
                  'training_updates': 0, 'final_opened': False, 'scope': plan['scope']}
        if net.all_owners.exchange(identity(result)) != [identity(result)]*net.world_size:
            raise ValueError('Source diagnostic reports disagree')
        save(output/'result.json', result)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    run(args.home, args.source)
