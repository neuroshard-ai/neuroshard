#!/usr/bin/env python3
"""Measure a frozen checked-streaming prescription on the actual owned models."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch.distributed as dist

from neuroshard.evolution.reference_data import identity, save, sha256
from neuroshard.evolution.sharded.batched_audit import verify
from neuroshard.evolution.sharded.fused_graph import generate_fused
from neuroshard.evolution.sharded.fused_service import FusedService
from neuroshard.evolution.sharded.graph_execution import GraphNetwork


def run(home, source):
    read = lambda name: json.loads((home/'inputs'/name).read_bytes())
    plan, graph, profile, freeze = [read(name+'.json') for name in ('plan', 'graph', 'profile', 'freeze')]
    if (plan['format'] != 'neuroshard-checked-streaming-trial-v1'
            or identity(plan) != freeze['plan'] or identity(graph) != freeze['graph']
            or sha256(source/'scripts/run_checked_streaming_trial.py') != freeze['driver']
            or sha256(source/'config/experiments/checked-streaming-trial.json') != sha256(home/'inputs/plan.json')
            or sha256(home/'inputs/dev.jsonl') != plan['data_sha256']
            or plan['service']['graph'] != identity(graph)):
        raise ValueError('Commit the complete streaming prescription and source before execution')
    rows = {row['id']: row for row in [json.loads(line) for line in (home/'inputs/dev.jsonl').read_text().splitlines()]}
    if len({case['id'] for case in plan['cases']}) != len(plan['cases']):
        raise ValueError('Streaming cases must be distinct and prospectively selected')
    rank = int(os.environ['RANK'])
    output = home/'results'
    output.mkdir(exist_ok=False)
    dist.init_process_group('gloo', timeout=timedelta(seconds=1200))
    try:
        net = GraphNetwork(graph, profile, objects=home/'objects', interpreter=home/'interpreter',
                           seed=home/'seed', source_home=source, rank=rank)
        service = FusedService(net, plan['service'], home/'inputs')
        alternate = FusedService(net, {**plan['service'], 'chunk_tokens': plan['different_chunk_tokens']}, home/'inputs')
        started = time.monotonic()
        records = []
        for index, case in enumerate(plan['cases']):
            messages = rows[case['id']]['messages'][:-1]
            prompt = net.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            observed = {}
            cached = generate_fused(net, service.gate, prompt, plan['service']['max_tokens'], observed,
                interface=service.interface, adapt_interfaces=True)
            if cached != case['cached_ids']:
                raise ValueError('The frozen cached response changed before its canonical comparison')
            cached_owners = net.all_owners.exchange(observed)
            folder = output/('case-'+str(index))
            began, first_event, count = time.monotonic(), None, 0
            for event in service.events(messages, plan['service']['max_tokens'], folder):
                if event['kind'] == 'tokens':
                    count += 1
                    if first_event is None:
                        first_event = time.monotonic()-began
            owners = net.all_owners.exchange({'seconds': time.monotonic()-began,
                'first_event_seconds': first_event, 'chunks': count})
            complete = json.loads((folder/'result.json').read_bytes())
            tokens = complete['tokens']
            audit = verify(net, service.gate, prompt, tokens, plan['service']['max_tokens'],
                plan['service']['context'], output/('audit-'+str(index)),
                interface=service.interface, adapt_interfaces=True)
            if not audit['result']['passed']:
                raise ValueError('A delivered stream failed complete fixed-context re-execution')
            chunks = [json.loads((folder/'execution'/('chunk-'+str(i)+'.json')).read_bytes()) for i in range(count)]
            record = {'id': case['id'], 'cached_ids': cached, 'canonical_ids': tokens,
                'cached_owners': cached_owners, 'stream_owners': owners, 'audit': audit,
                'checks_per_chunk': [len(chunk['checks']) for chunk in chunks],
                'stream_tensor_bytes': sum(
                    sum(owner['sent_tensor_bytes'] for owner in chunk['generation_owners'])
                    + sum(owner['sent_tensor_bytes'] for check in chunk['checks'] for owner in check['owners'])
                    for chunk in chunks),
                'same_as_cached': cached == tokens, 'text': complete['text']}
            if index < plan['forgery_cases']:
                forged = list(tokens)
                forged[0] = (forged[0]+1) % graph['parent']['config']['vocab_size']
                eos = net.tokenizer.eos_token_id
                if forged[0] == eos:
                    forged = forged[:1]
                elif len(forged) < plan['service']['max_tokens'] and forged[-1] != eos:
                    forged.append(eos)
                rejected = verify(net, service.gate, prompt, forged, plan['service']['max_tokens'],
                    plan['service']['context'], output/('forged-'+str(index)),
                    interface=service.interface, adapt_interfaces=True)
                if rejected['result']['passed']:
                    raise ValueError('A forged canonical response passed re-execution')
                record['forged_rejected'] = rejected
            if case['id'] in plan['different_chunk_cases']:
                # Same model and fixed-context numerical program; only draft
                # segmentation changes. No event reader is attached.
                repeat = output/('other-chunks-'+str(index))
                for _ in alternate.events(messages, plan['service']['max_tokens'], repeat):
                    pass
                repeated = json.loads((repeat/'result.json').read_bytes())
                if repeated['tokens'] != tokens:
                    raise ValueError('Changing draft chunk boundaries changed canonical output')
                record['different_chunks_equal'] = True
                record['different_chunk_service'] = alternate.root
            records.append(record)
            save(output/'cases.json', records)
            save(output/'progress.json', {'phase': 'canonical_streaming', 'cases': index+1,
                                         'seconds': time.monotonic()-started})
        result = {'plan': identity(plan), 'graph': identity(graph), 'service': service.root,
            'passed': True, 'cases': len(records), 'complete_checks_passed': len(records),
            'different_chunk_checks_passed': sum(row.get('different_chunks_equal', False) for row in records),
            'forgeries_rejected': sum('forged_rejected' in row for row in records),
            'changed_from_cached': sum(not row['same_as_cached'] for row in records),
            'result_root': identity(records), 'limitations': plan['limitations']}
        if net.all_owners.exchange(identity(result)) != [identity(result)]*net.world_size:
            raise ValueError('Owners disagree on the complete streaming result')
        save(output/'result.json', result)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    options = parser.parse_args()
    run(options.home, options.source)
