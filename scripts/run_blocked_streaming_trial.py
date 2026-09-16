#!/usr/bin/env python3
"""Measure a fixed-block target and a separately owned preserved drafter."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch.distributed as dist

from neuroshard.evolution.reference_data import identity, save, sha256
from neuroshard.evolution.sharded import blocked_inference as blocked
from neuroshard.evolution.sharded.expert_interface import ExpertInterface
from neuroshard.evolution.sharded.fused_graph import commitment
from neuroshard.evolution.sharded.fusion_trial import synchronize
from neuroshard.evolution.sharded.graph_execution import GraphNetwork
from neuroshard.evolution.sharded.interface_training import initialize_weights
from neuroshard.evolution.sharded.mixture import ProbabilityMixture


def run(home, source):
    read = lambda name: json.loads((home/'inputs'/name).read_bytes())
    plan, graph, profile, freeze = [read(name+'.json') for name in ('plan', 'graph', 'profile', 'freeze')]
    if (plan['format'] != 'neuroshard-blocked-streaming-trial-v1'
            or identity(plan) != freeze['plan'] or identity(graph) != freeze['graph']
            or identity(graph) != plan['graph']
            or sha256(source/'scripts/run_blocked_streaming_trial.py') != freeze['driver']
            or sha256(source/'config/experiments/blocked-streaming-trial.json') != sha256(home/'inputs/plan.json')
            or sha256(home/'inputs/dev.jsonl') != plan['data_sha256']):
        raise ValueError('Commit the complete block prescription before execution')
    rows = {row['id']: row for row in [json.loads(line) for line in (home/'inputs/dev.jsonl').read_text().splitlines()]}
    rank = int(os.environ['RANK'])
    output = home/'results'
    output.mkdir(exist_ok=False)
    dist.init_process_group('gloo', timeout=timedelta(seconds=1200))
    try:
        net = GraphNetwork(graph, profile, objects=home/'objects', interpreter=home/'interpreter',
                           seed=home/'seed', source_home=source, rank=rank)
        width, weights = graph['parent']['config']['hidden_size'], plan['weights']
        layout = weights['gate']['binding']['layout']
        gate = ProbabilityMixture(width, {name: width for name in ['parent', *graph['experts']]},
            rank=layout['rank'], max_context=layout['max_context']).to(net.shard.device_name).eval()
        if rank == 0:
            initialize_weights(gate, home/'inputs', weights['gate'])
        synchronize(net, gate)
        adapter = None
        if rank >= 3:
            name = graph['descriptor']['rules'][rank-3]['id']
            checkpoint = weights['interfaces'][name]
            adapter = ExpertInterface(net.shard, identity(graph['experts'][name]), checkpoint['binding']['layout']['rank']).eval()
            initialize_weights(adapter, home/'inputs', checkpoint)
        options = {'block_size': plan['block_size'], 'interface': adapter, 'adapt_interfaces': True}
        started, records = time.monotonic(), []

        def generate(prompt, maximum, folder, method='hub'):
            began, first = time.monotonic(), None
            events = []
            for event in blocked.stream(net, gate, prompt, maximum, plan['context'], folder,
                    draft_method=method, **options):
                if first is None:
                    first = time.monotonic()-began
                events.append(event)
            elapsed = net.all_owners.exchange({'seconds': time.monotonic()-began, 'first_event_seconds': first})
            prefill = json.loads((folder/'prefill.json').read_bytes())
            target_bytes = sum(owner['sent_tensor_bytes'] for trace in prefill+[event['target'] for event in events]
                               for owner in trace['owners'])
            draft_bytes = sum(owner.get('sent_tensor_bytes', 0) for event in events for owner in event['draft_owners'])
            return {'tokens': [token for event in events for token in event['tokens']],
                    'owners': elapsed, 'target_calls': len(events), 'prefill_blocks': len(prefill),
                    'target_tensor_bytes': target_bytes, 'draft_tensor_bytes': draft_bytes,
                    'tensor_bytes': target_bytes+draft_bytes}

        first_row = rows[plan['cases'][0]]
        first_prompt = net.tokenizer.apply_chat_template(first_row['messages'][:-1], tokenize=True, add_generation_prompt=True)
        warmup = generate(first_prompt, plan['warmup_tokens'], output/'warmup')
        save(output/'warmup.json', warmup)
        for index, key in enumerate(plan['cases']):
            prompt = net.tokenizer.apply_chat_template(rows[key]['messages'][:-1], tokenize=True, add_generation_prompt=True)
            generated = generate(prompt, plan['max_tokens'], output/('case-'+str(index)))
            tokens = generated['tokens']
            began = time.monotonic()
            audit = blocked.verify(net, gate, prompt, tokens, plan['max_tokens'], plan['context'],
                output/('audit-'+str(index)), **options)
            audit_owners = net.all_owners.exchange({'seconds': time.monotonic()-began})
            if not audit['passed']:
                raise ValueError('Emitted tokens failed fresh fixed-block reconstruction')
            record = {'id': key, 'generation': generated, 'audit_owners': audit_owners,
                'audit_tensor_bytes': sum(owner['sent_tensor_bytes'] for block in audit['blocks'] for owner in block['owners']),
                'audit_passed': True, 'text': net.tokenizer.decode(tokens, skip_special_tokens=True)}
            if index < plan['forgery_cases']:
                forged = list(tokens)
                forged[0] = (forged[0]+1) % graph['parent']['config']['vocab_size']
                eos = net.tokenizer.eos_token_id
                if forged[0] == eos:
                    forged = forged[:1]
                elif len(forged) < plan['max_tokens'] and forged[-1] != eos:
                    forged.append(eos)
                denied = blocked.verify(net, gate, prompt, forged, plan['max_tokens'], plan['context'],
                    output/('forged-'+str(index)), **options)
                if denied['passed'] or denied['mismatches'][0] != 0:
                    raise ValueError('First-token forgery was not rejected at its first position')
                record['forged_rejected'] = True
            if key in plan['padding_draft_cases']:
                independent = generate(prompt, plan['max_tokens'], output/('padding-'+str(index)), 'padding')
                if independent['tokens'] != tokens:
                    raise ValueError('Changing the speculative proposal changed canonical block output')
                record['padding_draft'] = independent
            records.append(record)
            save(output/'cases.json', records)
            save(output/'progress.json', {'phase': 'blocked_streaming', 'cases': index+1,
                                         'seconds': time.monotonic()-started})
        long = [row for row in records if row['id'] in plan['performance']['long_cases']]
        limits = plan['performance']
        checks = {
            'long_responses': len(long) == len(limits['long_cases']) and all(len(row['generation']['tokens']) >= limits['minimum_tokens'] for row in long),
            'seconds_per_token': all(max(owner['seconds'] for owner in row['generation']['owners'])/len(row['generation']['tokens']) <= limits['max_seconds_per_token'] for row in long),
            'tensor_bytes_per_token': all(row['generation']['tensor_bytes']/len(row['generation']['tokens']) <= limits['max_tensor_bytes_per_token'] for row in long),
            'first_checked_event': all(max(owner['first_event_seconds'] for owner in row['generation']['owners']) <= limits['max_first_event_seconds'] for row in records),
            'audit_seconds': all(max(owner['seconds'] for owner in row['audit_owners']) <= limits['max_audit_seconds'] for row in long),
            'audit_tensor_bytes': all(row['audit_tensor_bytes'] <= limits['max_audit_tensor_bytes'] for row in long)}
        result = {'plan': identity(plan), 'graph': identity(graph), 'method': blocked.FORMAT,
            'gate': commitment(gate), 'passed': all(checks.values()), 'performance_checks': checks,
            'cases': len(records), 'fresh_replay_passed': len(records),
            'forgeries_rejected': sum(row.get('forged_rejected', False) for row in records),
            'padding_draft_checks_passed': sum('padding_draft' in row for row in records),
            'result_root': identity(records), 'limitations': plan['limitations']}
        if net.all_owners.exchange(identity(result)) != [identity(result)]*net.world_size:
            raise ValueError('Owners disagree on the complete block result')
        save(output/'result.json', result)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    options = parser.parse_args()
    run(options.home, options.source)
