#!/usr/bin/env python3
"""Measure whether supervised planning makes the existing learned experts useful together."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from neuroshard.evolution.fusion_data import correct
from neuroshard.evolution.planner_data import INSTRUCTION
from neuroshard.evolution.reference_data import identity, save, sha256
from neuroshard.evolution.sharded.graph_execution import GraphNetwork
from neuroshard.evolution.sharded.learned_graph import configuration as learned_configuration
from neuroshard.evolution.sharded.planned_graph import PlannedGraphNetwork, configuration
from neuroshard.evolution.sharded.planner_training import PlannerTraining
from neuroshard.evolution.sharded.portable import tensor_path
from neuroshard.evolution.sharded.router_features import EmbeddingFeatures


def run(home, source):
    inputs = home/'inputs'
    read = lambda name: json.loads((inputs/name).read_bytes())
    plan, graph, profile, freeze = [read(name+'.json') for name in ('plan', 'graph', 'profile', 'freeze')]
    if (plan['format'] != 'neuroshard-owned-planner-trial-v1'
            or identity(plan) != freeze['plan'] or identity(graph) != freeze['graph']
            or identity(graph) != plan['graph'] or plan['planner']['instruction'] != INSTRUCTION
            or sha256(source/'scripts/run_planner_trial.py') != freeze['driver']
            or sha256(source/'config/experiments/owned-planner-trial.json') != sha256(inputs/'plan.json')):
        raise ValueError('Freeze the complete planner prescription before execution')

    def rows(role):
        path = inputs/('planner-'+role+'.jsonl')
        if sha256(path) != plan['data'][role]['sha256']:
            raise ValueError('The prepared planner role changed')
        result = [json.loads(line) for line in path.read_text().splitlines()]
        if identity(result) != plan['data'][role]['root'] or len(result) != plan['data'][role]['count']:
            raise ValueError('The planner role differs from its complete ordered selection')
        return result

    rank = int(os.environ['RANK'])
    output = home/'results'
    output.mkdir(exist_ok=False)
    dist.init_process_group('gloo', timeout=timedelta(seconds=1200))
    try:
        net = GraphNetwork(graph, profile, objects=home/'objects', interpreter=home/'interpreter',
            seed=home/'seed', source_home=source, rank=rank)
        features = None
        if rank == 0:
            fp = plan['routing']['feature_profile']
            digest = fp['embedding_sha256']
            features = EmbeddingFeatures(tensor_path(home/'interpreter', digest), digest,
                net.tokenizer, graph['tokenizer']['root'], max_tokens=fp['max_tokens'])
        learned = learned_configuration(graph, plan['routing']['model'], plan['routing']['feature_profile'],
            source, plan['routing']['route_models'])

        def service(weights=None):
            spec = configuration(graph, learned, plan['planner'], source,
                plan['expert_prompts'], plan['general_instruction'], plan['routing']['scopes'],
                weights, plan['composer'])
            return PlannedGraphNetwork(net, spec, source_home=source, features=features,
                planner_weights_home=output/'checkpoints')

        baseline = service()
        teacher_cache = {}
        started = time.monotonic()

        def evaluate(current, role, arm):
            selected = rows(role)
            original_path = inputs/('source-'+role+'.jsonl')
            if sha256(original_path) != plan['original_data'][role]:
                raise ValueError('The frozen response-quality references changed')
            originals = {row['id']: row for row in [json.loads(line) for line in original_path.read_text().splitlines()]}
            records = []
            for index, row in enumerate(selected):
                before, began = net.all_owners.sent_tensor_bytes, time.monotonic()
                response = current.answer(row['messages'], plan['max_tokens'])
                measurements = net.all_owners.exchange({'seconds': time.monotonic()-began,
                    'sent_tensor_bytes': net.all_owners.sent_tensor_bytes-before})
                category = 'coreference' if row['variant'] == 'coreference' else row['kind']
                # Labels enter only the evaluator, after ordinary execution.
                exact_plan = response['plan'] == row['questions'] and response['status'] == 'completed'
                if row['kind'] == 'general':
                    if row['id'] not in teacher_cache:
                        start = len(current.trace)
                        teacher_start, teacher_bytes = time.monotonic(), net.all_owners.sent_tensor_bytes
                        teacher_cache[row['id']] = current.call('interpreter', row['messages'],
                            plan['max_tokens'], 'retention_reference')
                        teacher_owners = net.all_owners.exchange({'seconds': time.monotonic()-teacher_start,
                            'sent_tensor_bytes': net.all_owners.sent_tensor_bytes-teacher_bytes})
                        save(output/'retention'/(row['id']+'.json'), {'text': teacher_cache[row['id']],
                            'execution': current.trace[start:], 'owners': teacher_owners})
                        del current.trace[start:]
                    quality = response['status'] == 'completed' and response['text'] == teacher_cache[row['id']]
                else:
                    quality = response['status'] == 'completed' and correct(response['text'], originals[row['source']])
                record = {'id': row['id'], 'category': category, 'exact_plan': exact_plan,
                    'correct': quality, 'response': response, 'owners': measurements}
                records.append(record)
                save(output/(arm+'-'+role+'.json'), records)
                save(output/'progress.json', {'phase': 'evaluate', 'arm': arm, 'role': role,
                    'done': index+1, 'count': len(selected), 'seconds': time.monotonic()-started})
            categories = sorted({row['category'] for row in records})
            return {'records': identity(records), 'categories': {category: {
                'count': sum(row['category'] == category for row in records),
                'correct': sum(row['correct'] for row in records if row['category'] == category),
                'exact_plans': sum(row['exact_plan'] for row in records if row['category'] == category)}
                for category in categories}}

        before = evaluate(baseline, 'dev', 'baseline')
        training_rows = rows('train')
        torch.manual_seed(plan['training_seed'])
        training = PlannerTraining(net, training_rows, plan['training'],
            adapter_rank=plan['adapter_rank'], max_length=plan['max_length'])
        initial = training.save(output/'checkpoints')
        updates, training_resources, boundary = [], [], None
        for step in range(plan['training']['steps']):
            if step == plan['training']['steps']-plan['restart_steps']:
                boundary = training.save(output/'checkpoints')
            began, wire_before = time.monotonic(), net.all_owners.sent_tensor_bytes
            update = training.advance()
            updates.append(update)
            measurements = net.all_owners.exchange({'seconds': time.monotonic()-began,
                'sent_tensor_bytes': net.all_owners.sent_tensor_bytes-wire_before})
            training_resources.append(measurements)
            save(output/'training.json', updates)
            save(output/'training-resources.json', training_resources)
            save(output/'progress.json', {'phase': 'training', 'step': step+1,
                'steps': plan['training']['steps'], 'owners': measurements, 'seconds': time.monotonic()-started})
        terminal = training.save(output/'checkpoints')
        resumed = PlannerTraining(net, training_rows, plan['training'],
            adapter_rank=plan['adapter_rank'], max_length=plan['max_length'])
        resumed.restore(output/'checkpoints', boundary)
        replay = [resumed.advance() for _ in range(plan['restart_steps'])]
        if replay != updates[-plan['restart_steps']:] or resumed.save(output/'checkpoints') != terminal:
            raise ValueError('A fresh planner optimizer did not reproduce the terminal updates')
        save(output/'restart.json', {'boundary': boundary, 'terminal': terminal, 'updates': replay, 'exact': True})
        candidate = service(terminal)
        after = evaluate(candidate, 'dev', 'candidate')

        def passes(score, rule):
            categories = score['categories']
            return all(categories[name]['correct']/categories[name]['count'] >= fraction
                for name, fraction in rule['minimum_accuracy'].items()) and all(
                categories[name]['exact_plans']/categories[name]['count'] >= fraction
                for name, fraction in rule['minimum_plan_accuracy'].items())

        def gain(candidate_score, baseline_score, rule):
            return all(candidate_score['categories'][name]['correct']/candidate_score['categories'][name]['count']-
                baseline_score['categories'][name]['correct']/baseline_score['categories'][name]['count'] >= fraction
                for name, fraction in rule['minimum_gain'].items())

        development_passed = passes(after, plan['quality']['dev']) and gain(after, before, plan['quality']['dev'])
        test_before, test_after = None, None
        if development_passed:
            test_before = evaluate(baseline, 'test', 'baseline')
            test_after = evaluate(candidate, 'test', 'candidate')
        result = {'plan': identity(plan), 'graph': identity(graph), 'initial': initial, 'terminal': terminal,
            'baseline_dev': before, 'candidate_dev': after, 'dev_passed': development_passed,
            'final_opened': test_after is not None, 'baseline_test': test_before, 'candidate_test': test_after,
            'passed': test_after is not None and passes(test_after, plan['quality']['test'])
                and gain(test_after, test_before, plan['quality']['test']),
            'optimizer_replay_exact': True, 'service': candidate.root, 'limitations': plan['limitations']}
        if net.all_owners.exchange(identity(result)) != [identity(result)]*net.world_size:
            raise ValueError('Owners disagree on the complete planner result')
        save(output/'result.json', result)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    options = parser.parse_args()
    run(options.home, options.source)
