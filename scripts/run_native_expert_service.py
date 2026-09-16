#!/usr/bin/env python3
"""Operate a bounded five-owner graph executor behind a local request queue.

This is an operator tool, not a public network API. A native audit daemon's
trusted backend can submit separate audit requests through its controller.
Every request executes anew; the queue never substitutes another auditor's
completed numerical result. Deployment must also impose an external deadline.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch.distributed as dist

from neuroshard.demo import protocol
from neuroshard.evolution.reference_data import identity, save
from neuroshard.evolution.schema import integer, root
from neuroshard.evolution.sharded.graph_execution import GraphNetwork
from neuroshard.evolution.sharded.graph_service import inference_report
from neuroshard.evolution.sharded.graph_quality import evaluate, quality_report


def read(path):
    raw = Path(path).read_bytes()
    if len(raw) > 8 * 1024**2:
        raise ValueError('Operator request exceeds its byte bound')
    return protocol.parse_json(raw)


def run(config):
    rank = int(os.environ['RANK'])
    maximum = integer(config['max_seconds'], 1, 21600)
    deadline = time.monotonic() + maximum
    home = Path(config['home'])
    home.mkdir(parents=True, exist_ok=True)
    queue = home / 'requests'
    queue.mkdir(exist_ok=True)
    results = home / 'results'
    results.mkdir(exist_ok=True)
    graph, profile, baseline, policy = [read(config[name]) for name in ('graph', 'profile', 'baseline', 'quality_policy')]
    dist.init_process_group('gloo', timeout=timedelta(seconds=300))
    try:
        net = GraphNetwork(graph, profile, objects=Path(config['objects']), interpreter=Path(config['interpreter']),
            seed=Path(config['seed']), source_home=Path(config['source_home']), rank=rank)
        learned, planned, fused, planned_tariff = None, None, None, None
        if config.get('fused_service'):
            from neuroshard.evolution.sharded.fused_service import FusedService
            fused = FusedService(net, read(config['fused_service']), Path(config['fused_weights']))
        if config.get('planned_service'):
            from neuroshard.evolution.sharded.planned_graph import PlannedGraphNetwork
            from neuroshard.evolution.sharded.router_features import EmbeddingFeatures
            from neuroshard.evolution.sharded.portable import tensor_path
            planned_config = read(config['planned_service'])
            feature_profile = planned_config['learned']['feature_profile']
            features = None
            if rank == 0:
                digest = feature_profile['embedding_sha256']
                features = EmbeddingFeatures(tensor_path(Path(config['interpreter']), digest), digest,
                    net.tokenizer, graph['tokenizer']['root'], max_tokens=feature_profile['max_tokens'])
            planned = PlannedGraphNetwork(net, planned_config, source_home=Path(config['source_home']), features=features,
                planner_weights_home=Path(config['planner_weights']) if config.get('planner_weights') else None)
            if config.get('planned_tariff'):
                from neuroshard.evolution import planned_metering
                planned_tariff = read(config['planned_tariff'])
                planned_metering.quote(planned.config, graph, 1, planned_tariff)
        elif config.get('planned_tariff'):
            raise ValueError('A conversation tariff requires its installed planned service')
        if config.get('learned_service'):
            from neuroshard.evolution.sharded.learned_graph import LearnedGraphNetwork
            from neuroshard.evolution.sharded.router_features import EmbeddingFeatures
            from neuroshard.evolution.sharded.portable import tensor_path
            learned_config = read(config['learned_service'])
            features = None
            if rank == 0:
                digest = learned_config['feature_profile']['embedding_sha256']
                features = EmbeddingFeatures(tensor_path(Path(config['interpreter']), digest), digest,
                    net.tokenizer, graph['tokenizer']['root'],
                    max_tokens=learned_config['feature_profile']['max_tokens'])
            learned = LearnedGraphNetwork(net, learned_config, source_home=Path(config['source_home']), features=features)
        ready = {'graph': identity(graph), 'executor': identity(profile),
            'rank': rank, 'runtime': net.runtime, 'owned_parameters': net.shard.resident_parameters + (
                net.preserved.shard.resident_parameters if net.preserved else 0)}
        if learned is not None:
            ready['learned_service'] = learned.root
        if planned is not None:
            ready['planned_service'] = planned.root
            if planned_tariff is not None:
                ready['planned_tariff'] = planned_tariff
                ready['planned_tariff_root'] = identity(planned_tariff)
        if fused is not None:
            ready['fused_service'] = fused.root
        save(home / 'ready.json', ready)
        handled = set()
        while True:
            command = None
            if rank == 0:
                heartbeat = time.monotonic() + 1
                while command is None:
                    if time.monotonic() >= deadline:
                        command = {'kind': 'stop'}
                        break
                    pending = sorted(path for path in queue.glob('*.json') if path.name not in handled)
                    if pending:
                        path = pending[0]
                        if path.is_symlink():
                            raise ValueError('Operator queue may not contain symlinks')
                        command = read(path)
                        if root(command['id']) != path.stem:
                            raise ValueError('Request filename differs from its identifier')
                        if (results / path.name).exists():
                            raise ValueError('Use a fresh identifier for a new execution')
                        handled.add(path.name)
                    else:
                        time.sleep(.1)
                        if time.monotonic() >= heartbeat:
                            command = {'kind': 'idle'}
            commands = net.all_owners.exchange(command)
            command = commands[0]
            if any(value is not None for value in commands[1:]):
                raise ValueError('Only the operator queue may dispatch execution')
            if command['kind'] == 'stop':
                break
            if command['kind'] == 'idle':
                continue
            began = time.monotonic()

            def progress(done, count):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Bounded graph service deadline expired')
                if rank == 0:
                    save(home / 'progress.json', {'id': command['id'], 'done': done, 'count': count})

            try:
                if command['kind'] == 'generate':
                    selected = {identity(graph): graph, identity(baseline): baseline}[command['graph']]
                    value = net.answer(command['question'], command['max_tokens'], selected)
                    report = None
                elif command['kind'] == 'generate_learned':
                    if learned is None or command['service'] != learned.root:
                        raise ValueError('Requested learned service is not installed')
                    value = learned.answer(command['question'], command['max_tokens'])
                    report = None
                elif command['kind'] == 'replay_learned':
                    if learned is None:
                        raise ValueError('Learned service is not installed')
                    valid, value = learned.replay(command['response'])
                    report = {'valid': valid, 'service': learned.root, 'response': identity(value)}
                elif command['kind'] == 'generate_planned':
                    if planned is None or command['service'] != planned.root:
                        raise ValueError('Requested planned service is not installed')
                    value = planned.answer(command['messages'], command['max_tokens'])
                    report = ({'metering': planned_metering.meter(planned.config, graph, value, planned_tariff)}
                              if planned_tariff is not None else None)
                elif command['kind'] == 'quote_planned':
                    if planned is None or command['service'] != planned.root or planned_tariff is None:
                        raise ValueError('The requested planned service has no installed tariff')
                    value = planned_metering.quote(planned.config, graph, command['max_tokens'], planned_tariff)
                    report = None
                elif command['kind'] == 'replay_planned':
                    if planned is None:
                        raise ValueError('Planned service is not installed')
                    valid, value = planned.replay(command['response'])
                    report = {'valid': valid, 'service': planned.root, 'response': identity(value)}
                    if valid and planned_tariff is not None:
                        report['metering'] = planned_metering.meter(planned.config, graph, value, planned_tariff)
                elif command['kind'] == 'stream_fused':
                    if fused is None or command['service'] != fused.root:
                        raise ValueError('Requested checked conversation service is not installed')
                    folder = home/'streams'/command['id']
                    # Consume independently of any external client reading the
                    # event files. A disconnected reader cannot strand peers.
                    for index, event in enumerate(fused.events(command['messages'], command['max_tokens'], folder)):
                        if rank == 0:
                            save(folder/'events'/(str(index)+'.json'), event)
                        progress(index+1, None)
                    value, report = event, None
                elif command['kind'] == 'verify_fused':
                    if fused is None or command['service'] != fused.root:
                        raise ValueError('Requested checked conversation service is not installed')
                    value = fused.verify(command['messages'], command['tokens'], command['max_tokens'],
                        home/'stream-audits'/command['id'])
                    report = {'valid': value['passed'], 'service': fused.root, 'response': identity(value)}
                elif command['kind'] == 'inference_audit':
                    report, value = inference_report(command['claim'], net)
                elif command['kind'] == 'quality_audit':
                    report, value = quality_report(command['claim'], policy, Path(config['inputs']), net, progress)
                elif command['kind'] == 'evaluate':
                    value = evaluate(policy, Path(config['inputs']), baseline, graph, net, progress)
                    report = None
                else:
                    raise ValueError('Unknown local graph execution request')
                progress(1, 1)
                result = {'id': command['id'], 'status': 'completed', 'report': report, 'result': value}
            except (ValueError, KeyError, FileNotFoundError, TimeoutError) as error:
                result = {'id': command['id'], 'status': 'unavailable', 'error': type(error).__name__}
            commitments = net.all_owners.exchange(identity(result))
            if commitments != [identity(result)] * net.world_size:
                raise ValueError('Owners disagree on the complete service result')
            save(results / (command['id'] + '.json'), {**result, 'seconds': time.monotonic() - began})
            if result['status'] != 'completed':
                # Resume requires a new explicit operator deployment; do not
                # retry an ambiguous collective or silently skip an audit.
                break
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    run(read(parser.parse_args().config))
