#!/usr/bin/env python3
"""Reconstruct an archived expert's update records without issuing rewards."""
import argparse
import json
from pathlib import Path

from neuroshard.evolution import incremental_capacity as base, reference, reference_data as data
from neuroshard.evolution import cohort_experiment
from neuroshard.evolution.sharded import expert_replay, feature_bank, incremental, incremental_state
from neuroshard.evolution.sharded.feature_probe import load_head
from neuroshard.evolution.sharded.incremental_job import tokenizer_for
from neuroshard.evolution.sharded.model import Partition
from transformers import LlamaConfig

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'config/experiments/expert-window-replay.json'


def read(path):
    return json.loads(Path(path).read_bytes())


def validate():
    plan = json.loads(base.committed(PLAN))
    if (plan['format'] != 'neuroshard-expert-window-replay-v1' or plan['operation'] != 'replay'
            or plan['tokens_issued'] != 0 or plan['native_activated'] is not False):
        raise ValueError('Require the frozen non-issuing replay experiment')
    for name, digest in plan['sources'].items():
        base.committed(ROOT / name)
        if data.sha256(ROOT / name) != digest:
            raise ValueError('Replay source changed after the plan was committed')
    training = json.loads(base.committed(ROOT / 'config/experiments/interpreted-cohort.json'))
    prepared = json.loads(base.committed(ROOT / 'config/experiments/interpreted-cohort-prepared.json'))
    if (data.identity(training) != plan['training_plan'] or data.identity(prepared) != plan['training_prepared']
            or any(plan['sources'][name] != prepared['sources'][name] for name in plan['unchanged_training_kernel'])
            or training['runtime'] != plan['runtime']):
        raise ValueError('Keep the actual original training kernel, inputs and runtime')
    return plan, training, prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    for name in ('parent', 'inputs', 'objects', 'seed', 'features', 'metrics', 'home'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    plan, training, prepared = validate()
    if args.check:
        print(json.dumps({'source_and_plan_valid': True, 'plan': data.identity(plan)}))
        return
    if any(getattr(args, name) is None for name in ('parent', 'inputs', 'objects', 'seed', 'features', 'metrics', 'home')):
        parser.error('Provide every frozen input and a new output directory')
    parent = read(args.parent)
    if data.identity(parent) != plan['parent'] or data.sha256(args.metrics) != plan['metrics_sha256']:
        raise ValueError('Parent or observed update trajectory changed')
    metrics = [json.loads(line) for line in args.metrics.read_text().splitlines()]
    runtime = reference.configure('cuda', training['threads'])
    import os
    runtime['allocator'] = os.environ['PYTORCH_CUDA_ALLOC_CONF']
    if {key: runtime[key] for key in plan['runtime']} != plan['runtime']:
        raise ValueError('Replay numerical runtime changed')
    tokenizer = tokenizer_for(training, args.seed)
    records = cohort_experiment.rows(prepared, args.inputs, 'train', tokenizer, training['max_length'])
    config = LlamaConfig(**parent['config'])
    config._attn_implementation = 'sdpa'
    shard = Partition(config, training['expert_layout'], 3, 'cuda', training['parameter_limit'])
    incremental_state.initialize(shard, parent, args.objects, 'tail-control', training['split'])
    optimizer = incremental.configure(shard, training['split'], training['training'])
    head = load_head(parent, args.objects, 'cuda')
    owned = shard.resident_parameters + sum(p.numel() for p in head.parameters())
    if owned > plan['parameter_limit']:
        raise ValueError('The replay owner must not hold a complete parent model')
    binding = {'plan': plan['training_plan'], 'prepared': plan['training_prepared'],
        'job': plan['training_job'], 'previous_graph': training['previous_graph'],
        'retention_cache': prepared['retention_cache'], 'cut': training['split'],
        'batches': data.identity(prepared['batches']), 'runtime': training['runtime']}
    bank = feature_bank.Reader(args.features, plan['feature_root'], binding, config,
                               training['microbatch'], len(prepared['batches']))
    expected = {int(step): digest for step, digest in plan['checkpoints'].items()}
    result = expert_replay.replay(shard, head, optimizer, parent, plan['training_job'],
        training['training'], training['objective'], bank, records, prepared['batches'], prepared['schedule'],
        metrics, expected, plan['numerical_profile'], args.home, plan['max_seconds'])
    data.save(args.home / 'runtime.json', {'runtime': runtime, 'owned_parameters': owned,
                                         'plan': data.identity(plan), 'training_job': plan['training_job']})
    print(json.dumps(result))


if __name__ == '__main__':
    main()
