#!/usr/bin/env python3
"""Audit one shard of the existing expert's frozen feature production."""
import argparse
import json
import os
from pathlib import Path

from transformers import LlamaConfig

from neuroshard.evolution import cohort_experiment, incremental_capacity, reference, reference_data as data
from neuroshard.evolution.sharded import prefix_audit
from neuroshard.evolution.sharded.incremental_job import tokenizer_for
from neuroshard.evolution.sharded.model import Partition

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'config/experiments/expert-prefix-audit.json'


def read(path):
    return json.loads(Path(path).read_bytes())


def validate():
    plan = json.loads(incremental_capacity.committed(PLAN))
    if (plan['format'] != 'neuroshard-expert-prefix-audit-plan-v1'
            or plan['tokens_issued'] != 0 or plan['native_activated'] is not False):
        raise ValueError('Require the fixed non-issuing production audit')
    for name, digest in plan['sources'].items():
        incremental_capacity.committed(ROOT / name)
        if data.sha256(ROOT / name) != digest:
            raise ValueError('Production audit source changed after freezing')
    training = json.loads(incremental_capacity.committed(ROOT / 'config/experiments/interpreted-cohort.json'))
    prepared = json.loads(incremental_capacity.committed(ROOT / 'config/experiments/interpreted-cohort-prepared.json'))
    if (data.identity(training) != plan['training_plan'] or data.identity(prepared) != plan['training_prepared']
            or training['runtime'] != plan['runtime']
            or any(plan['sources'][name] != prepared['sources'][name] for name in plan['unchanged_training_kernel'])):
        raise ValueError('Keep the original parent, preparation, forward kernel and runtime')
    return plan, training, prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--rank', type=int, choices=range(3))
    for name in ('parent', 'inputs', 'objects', 'seed', 'home', 'incoming'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    plan, training, prepared = validate()
    if args.check:
        print(json.dumps({'source_and_plan_valid': True, 'plan': data.identity(plan)}))
        return
    if args.rank is None or any(getattr(args, name) is None for name in ('parent', 'inputs', 'objects', 'seed', 'home')):
        parser.error('Provide the rank, every fixed input and a new output directory')
    parent = read(args.parent)
    if data.identity(parent) != plan['parent']:
        raise ValueError('Different immutable parent')
    runtime = reference.configure('cuda', training['threads'])
    runtime['allocator'] = os.environ['PYTORCH_CUDA_ALLOC_CONF']
    if {key: runtime[key] for key in plan['runtime']} != plan['runtime']:
        raise ValueError('Different numerical execution environment')
    tokenizer = tokenizer_for(training, args.seed)
    records = cohort_experiment.rows(prepared, args.inputs, 'train', tokenizer, training['max_length'])
    config = LlamaConfig(**parent['config'])
    config._attn_implementation = 'sdpa'
    shard = Partition(config, training['parent_layout'], args.rank, 'cuda', plan['parameter_limit'])
    binding = {'plan': plan['training_plan'], 'prepared': plan['training_prepared'],
        'job': plan['training_job'], 'previous_graph': training['previous_graph'],
        'retention_cache': prepared['retention_cache'], 'cut': training['split'],
        'batches': data.identity(prepared['batches']), 'runtime': training['runtime']}
    incoming = (args.incoming / 'features', read(args.incoming / 'result.json')) if args.incoming else None
    result = prefix_audit.replay_stage(shard, parent, args.objects, records, prepared['batches'],
        binding, training['split'], training['microbatch'], plan['feature_root'], args.home,
        incoming, plan['max_seconds_per_stage'])
    data.save(args.home / 'runtime.json', {'runtime': runtime, 'plan': data.identity(plan),
        'rank': args.rank, 'parameters': shard.resident_parameters})
    print(json.dumps(result))


if __name__ == '__main__':
    main()
