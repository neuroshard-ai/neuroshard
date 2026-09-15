#!/usr/bin/env python3
"""Run the separately frozen repeated-exposure learning comparison."""
import argparse
import os
from pathlib import Path

os.environ.setdefault('ATEN_CPU_CAPABILITY', 'default')
os.environ.setdefault('MKL_ENABLE_INSTRUCTIONS', 'SSE4_2')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from neuroshard.evolution import rehearsal, reference_data as data
from neuroshard.evolution.sharded.rehearsal_job import decide, run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'baseline', 'train', 'evaluate', 'select', 'score'])
    for name in ('parent', 'objects', 'inputs', 'seed', 'home', 'resume', 'baseline-report'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--arm', choices=['baseline', 'append', 'tail-control'])
    parser.add_argument('--candidate-reports', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    if args.home is None:
        parser.error('--home is required')
    if args.command == 'prepare':
        args.home.mkdir(parents=True, exist_ok=False)
        data.save(args.home / 'prepared.json', rehearsal.preparation())
    elif args.command in ('select', 'score'):
        if any(value is None for value in (args.inputs, args.seed, args.baseline_report)):
            parser.error('Decisions require inputs, tokenizer and complete baseline evidence')
        decide(args)
    else:
        if any(value is None for value in (args.parent, args.objects, args.inputs, args.seed, args.arm)):
            parser.error('Execution requires exact parent objects, inputs, tokenizer and arm')
        run(args)
