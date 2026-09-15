#!/usr/bin/env python3
"""Compare owner-local cached inference with the original generator, then replay."""
import argparse
import os
from pathlib import Path

os.environ.setdefault('ATEN_CPU_CAPABILITY', 'default')
os.environ.setdefault('MKL_ENABLE_INSTRUCTIONS', 'SSE4_2')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from neuroshard.evolution.sharded import cache_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'compare', 'replay', 'score'])
    for name in ('plan', 'seed', 'home', 'checkpoint'):
        parser.add_argument('--' + name, required=True, type=Path)
    for name in ('prepared', 'inputs', 'original-prepared', 'witness', 'reports', 'audits'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    needed = ['original_prepared'] if args.command == 'prepare' else ['prepared', 'inputs']
    if args.command == 'replay':
        needed.append('witness')
    if args.command == 'score':
        needed.extend(['reports', 'audits'])
    if any(getattr(args, name) is None for name in needed):
        parser.error('Missing input paths for ' + args.command)
    getattr(cache_probe, args.command)(args)


if __name__ == '__main__':
    main()
