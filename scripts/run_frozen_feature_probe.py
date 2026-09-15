#!/usr/bin/env python3
"""Check eight exact distributed-versus-cached updates after the frozen trial."""
import argparse
import os
from pathlib import Path

os.environ.setdefault('ATEN_CPU_CAPABILITY', 'default')
os.environ.setdefault('MKL_ENABLE_INSTRUCTIONS', 'SSE4_2')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from neuroshard.evolution.sharded.feature_probe import run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'objects', 'inputs', 'seed', 'home'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--arm', choices=['append', 'tail-control'], required=True)
    run(parser.parse_args())
