#!/usr/bin/env python3
"""Prepare, train and evaluate the frozen useful-capacity comparison."""
import os

os.environ.setdefault('ATEN_CPU_CAPABILITY', 'default')
os.environ.setdefault('MKL_ENABLE_INSTRUCTIONS', 'SSE4_2')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from neuroshard.evolution.sharded.incremental_job import main

if __name__ == '__main__':
    main()
