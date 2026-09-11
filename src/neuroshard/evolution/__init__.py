"""Continual full-model training and native settlement experiments.

Initialize the numerical profile before any helper imports PyTorch.
"""
import os

os.environ.update(ATEN_CPU_CAPABILITY='default', MKL_ENABLE_INSTRUCTIONS='SSE4_2',
                  OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
