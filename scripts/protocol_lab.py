#!/usr/bin/env python3
"""Run the candidate with CPU dispatch selected before PyTorch is imported."""
import os
import sys
from pathlib import Path

os.environ.setdefault("ATEN_CPU_CAPABILITY", "default")
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "SSE4_2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuroshard.lab.network import main

if __name__ == "__main__":
    main()
