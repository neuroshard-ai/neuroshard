#!/usr/bin/env python3
"""Run from a source checkout: python scripts/demo_neuroshard.py up."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuroshard.demo.network import main

if __name__ == "__main__":
    main()
