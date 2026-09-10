#!/usr/bin/env python3
"""Source-checkout wrapper for the installed native command."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from neuroshard.publicnet.entrypoints import worker

if __name__ == "__main__":
    worker()
