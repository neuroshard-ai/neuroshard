#!/usr/bin/env python3
"""Copy the current public protocol documents into the static site build."""
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
destination = root / "website/public/docs"
destination.mkdir(parents=True, exist_ok=True)
for name in ("PUBLIC_TESTNET.md", "PROTOCOL_CANDIDATE_V2.md", "PROTOCOL_EXPERIMENTS.md", "RESEARCH_ROADMAP.md"):
    shutil.copy2(root / "docs" / name, destination / name)
papers = root / "website/public/papers"
papers.mkdir(exist_ok=True)
shutil.copy2(root / "docs/FINE2026_neuroshard_short.pdf", papers / "FINE2026_neuroshard_short.pdf")
