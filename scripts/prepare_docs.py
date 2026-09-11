#!/usr/bin/env python3
"""Render documentation from canonical repo files; route remaining links to source."""
from pathlib import Path
import re
import shutil
root = Path(__file__).resolve().parents[1]
files = [*(root / "docs" / name for name in (
    "PUBLIC_TESTNET.md", "MODEL_CARD.md", "API.md", "DEPLOYMENT.md", "PROTOCOL_CANDIDATE_V2.md",
    "LLM_PROTOCOL.md", "LLM_EXPERIMENTS.md", "EVOLUTION_PROTOCOL.md", "DATA_PIPELINE.md", "PROTOCOL_EXPERIMENTS.md", "FUNDAMENTALS_REVIEW.md", "RESEARCH_ROADMAP.md")),
    *(root / name for name in ("CONTRIBUTING.md", "GOVERNANCE.md", "SECURITY.md", "RELEASES.md"))]
target = root / "docs-site/generated"
target.mkdir(parents=True, exist_ok=True)
names = {p.resolve(): p.stem for p in files}
for source in files:
    def link(match):
        url = match[1]
        if re.match(r"(?:[a-z]+:|#|/)", url):
            return match[0]
        path, _, fragment = url.partition("#")
        absolute = (source.parent / path).resolve()
        if absolute in names:
            value = "/generated/" + names[absolute] + ("#" + fragment if fragment else "")
        else:
            value = "https://github.com/neuroshard-ai/neuroshard/blob/main/" + absolute.relative_to(root).as_posix() + ("#" + fragment if fragment else "")
        return "](" + value + ")"
    (target / source.name).write_text(re.sub(r"\]\(([^)]+)\)", link, source.read_text()))
public = root / "docs-site/public"
public.mkdir(exist_ok=True)
for name in ("favicon.ico", "logo_large.png"):
    shutil.copy2(root / "website/public" / name, public / ("logo.png" if name == "logo_large.png" else name))
