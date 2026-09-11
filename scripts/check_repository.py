#!/usr/bin/env python3
"""Check the tracked source boundary and local Markdown links without network access."""
import posixpath
import re
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_ROOTS = {
    "archive", "website", "docs-site", "legacy", "assets", ".neuroshard",
    "venv_build", "dist", "node_modules", "logs", "reports",
}
EXCLUDED_DOCS = ("docs/archive/", "docs/whitepaper/", "docs/figures/",
                 "docs/experiments/", "docs/eval/results/")


def main():
    tracked = set(subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")) - {""}
    errors = []
    for name in sorted(tracked):
        path = PurePosixPath(name)
        if (path.parts[0] in EXCLUDED_ROOTS or name.startswith(EXCLUDED_DOCS)
                or path.suffix.lower() in {".tex", ".pdf", ".safetensors", ".ckpt"}):
            errors.append(f"Outside public source scope: {name}")
        if not (ROOT / name).is_file():
            errors.append(f"Tracked file missing; stage intended removals: {name}")
            continue
        if path.suffix != ".md":
            continue
        for match in re.finditer(r"\]\(([^)\n]+)\)", (ROOT / name).read_text()):
            target = match[1].strip().removeprefix("<").removesuffix(">")
            url = urlsplit(target)
            if url.scheme or url.netloc or not url.path:
                continue
            relative = unquote(url.path)
            resolved = posixpath.normpath(
                relative.lstrip("/") if relative.startswith("/")
                else posixpath.join(str(path.parent), relative))
            if resolved not in tracked and not any(p.startswith(resolved + "/") for p in tracked):
                line = (ROOT / name).read_text().count("\n", 0, match.start()) + 1
                errors.append(f"{name}:{line}: link is not in the tracked tree: {target}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Checked {len(tracked)} tracked files: public source boundaries and local Markdown links pass")


if __name__ == "__main__":
    main()
