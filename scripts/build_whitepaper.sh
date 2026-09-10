#!/usr/bin/env bash
set -euo pipefail
neuroshard_paper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../docs" && pwd)"
cd "$neuroshard_paper_dir"
pdflatex -halt-on-error -interaction=nonstopmode FINE2026_neuroshard_short.tex
pdflatex -halt-on-error -interaction=nonstopmode FINE2026_neuroshard_short.tex
printf '%s\n' 'Built docs/FINE2026_neuroshard_short.pdf (public, no login).'
