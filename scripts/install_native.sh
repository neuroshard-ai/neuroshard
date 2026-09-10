#!/usr/bin/env bash
# Install the CPU reference runtime locally; no keys or node state are created.
set -euo pipefail
neuroshard_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$neuroshard_repo_dir"
if [[ "$(uname -s)/$(uname -m)" != Linux/x86_64 ]]; then
  echo 'This release has been tested on Linux x86_64 only.' >&2
  exit 1
fi
neuroshard_python="${NEUROSHARD_PYTHON:-python3}"
"$neuroshard_python" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10–3.12"'
if [[ ! -x venv_build/bin/python ]]; then
  "$neuroshard_python" -m venv venv_build || {
    echo 'Install your distribution’s Python venv package (Ubuntu: sudo apt install python3-venv), then retry.' >&2
    exit 1
  }
fi
venv_build/bin/python -m pip install --disable-pip-version-check -r docs/llm-requirements.txt
venv_build/bin/python -m pip install --disable-pip-version-check --no-deps -e .
venv_build/bin/python -m pip install --disable-pip-version-check pytest==9.1.1 boto3==1.41.5 ijson==3.4.0
NEUROSHARD_STATE_DIR="$neuroshard_repo_dir/.neuroshard" \
  venv_build/bin/python -c 'from neuroshard.client.runtime import engine; print("Consensus executable:", engine())'
venv_build/bin/python scripts/neuroshard_chain.py --help
echo 'Runtime installed. Follow docs/PUBLIC_TESTNET.md to inspect genesis and initialize your own node.'
