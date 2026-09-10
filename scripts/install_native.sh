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
venv_build/bin/python -m pip install --disable-pip-version-check -r docs/demo-requirements.txt
venv_build/bin/python -m pip install --disable-pip-version-check --no-deps -e .
mkdir -p .neuroshard/tools .neuroshard/toolchain
neuroshard_go="${NEUROSHARD_GO:-$neuroshard_repo_dir/.neuroshard/toolchain/go/bin/go}"
if [[ ! -x "$neuroshard_go" ]]; then
  neuroshard_archive="$(mktemp .neuroshard/toolchain/go-download.XXXXXX)"
  trap 'rm -f "$neuroshard_archive"' EXIT
  curl --fail --location --retry 3 --proto '=https' --tlsv1.2 \
    https://go.dev/dl/go1.27.1.linux-amd64.tar.gz -o "$neuroshard_archive"
  printf '%s  %s\n' '63d339f0da5ab53635a56f2490a7984dfe12dfcff22ad749f63edaf590168445' "$neuroshard_archive" | sha256sum -c -
  tar -xzf "$neuroshard_archive" -C .neuroshard/toolchain
fi
if [[ "$("$neuroshard_go" version)" != 'go version go1.27.1 linux/amd64' ]]; then
  echo 'Use the pinned Go 1.27.1 toolchain for this release.' >&2
  exit 1
fi
if [[ ! -x .neuroshard/tools/cometbft ]] || [[ "$(.neuroshard/tools/cometbft version)" != 0.38.26 ]]; then
  GOBIN="$neuroshard_repo_dir/.neuroshard/tools" GOTOOLCHAIN=local \
    "$neuroshard_go" install github.com/cometbft/cometbft/cmd/cometbft@v0.38.26
fi
venv_build/bin/python scripts/neuroshard_chain.py --help
echo 'Runtime installed. Follow docs/PUBLIC_TESTNET.md to inspect genesis and initialize your own node.'
