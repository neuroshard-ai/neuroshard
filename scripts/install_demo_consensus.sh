#!/usr/bin/env bash
set -euo pipefail

# Go must be installed separately. Pin the consensus implementation exactly.
neuroshard_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$neuroshard_repo_dir/.neuroshard/tools"
GOBIN="$neuroshard_repo_dir/.neuroshard/tools" go install github.com/cometbft/cometbft/cmd/cometbft@v0.38.26
"$neuroshard_repo_dir/.neuroshard/tools/cometbft" version
