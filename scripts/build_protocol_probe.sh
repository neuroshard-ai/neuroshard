#!/usr/bin/env bash
set -euo pipefail
neuroshard_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$neuroshard_repo_dir/.neuroshard/tools"
cd "$neuroshard_repo_dir/tools/protocolprobe"
go build -mod=readonly -o "$neuroshard_repo_dir/.neuroshard/tools/protocolprobe" .
