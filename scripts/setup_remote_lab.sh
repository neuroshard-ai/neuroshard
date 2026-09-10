#!/usr/bin/env bash
# Prepare the owner's Ubuntu test host with the same source and pinned CPU runtime.
set -euo pipefail
neuroshard_remote_host="${1:?Usage: setup_remote_lab.sh user@host}"
case "$neuroshard_remote_host" in -*) exit 2 ;; esac
neuroshard_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$neuroshard_repo_dir"
test -x .neuroshard/tools/cometbft
test -x .neuroshard/tools/protocolprobe
test "$(.neuroshard/tools/cometbft version)" = "0.38.26"
ssh -o BatchMode=yes "$neuroshard_remote_host" '
  set -e
  mkdir -p /home/ubuntu/neuroshard-lab
  sudo env DEBIAN_FRONTEND=noninteractive apt-get update -qq
  sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv > /home/ubuntu/neuroshard-lab/system-setup.log 2>&1
  python3 -m venv /home/ubuntu/neuroshard-lab/venv_build
'
tar --exclude='__pycache__' --exclude='*.pyc' -czf .neuroshard/remote-lab-code.tar.gz \
  src docs/demo-requirements.txt docs/eval/data/input.txt \
  scripts/protocol_lab.py scripts/check_numerics_remote.py scripts/remote_lab_process.py scripts/protocol_lab_remote.py \
  scripts/install_demo_consensus.sh scripts/build_protocol_probe.sh tools/protocolprobe \
  tests/test_protocol_candidate.py tests/test_verified_demo.py \
  .neuroshard/tools/cometbft .neuroshard/tools/protocolprobe
scp -q .neuroshard/remote-lab-code.tar.gz "$neuroshard_remote_host:/home/ubuntu/neuroshard-lab/source.tar.gz"
ssh -o BatchMode=yes "$neuroshard_remote_host" '
  set -e
  cd /home/ubuntu/neuroshard-lab
  tar -xzf source.tar.gz
  venv_build/bin/python -m pip install --disable-pip-version-check -r docs/demo-requirements.txt > python-setup.log 2>&1
  .neuroshard/tools/cometbft version
'
