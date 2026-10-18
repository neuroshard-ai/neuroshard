#!/usr/bin/env bash
# Join the NeuroShard assistant network on this machine, as one role, with keys made here.
#
#   curl -LsSf https://raw.githubusercontent.com/neuroshard-ai/neuroshard/main/scripts/join.sh | bash -s -- ROLE [options]
#
# ROLE is one of:
#   owner --shard N        host one shard of the served model and earn for the work you serve (port 28700)
#   validator              order and judge transactions; owners vote you in (ports 26656, 26657, 28780, 28781)
#   auditor                replay owners' signed logs and earn half the bond of any fault you prove
#   contributor            train segments of growth jobs the network funds, and check others' steps
#   chat                   only try the assistant (no service is installed)
#
#   check                  only report which roles this machine can run; installs nothing
#
# Options: --public-host HOST (default: this machine's public address), --home DIR, --branch BRANCH.
# Everything runs as the current user under systemd. Keys never leave this machine. See
# docs/OPERATORS.md for what each role needs, earns and risks.
set -euo pipefail

ROLE="${1:-}"; shift || true
if [ "$ROLE" = check ] || [ "$ROLE" = "--check" ]; then
  # Report what this machine can run; install nothing.
  ok() { printf '  %-12s %s\n' "$1" "$2"; }
  ARCH="$(uname -s)-$(uname -m)"; MEM=$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || echo 0)
  CPUS=$(nproc 2>/dev/null || echo 1); DISK=$(df -BG --output=avail "$HOME" 2>/dev/null | tail -1 | tr -dc 0-9)
  AMX=no; grep -q amx_bf16 /proc/cpuinfo 2>/dev/null && AMX=yes
  echo "This machine: $ARCH, $CPUS threads, ${MEM} GB RAM, ${DISK} GB free disk, AMX: $AMX"
  [ "$ARCH" = Linux-x86_64 ] || { echo "  Every role needs Linux x86_64."; exit 1; }
  role() { if [ "$AMX" = yes ] || [ "$2" = any ]; then if [ "$MEM" -ge "$3" ] && [ "${DISK:-0}" -ge "$4" ]; then ok "$1" "yes"; else ok "$1" "no: needs ${3} GB RAM and ${4} GB disk"; fi; else ok "$1" "no: needs an Intel CPU with AMX (e.g. AWS m7i/r7i)"; fi; }
  role chat any 6 10; role owner amx 8 20; role validator amx 14 60; role auditor amx 14 30; role contributor amx 30 60
  exit 0
fi
SHARD="" ; PUBLIC_HOST="" ; BRANCH="main" ; HOME_DIR="$HOME/.neuroshard/assistant"
while [ $# -gt 0 ]; do
  case "$1" in
    --shard) SHARD="$2"; shift 2 ;;
    --public-host) PUBLIC_HOST="$2"; shift 2 ;;
    --home) HOME_DIR="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
case "$ROLE" in owner|validator|auditor|contributor|chat) ;; *)
  sed -n '2,15p' "$0" 2>/dev/null || true; echo "choose a role: owner, validator, auditor, contributor or chat" >&2; exit 2 ;; esac
[ "$ROLE" = owner ] && [ -z "$SHARD" ] && { echo "owner needs --shard 1 or --shard 2" >&2; exit 2; }

say() { printf '\n== %s\n' "$*"; }
[ "$(uname -s)-$(uname -m)" = "Linux-x86_64" ] || { echo "Linux x86_64 only" >&2; exit 1; }
if [ "$ROLE" != chat ] && ! grep -q amx_bf16 /proc/cpuinfo; then
  echo "This role replays work bit for bit and needs an Intel CPU with AMX (for example AWS m7i or r7i)." >&2
  echo "This CPU lacks it. 'chat' runs anywhere." >&2; exit 1
fi
MEM_GB=$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo)
NEED=8; [ "$ROLE" = validator ] && NEED=14; [ "$ROLE" = contributor ] && NEED=30; [ "$ROLE" = auditor ] && NEED=14
[ "$MEM_GB" -ge "$NEED" ] || { echo "role $ROLE needs about $NEED GB of RAM; this machine has $MEM_GB GB" >&2; exit 1; }
if [ -z "$PUBLIC_HOST" ] && [ "$ROLE" != chat ] && [ "$ROLE" != auditor ] && [ "$ROLE" != contributor ]; then
  PUBLIC_HOST=$(curl -s -m 5 https://checkip.amazonaws.com || true)
  [ -n "$PUBLIC_HOST" ] || { echo "pass --public-host HOST: the address peers reach this machine at" >&2; exit 1; }
fi

say "Pinned runtime: Python 3.12.12 through uv"
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12.12 >/dev/null
SRC="$HOME/neuroshard"
if [ -d "$SRC/.git" ]; then git -C "$SRC" fetch -q origin "$BRANCH" && git -C "$SRC" checkout -q -f --detach FETCH_HEAD
else git clone -q --branch "$BRANCH" https://github.com/neuroshard-ai/neuroshard.git "$SRC"; fi
VENV="$HOME/.venvs/neuroshard"
[ -x "$VENV/bin/python" ] || uv venv -q -p 3.12.12 "$VENV"
uv pip install -q --python "$VENV/bin/python" -r "$SRC/docs/granite-shard-chain-requirements.txt"
uv pip install -q --python "$VENV/bin/python" --no-deps --require-hashes -r "$SRC/docs/beacon-requirements.txt"
uv pip install -q --python "$VENV/bin/python" --no-deps -e "$SRC"
CLI="$VENV/bin/neuroshard assistant"
$CLI status --home "$HOME_DIR"

if [ "$ROLE" = chat ]; then say "Start a conversation with: $CLI chat --home $HOME_DIR"; exit 0; fi

ENVIRONMENT="OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false"
case "$ROLE" in
  owner) ARGS="host --shard $SHARD --endpoint $PUBLIC_HOST:28700 --mirror-port 28781"; PORTS="28700 28781" ;;
  validator)
    say "CometBFT v0.38.26 for consensus"
    if [ ! -x "$HOME/.local/bin/cometbft" ]; then
      # v0.38.26 publishes no binaries; build it from its tagged source with Go's official toolchain.
      GO=go1.23.4; mkdir -p "$HOME/.local/go-toolchain" "$HOME/.local/bin"
      [ -x "$HOME/.local/go-toolchain/go/bin/go" ] || curl -LsSf "https://go.dev/dl/$GO.linux-amd64.tar.gz" \
        | tar -xz -C "$HOME/.local/go-toolchain"
      GOBIN="$HOME/.local/bin" GOFLAGS=-mod=mod "$HOME/.local/go-toolchain/go/bin/go" install \
        github.com/cometbft/cometbft/cmd/cometbft@v0.38.26
    fi
    [ "$("$HOME/.local/bin/cometbft" version)" = 0.38.26 ] || { echo "CometBFT build failed" >&2; exit 1; }
    ENVIRONMENT="$ENVIRONMENT COMETBFT_BINARY=$HOME/.local/bin/cometbft"
    ARGS="validator --public-host $PUBLIC_HOST"; PORTS="26656 26657 28780 28781" ;;
  auditor) ARGS="audit --shard 1 --shard 2"; PORTS="" ;;
  contributor) ARGS="grow agent"; PORTS="" ;;
esac

say "Installing neuroshard-$ROLE as a service"
sudo tee "/etc/systemd/system/neuroshard-$ROLE.service" >/dev/null <<UNIT
[Unit]
Description=NeuroShard assistant $ROLE
After=network-online.target
Wants=network-online.target
[Service]
User=$(id -un)
Environment=$ENVIRONMENT
ExecStart=$VENV/bin/neuroshard assistant $ARGS --home $HOME_DIR
Restart=on-failure
RestartSec=30
[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable -q --now "neuroshard-$ROLE"
sleep 5
systemctl is-active "neuroshard-$ROLE" >/dev/null && say "neuroshard-$ROLE is running" || { journalctl -u "neuroshard-$ROLE" -n 20 --no-pager; exit 1; }
[ -n "$PORTS" ] && echo "Open inbound TCP $PORTS to this machine so peers and users reach it."
echo "Logs: journalctl -u neuroshard-$ROLE -f     Keys and state: $HOME_DIR (back it up; it holds your keys)"
case "$ROLE" in
  validator) echo "Your consensus key is printed in the log once the node starts. When the node has caught up (hours),"
             echo "ask owners to vote it in: neuroshard assistant govern propose-validator --consensus-key KEY --power 1" ;;
  owner) echo "The service bonds 5 test NEURO (from the faucet) for shard $SHARD and starts serving." ;;
esac
