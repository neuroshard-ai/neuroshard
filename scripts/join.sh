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
  role() {
    if [ "$AMX" != yes ] && [ "$2" != any ]; then ok "$1" "no: needs an Intel CPU with AMX (e.g. AWS m7i/r7i)"; return; fi
    short=""
    [ "$MEM" -ge "$3" ] || short="${3} GB RAM (has ${MEM})"
    [ "${DISK:-0}" -ge "$4" ] || short="${short:+$short and }${4} GB free disk (has ${DISK:-0})"
    if [ -z "$short" ]; then ok "$1" "yes"; else ok "$1" "no: needs $short"; fi
  }
  role chat any 8 10; role owner amx 8 20; role validator amx 14 60; role auditor amx 14 30; role contributor amx 30 60
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
# The code lives in its own directory, managed only by this script, never in a checkout someone works in.
SRC="$HOME/.neuroshard/release"
if [ -d "$SRC/.git" ]; then
  if [ -n "$(git -C "$SRC" status --porcelain --untracked-files=no)" ]; then
    echo "$SRC has local changes; this script will not overwrite them. Move them away and rerun." >&2; exit 1
  fi
  git -C "$SRC" fetch -q origin "$BRANCH" && git -C "$SRC" checkout -q --detach FETCH_HEAD
elif [ -e "$SRC" ]; then echo "$SRC exists but is not a NeuroShard release; move it away and rerun." >&2; exit 1
else mkdir -p "$(dirname "$SRC")"; git clone -q --branch "$BRANCH" https://github.com/neuroshard-ai/neuroshard.git "$SRC"; fi
say "Code at $(git -C "$SRC" rev-parse --short HEAD) ($BRANCH)"
VENV="$HOME/.venvs/neuroshard"
[ -x "$VENV/bin/python" ] || uv venv -q -p 3.12.12 "$VENV"
uv pip install -q --python "$VENV/bin/python" -r "$SRC/docs/assistant-runtime-2-requirements.txt"
uv pip install -q --python "$VENV/bin/python" --no-deps --require-hashes -r "$SRC/docs/beacon-requirements.txt"
uv pip install -q --python "$VENV/bin/python" --no-deps -e "$SRC"
CLI="$VENV/bin/neuroshard assistant"
$CLI status --home "$HOME_DIR"

if [ "$ROLE" = chat ]; then say "Start a conversation with: $CLI chat --home $HOME_DIR"; exit 0; fi

ENVIRONMENT="OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false"
case "$ROLE" in
  owner) ARGS="host --shard $SHARD --endpoint $PUBLIC_HOST:28700 --mirror-port 28781"; PORTS="28700 28781" ;;
  validator)
    say "CometBFT v0.38.26 for consensus (the repository's reviewed build: pinned Go toolchain and go.sum)"
    COMETBFT=$("$VENV/bin/python" -c 'from neuroshard.client import runtime; print(runtime.engine())' | tail -1)
    [ "$("$COMETBFT" version)" = 0.38.26 ] || { echo "CometBFT build failed" >&2; exit 1; }
    ENVIRONMENT="$ENVIRONMENT COMETBFT_BINARY=$COMETBFT"
    say "The full parent model, so this validator can judge growth steps (about 7 GB, verified)"
    $CLI prepare-parent --home "$HOME_DIR"
    ARGS="validator --public-host $PUBLIC_HOST"; PORTS="26656 26657 28780 28781" ;;
  auditor) ARGS="audit --shard 1 --shard 2"; PORTS="" ;;
  contributor)
    say "The full parent model contributors train on (about 7 GB, verified)"
    $CLI prepare-parent --home "$HOME_DIR"
    ARGS="grow agent"; PORTS="" ;;
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
sudo systemctl enable -q "neuroshard-$ROLE"
# restart, not start: on a rerun (an upgrade or rollback) the running service must load the code just installed.
sudo systemctl restart "neuroshard-$ROLE"
sleep 5
systemctl is-active "neuroshard-$ROLE" >/dev/null && say "neuroshard-$ROLE is running" || { journalctl -u "neuroshard-$ROLE" -n 20 --no-pager; exit 1; }
[ -n "$PORTS" ] && echo "Open inbound TCP $PORTS to this machine so peers and users reach it."
echo "Logs: journalctl -u neuroshard-$ROLE -f     Keys and state: $HOME_DIR (back it up; it holds your keys)"
case "$ROLE" in
  validator) echo "Your consensus key is printed in the log once the node starts. When the node has caught up (hours),"
             echo "ask owners to vote it in: neuroshard assistant govern propose-validator --consensus-key KEY --power 1" ;;
  owner) echo "The service bonds 5 test NEURO (from the faucet) for shard $SHARD and starts serving." ;;
esac
