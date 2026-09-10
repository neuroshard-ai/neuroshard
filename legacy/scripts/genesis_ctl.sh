#!/bin/bash
# ============================================================================
# NeuroShard Genesis Controller v2.0
# ============================================================================
# Utility script to manage the Genesis data population services.
# Now supports both sequential (v3) and parallel (v4) modes.
#
# Usage:
#   ./genesis_ctl.sh status        - Show current status
#   ./genesis_ctl.sh start         - Start parallel service (recommended)
#   ./genesis_ctl.sh stop          - Stop all genesis services
#   ./genesis_ctl.sh restart       - Restart services
#   ./genesis_ctl.sh logs          - Tail the logs
#   ./genesis_ctl.sh migrate       - Migrate from old to parallel service
#   ./genesis_ctl.sh install       - Install systemd service
#
# Legacy commands (for backward compatibility):
#   ./genesis_ctl.sh start-legacy  - Start old sequential service
# ============================================================================

NEUROSHARD_DIR="/home/ubuntu/neuroshard"
VENV_DIR="${NEUROSHARD_DIR}/venv_build"
SERVICE_NAME="neuroshard-genesis"
PARALLEL_SERVICE_NAME="neuroshard-genesis-parallel"
BUCKET="neuroshard-training-data"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║      NeuroShard Genesis Data Controller v2.0                 ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Check which services are running
check_running() {
    local parallel_running=false
    local legacy_running=false
    
    if pgrep -f "genesis_parallel.py" > /dev/null; then
        parallel_running=true
    fi
    
    if pgrep -f "populate_genesis_s3.py" > /dev/null; then
        legacy_running=true
    fi
    
    echo "${parallel_running}:${legacy_running}"
}

status() {
    print_header
    
    local running=$(check_running)
    local parallel_running=$(echo $running | cut -d: -f1)
    local legacy_running=$(echo $running | cut -d: -f2)
    
    echo -e "${YELLOW}=== Process Status ===${NC}"
    
    # Parallel populator
    if [ "$parallel_running" = "true" ]; then
        echo -e "${GREEN}● Genesis PARALLEL populator is RUNNING (v4.0)${NC}"
        ps aux | grep "genesis_parallel.py" | grep -v grep | head -3
    else
        echo -e "${BLUE}○ Genesis parallel populator is not running${NC}"
    fi
    
    echo ""
    
    # Legacy populator
    if [ "$legacy_running" = "true" ]; then
        echo -e "${YELLOW}● Genesis LEGACY populator is RUNNING (v3.0)${NC}"
        ps aux | grep "populate_genesis_s3.py" | grep -v grep | head -3
        echo -e "${YELLOW}  ⚠ Consider migrating to parallel mode: ./genesis_ctl.sh migrate${NC}"
    fi
    
    if [ "$parallel_running" = "false" ] && [ "$legacy_running" = "false" ]; then
        echo -e "${RED}○ No genesis populator is running${NC}"
    fi
    
    echo ""
    echo -e "${YELLOW}=== Systemd Services ===${NC}"
    if systemctl is-active --quiet ${PARALLEL_SERVICE_NAME} 2>/dev/null; then
        echo -e "${GREEN}● ${PARALLEL_SERVICE_NAME} is active${NC}"
    elif systemctl is-active --quiet ${SERVICE_NAME} 2>/dev/null; then
        echo -e "${YELLOW}● ${SERVICE_NAME} (legacy) is active${NC}"
    else
        echo -e "${BLUE}○ No systemd service active${NC}"
    fi
    
    echo ""
    echo -e "${YELLOW}=== Configured Sources ===${NC}"
    if [ -f "${NEUROSHARD_DIR}/scripts/genesis_sources.json" ]; then
        python3 -c "
import json
with open('${NEUROSHARD_DIR}/scripts/genesis_sources.json') as f:
    config = json.load(f)
for s in sorted(config.get('sources', []), key=lambda x: x.get('priority', 99)):
    status = '●' if s.get('enabled', True) else '○'
    print(f\"  {status} {s['name']}: target {s.get('target_shards', 0):,} shards - {s.get('description', '')}\")
"
    else
        echo "  No config file found"
    fi
    
    echo ""
    echo -e "${YELLOW}=== S3 Data Status ===${NC}"
    cd "${NEUROSHARD_DIR}"
    source "${VENV_DIR}/bin/activate" 2>/dev/null
    export PYTHONPATH="${NEUROSHARD_DIR}/src:${PYTHONPATH}"
    
    python3 - <<'EOF'
import os
import json
import boto3

# Load env
for env_path in ['website/.env', '.env']:
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    os.environ[k] = v.strip("'").strip('"')
        break

# Load source config for targets
source_targets = {}
config_path = 'scripts/genesis_sources.json'
if os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)
    for s in config.get('sources', []):
        if s.get('enabled', True):
            source_targets[s['name']] = s.get('target_shards', 500000)

try:
    s3 = boto3.client('s3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    )
    
    bucket = 'neuroshard-training-data'
    
    # Get manifest
    obj = s3.get_object(Bucket=bucket, Key='manifest.json')
    m = json.loads(obj['Body'].read())
    
    total_shards = m['total_shards']
    total_tokens = m.get('total_tokens', 0)
    
    # Calculate total target from all sources
    total_target = sum(source_targets.values()) if source_targets else 500000
    
    print(f"  Total Shards:    {total_shards:,} / {total_target:,} ({100*total_shards/total_target:.1f}%)")
    print(f"  Total Tokens:    {total_tokens/1e9:.2f}B")
    print(f"  Total Size:      {total_shards * 10 / 1000:.1f}GB / {total_target * 10 / 1000:.0f}GB")
    
    # Calculate rate based on number of active sources (parallel is faster)
    active_sources = len(m.get('sources', {}))
    rate_per_hour = 3600 * max(1, active_sources)  # ~60/min per source
    remaining = total_target - total_shards
    hours = remaining / rate_per_hour if rate_per_hour > 0 else 0
    print(f"  ETA:             ~{hours:.0f} hours remaining (at {rate_per_hour:,}/hour)")
    
    print(f"\n  Per-Source Progress:")
    for src, stats in m.get('sources', {}).items():
        target = source_targets.get(src, 500000)
        current = stats['shards']
        pct = 100 * current / target if target > 0 else 0
        bar_len = 20
        filled = int(bar_len * current / target) if target > 0 else 0
        bar = '█' * filled + '░' * (bar_len - filled)
        print(f"    {src}:")
        print(f"      [{bar}] {pct:.1f}%")
        print(f"      {current:,} / {target:,} shards, {stats['tokens']/1e9:.2f}B tokens")
    
    # Get checkpoint
    try:
        obj = s3.get_object(Bucket=bucket, Key='checkpoints.json')
        checkpoints = json.loads(obj['Body'].read())
        print(f"\n  Active Checkpoints:")
        for src, cp in checkpoints.items():
            print(f"    {src}: doc {cp['documents_processed']:,}, last shard {cp['last_shard_id']}")
    except:
        pass
    
    # Show tokenizer status
    try:
        obj = s3.get_object(Bucket=bucket, Key='tokenizer.json')
        tok = json.loads(obj['Body'].read())
        print(f"\n  Tokenizer Status:")
        print(f"    Vocabulary: {tok.get('next_merge_id', 0):,} / {tok.get('vocab_size', 32000):,} tokens")
        for src, merges in tok.get('sources_contributed', {}).items():
            print(f"    {src} contributed: {merges:,} BPE merges")
    except:
        pass
        
except Exception as e:
    print(f"  Error: {e}")
EOF
}

start() {
    print_header
    echo -e "${GREEN}Starting PARALLEL genesis service (v4.0)...${NC}"
    echo ""
    
    # Check if already running
    local running=$(check_running)
    local parallel_running=$(echo $running | cut -d: -f1)
    
    if [ "$parallel_running" = "true" ]; then
        echo -e "${YELLOW}Parallel populator already running!${NC}"
        return 0
    fi
    
    # Stop legacy if running
    local legacy_running=$(echo $running | cut -d: -f2)
    if [ "$legacy_running" = "true" ]; then
        echo -e "${YELLOW}Stopping legacy service first...${NC}"
        stop_legacy
        sleep 3
    fi
    
    # Start parallel
    cd "${NEUROSHARD_DIR}"
    source "${VENV_DIR}/bin/activate"
    export PYTHONPATH="${NEUROSHARD_DIR}/src:${PYTHONPATH}"
    
    mkdir -p logs
    
    echo "Starting parallel populator..."
    nohup python3 scripts/genesis_parallel.py \
        --bucket "${BUCKET}" \
        --max-sources 4 \
        --upload-workers 8 \
        > logs/genesis_parallel.log 2>&1 &
    
    local pid=$!
    echo -e "${GREEN}Started! PID: ${pid}${NC}"
    echo ""
    echo "Features:"
    echo "  • Multiple sources in parallel (up to 4)"
    echo "  • Vocabulary grows from ALL sources immediately"
    echo "  • ~40-60 shards/minute total throughput"
    echo ""
    echo "Logs: tail -f ${NEUROSHARD_DIR}/logs/genesis_parallel.log"
}

start_legacy() {
    print_header
    echo -e "${YELLOW}Starting LEGACY genesis service (v3.0)...${NC}"
    echo -e "${YELLOW}Consider using parallel mode instead: ./genesis_ctl.sh start${NC}"
    echo ""
    
    # Check if systemd service is installed
    if systemctl list-unit-files | grep -q ${SERVICE_NAME}; then
        echo "Starting systemd service..."
        sudo systemctl start ${SERVICE_NAME}
        sleep 2
        systemctl status ${SERVICE_NAME} --no-pager | head -5
    else
        echo "Systemd service not installed. Starting manually..."
        cd "${NEUROSHARD_DIR}"
        source "${VENV_DIR}/bin/activate"
        
        mkdir -p logs
        nohup bash scripts/genesis_service.sh > logs/genesis_service.log 2>&1 &
        
        echo -e "${GREEN}Started! PID: $!${NC}"
        echo "Logs: tail -f ${NEUROSHARD_DIR}/logs/genesis_service.log"
    fi
}

stop() {
    print_header
    echo "Stopping all genesis services..."
    
    # Stop parallel
    if pgrep -f "genesis_parallel.py" > /dev/null; then
        echo "Stopping parallel populator..."
        pkill -TERM -f "genesis_parallel.py"
        sleep 3
        if pgrep -f "genesis_parallel.py" > /dev/null; then
            pkill -9 -f "genesis_parallel.py"
        fi
        echo -e "${GREEN}Parallel populator stopped${NC}"
    fi
    
    # Stop legacy
    stop_legacy
    
    echo -e "${GREEN}All services stopped${NC}"
}

stop_legacy() {
    # Try systemd first
    if systemctl is-active --quiet ${SERVICE_NAME} 2>/dev/null; then
        sudo systemctl stop ${SERVICE_NAME}
        echo "Legacy systemd service stopped"
    fi
    
    # Also kill any manual processes
    if pgrep -f "populate_genesis_s3.py" > /dev/null; then
        echo "Stopping legacy populator processes..."
        pkill -TERM -f "populate_genesis_s3.py"
        sleep 5
        
        if pgrep -f "populate_genesis_s3.py" > /dev/null; then
            pkill -9 -f "populate_genesis_s3.py"
        fi
        echo "Legacy populator stopped"
    fi
}

migrate() {
    print_header
    echo -e "${CYAN}=== Migrating to Parallel Genesis Service ===${NC}"
    echo ""
    
    local running=$(check_running)
    local legacy_running=$(echo $running | cut -d: -f2)
    
    if [ "$legacy_running" = "true" ]; then
        echo -e "${YELLOW}Step 1: Stopping legacy service...${NC}"
        echo "  (This will save checkpoint, data is safe)"
        stop_legacy
        sleep 3
        echo -e "${GREEN}  ✓ Legacy service stopped${NC}"
    else
        echo -e "${GREEN}Step 1: No legacy service running ✓${NC}"
        fi
    
    echo ""
    echo -e "${YELLOW}Step 2: Starting parallel service...${NC}"
    start
    
    echo ""
    echo -e "${GREEN}=== Migration Complete ===${NC}"
    echo ""
    echo "The parallel service will:"
    echo "  • Continue from existing checkpoints (no data loss)"
    echo "  • Run multiple sources simultaneously"
    echo "  • Grow vocabulary from ALL sources immediately"
    echo ""
    echo "Monitor progress: ./genesis_ctl.sh status"
}

restart() {
    stop
    sleep 2
    start
}

logs() {
    echo "Tailing genesis logs (Ctrl+C to exit)..."
    
    # Try parallel logs first
    if [ -f "${NEUROSHARD_DIR}/logs/genesis_parallel.log" ]; then
        tail -f "${NEUROSHARD_DIR}/logs/genesis_parallel.log"
    elif [ -f "${NEUROSHARD_DIR}/logs/genesis_service.log" ]; then
        tail -f "${NEUROSHARD_DIR}/logs/genesis_service.log"
    else
    echo "No log files found"
        echo "Start the service first: ./genesis_ctl.sh start"
    fi
}

install_service() {
    print_header
    echo "Installing systemd service for PARALLEL populator..."
    
    # Create service file
    cat > /tmp/neuroshard-genesis-parallel.service <<EOF
[Unit]
Description=NeuroShard Genesis Parallel Data Population Service
Documentation=https://github.com/neuroshard/neuroshard
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/neuroshard
Environment="PATH=/home/ubuntu/neuroshard/venv_build/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONPATH=/home/ubuntu/neuroshard/src"
ExecStart=/home/ubuntu/neuroshard/venv_build/bin/python3 /home/ubuntu/neuroshard/scripts/genesis_parallel.py --bucket neuroshard-training-data --max-sources 4 --upload-workers 8
Restart=on-failure
RestartSec=30
StandardOutput=append:/home/ubuntu/neuroshard/logs/genesis_parallel.log
StandardError=append:/home/ubuntu/neuroshard/logs/genesis_parallel.log

# Resource limits
MemoryMax=14G
CPUQuota=700%

[Install]
WantedBy=multi-user.target
EOF
    
    # Make scripts executable
    chmod +x "${NEUROSHARD_DIR}/scripts/genesis_parallel.py"
    chmod +x "${NEUROSHARD_DIR}/scripts/genesis_parallel_service.sh"
    chmod +x "${NEUROSHARD_DIR}/scripts/genesis_ctl.sh"
    
    # Create logs directory
    mkdir -p "${NEUROSHARD_DIR}/logs"
    
    # Copy service file
    sudo cp /tmp/neuroshard-genesis-parallel.service /etc/systemd/system/${PARALLEL_SERVICE_NAME}.service
    
    # Reload systemd
    sudo systemctl daemon-reload
    
    # Enable on boot
    sudo systemctl enable ${PARALLEL_SERVICE_NAME}
    
    # Disable old service if exists
    if systemctl list-unit-files | grep -q ${SERVICE_NAME}; then
        sudo systemctl disable ${SERVICE_NAME} 2>/dev/null || true
    fi
    
    echo -e "${GREEN}Service installed and enabled!${NC}"
    echo ""
    echo "Commands:"
    echo "  sudo systemctl start ${PARALLEL_SERVICE_NAME}   - Start the service"
    echo "  sudo systemctl stop ${PARALLEL_SERVICE_NAME}    - Stop the service"
    echo "  sudo systemctl status ${PARALLEL_SERVICE_NAME}  - Check status"
    echo "  journalctl -u ${PARALLEL_SERVICE_NAME} -f       - View logs"
    echo ""
    echo "Or use this script:"
    echo "  ./genesis_ctl.sh start|stop|status|logs"
}

# Main
case "${1:-status}" in
    status)
        status
        ;;
    start)
        start
        ;;
    start-legacy)
        start_legacy
        ;;
    stop)
        stop
        ;;
    stop-legacy)
        stop_legacy
        ;;
    migrate)
        migrate
        ;;
    restart)
        restart
        ;;
    logs)
        logs
        ;;
    install)
        install_service
        ;;
    *)
        echo "Usage: $0 {status|start|stop|restart|migrate|logs|install}"
        echo ""
        echo "Commands:"
        echo "  status   - Show current status and progress"
        echo "  start    - Start parallel populator (recommended)"
        echo "  stop     - Stop all genesis services"
        echo "  restart  - Restart services"
        echo "  migrate  - Safely migrate from legacy to parallel"
        echo "  logs     - Tail the logs"
        echo "  install  - Install as systemd service"
        exit 1
        ;;
esac
