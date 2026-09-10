#!/bin/bash
# ============================================================================
# NeuroShard Genesis Data Service
# ============================================================================
# Robust startup script for continuous S3 data population.
# Processes multiple data sources sequentially with smart resume.
#
# Features:
# - Multi-source support (processes sources in priority order)
# - Auto-detects and resumes from S3 checkpoint per source
# - Graceful shutdown with checkpoint save
# - Configurable via genesis_sources.json
# ============================================================================

set -e

# Configuration
NEUROSHARD_DIR="/home/ubuntu/neuroshard"
VENV_DIR="${NEUROSHARD_DIR}/venv_build"
LOG_DIR="${NEUROSHARD_DIR}/logs"
SCRIPT="${NEUROSHARD_DIR}/scripts/populate_genesis_s3.py"
CONFIG="${NEUROSHARD_DIR}/scripts/genesis_sources.json"

# S3 Configuration
BUCKET="neuroshard-training-data"

# Ensure directories exist
mkdir -p "${LOG_DIR}"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_DIR}/genesis_service.log"
}

# Check if already running
check_running() {
    if pgrep -f "populate_genesis_s3.py" > /dev/null; then
        log "WARNING: Genesis populator already running!"
        return 0
    fi
    return 1
}

# Load sources from config
get_sources() {
    cd "${NEUROSHARD_DIR}"
    source "${VENV_DIR}/bin/activate"
    
    python3 - <<'PYEOF'
import json
import os

config_path = "scripts/genesis_sources.json"
if os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)
    
    # Sort by priority and filter enabled
    sources = sorted(
        [s for s in config.get('sources', []) if s.get('enabled', True)],
        key=lambda x: x.get('priority', 99)
    )
    
    for s in sources:
        print(f"{s['name']}:{s.get('target_shards', 500000)}")
else:
    # Default if no config
    print("fineweb-edu:500000")
    print("fineweb:1000000")
PYEOF
}

# Get current shard count for a source
get_source_shards() {
    local source=$1
    cd "${NEUROSHARD_DIR}"
    source "${VENV_DIR}/bin/activate"
    
    python3 - "$source" <<'PYEOF'
import os
import json
import boto3
import sys

source = sys.argv[1]

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

try:
    s3 = boto3.client('s3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    )
    
    obj = s3.get_object(Bucket='neuroshard-training-data', Key='manifest.json')
    manifest = json.loads(obj['Body'].read())
    
    source_stats = manifest.get('sources', {}).get(source, {})
    print(source_stats.get('shards', 0))
except:
    print(0)
PYEOF
}

# Run population for a specific source
run_source() {
    local source=$1
    local target=$2
    
    # Check current progress for this source
    local current=$(get_source_shards "$source")
    
    if [ "$current" -ge "$target" ]; then
        log "✓ Source ${source} complete: ${current}/${target} shards"
        return 0
    fi
    
    log "Starting ${source}: ${current}/${target} shards (need $((target - current)) more)"
    
    cd "${NEUROSHARD_DIR}"
    source "${VENV_DIR}/bin/activate"
    
    # Run with proper error handling
    # Note: target here is TOTAL shards across all sources
    # The script handles per-source tracking internally
    python3 "${SCRIPT}" \
        --bucket "${BUCKET}" \
        --target "${target}" \
        --source "${source}" \
        2>&1 | tee -a "${LOG_DIR}/genesis_${source}.log"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log "✓ Completed pass for: ${source}"
    else
        log "⚠ ${source} exited with code ${exit_code} (will retry on next run)"
    fi
    
    return $exit_code
}

# Main service loop
main() {
    log "=========================================="
    log "NeuroShard Genesis Service Starting"
    log "=========================================="
    
    # Check if already running
    if check_running; then
        log "Exiting to avoid duplicate processes"
        exit 0
    fi
    
    # Continuous loop - keeps running through sources
    while true; do
        log "--- Starting source processing cycle ---"
        
        # Get enabled sources from config
        local sources=$(get_sources)
        local all_complete=true
        
        while IFS=: read -r source target; do
            if [ -z "$source" ]; then continue; fi
            
            log "Processing: ${source} (target: ${target} shards)"
            
            # Run this source
            if ! run_source "$source" "$target"; then
                log "Source ${source} needs more work"
                all_complete=false
            fi
            
            # Check if we should stop
            if [ -f "${NEUROSHARD_DIR}/.genesis_stop" ]; then
                log "Stop file detected, exiting gracefully"
                rm -f "${NEUROSHARD_DIR}/.genesis_stop"
                exit 0
            fi
            
        done <<< "$sources"
        
        if [ "$all_complete" = true ]; then
            log "=========================================="
            log "All sources complete! Sleeping for 1 hour..."
            log "=========================================="
            sleep 3600
        else
            log "Cycle complete, continuing with next source..."
            sleep 10
        fi
    done
}

# Handle signals for graceful shutdown
cleanup() {
    log "Received shutdown signal, saving checkpoint..."
    # The Python script handles SIGTERM gracefully
    exit 0
}

trap cleanup SIGTERM SIGINT

# Run main
main "$@"
