#!/bin/bash
# ============================================================================
# NeuroShard Genesis Parallel Service
# ============================================================================
# Runs the parallel populator as a systemd service.
# Designed for c5.2xlarge (8 vCPU, 16GB RAM) and similar instances.
#
# Features:
# - TRUE parallel processing of multiple data sources
# - Automatic resource detection and allocation
# - Graceful shutdown with checkpoint save
# - Auto-restart on failure
# ============================================================================

set -e

# Configuration
NEUROSHARD_DIR="/home/ubuntu/neuroshard"
VENV_DIR="${NEUROSHARD_DIR}/venv_build"
LOG_DIR="${NEUROSHARD_DIR}/logs"
SCRIPT="${NEUROSHARD_DIR}/scripts/genesis_parallel.py"

# S3 Configuration
BUCKET="neuroshard-training-data"

# Parallel settings (auto-tuned by script, but can override)
MAX_SOURCES=4
UPLOAD_WORKERS=8

# Ensure directories exist
mkdir -p "${LOG_DIR}"

# Timestamp for logs
timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log() {
    echo "[$(timestamp)] $1" | tee -a "${LOG_DIR}/genesis_parallel_service.log"
}

# Check if already running
check_running() {
    if pgrep -f "genesis_parallel.py" > /dev/null; then
        return 0
    fi
    return 1
}

# Stop old sequential service if running
stop_old_service() {
    if pgrep -f "populate_genesis_s3.py" > /dev/null; then
        log "Stopping old sequential genesis service..."
        pkill -TERM -f "populate_genesis_s3.py" || true
        sleep 5
        
        # Force kill if still running
        if pgrep -f "populate_genesis_s3.py" > /dev/null; then
            log "Force killing old service..."
            pkill -9 -f "populate_genesis_s3.py" || true
        fi
        
        log "Old service stopped"
    fi
    
    # Also stop old systemd service
    if systemctl is-active --quiet neuroshard-genesis 2>/dev/null; then
        log "Stopping neuroshard-genesis systemd service..."
        sudo systemctl stop neuroshard-genesis || true
    fi
}

# Main run function
run_parallel() {
    log "=========================================="
    log "NeuroShard Genesis Parallel Service v4.0"
    log "=========================================="
    
    # Check if already running
    if check_running; then
        log "Parallel populator already running!"
        exit 0
    fi
    
    # Activate virtual environment
    cd "${NEUROSHARD_DIR}"
    source "${VENV_DIR}/bin/activate"
    
    # Add src to path
    export PYTHONPATH="${NEUROSHARD_DIR}/src:${PYTHONPATH}"
    
    # Log system info
    log "System: $(nproc) CPUs, $(free -h | awk '/^Mem:/{print $2}') RAM"
    log "Config: max ${MAX_SOURCES} sources, ${UPLOAD_WORKERS} upload workers"
    
    # Run the parallel populator
    log "Starting parallel populator..."
    
    python3 "${SCRIPT}" \
        --bucket "${BUCKET}" \
        --max-sources "${MAX_SOURCES}" \
        --upload-workers "${UPLOAD_WORKERS}" \
        2>&1 | tee -a "${LOG_DIR}/genesis_parallel.log"
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log "✓ Parallel populator completed successfully"
    else
        log "⚠ Parallel populator exited with code ${exit_code}"
    fi
    
    return $exit_code
}

# Handle signals
cleanup() {
    log "Received shutdown signal..."
    # The Python script handles SIGTERM gracefully
    exit 0
}

trap cleanup SIGTERM SIGINT

# Parse arguments
case "${1:-run}" in
    run)
        run_parallel
        ;;
    stop-old)
        stop_old_service
        ;;
    status)
        if check_running; then
            echo "Genesis parallel populator is RUNNING"
            pgrep -af "genesis_parallel.py"
        else
            echo "Genesis parallel populator is NOT running"
        fi
        ;;
    *)
        echo "Usage: $0 {run|stop-old|status}"
        exit 1
        ;;
esac

