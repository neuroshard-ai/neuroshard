#!/bin/bash
#
# NeuroShard Node - Headless Installer
#
# Downloads and runs the NeuroShard headless binary.
#
# Usage:
#   curl -sSL https://neuroshard.com/install.sh | bash
#   
# Or download and run:
#   chmod +x install.sh && ./install.sh
#

set -e

# ============================================================================
# CONFIG - Update these URLs to your CDN/release location
# ============================================================================

DOWNLOAD_URL="https://d1qsvy9420pqcs.cloudfront.net/releases/latest/neuroshard-headless"
INSTALL_DIR="$HOME/.neuroshard"
BINARY_PATH="$INSTALL_DIR/neuroshard-headless"
CONFIG_FILE="$INSTALL_DIR/gui_settings.json"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ============================================================================
# HELPERS
# ============================================================================

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

print_banner() {
    echo ""
    echo "  ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗ ███████╗██╗  ██╗ █████╗ ██████╗ ██████╗ "
    echo "  ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗██╔════╝██║  ██║██╔══██╗██╔══██╗██╔══██╗"
    echo "  ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║███████╗███████║███████║██████╔╝██║  ██║"
    echo "  ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║╚════██║██╔══██║██╔══██║██╔══██╗██║  ██║"
    echo "  ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝███████║██║  ██║██║  ██║██║  ██║██████╔╝"
    echo "  ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ "
    echo ""
    echo "  The People's AI Network - Distributed LLM Training & Inference"
    echo ""
}

# ============================================================================
# INSTALL
# ============================================================================

install() {
    info "Installing NeuroShard to $INSTALL_DIR..."
    
    # Create directory
    mkdir -p "$INSTALL_DIR"
    
    # Download binary
    info "Downloading headless binary..."
    if command -v curl &> /dev/null; then
        curl -fsSL "$DOWNLOAD_URL" -o "$BINARY_PATH"
    elif command -v wget &> /dev/null; then
        wget -q "$DOWNLOAD_URL" -O "$BINARY_PATH"
    else
        error "curl or wget required"
    fi
    
    # Make executable
    chmod +x "$BINARY_PATH"
    
    success "NeuroShard installed!"
    echo ""
    echo "  Binary:  $BINARY_PATH"
    echo "  Config:  $CONFIG_FILE"
    echo ""
}

# ============================================================================
# SETUP TOKEN
# ============================================================================

setup_token() {
    echo ""
    echo "=============================================="
    echo "  First Time Setup"
    echo "=============================================="
    echo ""
    echo "You need a wallet token to run a node."
    echo ""
    echo "Get one at: https://neuroshard.com/wallet"
    echo ""
    read -p "Enter your token/mnemonic (or press Enter to skip): " TOKEN
    
    if [ -z "$TOKEN" ]; then
        warn "No token provided."
        echo ""
        echo "Run later with:"
        echo "  export NEUROSHARD_TOKEN='your token'"
        echo "  $BINARY_PATH"
        echo ""
        echo "Or pass directly:"
        echo "  $BINARY_PATH --token 'your token'"
        exit 0
    fi
    
    # Save to config
    cat > "$CONFIG_FILE" << EOF
{
  "port": "8000",
  "tracker": "https://neuroshard.com/api/tracker",
  "token": "$TOKEN",
  "enable_training": true,
  "max_storage_mb": 100,
  "max_memory_mb": 4096,
  "max_cpu_threads": 4
}
EOF
    
    success "Token saved!"
}

# ============================================================================
# RUN
# ============================================================================

run() {
    if [ ! -f "$BINARY_PATH" ]; then
        install
    fi
    
    # Check for token
    if [ -z "$NEUROSHARD_TOKEN" ]; then
        if [ -f "$CONFIG_FILE" ]; then
            TOKEN=$(grep -o '"token"[[:space:]]*:[[:space:]]*"[^"]*"' "$CONFIG_FILE" 2>/dev/null | cut -d'"' -f4)
            if [ -z "$TOKEN" ] || [ "$TOKEN" = "PASTE_YOUR_12_WORD_MNEMONIC_OR_TOKEN_HERE" ]; then
                setup_token
            fi
        else
            setup_token
        fi
    fi
    
    # Run
    exec "$BINARY_PATH" "$@"
}

# ============================================================================
# UPDATE
# ============================================================================

update() {
    info "Updating NeuroShard..."
    
    # Backup old binary
    if [ -f "$BINARY_PATH" ]; then
        mv "$BINARY_PATH" "$BINARY_PATH.old"
    fi
    
    # Download new
    if command -v curl &> /dev/null; then
        curl -fsSL "$DOWNLOAD_URL" -o "$BINARY_PATH"
    else
        wget -q "$DOWNLOAD_URL" -O "$BINARY_PATH"
    fi
    
    chmod +x "$BINARY_PATH"
    
    # Remove backup
    rm -f "$BINARY_PATH.old"
    
    success "Updated!"
}

# ============================================================================
# UNINSTALL
# ============================================================================

uninstall() {
    warn "This will remove NeuroShard (config preserved)."
    read -p "Continue? [y/N]: " CONFIRM
    
    if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
        echo "Aborted."
        exit 0
    fi
    
    rm -f "$BINARY_PATH"
    success "Uninstalled. Config at: $CONFIG_FILE"
}

# ============================================================================
# HELP
# ============================================================================

show_help() {
    echo "Usage: $0 [OPTIONS] [-- NODE_ARGS]"
    echo ""
    echo "Options:"
    echo "  (none)       Install (if needed) and run"
    echo "  --install    Install only"
    echo "  --update     Update to latest"
    echo "  --uninstall  Remove binary"
    echo "  --help       Show this help"
    echo ""
    echo "Node Arguments (after --):"
    echo "  --port PORT       HTTP port (default: 8000)"
    echo "  --token TOKEN     Wallet token or mnemonic"
    echo "  --no-training     Inference only"
    echo "  --memory MB       Memory limit"
    echo "  --cpu THREADS     CPU thread limit"
    echo "  --init            Create config file"
    echo "  --show-config     Show current config"
    echo ""
    echo "Examples:"
    echo "  $0                              # Install and run"
    echo "  $0 -- --port 8001               # Custom port"
    echo "  $0 -- --no-training             # Inference only"
    echo "  NEUROSHARD_TOKEN=xxx $0         # With token env var"
    echo ""
    echo "Or run binary directly:"
    echo "  $BINARY_PATH --help"
    echo ""
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    print_banner
    
    case "${1:-}" in
        --install)
            install
            ;;
        --update)
            update
            ;;
        --uninstall)
            uninstall
            ;;
        --help|-h)
            show_help
            ;;
        --)
            shift
            run "$@"
            ;;
        "")
            run
            ;;
        *)
            # Pass through to run
            run "$@"
            ;;
    esac
}

main "$@"
