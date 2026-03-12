# Quick Start

Get your NeuroShard node running and earning NEURO in under 5 minutes.

## Prerequisites

- **Python 3.10+** installed
- **4GB+ RAM** (more RAM = more layers = more rewards)
- **Internet connection** for peer discovery and training sync

## Installation

### Install via pip

```bash
# Basic install (CPU only)
pip install neuroshard-ai

# With GPU support (recommended)
pip install neuroshard-ai[gpu]
```

### Platform-Specific

| Platform | Install Command |
|----------|----------------|
| **x86 Linux/Windows** | `pip install neuroshard-ai[gpu]` |
| **macOS (Apple Silicon)** | `pip install neuroshard-ai[gpu]` |
| **NVIDIA Jetson** | See [Jetson Install](/guide/installation#nvidia-jetson-arm64) |

## Get Your Wallet Token

1. Go to [neuroshard.com/signup](https://neuroshard.com/signup)
2. Create an account
3. Navigate to your Dashboard → Wallet
4. Copy your wallet token (64-character hex string or 12-word mnemonic)

::: warning Keep Your Token Safe
Your wallet token is the key to your NEURO earnings. **Never share it** with anyone. Store it securely.
:::

## Start Your Node

```bash
neuroshard --token YOUR_WALLET_TOKEN
```

That's it! Your node will:
1. **Detect available memory** and GPU automatically
2. **Register with the network** and get layer assignments
3. **Start training** NeuroLLM and earning NEURO
4. **Open a dashboard** at `http://localhost:8000/`

## Verify It's Working

### Check the Dashboard

Open `http://localhost:8000/` in your browser. You should see:
- Your node ID and network status
- Layers assigned to your node
- Training progress and loss
- NEURO balance and earnings rate

### Check the Logs

```bash
# Look for these messages:
[NODE] ✅ Wallet recovered from mnemonic
[NODE] Starting on port 8000...
[NODE] Dashboard: http://localhost:8000/
[NODE] Assigned 24 layers: [0, 1, 2, ...]
[NODE] GPU detected: CUDA (NVIDIA GeForce RTX 3080)
[GENESIS] Data loader ready: 1024 shards available
```

### Check the Ledger

Visit [neuroshard.com/ledger](https://neuroshard.com/ledger) and search for your wallet address to see your balance and transaction history.

## Common Options

```bash
# Use a custom port
neuroshard --token YOUR_TOKEN --port 9000

# Force specific device (auto-detected by default)
neuroshard --token YOUR_TOKEN --device cuda   # NVIDIA GPU
neuroshard --token YOUR_TOKEN --device mps    # Apple Silicon
neuroshard --token YOUR_TOKEN --device cpu    # CPU only

# Limit memory usage (in MB)
neuroshard --token YOUR_TOKEN --memory 4096

# Inference-only mode (no training)
neuroshard --token YOUR_TOKEN --no-training

# Run without opening browser
neuroshard --token YOUR_TOKEN --no-browser

# Run as background daemon (Linux/macOS)
neuroshard --daemon --token YOUR_TOKEN

# Set CPU thread limit
neuroshard --token YOUR_TOKEN --cpu-threads 4
```

## Expected Earnings

Your earnings depend on:
- **Memory**: More RAM = more layers = higher rewards
- **Training Activity**: Active training earns ~300x more than idle
- **Role**: Drivers (Layer 0) and Validators (Last Layer) earn bonuses

| Node Type | Memory | Daily Earnings (Active) |
|-----------|--------|------------------------|
| Raspberry Pi | 2GB | ~10-20 NEURO |
| Laptop | 8GB | ~40-60 NEURO |
| Gaming PC | 16GB | ~80-120 NEURO |
| Server | 64GB+ | ~200-400 NEURO |

::: info Earnings Fluctuate
Earnings depend on network activity, model quality, and inference demand. These are estimates based on typical conditions.
:::

## Troubleshooting

### "No GPU detected"

If you have an NVIDIA GPU but it's not detected:

```bash
# Reinstall with GPU support
pip install neuroshard-ai[gpu]

# Or install CUDA PyTorch manually
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

You can also force a specific device:
```bash
neuroshard --token YOUR_TOKEN --device cuda
```

### "Insufficient memory"

Limit the memory usage:
```bash
neuroshard --token YOUR_TOKEN --memory 2048
```

### "Connection refused"

Check your firewall settings. NeuroShard uses:
- Port 8000 (HTTP dashboard and REST API)
- Port 9000 (gRPC for peer communication)

The gRPC port is always HTTP port + 1000. If you change the HTTP port with `--port`, the gRPC port changes accordingly.

### "Data not ready"

The Genesis data loader needs time to download training shards. Wait 30-60 seconds for initialization.

## Next Steps

- [Running a Node](/guide/running-a-node) — Detailed node configuration
- [Network Roles](/guide/network-roles) — Understand Driver, Worker, Validator roles
- [CLI Reference](/guide/cli-reference) — All command-line options
- [NEURO Economics](/economics/rewards) — Maximize your earnings

