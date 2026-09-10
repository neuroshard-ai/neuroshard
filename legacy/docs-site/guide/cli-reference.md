# CLI Reference

Complete command-line reference for `neuroshard`.

## Basic Usage

```bash
neuroshard [OPTIONS]
```

## Options

### Required

| Option | Description |
|--------|-------------|
| `--token <TOKEN>` | Your wallet token (64-char hex or 12-word mnemonic) |

### Core Options

| Option | Default | Description |
|--------|---------|-------------|
| `--port <PORT>` | 8000 | HTTP port for the node |
| `--tracker <URL>` | `https://neuroshard.com/api/tracker` | Tracker URL for peer discovery |
| `--device <DEVICE>` | auto | Compute device: `auto`, `cuda`, `mps`, or `cpu` |
| `--version` | - | Show version and exit |
| `--help` | - | Show help message and exit |

### Daemon Options (Linux/macOS)

| Option | Description |
|--------|-------------|
| `--daemon` / `-d` | Run as background daemon (logs to `~/.neuroshard/node.log`) |
| `--stop` | Stop the running daemon |
| `--status` | Check if daemon is running |
| `--logs` | Show recent daemon logs |
| `--log-lines <N>` | Number of log lines to show (default: 50) |

### Network Options

| Option | Default | Description |
|--------|---------|-------------|
| `--announce-ip <IP>` | None (auto-detect) | Force this IP address for peer announcements |
| `--announce-port <PORT>` | None (uses `--port`) | Force this port for peer announcements |

### Resource Limits

| Option | Default | Description |
|--------|---------|-------------|
| `--memory <MB>` | None (auto-detect ~70% of RAM) | Max memory in MB |
| `--cpu-threads <N>` | None (all cores) | Max CPU threads to use |
| `--max-storage <MB>` | 100 | Max disk space for training data in MB |

### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--no-training` | false | Disable training (inference only) |
| `--diloco-steps <N>` | 500 | DiLoCo inner steps before gradient sync |

### UI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--no-browser` | false | Don't auto-open dashboard in browser |
| `--headless` | false | Run without any UI (server mode) |

## Examples

### Basic Startup

```bash
neuroshard --token YOUR_WALLET_TOKEN
```

### Force GPU Device

```bash
# Force CUDA (NVIDIA)
neuroshard --token YOUR_TOKEN --device cuda

# Force Apple Metal (MPS)
neuroshard --token YOUR_TOKEN --device mps

# Force CPU (useful for debugging)
neuroshard --token YOUR_TOKEN --device cpu
```

### Run as Daemon (Linux/macOS)

```bash
# Start as background daemon
neuroshard --daemon --token YOUR_TOKEN

# Check daemon status
neuroshard --status

# View daemon logs
neuroshard --logs

# Stop the daemon
neuroshard --stop
```

### Custom Port

```bash
neuroshard --token YOUR_TOKEN --port 9000
```

### Limited Resources

```bash
neuroshard --token YOUR_TOKEN \
  --memory 4096 \
  --cpu-threads 4 \
  --max-storage 50
```

### Server Mode

```bash
neuroshard --token YOUR_TOKEN \
  --headless \
  --no-browser
```

### Inference Only

```bash
neuroshard --token YOUR_TOKEN --no-training
```

### Behind NAT/Firewall

```bash
neuroshard --token YOUR_TOKEN \
  --announce-ip 203.0.113.50 \
  --announce-port 8001
```

### Custom Tracker

```bash
neuroshard --token YOUR_TOKEN \
  --tracker http://localhost:3000
```

### Fast DiLoCo Sync

```bash
neuroshard --token YOUR_TOKEN \
  --diloco-steps 100
```

## Getting Your Token

The `--token` argument is required. You can get your wallet token at:
- **Web Wallet**: https://neuroshard.com/wallet

The token can be provided as:
- A 64-character hexadecimal string
- A 12-word BIP39 mnemonic phrase

## Token Formats

### Hexadecimal (64 characters)

```bash
neuroshard --token a1b2c3d4e5f6...
```

### Mnemonic (12 words)

```bash
neuroshard --token "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12"
```

The mnemonic is converted to a seed using BIP39.

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Clean shutdown |
| 1 | Invalid arguments or missing token |

## Signals

| Signal | Action |
|--------|--------|
| `SIGINT` (Ctrl+C) | Graceful shutdown |
| `SIGTERM` | Graceful shutdown |

## Output

### Console Output

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗               ║
║   ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗              ║
║   ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║              ║
║   ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║              ║
║   ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝              ║
║   ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝              ║
║                                                              ║
║            Decentralized AI Training Network                 ║
║                     v0.0.6                                   ║
╚══════════════════════════════════════════════════════════════╝

  🖥️  Device: CUDA (NVIDIA GeForce RTX 3080)

[NODE] ✅ Wallet recovered from mnemonic
[NODE] Starting on port 8000...
[NODE] Dashboard: http://localhost:8000/
[NODE] Assigned 24 layers: [0, 1, 2, ...]
[GENESIS] Data loader ready: 1024 shards available
```

## Error Messages

If you don't provide a token, you'll see:

```
[ERROR] Wallet token required!

Get your token at: https://neuroshard.com/wallet
Or generate a new wallet with: neuroshard --help

Usage: neuroshard --token YOUR_TOKEN
```

The node will exit with code 1 if the token is missing or invalid.

## See Also

- [Installation](/guide/installation)
- [Running a Node](/guide/running-a-node)
- [Configuration](/guide/configuration)

