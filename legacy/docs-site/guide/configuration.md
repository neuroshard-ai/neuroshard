# Configuration

Complete guide to configuring your NeuroShard node.

## Configuration Methods

NeuroShard can be configured via:
1. Command-line arguments (highest priority)
2. Environment variables
3. Configuration file

## Command-Line Arguments

See [CLI Reference](/guide/cli-reference) for complete list.

```bash
neuroshard --token YOUR_TOKEN --port 8000 --memory 8192
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `NEUROSHARD_TOKEN` | Wallet token | `abc123...` |
| `NEUROSHARD_PORT` | HTTP port | `8000` |
| `NEUROSHARD_TRACKER` | Tracker URL | `https://neuroshard.com/api/tracker` |
| `NEUROSHARD_MEMORY` | Memory limit (MB) | `8192` |
| `NEUROSHARD_LOG_LEVEL` | Log verbosity | `INFO` |

```bash
export NEUROSHARD_TOKEN=your_token
export NEUROSHARD_PORT=8000
neuroshard
```

## Resource Configuration

### Memory Limits

By default, NeuroShard uses 70% of available RAM.

```bash
# Limit to 4GB
neuroshard --token YOUR_TOKEN --memory 4096

# Limit to 50% of system RAM
neuroshard --token YOUR_TOKEN --memory $(free -m | awk '/^Mem:/{print int($2*0.5)}')
```

**Memory allocation**:
- Model weights: ~40%
- Gradients: ~20%
- Optimizer states: ~20%
- Activations: ~15%
- Overhead: ~5%

### CPU Threads

```bash
# Use only 4 threads
neuroshard --token YOUR_TOKEN --cpu-threads 4
```

::: tip
Leave at least 1-2 cores free for the HTTP server and network operations.
:::

### Storage Limits

```bash
# Limit Genesis data cache to 50MB
neuroshard --token YOUR_TOKEN --max-storage 50
```

## Network Configuration

### Port Configuration

NeuroShard uses two ports:
- **HTTP Port**: Dashboard and API (default: 8000)
- **gRPC Port**: Peer communication (HTTP port + 1000, default: 9000)

```bash
neuroshard --token YOUR_TOKEN --port 8000
# Opens: 8000 (HTTP), 9000 (gRPC)
```

### NAT Traversal

If behind NAT, configure announce addresses:

```bash
neuroshard --token YOUR_TOKEN \
  --announce-ip 203.0.113.50 \
  --announce-port 8001
```

### Firewall Rules

```bash
# UFW (Ubuntu)
sudo ufw allow 8000/tcp  # HTTP
sudo ufw allow 9000/tcp  # gRPC

# iptables
sudo iptables -A INPUT -p tcp --dport 8000 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 9000 -j ACCEPT
```

## Training Configuration

### DiLoCo Settings

```bash
# Sync more frequently (faster convergence, more bandwidth)
neuroshard --token YOUR_TOKEN --diloco-steps 100

# Sync less frequently (slower convergence, less bandwidth)
neuroshard --token YOUR_TOKEN --diloco-steps 1000
```

| Inner Steps | Bandwidth | Convergence | Use Case |
|-------------|-----------|-------------|----------|
| 100 | High | Fast | Good network |
| 500 | Medium | Normal | Default |
| 1000 | Low | Slow | Bad network |

### Disable Training

```bash
# Inference-only mode
neuroshard --token YOUR_TOKEN --no-training
```

## GPU Configuration

### CUDA (NVIDIA)

NeuroShard auto-detects CUDA. To use a specific GPU:

```bash
CUDA_VISIBLE_DEVICES=0 neuroshard --token YOUR_TOKEN
```

### Apple Metal (MPS)

Auto-detected on Apple Silicon. No configuration needed.

### Force CPU

```bash
CUDA_VISIBLE_DEVICES="" neuroshard --token YOUR_TOKEN
```

## File Locations

### Data Directory

Default: `~/.neuroshard/`

| Path | Contents |
|------|----------|
| `~/.neuroshard/checkpoints/` | Model checkpoints |
| `~/.neuroshard/data_cache/` | Genesis data cache |
| `~/.neuroshard/logs/` | Node logs |

### Custom Data Directory

```bash
export NEUROSHARD_DATA_DIR=/data/neuroshard
neuroshard --token YOUR_TOKEN
```

## Docker Configuration

### Docker Compose

```yaml
version: '3.8'
services:
  neuroshard:
    image: neuroshard/node:latest
    ports:
      - "8000:8000"
      - "9000:9000"
    environment:
      - NEUROSHARD_TOKEN=${NEUROSHARD_TOKEN}
      - NEUROSHARD_MEMORY=8192
      - NEUROSHARD_LOG_LEVEL=INFO
    volumes:
      - neuroshard_data:/data
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  neuroshard_data:
```

### Docker Run

```bash
docker run -d \
  --name neuroshard \
  --gpus all \
  -p 8000:8000 \
  -p 9000:9000 \
  -e NEUROSHARD_TOKEN=YOUR_TOKEN \
  -e NEUROSHARD_MEMORY=8192 \
  -v neuroshard_data:/data \
  neuroshard/node:latest
```

## Systemd Configuration

`/etc/systemd/system/neuroshard.service`:

```ini
[Unit]
Description=NeuroShard Node
After=network.target

[Service]
Type=simple
User=ubuntu
Environment="NEUROSHARD_TOKEN=your_token_here"
Environment="NEUROSHARD_PORT=8000"
Environment="NEUROSHARD_MEMORY=8192"
Environment="NEUROSHARD_LOG_LEVEL=INFO"
ExecStart=/usr/local/bin/neuroshard --headless
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable neuroshard
sudo systemctl start neuroshard
```

## Tuning Guide

### Low Memory (<4GB)

```bash
neuroshard --token YOUR_TOKEN \
  --memory 2048 \
  --max-storage 25 \
  --diloco-steps 500
```

### High Performance (>16GB)

```bash
neuroshard --token YOUR_TOKEN \
  --memory 32768 \
  --max-storage 500 \
  --diloco-steps 500
```

### Low Bandwidth

```bash
neuroshard --token YOUR_TOKEN \
  --diloco-steps 1000 \
  --no-training  # Or inference-only
```

## See Also

- [CLI Reference](/guide/cli-reference)
- [Running a Node](/guide/running-a-node)
- [Troubleshooting](/guide/troubleshooting)

