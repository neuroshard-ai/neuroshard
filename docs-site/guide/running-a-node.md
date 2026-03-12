# Running a Node

Complete guide to configuring and operating a NeuroShard node.

## Basic Startup

```bash
neuroshard --token YOUR_WALLET_TOKEN
```

Your node will:
1. Detect available system resources (RAM, GPU)
2. Connect to the tracker for peer discovery
3. Register with the network and receive layer assignments
4. Load the NeuroLLM model layers assigned to it
5. Start training and processing inference requests

## Command Line Options

### Core Options

| Option | Default | Description |
|--------|---------|-------------|
| `--token` | Required | Your wallet token (64-char hex or 12-word mnemonic) |
| `--port` | 8000 | HTTP port for dashboard and API |
| `--tracker` | `https://neuroshard.com/api/tracker` | Tracker URL for peer discovery |

### Network Options

| Option | Default | Description |
|--------|---------|-------------|
| `--announce-ip` | Auto-detect | IP address for peer announcements |
| `--announce-port` | Same as `--port` | Port for peer announcements |

### Resource Limits

| Option | Default | Description |
|--------|---------|-------------|
| `--memory` | 70% of system RAM | Maximum memory in MB |
| `--cpu-threads` | All cores | Maximum CPU threads |
| `--max-storage` | 100 | Maximum disk space for training data (MB) |

### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--no-training` | false | Disable training (inference only) |
| `--diloco-steps` | 500 | Inner steps before gradient sync |

### UI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--no-browser` | false | Don't auto-open dashboard |
| `--headless` | false | Run without any UI (server mode) |

## Example Configurations

### Maximum Performance (Gaming PC)

```bash
neuroshard \
  --token YOUR_TOKEN \
  --port 8000 \
  --memory 16384 \
  --cpu-threads 8 \
  --max-storage 500
```

### Resource-Constrained (Raspberry Pi)

```bash
neuroshard \
  --token YOUR_TOKEN \
  --memory 1024 \
  --cpu-threads 2 \
  --max-storage 50 \
  --no-browser
```

### Server/Headless

```bash
neuroshard \
  --token YOUR_TOKEN \
  --port 8000 \
  --headless \
  --no-browser
```

### Inference Only (No Training)

```bash
neuroshard \
  --token YOUR_TOKEN \
  --no-training
```

### Behind NAT/Firewall

```bash
neuroshard \
  --token YOUR_TOKEN \
  --announce-ip YOUR_PUBLIC_IP \
  --announce-port 8001
```

## Understanding the Dashboard

Open `http://localhost:8000/` to access the node dashboard.

### Network Status

- **Node ID**: Your unique network identifier
- **Status**: Online, Training, Syncing, or Idle
- **Peers**: Number of connected peer nodes

### Layer Information

- **Assigned Layers**: Which model layers this node holds
- **Total Network Layers**: Current model depth
- **Role**: Driver (Layer 0), Worker (middle), or Validator (Last Layer)

### Training Stats

- **Current Loss**: Training loss (lower is better)
- **Training Rounds**: Total training batches completed
- **Tokens Processed**: Total tokens processed for training/inference

### Economics

- **NEURO Balance**: Current balance in your wallet
- **Earnings Rate**: Current earning rate per minute
- **Reward Multiplier**: Your bonus multiplier (from layers, roles, stake)

## File Locations

NeuroShard stores data in `~/.neuroshard/`:

```
~/.neuroshard/
├── checkpoints/           # Model checkpoints
│   └── dynamic_node_*.pt  # Your node's checkpoint
├── data_cache/            # Cached training data
└── logs/                  # Node logs
```

### Checkpoint Files

Checkpoints are saved automatically:
- Every 50 training steps
- On graceful shutdown
- Contains: layer weights, optimizer state, DiLoCo state

To reset your node (start fresh):
```bash
rm -rf ~/.neuroshard/checkpoints
```

## Running in Background

### Built-in Daemon Mode (Recommended)

The simplest way to run NeuroShard as a background service:

```bash
# Start as daemon
neuroshard --daemon --token YOUR_TOKEN

# Check status
neuroshard --status

# View logs
neuroshard --logs

# Stop the daemon
neuroshard --stop
```

Logs are written to `~/.neuroshard/node.log` and the PID is stored in `~/.neuroshard/node.pid`.

::: tip
The dashboard at `http://localhost:8000` remains accessible when running as a daemon.
:::

### Alternative: screen or tmux

For interactive background sessions:

```bash
# Using screen
screen -S neuroshard
neuroshard --token YOUR_TOKEN
# Press Ctrl+A, then D to detach
# Reattach later: screen -r neuroshard

# Using tmux
tmux new -s neuroshard
neuroshard --token YOUR_TOKEN
# Press Ctrl+B, then D to detach
# Reattach later: tmux attach -t neuroshard
```

## Running as a Service

### systemd (Linux)

Create `/etc/systemd/system/neuroshard.service`:

```ini
[Unit]
Description=NeuroShard Node
After=network.target

[Service]
Type=simple
User=ubuntu
Environment="NEUROSHARD_TOKEN=your_token_here"
ExecStart=/usr/local/bin/neuroshard --token ${NEUROSHARD_TOKEN} --headless
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable neuroshard
sudo systemctl start neuroshard
sudo systemctl status neuroshard
```

### Docker

```bash
docker run -d \
  --name neuroshard \
  --restart unless-stopped \
  -p 8000:8000 \
  -p 9000:9000 \
  -e NEUROSHARD_TOKEN=YOUR_TOKEN \
  -v neuroshard_data:/data \
  neuroshard/node:latest
```

### Windows Task Scheduler

1. Open Task Scheduler
2. Create Basic Task → "NeuroShard Node"
3. Trigger: At startup
4. Action: Start a program
5. Program: `C:\path\to\neuroshard.exe`
6. Arguments: `--token YOUR_TOKEN --headless`

## Monitoring

### Logs

```bash
# View live logs (if running in foreground)
# Logs are printed to stdout

# Check systemd logs
journalctl -u neuroshard -f

# Docker logs
docker logs -f neuroshard
```

### API Health Check

```bash
curl http://localhost:8000/api/stats
```

Returns:
```json
{
  "node_id": "abc123...",
  "my_layers": [0, 1, 2, 3, 4],
  "total_training_rounds": 1234,
  "current_loss": 4.567,
  "total_tokens_processed": 1000000
}
```

### Metrics

The node exposes metrics at `/api/stats`:
- Training loss over time
- Tokens processed per minute
- Network peer count
- GPU/memory utilization

## Multi-Node Setup

You can run multiple nodes on different machines with the same wallet token:

```bash
# Machine 1
neuroshard --token YOUR_TOKEN --port 8000

# Machine 2
neuroshard --token YOUR_TOKEN --port 8000

# Machine 3
neuroshard --token YOUR_TOKEN --port 8000
```

Each node gets a unique network identity but earnings accumulate to the same wallet.

::: info Multi-Node Earnings
Running multiple nodes with the same token multiplies your earning potential. Each node earns independently based on its contribution.
:::

## Graceful Shutdown

To stop your node cleanly:

```bash
# If running in foreground
Ctrl+C

# If running as systemd service
sudo systemctl stop neuroshard

# Docker
docker stop neuroshard
```

Graceful shutdown:
1. Saves checkpoint to disk
2. Unregisters from network
3. Closes peer connections

## Next Steps

- [Network Roles](/guide/network-roles) — Understand Driver, Worker, Validator roles
- [Training Pipeline](/guide/training-pipeline) — How distributed training works
- [CLI Reference](/guide/cli-reference) — Complete command documentation

