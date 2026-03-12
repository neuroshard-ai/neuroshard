<p align="center">
  <img src="https://neuroshard.com/static/logo.png" alt="NeuroShard Logo" width="120" height="120">
</p>

<h1 align="center">NeuroShard</h1>

<p align="center">
  <strong>Decentralized LLM Training Network</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/neuroshard-ai/"><img src="https://badge.fury.io/py/neuroshard-ai.svg" alt="PyPI version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
  <a href="https://github.com/neuroshard-ai/neuroshard/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="https://discord.gg/4R49xpj7vn"><img src="https://img.shields.io/discord/1234567890?color=7289da&label=Discord&logo=discord&logoColor=white" alt="Discord"></a>
</p>

<p align="center">
  <a href="https://neuroshard.com">Website</a> •
  <a href="https://docs.neuroshard.com">Documentation</a> •
  <a href="docs/whitepaper/neuroshard_whitepaper.pdf">Whitepaper</a> •
  <a href="https://discord.gg/4R49xpj7vn">Discord</a> •
  <a href="https://x.com/shardneuro">Twitter</a>
</p>

---

## What is NeuroShard?

NeuroShard is a **decentralized network** for training large language models. Anyone can contribute GPU/CPU power and earn **NEURO tokens** through Proof of Neural Work.

Unlike centralized AI companies, NeuroShard distributes both the compute AND the rewards across all participants.

### Key Features

| Feature | Description |
|---------|-------------|
| **DiLoCo Training** | Distributed Low-Communication training - sync every 500 steps, not every step |
| **Byzantine Tolerance** | Robust gradient aggregation (Krum, Trimmed Mean) handles malicious nodes |
| **NEURO Rewards** | Earn tokens for contributing compute via Proof of Neural Work |
| **Cryptographic Proofs** | ECDSA-signed proofs ensure trustless verification |
| **Web Dashboard** | Real-time monitoring at `http://localhost:8000` |
| **P2P Network** | Decentralized peer discovery and gossip protocol |

---

## Quick Start

### Installation

```bash
pip install neuroshard
```

### Run a Node

```bash
# Get your token from neuroshard.com
neuroshard --token YOUR_TOKEN
```

That's it! Your node will:
1. Connect to the network
2. Start training model layers
3. Earn NEURO for your contribution

### Web Dashboard

Open `http://localhost:8000` to see:
- Node status and role
- Training progress (DiLoCo inner/outer steps)
- NEURO balance
- Network statistics

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 4 GB | 8+ GB |
| **Python** | 3.9+ | 3.10+ |
| **GPU** | Optional | NVIDIA 8GB+ VRAM |

### GPU Support (Optional)

For NVIDIA GPUs with CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## How It Works

### DiLoCo Distributed Training

NeuroShard uses [DiLoCo](https://arxiv.org/abs/2311.08105) (Distributed Low-Communication) for efficient distributed training:

```
┌─────────────────────────────────────────────────┐
│  INNER LOOP (500 steps - no communication)      │
│  • Each node trains independently               │
│  • Local AdamW optimization                     │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│  OUTER LOOP (sync with peers)                   │
│  • Compute pseudo-gradient: Δθ = θ₀ - θ₅₀₀     │
│  • Gossip to peers                              │
│  • Byzantine-tolerant aggregation               │
│  • Nesterov momentum update                     │
└─────────────────────────────────────────────────┘
                      ↓
              (Repeat)
```

This reduces network communication by **500x** compared to synchronous training!

### Proof of Neural Work

Nodes earn NEURO by submitting cryptographically signed proofs of their work:

- Training batches processed
- Inference requests served
- Uptime contribution
- Data samples provided

All proofs are verified using ECDSA signatures (secp256k1).

---

## Configuration

### CLI Options

```bash
neuroshard --token YOUR_TOKEN \
           --port 8000 \
           --tracker https://tracker.neuroshard.com \
           --training \
           --diloco-steps 500
```

| Option | Default | Description |
|--------|---------|-------------|
| `--token` | Required | Your node authentication token |
| `--port` | 8000 | HTTP server port |
| `--tracker` | Auto | Tracker server URL |
| `--training` | False | Enable training mode |
| `--diloco-steps` | 500 | Inner steps before sync |

See [full CLI reference](https://docs.neuroshard.com/guide/cli-reference) for all options.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      NeuroShard Node                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  NeuroLLM   │  │   DiLoCo    │  │  Proof of Neural    │  │
│  │  (Model)    │  │  Trainer    │  │  Work Ledger        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  P2P/DHT    │  │  Gradient   │  │  ECDSA Crypto       │  │
│  │  Network    │  │  Aggregator │  │  (secp256k1)        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Documentation

- **[Whitepaper](docs/whitepaper/neuroshard_whitepaper.pdf)** - Technical whitepaper (PDF)
- **[Getting Started](https://docs.neuroshard.com/guide/quick-start)** - First steps
- **[Running a Node](https://docs.neuroshard.com/guide/running-a-node)** - Detailed setup
- **[Architecture](https://docs.neuroshard.com/architecture/overview)** - System design
- **[Economics](https://docs.neuroshard.com/economics/overview)** - NEURO tokenomics
- **[API Reference](https://docs.neuroshard.com/api/overview)** - SDK & endpoints

---

## Links

| Resource | Link |
|----------|------|
| Website | [neuroshard.com](https://neuroshard.com) |
| Documentation | [docs.neuroshard.com](https://docs.neuroshard.com) |
| Whitepaper | [PDF](docs/whitepaper/neuroshard_whitepaper.pdf) |
| Discord | [discord.gg/4R49xpj7vn](https://discord.gg/4R49xpj7vn) |
| Twitter | [@shardneuro](https://x.com/shardneuro) |
| PyPI | [pypi.org/project/neuroshard](https://pypi.org/project/neuroshard/) |

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Clone the repo
git clone https://github.com/neuroshard-ai/neuroshard.git
cd neuroshard

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

---

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Train AI. Earn NEURO. Own the Network.</strong>
</p>
