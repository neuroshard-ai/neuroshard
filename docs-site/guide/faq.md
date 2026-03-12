# Frequently Asked Questions

## General

### What is NeuroShard?

NeuroShard is a decentralized protocol for training Large Language Models (LLMs). Instead of a company training a model in a data center, NeuroShard coordinates thousands of individual nodes to collectively train a shared model called **NeuroLLM**.

### How is this different from other decentralized AI projects?

| Feature | NeuroShard | Bittensor | Gensyn |
|---------|-----------|-----------|--------|
| Model ownership | Community trains from scratch | Pre-trained models | User-provided |
| Architecture | Dynamic, grows with network | Fixed | Fixed |
| Verification | Proof of Neural Work | Output validation | Compute verification |
| Training | Full distributed training | Inference/fine-tune | Verification |

### Is NeuroLLM any good?

NeuroLLM starts from **random weights** and improves over time as the network trains it. Early on, output quality will be low. As the network grows and more training happens, quality improves. This is intentional — we refuse to bootstrap from a corporate model.

### Why train from scratch instead of fine-tuning an existing model?

1. **True Ownership**: Models trained by corporations have unknown biases, filters, and training data
2. **Decentralization**: Starting from scratch means no single entity ever controlled the model
3. **Transparency**: Every gradient, every training round is recorded on the Intelligence Ledger
4. **Philosophy**: We're building collective intelligence, not redistributing corporate AI

## NEURO Token

### What is NEURO?

NEURO is the utility token of NeuroShard. It's earned by contributing compute (training) and spent to use inference. NEURO is **not** a cryptocurrency investment — it's a utility token for network participation.

### How do I earn NEURO?

1. **Training**: 0.0005 NEURO per batch (~43 NEURO/day active)
2. **Uptime**: 0.0001 NEURO per minute (~0.14 NEURO/day idle)
3. **Data**: 0.00001 NEURO per sample served
4. **Inference**: Market-based (supply/demand)

### Was there a pre-mine or ICO?

**No**. The Genesis Block starts with `total_minted = 0.0`. All NEURO must be earned through verified Proof of Neural Work. Even the project creators must run nodes to earn NEURO.

### What's the total supply?

There's no hard cap. NEURO is minted through work. However:
- 5% of all spending is burned (deflationary)
- Rewards have diminishing returns
- Circulating supply = minted - burned

## Running a Node

### What hardware do I need?

**Minimum**:
- 2GB RAM
- Internet connection
- Any CPU

**Recommended**:
- 8GB+ RAM
- NVIDIA GPU with CUDA or Apple Silicon
- Stable internet

### Can I run multiple nodes?

Yes! You can run multiple nodes with the same wallet token:
- Each node gets a unique network identity
- All earnings go to the same wallet
- More nodes = more earnings

### Do I need to stake to run a node?

**No** for basic participation. You can run a Driver or Worker without staking.

**Yes** (100 NEURO) to become a **Validator** who verifies proofs from other nodes.

### How much can I earn?

| Node Type | Memory | Daily Earnings (Active) |
|-----------|--------|------------------------|
| Raspberry Pi | 2GB | ~10-20 NEURO |
| Laptop | 8GB | ~40-60 NEURO |
| Gaming PC | 16GB | ~80-120 NEURO |
| Server | 64GB+ | ~200-400 NEURO |

Earnings depend on network activity and model quality.

### Why is my node holding all layers?

In early network stages (few nodes), each node holds many layers. This is **temporary**. As more nodes join, layers are redistributed across the network.

### Can I run inference without training?

Yes:
```bash
neuroshard --token YOUR_TOKEN --no-training
```

## Technical

### What model architecture is NeuroLLM?

NeuroLLM is a modern transformer with:
- RMSNorm (stable distributed training)
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA)
- SwiGLU activation

Architecture scales dynamically with network capacity.

### How does DiLoCo work?

DiLoCo (Distributed Low-Communication) is our training protocol:
1. Train locally for 500 steps
2. Compute pseudo-gradient (what we learned)
3. Sync with peers (rarely!)
4. Apply outer optimizer with Nesterov momentum

This reduces communication by 500×.

### How are gradients aggregated?

We use Byzantine-tolerant aggregation:
- **Trimmed Mean**: Remove outliers, then average
- **Krum**: Select gradient closest to majority
- **Coordinate Median**: Median of each parameter

This protects against malicious nodes.

### What is Proof of Neural Work?

PoNW is our consensus mechanism. It validates that nodes are doing real neural network computation (training/inference). Unlike Bitcoin's PoW which wastes energy on hashing, PoNW produces useful AI training.

Each proof includes:
- Work metrics (tokens, batches)
- ECDSA signature
- Timestamp and nonce
- Role and layer information

### How is the network secured?

1. **ECDSA Signatures**: Proofs are cryptographically signed
2. **Robust Aggregation**: Malicious gradients are rejected
3. **Validator Stakes**: Validators stake 100 NEURO (slashed for bad behavior)
4. **Rate Limiting**: Prevents reward inflation
5. **Slashing**: Fraud results in token loss

## Troubleshooting

### My node says "No GPU detected"

Install CUDA-enabled PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### I'm not earning any NEURO

1. Ensure training is enabled (no `--no-training` flag)
2. Wait for Genesis data to load (30-60 seconds)
3. Check your balance at the Ledger Explorer

### Training loss isn't decreasing

Normal in early network stages. The model improves as:
- More nodes join
- More data is processed
- More training rounds complete

### Can I see my earnings history?

Yes, visit the [Ledger Explorer](https://neuroshard.com/ledger) and search for your wallet address.

## Security

### Is my wallet token safe?

Your wallet token should be treated like a private key:
- Never share it
- Don't post it in logs or screenshots
- Store it securely

### Can someone steal my NEURO?

Only if they have your wallet token. Keep it secret.

### What happens if I lose my token?

Your NEURO is unrecoverable without your token. **Back it up securely**.

### Can nodes poison the model?

Our defenses make this difficult:
1. Robust aggregation rejects outlier gradients
2. Validators verify proof plausibility
3. Slashing punishes detected fraud

## Future

### Will there be a token launch/ICO?

No. NEURO is only earned through work. There's no sale, no pre-mine, no investor allocation.

### Will the model be open-sourced?

The model is inherently open — it exists across the network. Checkpoints can be assembled from shards and are community property.

### How will governance work?

NeuroDAO (coming) will enable token holders to vote on:
- Reward rates
- Slashing conditions
- Protocol parameters
- Architecture changes (tokenizer, attention mechanism)

What NeuroDAO will **not** control:
- Model size (automatic based on network capacity)
- Who can participate (open to all)

## See Also

- [Introduction](/guide/introduction)
- [How It Works](/guide/how-it-works)
- [NEURO Token](/economics/overview)
- [Troubleshooting](/guide/troubleshooting)

