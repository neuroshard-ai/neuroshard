# Introduction

NeuroShard is a **decentralized architecture for collective intelligence** — a protocol that turns a global network of heterogeneous devices into one continuously evolving optimizer that collectively trains a shared model, **NeuroLLM**, from random initialization.

## The Problem with Centralized AI

The current landscape of artificial intelligence is defined by a small number of vertically integrated platforms that own the models, the data, and the distribution channels:

- **Centralized Control**: Alignment, safety, and access policies are set unilaterally with limited transparency
- **Infrastructure Barriers**: Training frontier models requires billion-dollar clusters, shutting out independent researchers
- **Compute Waste**: While hyperscale data centers consume enormous energy, the world's edge compute remains unused
- **Data Privacy**: Centralized training creates systemic privacy risks

## The NeuroShard Solution

NeuroShard proposes a decentralized alternative: a protocol for **collaborative model creation** where the network, not any single operator, is where the model lives.

### Core Principles

#### 1. Organic Scalability
The model architecture is not fixed but elastic. Using a **Dynamic Layer Pool**, the model expands as new nodes join — from 85M (Nano) to 123B (XL) parameters — with no fixed tier jumps.

#### 2. Verifiable Computation
Through **Proof of Neural Work (PoNW)**, the network cryptographically validates that participants are performing useful training operations. Gradients are proofs of work. Chained PoNW creates tamper-evident epoch chains via ECDSA-signed gradient commitments.

#### 3. Byzantine Tolerance
The training process is secured against malicious actors through robust statistics (Krum, Multi-Krum, Trimmed Mean, Geometric Median, Coordinate-wise Median) and a fraud-proof slashing mechanism.

#### 4. Economic Alignment
The NEURO token aligns incentives, rewarding participants for verifiable contributions of compute, data, and uptime. Role multipliers (Driver 1.0×, Worker 0.8×, Validator 1.2×) and stake multipliers (up to 2×) shape earnings.

## NeuroLLM: The Model

NeuroLLM is not a fork or fine-tune of an existing model. It is a new transformer architecture built from scratch for distributed, verifiable training:

| Feature | Description |
|---------|-------------|
| **Efficient Gradients** | Optimized for gradient compression and gossip transmission |
| **Stable Training** | Robust to asynchronous, noisy gradient updates |
| **Scalable Architecture** | Grows from 85M to 123B+ parameters as the network matures |
| **Privacy-Compatible** | Supports differential privacy in training data |

### Architecture Components

- **RMSNorm** — More stable for distributed training than LayerNorm
- **Rotary Position Embeddings (RoPE)** — No fixed maximum sequence length
- **Grouped Query Attention (GQA)** — 3× reduction in KV cache size
- **SwiGLU Activation** — Better gradient flow than ReLU or GELU
- **Mixture of Experts (MoE)** — 8 expert networks per layer, top-2 routing, scarcity bonuses for rare experts
- **NeuroMemory** — Three-tier persistent memory (Cortex / Commons / Soul) with personal LoRA adapters

## How It Works

```mermaid
graph TB
    subgraph Network["NeuroShard Network"]
        direction TB

        subgraph Roles["Training Pipeline"]
            Driver["Driver (Layer 0)"]
            Workers["Workers (Layers 1…N)"]
            Validator["Validator (LM Head)"]
        end

        Driver --> Workers
        Workers --> Validator

        Gossip["Gradient Gossip Protocol<br/>DiLoCo: Sync every 500 steps"]
        Agg["Robust Aggregation<br/>Krum / Geometric Median / Trimmed Mean"]
        Ledger["NEURO Ledger<br/>Rewards, PoNW Proofs, Chained Epochs"]

        Validator --> Gossip
        Gossip --> Agg
        Agg --> Ledger
    end
```

## Key Differentiators

| Feature | NeuroShard | Other Systems |
|---------|-----------|---------------|
| **Model Ownership** | Community-owned, trained from scratch | Pre-trained by corporations |
| **Architecture** | Dynamic, grows with network | Fixed model sizes |
| **Training** | Decentralized DiLoCo, 500× less comms | Centralized data centers |
| **Experts** | MoE per layer — efficient sparse activation | Dense models |
| **Memory** | Persistent NeuroMemory with personal LoRA | Stateless inference |
| **Rewards** | Proof of Neural Work — role + stake multipliers | Proof of Stake / Output |
| **Data** | Genesis dataset (FineWeb-Edu, RedPajama) | Unknown training data |
| **No Pre-mine** | Zero pre-mine, zero ICO | Founder allocations common |

## Whitepaper

The full technical whitepaper is available to registered members at [neuroshard.com/whitepaper](https://neuroshard.com/whitepaper). Create a free account to access it.

## Next Steps

- [Quick Start Guide](/guide/quick-start) — Get your node running in 5 minutes
- [How It Works](/guide/how-it-works) — Deep dive into the technology
- [NEURO Economics](/economics/overview) — Understand rewards and tokenomics
- [Architecture Overview](/architecture/overview) — Technical deep dive
- [Mixture of Experts](/architecture/mixture-of-experts) — MoE routing and rewards
- [NeuroMemory](/architecture/memory) — Persistent memory architecture
