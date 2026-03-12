# Mixture of Experts (MoE)

NeuroShard uses sparse Mixture of Experts layers that distribute computation across the network. Each layer has multiple expert FFNs, and a router selects which experts process each token.

## Overview

```
                    ┌─────────────────────────────────────────┐
                    │           MoE Transformer Layer          │
                    │                                          │
   Input ──────────►│  ┌──────────────────────────────────┐   │
                    │  │           Self-Attention         │   │
                    │  └──────────────┬───────────────────┘   │
                    │                 │                        │
                    │                 ▼                        │
                    │  ┌──────────────────────────────────┐   │
                    │  │            Router                 │   │
                    │  │     "Which 2 experts?"           │   │
                    │  └─────────────┬────────────────────┘   │
                    │                │                        │
                    │     ┌──────────┼──────────┐            │
                    │     ▼          ▼          ▼            │
                    │  ┌─────┐   ┌─────┐   ┌─────┐          │
                    │  │ E0  │   │ E3  │   │ E7  │  ...     │
                    │  │Local│   │Peer1│   │Peer2│          │
                    │  └──┬──┘   └──┬──┘   └──┬──┘          │
                    │     │         │         │              │
                    │     └─────────┴─────────┘              │
                    │               │                        │
                    │               ▼                        │
                    │         Weighted Sum ──────────────────► Output
                    └─────────────────────────────────────────┘
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_experts` | 8 | Experts per MoE layer |
| `experts_per_token` | 2 | Top-k routing (k experts per token) |
| `target_replicas` | 2 | Minimum replicas per expert |
| `capacity_factor` | 1.25 | Buffer for load balancing |

## Network Scaling

MoE works at any network size:

### Single Node (Bootstrap)

When only one node is in the network, it holds all experts locally:

```
┌────────────────────────────────────────┐
│            Single Node                  │
│                                         │
│   Layer 0: E0, E1, E2, E3, E4, E5, E6, E7  │
│   Layer 1: E0, E1, E2, E3, E4, E5, E6, E7  │
│   ...                                   │
│                                         │
│   All experts local → no remote calls   │
└────────────────────────────────────────┘
```

- Full model runs locally
- No gRPC overhead
- Training works normally

### Small Network (2-8 nodes)

Experts distribute across nodes:

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│   Node A    │   │   Node B    │   │   Node C    │
├─────────────┤   ├─────────────┤   ├─────────────┤
│ E0, E1, E2  │   │ E2, E3, E4  │   │ E5, E6, E7  │
│ (3 experts) │   │ (3 experts) │   │ (3 experts) │
└─────────────┘   └─────────────┘   └─────────────┘

Note: E2 is replicated on A and B for redundancy
```

- Each node holds a subset of experts
- Some experts replicated for redundancy
- Cross-node calls when needed

### Large Network (100+ nodes)

Experts replicated for availability and load balancing:

```
Expert 0: Nodes A, D, K, P, ...  (high demand → many replicas)
Expert 1: Nodes B, E, L, Q, ...
...
Expert 7: Nodes C, G, M, R, ...  (low demand → fewer replicas)
```

- Popular experts get more replicas
- Scarcity bonus incentivizes holding rare experts
- Load naturally balances

## Node Capacity Handling

Nodes with different computational power are all supported:

### Small Nodes (e.g., Jetson Orin, 8-16GB)

```python
# Assignment based on available memory
available_memory = 8000  # MB

# Small node might hold 2 experts per layer
experts_per_layer = available_memory // expert_memory_mb
# Result: 1-2 experts per layer
```

- Hold fewer experts
- Rely on network for remaining experts
- Still contribute to training

### Large Nodes (e.g., A100, 80GB)

```python
available_memory = 80000  # MB

# Large node can hold all 8 experts per layer
experts_per_layer = min(8, available_memory // expert_memory_mb)
# Result: 8 experts per layer
```

- Hold all or most experts locally
- Serve as expert providers for smaller nodes
- Higher NEURO rewards (more work)

### Expert Assignment Algorithm

```python
def assign_experts(node_id, layer_ids, available_memory):
    """Assign experts based on node capacity."""
    
    # Calculate how many experts this node can hold per layer
    expert_memory = hidden_dim * intermediate_dim * 4 * 2 / 1e6  # MB
    max_experts = int(available_memory / (len(layer_ids) * expert_memory))
    experts_per_layer = min(NUM_EXPERTS_PER_LAYER, max(1, max_experts))
    
    assigned = {}
    for layer_id in layer_ids:
        # Prefer under-replicated experts (scarcity bonus)
        candidates = get_underreplicated_experts(layer_id)
        assigned[layer_id] = candidates[:experts_per_layer]
    
    return assigned
```

## Handling Missing Experts

When a required expert isn't available locally:

### 1. Try Remote Call

```python
if expert_id not in local_experts:
    # Find peer with this expert
    peer = find_expert_holder(layer_id, expert_id)
    if peer:
        output = grpc_expert_forward(peer, activations)
```

### 2. Fallback: Skip Token

If no peer has the expert (rare edge case):

```python
if peer is None:
    # Token doesn't get this expert's contribution
    # Other selected expert(s) still contribute
    logger.warning(f"Expert {expert_id} unavailable, skipping")
    output = torch.zeros_like(input)
```

This is rare because:
- Target replication ensures 2+ copies of each expert
- Network incentivizes holding rare experts (scarcity bonus)
- New nodes prioritize under-replicated experts

### 3. Single-Node Mode

When network has only 1 node, all experts are local:

```python
def initialize_layers(self, layer_ids):
    # Check if we're the only node
    if len(self.layer_pool.node_capacities) <= 1:
        # Single node: hold ALL experts
        for layer_id in layer_ids:
            self.my_experts[layer_id] = list(range(NUM_EXPERTS_PER_LAYER))
```

## Router Synchronization

Each node has a copy of the router. Routers stay in sync via DiLoCo:

```
Node A            Node B            Node C
┌─────────┐       ┌─────────┐       ┌─────────┐
│ Router  │       │ Router  │       │ Router  │
│ (copy)  │       │ (copy)  │       │ (copy)  │
└────┬────┘       └────┬────┘       └────┬────┘
     │                 │                 │
     └────────────┬────┴─────────────────┘
                  ▼
          DiLoCo Sync
    (includes router weights)
```

- Router weights: ~6KB per layer
- Synced every 500 steps
- Part of normal DiLoCo pseudo-gradient aggregation

## Training

Training works the same regardless of network size:

```python
# Forward pass (router selects experts)
output = model(input_ids)

# Compute loss
loss = cross_entropy(output, labels)

# Add load balancing loss (prevents routing collapse)
aux_loss = model.get_moe_aux_loss()
if aux_loss is not None:
    loss = loss + aux_loss

# Backward and update
loss.backward()
optimizer.step()
```

### Load Balancing Loss

Prevents all tokens routing to the same experts:

```python
# Fraction of tokens per expert
f = tokens_per_expert / total_tokens

# Probability mass per expert  
P = mean_router_probability_per_expert

# Loss = num_experts * sum(f * P)
# Minimized when both are uniform (1/num_experts)
load_balance_loss = num_experts * (f * P).sum()
```

## Inference

Inference follows the same routing:

```python
def generate(prompt, max_tokens):
    tokens = tokenize(prompt)
    
    for _ in range(max_tokens):
        # Forward through MoE layers
        logits = model(tokens)
        
        # Sample next token
        next_token = sample(logits[:, -1])
        tokens = concat(tokens, next_token)
    
    return tokens
```

## NEURO Rewards

Expert hosting affects rewards:

| Factor | Bonus |
|--------|-------|
| Hosting rare expert (scarcity) | Up to 1.5× |
| High expert utilization | Up to 1.2× |
| Multiple experts per layer | Linear with count |

## Next Steps

- [DiLoCo Protocol](/architecture/diloco) — How weights synchronize
- [P2P Network](/architecture/p2p-network) — Expert discovery and routing
- [Economics](/economics/rewards) — NEURO reward calculation
