# NeuroShard v2: Complete Architecture Design

> **Status**: Design Document  
> **Version**: 2.1  
> **Date**: December 2025  
> **Authors**: NeuroShard Team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Core Design Philosophy](#2-core-design-philosophy)
3. [Dynamic Network Topology](#3-dynamic-network-topology)
4. [Quorum-Based Training](#4-quorum-based-training)
5. [Contribution Modes](#5-contribution-modes)
6. [Layer Growth System](#6-layer-growth-system)
7. [Vocabulary Governance](#7-vocabulary-governance)
8. [Training Architecture](#8-training-architecture)
9. [Inference Architecture](#9-inference-architecture)
10. [Proof of Neural Work v2](#10-proof-of-neural-work-v2)
11. [Reward System](#11-reward-system)
12. [Adversarial Resistance](#12-adversarial-resistance)
13. [Scale Adaptation](#13-scale-adaptation)
14. [Data Structures](#14-data-structures)
15. [Implementation Plan](#15-implementation-plan)

---

## 1. Executive Summary

### 1.1 Vision

NeuroShard v2 is a **fully dynamic**, decentralized architecture for training and running an ever-growing Large Language Model. Every aspect of the system adapts automatically:

| Dynamic Element | How It Adapts |
|-----------------|---------------|
| **Network Size** | 1 node to millions, same protocol |
| **Model Depth** | Layers grow as network capacity increases |
| **Model Width** | Hidden dimensions scale with depth |
| **Vocabulary** | Expands via governance proposals |
| **Node Power** | Any device contributes appropriately |
| **Quorums** | Form, dissolve, reorganize dynamically |
| **Pricing** | Market-driven with demand response |
| **Verification** | Scales with network size |

### 1.2 Key Innovations

| Innovation | Description |
|------------|-------------|
| **Quorum-Based Training** | Speed-matched node groups form complete pipelines |
| **Multi-Mode Contribution** | Every node contributes something (pipeline, async, data, verify) |
| **Adaptive Protocol** | Same code works from 1 to 1M nodes |
| **Governance-Based Vocab** | No coordinator attack vector |
| **Sqrt-Weighted Gradients** | Fair influence proportional to work |

### 1.3 The Fundamental Principle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     EVERYTHING IS DYNAMIC                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   There are NO fixed:                                                       │
│   • Roles (nodes change what they do based on context)                      │
│   • Sizes (model grows, network grows, vocab grows)                         │
│   • Speeds (quorums match similar speeds together)                          │
│   • Prices (market discovers fair rates)                                    │
│   • Quorums (form, adapt, dissolve as needed)                               │
│                                                                             │
│   The only constants are the RULES OF THE GAME:                             │
│   • How to discover peers (DHT)                                             │
│   • How to verify work (PoNW)                                               │
│   • How to aggregate gradients (robust methods)                             │
│   • How to resolve disputes (stake slashing)                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Design Philosophy

### 2.1 The Model is a Living Entity

The NeuroLLM model is not a fixed artifact - it's a living, growing entity that exists only as the collective state of the network.

```
TIME →

T=0 (Genesis):
  ┌─────────────────┐
  │ 11 layers       │
  │ 512 hidden      │  ← Fits on 1 node
  │ 266 vocab       │
  │ ~125M params    │
  └─────────────────┘

T=1 year:
  ┌─────────────────────────────────────────────────────┐
  │ 48 layers                                           │
  │ 4096 hidden                                         │  ← Needs 20+ nodes
  │ 50,000 vocab                                        │
  │ ~25B params                                         │
  └─────────────────────────────────────────────────────┘

T=5 years:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 128 layers                                                              │
  │ 8192 hidden                                                             │
  │ 500,000 vocab                                            ← Needs 500+ nodes
  │ ~500B params                                                            │
  └─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 No Single Node Holds Everything

After the bootstrap phase, no single node can hold the entire model. This is by design:
- Enforces true decentralization
- Prevents single points of failure
- Enables unlimited growth

### 2.3 Every Node Contributes Something

From an H100 server to a Raspberry Pi, every node can contribute:

| Node Capability | Contribution Mode | Value Provided |
|-----------------|-------------------|----------------|
| Fast + Large | Pipeline Member | Real-time training/inference |
| Medium | Pipeline Member | Training at moderate speed |
| Slow + Small | Async Contributor | Gradient diversity |
| Any | Data Provider | Training data availability |
| Any | Verifier | Network security |

---

## 3. Dynamic Network Topology

### 3.1 Node Capability Assessment

Every node self-benchmarks on join and periodically thereafter:

```python
@dataclass
class NodeCapabilities:
    node_id: str
    public_key: bytes
    endpoint: str
    
    # Hardware (measured)
    memory_mb: int              # Available RAM/VRAM
    compute_speed_ms: float     # Time for 1 layer forward pass
    bandwidth_mbps: float       # Network throughput
    has_gpu: bool
    
    # Derived
    speed_tier: SpeedTier       # T1, T2, T3, T4, T5
    max_layers: int             # How many layers can hold
    
    # Current state (dynamic)
    layer_range: Tuple[int, int]
    mode: ContributionMode
    quorum_id: Optional[str]
    available_capacity: int     # For inference
    
    # Reputation (earned)
    uptime_ratio: float         # 0.0 to 1.0
    success_rate: float         # Completed vs failed requests
    stake: float                # NEURO staked

class SpeedTier(Enum):
    T1 = "tier1"  # < 10ms/layer (H100, A100)
    T2 = "tier2"  # 10-50ms/layer (RTX 4090, 3090)
    T3 = "tier3"  # 50-200ms/layer (RTX 3060, good CPU)
    T4 = "tier4"  # 200-1000ms/layer (older GPU, standard CPU)
    T5 = "tier5"  # > 1000ms/layer (Raspberry Pi, old hardware)
```

### 3.2 DHT Structure

The DHT stores all dynamic network state:

```python
# Network-wide state
dht["network:state"] = {
    "arch_version": 5,
    "num_layers": 48,
    "hidden_dim": 4096,
    "vocab_version": 12,
    "vocab_size": 50000,
    "total_nodes": 247,
    "total_memory_mb": 1500000,
    "active_quorums": 23,
    "current_step": 5000000
}

# Per-layer state
dht["layer:15"] = {
    "layer_id": 15,
    "holders": [
        {"node_id": "abc", "quorum_id": "q1", "speed_tier": "T2"},
        {"node_id": "def", "quorum_id": "q2", "speed_tier": "T3"},
        {"node_id": "ghi", "quorum_id": None, "speed_tier": "T4"}  # async
    ],
    "replica_count": 3,
    "scarcity_score": 0.2  # 0=abundant, 1=scarce
}

# Per-quorum state
dht["quorum:q1"] = {
    "quorum_id": "q1",
    "speed_tier": "T2",
    "members": ["abc", "jkl", "mno", "pqr"],
    "layer_coverage": [[0,12], [13,24], [25,36], [37,48]],
    "throughput_batches_per_sec": 8.5,
    "session_start": 1702400000,
    "session_end": 1702403600,
    "status": "active"
}

# Per-node state
dht["node:abc"] = {
    "node_id": "abc",
    "endpoint": "192.168.1.1:8080",
    "speed_tier": "T2",
    "layer_range": [0, 12],
    "quorum_id": "q1",
    "mode": "pipeline",
    "last_heartbeat": 1702400500,
    "reputation": 0.95
}
```

### 3.3 Heartbeat Protocol

Nodes broadcast lightweight heartbeats every 30 seconds:

```python
heartbeat = {
    "node_id": "abc",
    "timestamp": time.now(),
    "quorum_id": "q1",
    "queue_depth": 2,
    "available_capacity": 5,
    "current_step": 5000123
}

# Stale detection
HEARTBEAT_INTERVAL = 30      # seconds
STALE_THRESHOLD = 4          # missed heartbeats
OFFLINE_THRESHOLD = 120      # seconds = 4 × 30

# On stale detection:
# 1. Mark node as "degraded"
# 2. Quorum initiates replacement search
# 3. After OFFLINE_THRESHOLD, node marked offline
# 4. Quorum reorganizes or dissolves
```

---

## 4. Quorum-Based Training

### 4.1 What is a Quorum?

A **Quorum** is a self-organized group of nodes that together hold a complete model and train as a unit.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           QUORUM CONCEPT                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  A Quorum is a complete pipeline:                                           │
│                                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐                  │
│  │ Node A  │───→│ Node B  │───→│ Node C  │───→│ Node D  │                  │
│  │ L0-12   │    │ L13-24  │    │ L25-36  │    │ L37-48  │                  │
│  │ (T2)    │    │ (T2)    │    │ (T2)    │    │ (T2)    │                  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘                  │
│       ↑                                             │                       │
│       └─────────── gradients ──────────────────────┘                       │
│                                                                             │
│  Properties:                                                                │
│  • Complete: Covers all layers (0 to N+head)                                │
│  • Speed-matched: All members in compatible tiers                           │
│  • Self-sufficient: Can train independently                                 │
│  • Temporary: Lasts 1 session (~1 hour), then renews or dissolves           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Quorum Formation Algorithm

```python
def form_quorum(initiator: Node) -> Optional[Quorum]:
    """
    Form a new quorum starting from the initiator node.
    Uses DHT to discover compatible peers.
    """
    
    my_tier = initiator.speed_tier
    my_layers = set(initiator.layer_range)
    all_layers = set(range(network_state.num_layers + 1))  # +1 for LM head
    missing_layers = all_layers - my_layers
    
    # Find compatible speed tiers
    compatible = {
        SpeedTier.T1: [SpeedTier.T1, SpeedTier.T2],
        SpeedTier.T2: [SpeedTier.T1, SpeedTier.T2, SpeedTier.T3],
        SpeedTier.T3: [SpeedTier.T2, SpeedTier.T3, SpeedTier.T4],
        SpeedTier.T4: [SpeedTier.T3, SpeedTier.T4],
        SpeedTier.T5: [],  # T5 doesn't form quorums
    }
    
    if my_tier == SpeedTier.T5:
        return None  # Use async mode instead
    
    # Query DHT for candidates
    candidates = []
    for layer_id in missing_layers:
        layer_info = dht.get(f"layer:{layer_id}")
        for holder in layer_info.holders:
            if holder.speed_tier in compatible[my_tier]:
                if holder.quorum_id is None:  # Not already in a quorum
                    candidates.append(holder)
    
    # Greedy selection: maximize coverage, minimize latency
    selected = [initiator]
    covered = my_layers.copy()
    
    while covered != all_layers:
        # Score candidates by: new coverage / expected latency
        def score(c):
            new_coverage = len(set(c.layer_range) - covered)
            if new_coverage == 0:
                return 0
            latency_penalty = estimate_latency(initiator, c)
            return new_coverage / (1 + latency_penalty / 100)
        
        best = max(candidates, key=score, default=None)
        if best is None or score(best) == 0:
            return None  # Can't form complete quorum
        
        selected.append(best)
        covered |= set(best.layer_range)
        candidates.remove(best)
    
    # Propose quorum to all selected members
    proposal = QuorumProposal(
        quorum_id=generate_quorum_id(),
        members=[n.node_id for n in selected],
        layer_map={n.node_id: n.layer_range for n in selected},
        speed_tier=my_tier,
        session_duration=3600  # 1 hour
    )
    
    # All must accept
    responses = parallel_request([
        (node, "quorum_invite", proposal) 
        for node in selected if node != initiator
    ])
    
    if all(r.accepted for r in responses):
        quorum = Quorum(proposal)
        dht.set(f"quorum:{quorum.id}", quorum.to_dict())
        return quorum
    else:
        # Some declined, retry with alternates or fail
        return retry_with_alternates(proposal, responses)
```

### 4.3 Quorum Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         QUORUM LIFECYCLE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. FORMATION                                                               │
│     ───────────                                                             │
│     • Initiator discovers compatible peers via DHT                          │
│     • Proposes quorum, all must accept                                      │
│     • Register in DHT, set session_end timestamp                            │
│     • Begin training                                                        │
│                                                                             │
│  2. ACTIVE TRAINING                                                         │
│     ────────────────                                                        │
│     • Process batches as a synchronized pipeline                            │
│     • Track per-member performance metrics                                  │
│     • Participate in cross-quorum cohort sync (DiLoCo)                      │
│                                                                             │
│  3. HEALTH MONITORING (continuous)                                          │
│     ────────────────────────────────                                        │
│     For each member:                                                        │
│       if missed_heartbeats >= 2:                                            │
│           mark_degraded(member)                                             │
│           start_replacement_search(member.layers)                           │
│       if latency > tier_average × 2 for 5 minutes:                          │
│           initiate_replacement(member)                                      │
│                                                                             │
│  4. RENEWAL (at 80% of session duration)                                    │
│     ──────────────────────────────────                                      │
│     • All members healthy? → Auto-extend session                            │
│     • Member underperforming? → Replace during renewal                      │
│     • Better options available? → Consider upgrade (if >20% better)         │
│     • Member wants to leave? → Graceful replacement                         │
│                                                                             │
│  5. DISSOLUTION                                                             │
│     ────────────                                                            │
│     Triggers:                                                               │
│       • Session ends without renewal                                        │
│       • Critical member goes offline with no replacement                    │
│       • Network architecture upgrade (all quorums reform)                   │
│     Process:                                                                │
│       • Save current weights to DHT                                         │
│       • Remove quorum from registry                                         │
│       • Members become available for new quorums                            │
│                                                                             │
│  SESSION PARAMETERS:                                                        │
│  ───────────────────                                                        │
│  BASE_SESSION = 3600 seconds (1 hour)                                       │
│  MAX_SESSION = 14400 seconds (4 hours)                                      │
│  RENEWAL_CHECK = 0.8 × session (48 minutes for 1-hour session)              │
│  MIN_BATCHES_TO_RENEW = 1000                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.4 Multiple Quorums, Different Speeds

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SPEED-TIERED QUORUMS                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FAST QUORUM (T1-T2):           MEDIUM QUORUM (T2-T3):                      │
│  ┌───────────────────────┐      ┌───────────────────────┐                   │
│  │ 4 nodes, all fast GPU │      │ 5 nodes, mixed GPU    │                   │
│  │ ~60ms per batch       │      │ ~300ms per batch      │                   │
│  │ ~16 batches/sec       │      │ ~3 batches/sec        │                   │
│  └───────────────────────┘      └───────────────────────┘                   │
│                                                                             │
│  SLOW QUORUM (T3-T4):           ASYNC CONTRIBUTORS (T5):                    │
│  ┌───────────────────────┐      ┌───────────────────────┐                   │
│  │ 6 nodes, CPU-heavy    │      │ Not in any quorum     │                   │
│  │ ~2000ms per batch     │      │ Train offline         │                   │
│  │ ~0.5 batches/sec      │      │ Submit when ready     │                   │
│  └───────────────────────┘      └───────────────────────┘                   │
│                                                                             │
│  KEY INSIGHT: Each quorum is internally synchronized but operates           │
│  asynchronously relative to other quorums. Cross-quorum sync via DiLoCo.    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Contribution Modes

Every node operates in one of these modes based on its capabilities and network needs:

### 5.1 Mode Definitions

```python
class ContributionMode(Enum):
    PIPELINE = "pipeline"      # Real-time quorum member
    ASYNC = "async"            # Offline training, submit gradients
    DATA = "data"              # Store and serve training data
    VERIFY = "verify"          # Re-execute proofs for verification
    INFERENCE = "inference"    # Serve inference requests only
    IDLE = "idle"              # Available but not active
```

### 5.2 Mode Selection Logic

```python
def select_contribution_mode(node: Node) -> ContributionMode:
    """Automatically select best mode for this node."""
    
    # T5 nodes can't do real-time pipeline
    if node.speed_tier == SpeedTier.T5:
        if node.has_genesis_data:
            return ContributionMode.DATA
        else:
            return ContributionMode.ASYNC
    
    # Check if node can join/form a quorum
    quorum = try_join_existing_quorum(node, timeout=30)
    if quorum:
        return ContributionMode.PIPELINE
    
    quorum = try_form_new_quorum(node, timeout=60)
    if quorum:
        return ContributionMode.PIPELINE
    
    # No compatible quorum found
    # Maybe network is small or node's layers are well-covered
    
    if network_needs_verifiers():
        return ContributionMode.VERIFY
    
    if network_needs_data_providers() and node.has_storage:
        return ContributionMode.DATA
    
    # Fall back to async contribution
    return ContributionMode.ASYNC
```

### 5.3 Pipeline Mode (Real-Time)

```
Requirements:
  • Speed tier T1-T4
  • Part of an active quorum
  • Reliable uptime (>90%)

Responsibilities:
  • Receive activations from previous node
  • Forward through local layers
  • Send to next node (forward) or previous (backward)
  • Participate in cohort sync every N batches

Rewards:
  • Per-batch payment (highest rate)
  • Multipliers for: scarcity, position (embed/head), stake
```

### 5.4 Async Mode (Offline Training)

```
Requirements:
  • Any speed tier (even T5)
  • Can hold at least 1 layer

Process:
  1. Download current weights for held layers from DHT/peers
  2. Download training data (from Genesis providers)
  3. Train locally for N steps (can take hours/days)
  4. Compute pseudo-gradient: Δw = w_initial - w_final
  5. Submit to layer cohort for next sync round
  6. Include: steps_trained, data_hash, initial_weights_hash

Rewards:
  • Per-gradient payment
  • Weighted by: sqrt(batches) × freshness
  • Freshness decay: 1.0 (<1hr), 0.9 (<1day), 0.7 (<1week), 0.3 (older)
```

### 5.5 Data Provider Mode

```
Requirements:
  • Storage capacity for Genesis shards
  • Reliable bandwidth

Responsibilities:
  • Store assigned Genesis data shards
  • Serve data to requesting nodes
  • Maintain data integrity (hashes verified)

Rewards:
  • Storage payment (per GB per day)
  • Bandwidth payment (per GB served)
```

### 5.6 Verifier Mode

```
Requirements:
  • Can execute forward pass (any speed)
  • Has stake to challenge

Responsibilities:
  • Monitor submitted proofs
  • Randomly select proofs to verify
  • Re-execute and compare results
  • Challenge if mismatch found

Rewards:
  • No base payment
  • Large reward if fraud detected (gets slashed stake)
  • Risk: lose stake if challenge fails
```

---

## 6. Layer Growth System

### 6.1 Adaptive Growth

Model depth grows automatically as network capacity increases:

```python
def check_layer_growth() -> Optional[ArchitectureUpgrade]:
    """Called periodically to check if model should grow."""
    
    state = dht.get("network:state")
    
    # Current model memory usage
    current_memory = estimate_model_memory(
        num_layers=state.num_layers,
        hidden_dim=state.hidden_dim,
        vocab_size=state.vocab_size
    )
    
    # Total network capacity
    total_capacity = sum(n.memory_mb for n in get_active_nodes())
    
    # Growth threshold (more conservative as network grows)
    # Small network: grow at 1.5×, Large network: grow at 2.0×
    threshold = 1.5 + 0.1 * math.log2(max(1, state.total_nodes))
    
    if total_capacity < current_memory * threshold:
        return None  # Not enough capacity
    
    # Check layer coverage
    min_replicas = get_adaptive_min_replicas(state.total_nodes)
    for layer_id in range(state.num_layers):
        layer_info = dht.get(f"layer:{layer_id}")
        if layer_info.replica_count < min_replicas:
            return None  # Existing layers need more coverage first
    
    # Check stability
    if state.steps_since_last_upgrade < 10000:
        return None  # Too soon after last upgrade
    
    # Calculate new architecture
    new_arch = calculate_optimal_architecture(total_capacity)
    
    if new_arch.num_layers <= state.num_layers + 2:
        return None  # Not significant enough
    
    return ArchitectureUpgrade(
        new_num_layers=new_arch.num_layers,
        new_hidden_dim=new_arch.hidden_dim,
        grace_period=600,  # 10 minutes
        warmup_steps=1000
    )

def get_adaptive_min_replicas(num_nodes: int) -> int:
    """Minimum replicas required, scales with network size."""
    if num_nodes < 5:
        return 1
    elif num_nodes < 20:
        return 2
    else:
        return 3
```

### 6.2 Layer Addition Process

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LAYER ADDITION SEQUENCE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1: ANNOUNCEMENT (T=0)                                                │
│  ───────────────────────────                                                │
│  • Broadcast: "Architecture upgrade: 24L → 28L"                             │
│  • New arch_version assigned                                                │
│  • Grace period begins (10 minutes)                                         │
│  • Existing quorums continue with old architecture                          │
│                                                                             │
│  PHASE 2: PREPARATION (T+0 to T+10min)                                      │
│  ─────────────────────────────────────                                      │
│  • New layer entries created in DHT (status="pending")                      │
│  • Nodes decide which new layers to claim                                   │
│  • High scarcity bonus attracts nodes to new layers                         │
│  • Nodes download initial weights (identity-initialized)                    │
│                                                                             │
│  PHASE 3: ACTIVATION (T+10min)                                              │
│  ─────────────────────────────                                              │
│  Activation requires:                                                       │
│    • Each new layer has >= MIN_REPLICAS holders                             │
│    • At least one "fast" holder (T1-T3) per layer                           │
│  If not met:                                                                │
│    • Extend grace period by 10 more minutes                                 │
│    • Increase scarcity bonus further                                        │
│    • After 3 extensions, abort upgrade                                      │
│                                                                             │
│  PHASE 4: QUORUM REFORMATION (T+10min to T+15min)                           │
│  ────────────────────────────────────────────────                           │
│  • All existing quorums dissolve gracefully                                 │
│  • Nodes reform quorums with new layer coverage                             │
│  • New pipelines include new layers                                         │
│                                                                             │
│  PHASE 5: WARMUP (T+15min to T+warmup_complete)                             │
│  ──────────────────────────────────────────────                             │
│  • New layers train with 0.1× learning rate                                 │
│  • Gradually ramp to full LR over warmup_steps                              │
│  • Reduced rewards during warmup (nodes are "investing")                    │
│                                                                             │
│  PHASE 6: STABLE                                                            │
│  ───────────────                                                            │
│  • New layers at full learning rate                                         │
│  • Full rewards for new layer holders                                       │
│  • arch_version fully adopted                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 New Layer Initialization

New layers are initialized as near-identity functions to minimize disruption:

```python
def initialize_new_layer(config: ModelConfig) -> TransformerLayer:
    """Initialize layer so output ≈ input initially."""
    
    layer = TransformerDecoderLayer(config)
    
    with torch.no_grad():
        # Attention output projection: near-zero
        # This makes attention contribution negligible initially
        layer.self_attn.o_proj.weight *= 0.01
        layer.self_attn.o_proj.bias.zero_()
        
        # FFN output projection: near-zero
        # This makes FFN contribution negligible initially
        layer.mlp.down_proj.weight *= 0.01
        layer.mlp.down_proj.bias.zero_()
        
        # Result: layer output ≈ input (identity + tiny residual)
        # Training will gradually learn useful representations
    
    return layer
```

---

## 7. Vocabulary Governance

### 7.1 No Automatic Coordinator

Unlike layer growth, vocabulary changes go through governance to prevent attacks:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHY GOVERNANCE FOR VOCAB?                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ATTACK SCENARIO (without governance):                                      │
│  ─────────────────────────────────────                                      │
│  1. Malicious "merge coordinator" is elected/selected                       │
│  2. Coordinator proposes merge: "api" + "_key" → "api_key"                  │
│  3. This token gets heavily trained on API key patterns                     │
│  4. Model learns to output API keys more readily                            │
│  5. Attacker can extract sensitive data via inference                       │
│                                                                             │
│  SOLUTION: GOVERNANCE                                                       │
│  ───────────────────────                                                    │
│  • Any merge must be publicly proposed                                      │
│  • 7-day review period for community inspection                             │
│  • 66% stake-weighted approval required                                     │
│  • Suspicious merges get rejected                                           │
│  • Proposer stake at risk if proposal is malicious                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Vocab Proposal Process

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VOCAB GOVERNANCE FLOW                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1: PROPOSAL (Requires 100+ NEURO stake)                              │
│  ─────────────────────────────────────────────                              │
│  proposal = {                                                               │
│      "id": "vocab-2025-042",                                                │
│      "proposer": "node_xyz",                                                │
│      "proposer_stake": 500,  # At risk if malicious                         │
│      "new_merges": [                                                        │
│          {"tokens": ["block", "chain"], "new_id": 50001, "freq": 85000},    │
│          {"tokens": ["smart", "contract"], "new_id": 50002, "freq": 72000}, │
│          ...                                                                │
│      ],                                                                     │
│      "data_sources": ["hash1", "hash2"],  # Verifiable                      │
│      "rationale": "High-frequency blockchain terminology"                   │
│  }                                                                          │
│                                                                             │
│  Automatic checks:                                                          │
│  • All merges have frequency > 1000 (prevent low-freq attacks)              │
│  • No merges contain sensitive patterns (password, secret, key, etc.)       │
│  • Total new tokens reasonable (max 1000 per proposal)                      │
│                                                                             │
│  PHASE 2: DISCUSSION (7 days)                                               │
│  ─────────────────────────────                                              │
│  • Proposal visible to all nodes                                            │
│  • Anyone can verify frequencies against their own data                     │
│  • Raise objections with evidence                                           │
│  • Proposer can amend (resets discussion period)                            │
│                                                                             │
│  PHASE 3: VOTING (7 days)                                                   │
│  ─────────────────────────                                                  │
│  • Stake-weighted voting                                                    │
│  • Options: APPROVE, REJECT, ABSTAIN                                        │
│  • Quorum requirement: 30% of total stake must vote                         │
│  • Approval threshold: 66% of votes (by stake weight)                       │
│  • Votes are final once cast                                                │
│                                                                             │
│  PHASE 4: IMPLEMENTATION (if approved)                                      │
│  ──────────────────────────────────────                                     │
│  • 7-day grace period for nodes to prepare                                  │
│  • Nodes expand embedding/LM head                                           │
│  • On activation: vocab_version increments                                  │
│  • Proposer stake returned + small reward                                   │
│                                                                             │
│  PHASE 5: EMERGENCY ROLLBACK (if issues found post-activation)              │
│  ─────────────────────────────────────────────────────────────              │
│  • 10% of total stake can trigger emergency vote                            │
│  • Fast-track: 48-hour voting period                                        │
│  • If approved: revert to previous vocab_version                            │
│  • Original proposer stake slashed                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Embedding/LM Head Expansion

When vocabulary grows, relevant nodes expand their weights:

```python
def expand_for_vocab_update(
    model: NeuroshardModel, 
    old_vocab: int, 
    new_vocab: int
) -> None:
    """Expand embedding and LM head for new vocabulary."""
    
    new_tokens = new_vocab - old_vocab
    hidden_dim = model.config.hidden_dim
    
    if model.has_embedding:
        # Expand embedding layer
        old_weight = model.embedding.weight.data
        model.embedding = nn.Embedding(new_vocab, hidden_dim)
        with torch.no_grad():
            model.embedding.weight[:old_vocab] = old_weight
            # Initialize new tokens with small random values
            model.embedding.weight[old_vocab:] = torch.randn(
                new_tokens, hidden_dim
            ) * 0.01
    
    if model.has_lm_head:
        # Expand LM head
        old_weight = model.lm_head.weight.data
        model.lm_head = nn.Linear(hidden_dim, new_vocab, bias=False)
        with torch.no_grad():
            model.lm_head.weight[:old_vocab] = old_weight
            model.lm_head.weight[old_vocab:] = torch.randn(
                new_tokens, hidden_dim
            ) * 0.01
```

---

## 8. Training Architecture

### 8.1 Complete Training Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE TRAINING ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                          ┌──────────────────┐                               │
│                          │   GENESIS DATA   │                               │
│                          │   (distributed)  │                               │
│                          └────────┬─────────┘                               │
│                                   │                                         │
│              ┌────────────────────┼────────────────────┐                    │
│              ▼                    ▼                    ▼                    │
│       ┌───────────┐        ┌───────────┐        ┌───────────┐              │
│       │   FAST    │        │  MEDIUM   │        │   SLOW    │              │
│       │  QUORUMS  │        │  QUORUMS  │        │  QUORUMS  │              │
│       │  (T1-T2)  │        │  (T2-T3)  │        │  (T3-T4)  │              │
│       └─────┬─────┘        └─────┬─────┘        └─────┬─────┘              │
│             │                    │                    │                     │
│      ~16 batch/s           ~3 batch/s           ~0.5 batch/s               │
│             │                    │                    │                     │
│             │                    │                    │      ┌───────────┐  │
│             │                    │                    │      │   ASYNC   │  │
│             │                    │                    │      │   (T5)    │  │
│             │                    │                    │      └─────┬─────┘  │
│             │                    │                    │            │        │
│             ▼                    ▼                    ▼            ▼        │
│       ┌─────────────────────────────────────────────────────────────┐      │
│       │                   LAYER COHORT SYNC                         │      │
│       │  ───────────────────────────────────────────────────────    │      │
│       │  For each layer L:                                          │      │
│       │    • Gather pseudo-gradients from all L holders             │      │
│       │    • Weight by sqrt(batches) × freshness                    │      │
│       │    • Robust aggregation (trimmed mean)                      │      │
│       │    • All L holders apply aggregated update                  │      │
│       └─────────────────────────────────────────────────────────────┘      │
│                                   │                                         │
│                                   ▼                                         │
│                          ┌──────────────────┐                               │
│                          │  UPDATED MODEL   │                               │
│                          │   (all synced)   │                               │
│                          └──────────────────┘                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Within-Quorum Training (Synchronous Pipeline)

```python
class QuorumTrainer:
    """Training coordinator for a single quorum."""
    
    def __init__(self, quorum: Quorum, node: Node):
        self.quorum = quorum
        self.node = node
        self.is_initiator = node.layer_range[0] == 0
        self.is_finisher = node.has_lm_head
        
        # DiLoCo state
        self.initial_weights = self.snapshot_weights()
        self.batches_since_sync = 0
    
    async def run_training_loop(self):
        """Main training loop for quorum member."""
        
        while self.quorum.is_active:
            if self.is_initiator:
                # Start new batch
                batch = self.genesis_loader.get_batch()
                await self.initiate_forward(batch)
            else:
                # Wait for activation from previous node
                packet = await self.receive_activation()
                await self.process_and_forward(packet)
            
            self.batches_since_sync += 1
            
            # Check if time for cohort sync
            if self.batches_since_sync >= SYNC_INTERVAL:
                await self.cohort_sync()
    
    async def initiate_forward(self, batch: Batch):
        """Initiator: embed and start pipeline."""
        
        embeddings = self.model.embed(batch.input_ids)
        hidden = self.model.forward_my_layers(embeddings)
        
        # Save for backward
        self.save_activation(batch.id, embeddings, hidden)
        
        # Send to next quorum member
        next_node = self.quorum.get_next_node(self.node)
        await self.send_activation(next_node, hidden, batch)
    
    async def process_and_forward(self, packet: ActivationPacket):
        """Processor or Finisher: continue pipeline."""
        
        hidden = deserialize(packet.activations)
        hidden.requires_grad_(True)
        
        output = self.model.forward_my_layers(hidden)
        
        if self.is_finisher:
            # Compute loss and start backward
            loss = self.compute_loss(output, packet.labels)
            loss.backward()
            self.optimizer.step()
            
            # Send gradients back
            await self.send_gradient(packet.sender, hidden.grad, loss)
        else:
            # Save and forward
            self.save_activation(packet.id, hidden, output)
            next_node = self.quorum.get_next_node(self.node)
            await self.send_activation(next_node, output, packet)
```

### 8.3 Cross-Quorum Sync (DiLoCo)

```python
SYNC_INTERVAL = 500  # batches per quorum (adaptive)

async def cohort_sync(self):
    """Synchronize with other holders of same layers across quorums."""
    
    # 1. Compute pseudo-gradient
    pseudo_grad = {}
    for name, param in self.model.named_parameters():
        pseudo_grad[name] = self.initial_weights[name] - param.data
    
    # 2. Find cohort (all nodes holding my layers)
    cohort = await self.find_layer_cohort(self.node.layer_range)
    
    # 3. Exchange pseudo-gradients with weights
    contributions = [
        GradientContribution(
            node_id=self.node.id,
            gradient=pseudo_grad,
            batches=self.batches_since_sync,
            timestamp=time.now()
        )
    ]
    
    for peer in cohort:
        peer_contribution = await self.request_gradient(peer)
        if peer_contribution:
            contributions.append(peer_contribution)
    
    # 4. Weighted robust aggregation
    aggregated = weighted_robust_aggregate(contributions)
    
    # 5. Apply outer update
    OUTER_LR = 0.7
    with torch.no_grad():
        for name, param in self.model.named_parameters():
            param.data = self.initial_weights[name] + OUTER_LR * aggregated[name]
    
    # 6. Reset for next round
    self.initial_weights = self.snapshot_weights()
    self.batches_since_sync = 0


def weighted_robust_aggregate(
    contributions: List[GradientContribution]
) -> Dict[str, Tensor]:
    """Aggregate gradients with sqrt-batch weighting and trimmed mean."""
    
    # Compute weights
    weights = []
    for c in contributions:
        batch_weight = math.sqrt(c.batches)
        
        age = time.now() - c.timestamp
        if age < 60:
            freshness = 1.0
        elif age < 3600:
            freshness = 0.9
        elif age < 86400:
            freshness = 0.7
        else:
            freshness = 0.3
        
        weights.append(max(0.1, batch_weight * freshness))
    
    # Normalize weights
    total = sum(weights)
    weights = [w / total for w in weights]
    
    # Trimmed mean aggregation (trim extreme 20%)
    aggregated = {}
    for param_name in contributions[0].gradient.keys():
        grads = [c.gradient[param_name] for c in contributions]
        aggregated[param_name] = trimmed_mean(grads, weights, trim=0.2)
    
    return aggregated
```

---

## 9. Inference Architecture

### 9.1 Decentralized Inference Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DECENTRALIZED INFERENCE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 1: DISCOVERY                                                          │
│  ─────────────────                                                          │
│  User queries DHT: "Available inference endpoints?"                         │
│                                                                             │
│  Response:                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Quorum │ Latency │ Price/Token │ Reputation │ Capacity │            │    │
│  ├────────┼─────────┼─────────────┼────────────┼──────────┤            │    │
│  │ Q1     │ 50ms    │ 0.00018     │ 98%        │ 5 slots  │            │    │
│  │ Q7     │ 200ms   │ 0.00010     │ 95%        │ 12 slots │            │    │
│  │ Q12    │ 2000ms  │ 0.00005     │ 90%        │ 20 slots │            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  STEP 2: REQUEST                                                            │
│  ────────────────                                                           │
│  User → Q1 Initiator:                                                       │
│  {                                                                          │
│      "prompt": "Explain blockchain...",                                     │
│      "max_tokens": 200,                                                     │
│      "temperature": 0.7,                                                    │
│      "payment_lock": 0.036  // 200 × 0.00018                                │
│  }                                                                          │
│                                                                             │
│  Payment held in escrow (DHT-based multisig or smart contract)              │
│                                                                             │
│  STEP 3: EXECUTION                                                          │
│  ─────────────────                                                          │
│  For each token:                                                            │
│    Initiator → embed → Processor1 → ... → Finisher → sample → Initiator    │
│    (Full pipeline round-trip per token)                                     │
│                                                                             │
│  Streaming: Tokens sent to user as generated                                │
│                                                                             │
│  STEP 4: PAYMENT                                                            │
│  ────────────────                                                           │
│  On completion:                                                             │
│  • Verify response was delivered                                            │
│  • Release payment proportionally:                                          │
│      - Initiator (embed): 15%                                               │
│      - Processors: proportional to layers                                   │
│      - Finisher (head): 20%                                                 │
│      - Network burn: 5%                                                     │
│                                                                             │
│  STEP 5: FAILURE HANDLING                                                   │
│  ─────────────────────────                                                  │
│  Timeout (30s no progress):                                                 │
│    • User can cancel, retry with different quorum                           │
│    • Partial refund for incomplete tokens                                   │
│    • Failing node gets reputation penalty                                   │
│                                                                             │
│  Complete failure:                                                          │
│    • Full refund                                                            │
│    • Quorum members lose reputation                                         │
│    • Repeated failures → stake slashing                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Dynamic Pricing

```python
def calculate_inference_price(quorum: Quorum) -> float:
    """Calculate per-token price for this quorum."""
    
    BASE_PRICE = 0.0001  # NEURO per token (network floor)
    
    # Demand multiplier: increases with load
    utilization = quorum.active_requests / quorum.max_capacity
    demand_mult = 1.0 + utilization ** 2  # 1.0 to 2.0
    
    # Speed factor: faster quorums charge more
    speed_mult = {
        SpeedTier.T1: 1.5,
        SpeedTier.T2: 1.3,
        SpeedTier.T3: 1.0,
        SpeedTier.T4: 0.8,
    }.get(quorum.speed_tier, 1.0)
    
    # Reputation factor: proven quorums charge more
    rep_mult = 0.8 + 0.4 * quorum.reputation  # 0.8 to 1.2
    
    price = BASE_PRICE * demand_mult * speed_mult * rep_mult
    
    # Ensure minimum profitability
    min_price = BASE_PRICE * 0.5
    return max(price, min_price)

# Example prices:
# Fast quorum, low demand, high rep: 0.0001 × 1.0 × 1.5 × 1.1 = 0.000165
# Slow quorum, high demand, new: 0.0001 × 2.0 × 0.8 × 0.9 = 0.000144
```

---

## 10. Proof of Neural Work v2

### 10.1 Optimistic Verification

Proofs are accepted optimistically and can be challenged:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OPTIMISTIC PONW                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. SUBMISSION                                                              │
│     ──────────                                                              │
│     Node completes work → generates proof → submits to DHT                  │
│     Proof status: "pending"                                                 │
│     Challenge window: 10 minutes                                            │
│                                                                             │
│  2. NO CHALLENGE (common case)                                              │
│     ─────────────────────────                                               │
│     Window expires → proof auto-accepted → rewards distributed              │
│                                                                             │
│  3. CHALLENGE (rare case)                                                   │
│     ──────────────────────                                                  │
│     Challenger stakes CHALLENGE_STAKE (10 NEURO)                            │
│     Challenger requests: specific gradients, activations, batch data        │
│     Prover must respond within 5 minutes                                    │
│                                                                             │
│  4. RESOLUTION                                                              │
│     ──────────                                                              │
│     Challenger recomputes and compares:                                     │
│                                                                             │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │ Outcome          │ Prover             │ Challenger             │     │
│     ├───────────────────────────────────────────────────────────────────    │
│     │ No challenge     │ Full reward        │ N/A                    │     │
│     │ Valid proof      │ Reward + 50% stake │ Loses stake            │     │
│     │ Fraud detected   │ Loses 2× stake     │ Gets prover stake      │     │
│     │ Prover timeout   │ Loses 2× stake     │ Gets prover stake      │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  GAME THEORY:                                                               │
│  ────────────                                                               │
│  • Honest provers: almost always get rewards (challenges rare)              │
│  • Random challengers: lose stake (can't profit from random checks)         │
│  • Targeted challengers: profitable if they detect real fraud               │
│  • Cheaters: expected value negative (high probability of detection)        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Proof Structure

```python
@dataclass
class PipelineProof:
    """Proof of work for a single batch in pipeline."""
    
    # Identity
    node_id: str
    quorum_id: str
    layer_range: Tuple[int, int]
    
    # Work reference
    batch_id: str
    step_id: int
    
    # Commitments (hashes, not full data)
    input_activation_hash: bytes
    output_activation_hash: bytes
    gradient_merkle_root: bytes
    
    # Chain links
    prev_node_id: str  # For verification continuity
    next_node_id: str
    
    # Metadata
    timestamp: float
    signature: bytes


@dataclass  
class CohortSyncProof:
    """Proof of participation in cohort sync."""
    
    node_id: str
    layer_range: Tuple[int, int]
    sync_round: int
    
    # Work done
    batches_processed: int
    pseudo_gradient_hash: bytes
    
    # Aggregation participation
    cohort_members: List[str]
    aggregated_gradient_hash: bytes
    
    timestamp: float
    signature: bytes
```

### 10.3 Adaptive Verification Rate

```python
def get_verification_probability(network_size: int) -> float:
    """Spot-check probability, scales with network size."""
    
    if network_size < 20:
        return 0.0  # Too small, trust everyone
    elif network_size < 100:
        return 0.01  # 1% spot-check
    elif network_size < 500:
        return 0.03  # 3% spot-check
    else:
        return 0.05  # 5% spot-check (full adversarial mode)
```

---

## 11. Reward System

### 11.1 Training Rewards

Training is the **dominant** reward activity. More layers = more work = more reward.

```python
# Constants (from neuroshard/core/economics/constants.py)
TRAINING_REWARD_PER_BATCH_PER_LAYER = 0.0005  # Pipeline training
ASYNC_TRAINING_REWARD_PER_BATCH_PER_LAYER = 0.0003  # Async gradient submission

# Role bonuses (ADDITIVE - sum together, not multiply!)
INITIATOR_BONUS = 0.2   # +20% for holding embedding layer
FINISHER_BONUS = 0.3    # +30% for holding LM head
TRAINING_BONUS = 0.1    # +10% when actively training

def calculate_training_reward(
    batches: int,
    layers_held: int,
    is_async: bool = False,
    age_seconds: float = 0.0,
) -> float:
    """
    Calculate training reward for a node.
    
    Args:
        batches: Number of training batches processed
        layers_held: Number of layers this node holds
        is_async: Whether this is an async contribution (vs pipeline)
        age_seconds: Age of contribution in seconds (for async freshness)
        
    Returns:
        Base training reward in NEURO (before multipliers)
        
    Examples:
        # Pipeline: 60 batches, 10 layers
        >>> calculate_training_reward(60, 10)
        0.3  # 60 × 0.0005 × 10 = 0.3 NEURO
        
        # Async, 2 hours old: 60 batches, 10 layers
        >>> calculate_training_reward(60, 10, is_async=True, age_seconds=7200)
        0.126  # 60 × 0.0003 × 10 × 0.7 = 0.126 NEURO
    """
    if layers_held <= 0:
        return 0.0
    
    if is_async:
        base_rate = ASYNC_TRAINING_REWARD_PER_BATCH_PER_LAYER
        freshness = calculate_async_freshness(age_seconds)
        return batches * base_rate * layers_held * freshness
    else:
        base_rate = TRAINING_REWARD_PER_BATCH_PER_LAYER
        return batches * base_rate * layers_held


def calculate_role_bonus(
    has_embedding: bool = False,
    has_lm_head: bool = False,
    is_training: bool = False,
) -> float:
    """
    Calculate the role bonus multiplier (ADDITIVE).
    
    Returns 1.0 + sum of applicable bonuses.
    
    Examples:
        >>> calculate_role_bonus()  # Plain processor
        1.0
        >>> calculate_role_bonus(has_embedding=True, has_lm_head=True, is_training=True)
        1.6  # 1.0 + 0.2 + 0.3 + 0.1 = 1.6x
    """
    return 1.0 + (INITIATOR_BONUS if has_embedding else 0) \
               + (FINISHER_BONUS if has_lm_head else 0) \
               + (TRAINING_BONUS if is_training else 0)
```

### 11.2 Async Contributor Rewards

Async gradients are less valuable because they may be stale.

```python
# Freshness decay - gradients lose value over time
ASYNC_FRESHNESS_DECAY = {
    3600: 1.0,      # < 1 hour: full value (100%)
    86400: 0.7,     # < 1 day: 70% value
    604800: 0.5,    # < 1 week: 50% value
    float('inf'): 0.3,  # > 1 week: 30% value (still has diversity value)
}

def calculate_async_freshness(age_seconds: float) -> float:
    """
    Calculate freshness multiplier for async gradient contributions.
    
    Examples:
        >>> calculate_async_freshness(1800)   # 30 minutes
        1.0
        >>> calculate_async_freshness(43200)  # 12 hours
        0.7
        >>> calculate_async_freshness(259200) # 3 days
        0.5
    """
    for threshold, freshness in sorted(ASYNC_FRESHNESS_DECAY.items()):
        if age_seconds < threshold:
            return freshness
    return 0.3
```

### 11.3 Scarcity-Based Incentives

```python
def update_scarcity_scores():
    """Update scarcity scores for all layers."""
    
    TARGET_REPLICAS = get_adaptive_target_replicas()
    
    for layer_id in range(network_state.num_layers):
        layer_info = dht.get(f"layer:{layer_id}")
        actual = layer_info.replica_count
        
        if actual >= TARGET_REPLICAS:
            scarcity = 0.0
        else:
            # Linear scarcity up to 1.0
            scarcity = (TARGET_REPLICAS - actual) / TARGET_REPLICAS
        
        # Boost for critical layers (embed, head)
        if layer_id == 0:
            scarcity *= 1.5
        if layer_id == network_state.num_layers - 1:
            scarcity *= 1.5
        
        layer_info.scarcity_score = min(scarcity, 1.0)
        dht.set(f"layer:{layer_id}", layer_info)
```

---

## 12. Adversarial Resistance

### 12.1 Defense Matrix

| Attack | Detection | Defense | Penalty |
|--------|-----------|---------|---------|
| **Lazy compute** | PoNW spot-checks | Merkle proofs, recomputation | 2× stake slash |
| **Slow griefing** | Latency monitoring | Speed tiers, quorum kick | Reputation loss |
| **Gradient poison** | Magnitude/direction checks | Robust aggregation | Gradient rejected |
| **Sybil attack** | IP/hardware analysis | Min stake, attestation | All nodes slashed |
| **Quorum collusion** | Cross-quorum verification | Random audits | Quorum dissolved |
| **Inference cheat** | Multi-quorum verification | Response comparison | Stake slash, blacklist |
| **Vocab attack** | Community review | Governance voting | Proposal rejected, slash |

### 12.2 Robust Aggregation

```python
def robust_aggregate(
    gradients: List[Tensor],
    weights: List[float],
    method: str = "trimmed_mean"
) -> Tensor:
    """Aggregate gradients robustly against poisoning."""
    
    if method == "trimmed_mean":
        # Sort by magnitude, trim top and bottom 20%
        trim = int(len(gradients) * 0.2)
        sorted_grads = sorted(gradients, key=lambda g: g.norm())
        trimmed = sorted_grads[trim:-trim] if trim > 0 else sorted_grads
        return weighted_mean(trimmed, weights[trim:-trim])
    
    elif method == "coordinate_median":
        # Take median at each coordinate
        stacked = torch.stack(gradients)
        return torch.median(stacked, dim=0).values
    
    elif method == "krum":
        # Select gradient closest to others
        n = len(gradients)
        f = int(n * 0.2)  # Assume up to 20% Byzantine
        
        scores = []
        for i, g in enumerate(gradients):
            distances = [torch.norm(g - other) for other in gradients]
            distances.sort()
            scores.append(sum(distances[:n - f - 1]))
        
        best_idx = scores.index(min(scores))
        return gradients[best_idx]
```

### 12.3 Cross-Quorum Verification

```python
async def cross_quorum_audit(target_quorum: Quorum) -> AuditResult:
    """Random verification of another quorum's work."""
    
    # Select random proof from target quorum
    proof = random.choice(target_quorum.recent_proofs)
    
    # Request verification data
    verif_data = await request_verification_data(
        target_quorum,
        proof.batch_id,
        gradient_indices=random.sample(range(num_params), k=10)
    )
    
    # Recompute forward pass
    my_output = forward_layers(
        verif_data.input_activation,
        proof.layer_range
    )
    
    if hash(my_output) != proof.output_activation_hash:
        return AuditResult.FRAUD_DETECTED
    
    # Verify gradient samples
    for idx, sample in zip(verif_data.indices, verif_data.gradient_samples):
        if not verify_merkle_proof(sample, idx, proof.gradient_merkle_root):
            return AuditResult.FRAUD_DETECTED
    
    return AuditResult.VALID
```

---

## 13. Scale Adaptation

### 13.1 Adaptive Parameters

The same protocol works at all scales with adaptive parameters:

```python
def get_adaptive_params(network_size: int) -> AdaptiveParams:
    """Get parameters adapted to current network size."""
    
    return AdaptiveParams(
        # Quorum requirements
        min_quorum_size=1 if network_size < 5 else (2 if network_size < 20 else 3),
        
        # Sync frequency (more nodes → less frequent sync needed)
        cohort_sync_interval=100 if network_size < 10 else (500 if network_size < 100 else 1000),
        
        # Verification intensity
        challenge_probability=0.0 if network_size < 20 else (0.01 if network_size < 100 else 0.05),
        
        # Layer coverage targets
        target_replicas=1 if network_size < 5 else (2 if network_size < 20 else 3),
        
        # Growth threshold (more conservative as network grows)
        layer_growth_threshold=1.5 + 0.1 * math.log2(max(1, network_size)),
        
        # Session duration (longer for stable networks)
        quorum_session_duration=1800 if network_size < 20 else 3600,
    )
```

### 13.2 Network Phases

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NETWORK PHASE BEHAVIOR                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────┬─────────────────┬────────────────────────────────────────────┐   │
│  │ NODES │ PHASE           │ EMERGENT BEHAVIOR                          │   │
│  ├───────┼─────────────────┼────────────────────────────────────────────┤   │
│  │   1   │ GENESIS         │ Solo training, small model                 │   │
│  │       │                 │ Only time solo mode is allowed             │   │
│  ├───────┼─────────────────┼────────────────────────────────────────────┤   │
│  │  2-4  │ MICRO           │ Single quorum, all nodes know each other   │   │
│  │       │                 │ No cross-quorum sync (only 1 quorum)       │   │
│  ├───────┼─────────────────┼────────────────────────────────────────────┤   │
│  │  5-19 │ SMALL           │ 1-3 quorums, speed matching begins         │   │
│  │       │                 │ First layer growth possible                │   │
│  ├───────┼─────────────────┼────────────────────────────────────────────┤   │
│  │ 20-99 │ MEDIUM          │ Multiple quorums at different tiers        │   │
│  │       │                 │ Async contributors welcome                 │   │
│  │       │                 │ Full DiLoCo cross-quorum sync              │   │
│  ├───────┼─────────────────┼────────────────────────────────────────────┤   │
│  │100-499│ GROWING         │ Adversarial resistance fully active        │   │
│  │       │                 │ Governance proposals common                │   │
│  │       │                 │ Model grows significantly                  │   │
│  ├───────┼─────────────────┼────────────────────────────────────────────┤   │
│  │ 500+  │ LARGE           │ Regional clustering emerges                │   │
│  │       │                 │ Specialized quorums (train vs inference)   │   │
│  │       │                 │ Model very large (100B+ params)            │   │
│  ├───────┼─────────────────┼────────────────────────────────────────────┤   │
│  │ 5000+ │ MASSIVE         │ Hierarchical coordination possible         │   │
│  │       │                 │ Model approaching frontier scale           │   │
│  │       │                 │ Full decentralized AI infrastructure       │   │
│  └───────┴─────────────────┴────────────────────────────────────────────┘   │
│                                                                             │
│  KEY: Same code at all phases. Behavior emerges from conditions.            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 13.3 Mode Selection Logic

```python
def decide_contribution_mode(node: Node) -> ContributionMode:
    """Automatically decide best mode based on conditions."""
    
    network_size = dht.get("network:state").total_nodes
    
    # Genesis: solo training allowed
    if network_size == 1 and node.holds_all_layers():
        return ContributionMode.SOLO
    
    # T5 nodes: async only (too slow for pipeline)
    if node.speed_tier == SpeedTier.T5:
        return ContributionMode.ASYNC
    
    # Try to join existing quorum
    quorum = find_compatible_quorum(node)
    if quorum:
        return ContributionMode.PIPELINE
    
    # Try to form new quorum
    quorum = form_new_quorum(node)
    if quorum:
        return ContributionMode.PIPELINE
    
    # No quorum possible, check other options
    if network_needs_verifiers() and node.has_stake:
        return ContributionMode.VERIFY
    
    if network_needs_data_providers() and node.has_genesis_data:
        return ContributionMode.DATA
    
    # Default: async contribution
    return ContributionMode.ASYNC
```

---

## 14. Data Structures

### 14.1 Core Types

```python
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from enum import Enum

class SpeedTier(Enum):
    T1 = "tier1"  # < 10ms/layer
    T2 = "tier2"  # 10-50ms/layer
    T3 = "tier3"  # 50-200ms/layer
    T4 = "tier4"  # 200-1000ms/layer
    T5 = "tier5"  # > 1000ms/layer

class ContributionMode(Enum):
    SOLO = "solo"            # Genesis only
    PIPELINE = "pipeline"    # Quorum member
    ASYNC = "async"          # Offline training
    DATA = "data"            # Data provider
    VERIFY = "verify"        # Proof verifier
    INFERENCE = "inference"  # Inference only
    IDLE = "idle"            # Available

@dataclass
class NetworkState:
    arch_version: int
    num_layers: int
    hidden_dim: int
    vocab_version: int
    vocab_size: int
    total_nodes: int
    active_quorums: int
    current_step: int

@dataclass
class NodeCapabilities:
    node_id: str
    endpoint: str
    speed_tier: SpeedTier
    memory_mb: int
    layer_range: Tuple[int, int]
    mode: ContributionMode
    quorum_id: Optional[str]
    stake: float
    reputation: float

@dataclass
class Quorum:
    quorum_id: str
    speed_tier: SpeedTier
    members: List[str]
    layer_map: Dict[str, Tuple[int, int]]
    session_start: float
    session_end: float
    throughput: float
    status: str

@dataclass
class ActivationPacket:
    arch_version: int
    vocab_version: int
    quorum_id: str
    batch_id: str
    step_id: int
    current_layer: int
    activations: bytes
    labels: Optional[bytes]
    sender_id: str
    timestamp: float

@dataclass
class GradientContribution:
    node_id: str
    layer_range: Tuple[int, int]
    gradient: Dict[str, bytes]
    batches: int
    timestamp: float
    signature: bytes
```

---

## 15. Implementation Plan

### Phase 1: Foundation (Week 1-2)

- [ ] Add versioning to `DynamicLayerPool` (arch_version, vocab_version)
- [ ] Update protobuf with version fields
- [ ] Implement version compatibility checks
- [ ] Add SpeedTier benchmarking
- [ ] Implement ContributionMode selection

### Phase 2: Quorum System (Week 2-3)

- [ ] Implement quorum formation algorithm
- [ ] Implement quorum lifecycle (formation, active, renewal, dissolution)
- [ ] Add quorum registry to DHT
- [ ] Implement within-quorum pipeline training
- [ ] Add quorum health monitoring

### Phase 3: Cross-Quorum Sync (Week 3-4)

- [ ] Implement cohort discovery
- [ ] Implement weighted gradient aggregation
- [ ] Add robust aggregation methods
- [ ] Implement async contributor flow
- [ ] Add freshness weighting

### Phase 4: Layer Growth (Week 4-5)

- [ ] Implement adaptive growth triggers
- [ ] Implement layer addition sequence
- [ ] Add identity initialization
- [ ] Implement warmup handling
- [ ] Test growth with multiple quorums

### Phase 5: Vocab Governance (Week 5-6)

- [ ] Implement proposal submission
- [ ] Implement discussion/voting periods
- [ ] Add automatic merge validation
- [ ] Implement embedding/LM head expansion
- [ ] Add emergency rollback

### Phase 6: Inference & Rewards (Week 6-7)

- [ ] Implement inference routing
- [ ] Add dynamic pricing
- [ ] Implement payment escrow
- [ ] Add training reward calculation
- [ ] Implement scarcity incentives

### Phase 7: Adversarial Resistance (Week 7-8)

- [ ] Implement PoNW proof generation
- [ ] Add optimistic verification
- [ ] Implement challenge protocol
- [ ] Add cross-quorum auditing
- [ ] Implement stake slashing

### Phase 8: Testing & Hardening (Week 8-10)

- [ ] Unit tests for all components
- [ ] Integration tests (multi-quorum scenarios)
- [ ] Adversarial simulation tests
- [ ] Scale testing (100+ simulated nodes)
- [ ] Performance optimization

**Total Estimated Time: 10 weeks**

---

## Appendix: Constants

```python
# Speed tier thresholds (ms per layer)
SPEED_TIER_THRESHOLDS = {
    SpeedTier.T1: 10,
    SpeedTier.T2: 50,
    SpeedTier.T3: 200,
    SpeedTier.T4: 1000,
    SpeedTier.T5: float('inf')
}

# Quorum settings
BASE_SESSION_DURATION = 3600      # 1 hour
MAX_SESSION_DURATION = 14400      # 4 hours
RENEWAL_CHECK_RATIO = 0.8         # Check at 80% of session

# Sync settings
BASE_SYNC_INTERVAL = 500          # batches
OUTER_LR = 0.7                    # DiLoCo outer learning rate

# Rewards (per-layer rates)
TRAINING_REWARD_PER_BATCH_PER_LAYER = 0.0005  # Pipeline training
ASYNC_TRAINING_REWARD_PER_BATCH_PER_LAYER = 0.0003  # Async (40% less)
BASE_INFERENCE_PRICE = 0.0001     # NEURO per token (market-driven)

# Role bonuses (ADDITIVE)
INITIATOR_BONUS = 0.2             # +20% for embedding
FINISHER_BONUS = 0.3              # +30% for LM head
TRAINING_BONUS = 0.1              # +10% when training

# Uptime (minimal, discourages idle farming)
UPTIME_REWARD_PER_MINUTE = 0.0001 # ~0.14 NEURO/day idle

# Verification
BASE_CHALLENGE_WINDOW = 600       # 10 minutes
MIN_CHALLENGE_STAKE = 10          # NEURO
SLASH_MULTIPLIER = 2.0            # Lose 2× stake on fraud

# Governance
PROPOSAL_MIN_STAKE = 100          # NEURO to propose
VOTING_PERIOD = 604800            # 7 days
APPROVAL_THRESHOLD = 0.66         # 66% to pass
QUORUM_REQUIREMENT = 0.30         # 30% must vote

# Heartbeat
HEARTBEAT_INTERVAL = 30           # seconds
STALE_THRESHOLD = 4               # missed beats
OFFLINE_THRESHOLD = 120           # seconds
```

---

*This document is a living specification. The architecture is designed to be fully dynamic and self-adapting.*
